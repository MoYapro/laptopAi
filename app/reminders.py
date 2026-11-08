"""Reminders: the conversation model sets them with a tool; when one is due, the model phrases it for the
moment and it is spoken aloud, also to an empty room and in quiet mode (it was asked for).

The model gives a delay in minutes or a clock time; the code computes the due time (a small model gets
date arithmetic wrong). Reminders live in the database, so they survive restarts; one that came due
while the system was off is still said if it is less than MISSED_AFTER_S late.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from util import ago, fmt_dur, hhmm, split_sentences

if TYPE_CHECKING:
    from agent import Agent

log = logging.getLogger("reminders")

CHECK_S = 10
MISSED_AFTER_S = 6 * 3600

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": (
                "Remind someone of something later: you will say it aloud when it is due. Give either in_minutes "
                "(e.g. 20 for 'in 20 minutes', 90 for 'in an hour and a half') or at (a clock time)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "what to remind of, e.g. 'take the pasta off the stove'"},
                    "in_minutes": {"type": "number", "description": "minutes from now"},
                    "at": {"type": "string", "description": "24-hour clock time 'HH:MM', e.g. '18:30'; the next time it comes"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Cancel one of the pending reminders listed in your situation (by its #number).",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "integer", "description": "the reminder's #number"}},
                "required": ["id"],
            },
        },
    },
]
NAMES = {t["function"]["name"] for t in TOOLS}

# The small model sometimes answers "Okay, I'll remind you" without calling the tool; a message like that
# gets a second, JSON-constrained look (CATCH_PROMPT).
MENTIONS_REMINDER = re.compile(r"\b(remind|reminder|erinner)", re.I)
CATCH_SCHEMA = {
    "type": "object",
    "properties": {  # keys are written alphabetically: quote the request first, then decide
        "a_request": {"type": "string"},
        "b_action": {"type": "string", "enum": ["set", "cancel", "none"]},
        "c_text": {"type": "string"},
        "d_in_minutes": {"type": "number"},
        "e_at": {"type": "string"},
        "f_id": {"type": "integer"},
    },
    "required": ["a_request", "b_action", "c_text", "d_in_minutes", "e_at", "f_id"],
}
CATCH_PROMPT = """It is {now}. Someone said to Housemate, a home AI:
"{said}"
Housemate answered: "{reply}"
Housemate forgot to store anything, so whatever they asked for is NOT set up yet.
Reminders already stored: {pending}

Return JSON:
- a_request: the words in which they ask to be reminded of something or to cancel a reminder, copied; "" if they don't
- b_action: "set" if they ask to be reminded of something, "cancel" if they want a stored reminder removed, "none" if
  a_request is ""
- c_text: for "set", what to remind of (e.g. "take the pasta off the stove"); otherwise ""
- d_in_minutes: for "set" with a delay, the minutes from now (half an hour = 30); otherwise 0
- e_at: for "set" with a clock time, "HH:MM" (24-hour); otherwise ""
- f_id: for "cancel", the stored reminder's number; otherwise 0"""

DUE_PROMPT = """[Internal check, not said by anyone. A reminder is due now.]
At {set_at} {who} asked you to remind them: "{text}". It was due at {due_at}{late}.
Say the reminder now, out loud, in one short sentence (two at most): address them by name if you know who asked,
and say plainly what it is about. No small talk or guesses about how things are going. Just the words to say."""


def due_time(args: dict, now: float) -> float | None:
    at = str(args.get("at") or "").strip()
    if m := re.fullmatch(r"(\d{1,2})[:.](\d{2})", at):
        hour, minute = int(m.group(1)), int(m.group(2))
        if hour > 23 or minute > 59:
            return None
        base = datetime.fromtimestamp(now)
        due = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if due.timestamp() <= now:
            due += timedelta(days=1)
        return due.timestamp()
    try:
        minutes = float(args.get("in_minutes"))
    except (TypeError, ValueError):
        return None
    return now + minutes * 60 if 0 < minutes <= 7 * 24 * 60 else None


class Reminders:
    def __init__(self, agent: "Agent"):
        self.agent, self.db = agent, agent.db
        self.last_call = 0.0  # when the model last called one of the tools

    async def catch_missed(self, said: str, reply: str) -> None:
        """A reply that talked about a reminder without calling the tool: ask again, constrained to JSON."""
        pending = self.db.pending_reminders()
        prompt = CATCH_PROMPT.format(
            now=datetime.now().strftime("%A %H:%M"), said=said, reply=reply,
            pending="; ".join(f"#{r['id']} at {hhmm(r['due'])}: {r['text']}" for r in pending) or "none",
        )
        async with self.agent.lock:
            raw = await self.agent.llm.complete([{"role": "user", "content": prompt}], schema=CATCH_SCHEMA,
                                                max_tokens=150, temperature=0)
        result = json.loads(raw)
        action = result.get("b_action")
        if action in ("set", "cancel"):
            name = "set_reminder" if action == "set" else "cancel_reminder"
            args = {"text": result.get("c_text"), "in_minutes": result.get("d_in_minutes"), "at": result.get("e_at"),
                    "id": result.get("f_id")}
            outcome = await self.run_tool(name, args)
            log.info("reminder the reply missed: %s %s -> %s", name, result, outcome)

    # ---------------------------------------------------------------- tools

    async def run_tool(self, name: str, args: dict) -> dict:
        now = self.last_call = time.time()
        if name == "cancel_reminder":
            r = self.db.reminder(int(args.get("id") or 0))
            if not r or r["status"] != "pending":
                return {"ok": False, "reason": "no pending reminder with that number"}
            self.db.set_reminder_status(r["id"], "cancelled")
            await self.agent.emit("reminder", f'reminder #{r["id"]} cancelled: {r["text"]}')
            return {"ok": True, "cancelled": r["text"]}

        text = str(args.get("text") or "").strip()
        due = due_time(args, now)
        if not text or due is None:
            return {"ok": False, "reason": "need the text and either in_minutes (more than 0) or at ('HH:MM')"}
        who = ", ".join(self.agent.talking_to()) or None
        r = self.db.add_reminder(due, text, who)
        when = f"{hhmm(due)}{' tomorrow' if datetime.fromtimestamp(due).date() > datetime.fromtimestamp(now).date() else ''}"
        await self.agent.emit("reminder", f"reminder #{r} set for {when} (in {fmt_dur(due - now)}): {text}",
                              {"id": r, "due": due, "who": who})
        return {"ok": True, "id": r, "due": f"{when}, in {fmt_dur(due - now)}"}

    def context_lines(self) -> list[str]:
        pending = self.db.pending_reminders()
        if not pending:
            return []
        now = time.time()
        return ["Reminders you set (you will say them when due; cancel_reminder removes one):", *[
            f"- #{r['id']} at {hhmm(r['due'])} (in {fmt_dur(max(0, r['due'] - now))}) for {r['who'] or 'someone'}: {r['text']}"
            for r in pending
        ]]

    # ---------------------------------------------------------------- firing

    async def loop(self) -> None:
        while True:
            await asyncio.sleep(CHECK_S)
            for r in self.db.pending_reminders(due_before=time.time()):
                try:
                    await self.fire(r)
                except Exception:
                    log.exception("reminder #%s failed, saying it plainly", r["id"])
                    self.db.set_reminder_status(r["id"], "failed")
                    try:
                        await self.say(r, self.plain(r))
                    except Exception:
                        log.exception("reminder #%s could not be said", r["id"])

    async def fire(self, r: dict) -> None:
        now = time.time()
        late = now - r["due"]
        if late > MISSED_AFTER_S:  # the system was off for long
            self.db.set_reminder_status(r["id"], "missed")
            await self.agent.emit("reminder", f"missed reminder #{r['id']} (due {ago(r['due'])}, the system was off): "
                                  f"{r['text']}")
            return
        self.db.set_reminder_status(r["id"], "done")  # first: a crash while speaking must not repeat it forever
        prompt = DUE_PROMPT.format(
            set_at=hhmm(r["ts"]), who=r["who"] or "someone", text=r["text"], due_at=hhmm(r["due"]),
            late=f", so it is {fmt_dur(late)} late (the system was off)" if late > 120 else "",
        )
        text = ""
        try:
            async with self.agent.lock:
                messages = [{"role": "system", "content": self.agent.system_prompt(query=r["text"])},
                            *self.agent.history(), {"role": "user", "content": prompt}]
                text = (await self.agent.llm.complete(messages, max_tokens=120, temperature=0.6)).strip()
        except Exception:
            log.exception("phrasing reminder #%s failed", r["id"])
        await self.say(r, text if text and not text.startswith("{") else self.plain(r))

    @staticmethod
    def plain(r: dict) -> str:
        return f"{r['who'] + ', a' if r['who'] else 'A'} reminder: {r['text']}."

    async def say(self, r: dict, text: str) -> None:
        await self.agent.emit("reminder", f'reminder #{r["id"]} said: "{text}"', {"id": r["id"], "text": r["text"]})
        self.db.add_message("assistant", text, "reminder", self.agent.present_names())
        await self.agent.hub.broadcast({"type": "assistant", "text": text, "source": "reminder", "ts": time.time()})
        for sentence in split_sentences(text):
            await self.agent.say_audio(sentence)


def parse_args(call: dict) -> dict:
    args = call.get("function", {}).get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}
