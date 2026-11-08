"""Curiosity: learn about the surroundings instead of just watching them.

- reflect():  every few minutes, turn recent scene descriptions into observations ("A cat lives here.") and
              open questions ("What's the cat's name?"). Observations are short-term notes: single camera
              images are often momentary or misread, so only the memory review (consolidation.py) turns notes
              that recur at different times into lasting facts; the rest expire.
- maybe_ask(): when someone is in view and it has been quiet for a while, give the initiative
              step a chance to ask one of those questions.
- answers are picked up by the agent's memory extraction after each reply (see Agent.extract_memories).
"""

import json
import logging
import time
from typing import TYPE_CHECKING

import config
from util import hhmm, word_similarity

if TYPE_CHECKING:
    from agent import Agent

log = logging.getLogger("curiosity")

REFLECT_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["observations", "questions"],
}

REFLECT_PROMPT = """Here is what your webcam noticed recently:
{events}

People you know by name: {names}
What you already know about this place: {known}
Questions you already have or asked before: {questions}

Return JSON:
- observations: up to 3 lasting facts about this place, its things or the habits of the people in it that you can reasonably conclude from what you saw, as short sentences (e.g. "A cat lives here.", "There is a sim-racing steering wheel at the desk.", "Tom often wears a headset at the desk."). Not momentary things (someone sitting, looking down, touching their face, a thing that moved), not guesses about visitors or what didn't happen. Nothing you already know. Often empty.
- questions: up to 2 short, friendly questions you would genuinely like to ask the people here to understand your surroundings and their lives better (e.g. "What's the cat's name?", "What do you use the steering wheel for?"). Only things a person in the room could answer. Not ones you already have or know. Often empty."""


class Curiosity:
    def __init__(self, agent: "Agent"):
        self.agent, self.db = agent, agent.db
        self.last_reflect = time.time()  # first reflection only after REFLECT_INTERVAL_S
        self.reflected_until = float(self.db.get_setting("reflected_until", "0"))
        self._last_ask_check = 0.0

    # -------------------------------------------------------------- questions

    def askable(self, limit: int = 4) -> list[dict]:
        """Open questions that may be asked now (not asked too often or too recently)."""
        now = time.time()
        return [
            q for q in self.db.open_questions()
            if q["asked_count"] < config.QUESTION_MAX_ASKS
            and (not q["last_asked"] or now - q["last_asked"] > config.QUESTION_REASK_S)
        ][:limit]

    def context_lines(self) -> list[str]:
        qs = self.askable()
        if not qs:
            return []
        return [
            "Things you are curious about and would like to find out (ask when it fits naturally, one at a time):",
            *[f"- #{q['id']} {q['text']}" for q in qs],
        ]

    def public(self) -> list[dict]:
        return [{"id": q["id"], "text": q["text"], "asked": q["asked_count"]} for q in self.db.open_questions(20)]

    async def broadcast(self) -> None:
        await self.agent.hub.broadcast({"type": "questions", "questions": self.public()})

    async def asked(self, qid: int) -> None:
        if self.db.question(qid):
            self.db.mark_asked(qid)
            await self.broadcast()

    async def answered(self, qid: int, answer: str) -> None:
        q = self.db.question(qid)
        if not q or q["status"] != "open" or not answer.strip():
            return
        answer = answer.strip()
        self.db.set_question_status(qid, "answered", answer)
        # "Tom." means nothing on its own later; keep the question with short answers.
        self.db.add_memory(answer if len(answer.split()) >= 4 else f"{q['text']} {answer}", None)
        await self.agent.emit("learned", f"{q['text']} → {answer}", {"question_id": qid})
        await self.broadcast()

    # ------------------------------------------------------------- reflection

    async def maybe_reflect(self) -> None:
        if self.agent.user_busy or time.time() - self.last_reflect < config.REFLECT_INTERVAL_S:
            return
        events = [
            e for e in self.db.recent_events(30, kinds=["scene", "person_arrived", "person_left"])
            if e["ts"] > self.reflected_until
        ]
        if sum(e["kind"] == "scene" for e in events) < 2:
            return
        self.last_reflect = time.time()
        await self.reflect(events)

    async def reflect(self, events: list[dict]) -> None:
        names = sorted({p["name"] for p in self.db.people()}) or ["nobody yet"]
        known = [m["text"] for m in self.db.memories(None, 30)]
        previous = self.db.all_questions(60)
        prompt = REFLECT_PROMPT.format(
            events="\n".join(f"- {hhmm(e['ts'])} {e['text']}" for e in events),
            names=", ".join(names),
            known="; ".join(known) or "nothing yet",
            questions="; ".join(q["text"] for q in previous) or "none",
        )
        async with self.agent.lock:
            raw = await self.agent.llm.complete(
                [{"role": "system", "content": "You are the reflective, curious mind of a home AI that watches a room."},
                 {"role": "user", "content": prompt}],
                schema=REFLECT_SCHEMA, max_tokens=300, temperature=0.4,
            )
        result = json.loads(raw)
        self.reflected_until = events[-1]["ts"]
        self.db.set_setting("reflected_until", str(self.reflected_until))

        # Repeats of a note at different times are what the memory review needs as evidence, so known notes
        # are only skipped when they were noted in the last half hour.
        recent = [n["text"] for n in self.db.notes(since=time.time() - 1800)]
        for obs in [o.strip() for o in result.get("observations", [])][:3]:
            if (obs and not any(word_similarity(obs, k) >= 0.6 for k in known + recent)
                    and self.db.add_note(obs)):
                recent.append(obs)
                await self.agent.emit("memory", f"noticed (short-term note): {obs}")

        added = False
        for text in [q.strip() for q in result.get("questions", [])][:2]:
            if not text or any(word_similarity(text, q["text"]) >= 0.5 for q in previous):
                continue
            q = self.db.add_question(text)
            previous.append(q)
            added = True
            await self.agent.emit("curious", f"wondering: {text}", {"question_id": q["id"]})

        # keep the list short: drop the oldest open questions beyond the limit
        for q in self.db.open_questions()[config.MAX_OPEN_QUESTIONS:]:
            self.db.set_question_status(q["id"], "dropped")
        if added:
            await self.broadcast()

    # ------------------------------------------------------------------ asking

    async def maybe_ask(self) -> None:
        """Someone is here, it's been quiet, and there's something to find out: consider asking."""
        now = time.time()
        if now - self._last_ask_check < 60 or self.agent.user_busy:
            return
        self._last_ask_check = now
        if not self.agent.presence.present() or not self.askable():
            return
        last = max(
            self.db.last_message_ts() or 0,
            (self.db.last_event(["initiative", "initiative_quiet"]) or {}).get("ts", 0),
        )
        if now - last < config.CURIOSITY_IDLE_S:
            return
        await self.agent.consider_initiative(
            "someone is here and it has been quiet for a while; a good moment to ask about something you are curious about"
        )
