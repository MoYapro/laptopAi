"""The agent: conversation, initiative, scene descriptions, memory and lifecycle.

All model calls go through one lock, since there is one GPU and conversation
should never queue behind more than a single background job.
"""

import asyncio
import base64
import json
import logging
import re
import time

import config
from db import DB
from curiosity import Curiosity
from faces import Presence, Track
from llm import Ollama
from speech import TTS
from util import ago, clean_for_speech, fmt_dur, fmt_ts, hhmm, pop_sentence, split_sentences

log = logging.getLogger("agent")


class Hub:
    """Fan-out to every connected page."""

    def __init__(self):
        self.clients: set = set()

    async def broadcast(self, msg: dict) -> None:
        for client in list(self.clients):
            try:
                await client.send(msg)
            except Exception:
                self.clients.discard(client)


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "action": {"type": "string", "enum": ["speak", "none"]},
        "text": {"type": "string"},
        "question_id": {"type": "integer"},
    },
    "required": ["reason", "action", "text", "question_id"],
}

DECISION_PROMPT = """[Internal check, not said by anyone. Trigger: {trigger}]
Decide whether to say something to the people in the room right now, on your own initiative.
You are curious about your surroundings and the people in them, and you like to learn. Good reasons to speak:
- greeting someone who just arrived (by name if you know them; you may bring up something you remember about them)
- welcoming someone you don't know and asking their name
- asking about something new or interesting you notice, or one of the things you are curious about (listed with #numbers in your situation)
- after a restart, a brief hello to whoever is there
Ask at most one question at a time, and keep it light and friendly; people may be busy.
Stay silent if nobody is there to hear you, if you would repeat something you said recently, or if you have nothing worthwhile to add. Keep it short (one or two sentences) and natural.
Answer as JSON: reason (short, for your log), action ("speak" or "none"), text (what to say; empty for none), question_id (the #number if you ask one of your listed questions, otherwise 0)."""

MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "introduced_name": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"about": {"type": "string"}, "fact": {"type": "string"}},
                "required": ["about", "fact"],
            },
        },
        "answered": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "answer": {"type": "string"}},
                "required": ["id", "answer"],
            },
        },
    },
    "required": ["introduced_name", "facts", "answered"],
}


# Where an incoming message came from. Every user-role message reaches the model with this header,
# so it can tell people in the room from developer tools and from its own internal checks.
# (A plain "Message from X (...):" line works far more reliably with small models than a [bracket tag].)
SOURCES = {
    "voice": "Spoken aloud by someone in the room (heard through the microphone; speech recognition may mishear words)",
    "text": "Typed on the laptop's screen by someone in the room",
    "developer": "Message from {sender} (a developer tool used to build and test you, not a person in the room; "
    "nothing was said aloud)",
}


def tag_message(content: str, source: str, sender: str | None = None, image: bool = False) -> str:
    header = SOURCES.get(source, f"Message via {source}").format(sender=sender or "an unnamed developer tool")
    if image:
        header += ". Attached: a camera image of the room right now"
    return f"{header}:\n{content}"


STOPWORDS = set("""a an the is are was were be been it its it's this that these those he she they his her their
him them i you we my your our of to in on at for with and or but not no do does did what what's who whose where
when why how which name names called user answer answered question""".split())


def answer_supported(answer: str, question: str, user_text: str) -> bool:
    """Small models happily 'answer' questions nobody answered. Accept an answer only if the new
    information in it (words not already in the question) actually appears in what the person said."""
    words = lambda s: set(re.findall(r"[a-zäöüß0-9]+", s.lower()))  # noqa: E731
    novel = words(answer) - words(question) - STOPWORDS
    return bool(novel & words(user_text))


class Agent:
    def __init__(self, db: DB, llm: Ollama, tts: TTS, presence: Presence, hub: Hub, base_prompt):
        self.db, self.llm, self.tts, self.presence, self.hub = db, llm, tts, presence, hub
        self.base_prompt = base_prompt  # callable -> str, re-read every turn
        self.lock = asyncio.Lock()
        self.user_busy = False
        self.latest_scene: dict | None = db.last_event(["scene"])
        self.claimed: tuple[int, float] | None = None  # (person_id, ts) of a name said in conversation
        self.curiosity = Curiosity(self)

    # ---------------------------------------------------------------- state

    @property
    def quiet(self) -> bool:
        return self.db.get_setting("quiet", "1" if config.QUIET_DEFAULT else "0") == "1"

    async def set_quiet(self, quiet: bool) -> None:
        self.db.set_setting("quiet", "1" if quiet else "0")
        await self.hub.broadcast({"type": "settings", "quiet": quiet})
        await self.emit("setting", f"quiet mode {'on' if quiet else 'off'}")

    async def emit(self, kind: str, text: str, data: dict | None = None, ts: float | None = None) -> dict:
        event = self.db.add_event(kind, text, data, ts)
        log.info("[%s] %s", kind, text)
        await self.hub.broadcast({"type": "event", "event": event})
        return event

    async def broadcast_presence(self) -> None:
        await self.hub.broadcast({"type": "presence", "people": [t.public() for t in self.presence.present()]})

    def history(self) -> list[dict]:
        msgs = self.db.recent_messages(config.HISTORY_TURNS * 2, since=time.time() - config.HISTORY_MAX_AGE_S)
        return [
            {
                "role": m["role"],
                "content": tag_message(m["content"], m["source"], m["sender"]) if m["role"] == "user" else m["content"],
            }
            for m in msgs
        ]

    def present_names(self) -> list[str]:
        return [t.name for t in self.presence.present() if t.name]

    # -------------------------------------------------------------- context

    def system_prompt(self, answering: str | None = None) -> str:
        """answering: where the message being answered came from; restated last, where small models attend most."""
        prompt = (
            self.base_prompt().strip()
            + "\n\n# Current situation\n"
            + self.context()
            + "\n\nReminder: the facts above and this conversation are everything you know. "
            "Do not invent past meetings, dates, or things you supposedly saw or did. Reply in plain spoken words."
        )
        if answering:
            prompt += f"\nThe message you are answering now: {answering}."
        return prompt

    def context(self) -> str:
        now = time.time()
        lines = [f"Current time: {fmt_ts(now)}."]

        present = self.presence.present()
        if present:
            lines.append("People you can see right now:")
            for t in present:
                if t.person_id is None:
                    lines.append("- someone you don't know yet (you could ask their name)")
                    continue
                visit = f"in view for {fmt_dur(now - t.first_seen)}"
                if t.prev_seen:
                    visit += f", before that last seen {ago(t.prev_seen)}"
                else:
                    visit += ", first time you see them"
                facts = self.db.memories(t.person_id, 8)
                known = "; ".join(f["text"] for f in facts) or "nothing yet"
                lines.append(f"- {t.name} ({visit}). What you remember about {t.name}: {known}")
        else:
            lines.append("You don't recognize any face right now (the room may be empty, or faces are turned away).")

        # Someone who told us their name recently, but whose face we can't link (yet).
        if self.claimed and now - self.claimed[1] < 1800:
            pid = self.claimed[0]
            if pid not in {t.person_id for t in present}:
                person = self.db.person(pid)
                facts = self.db.memories(pid, 8)
                known = "; ".join(f["text"] for f in facts) or "nothing yet"
                lines.append(
                    f"The person you are talking to said they are {person['name']} "
                    f"(face not recognized). What you remember about {person['name']}: {known}"
                )

        if household := self.db.memories(None, 10):
            lines.append("What you know about this place: " + "; ".join(f["text"] for f in household))
        lines += self.curiosity.context_lines()

        if self.latest_scene:
            lines.append(f"Latest camera description ({ago(self.latest_scene['ts'])}): {self.latest_scene['text']}")

        timeline = self.db.recent_events(15, exclude=["decision", "memory", "setting", "initiative_quiet"])
        if timeline:
            lines.append("Recent timeline (oldest first):")
            lines += [f"- {hhmm(e['ts'])} {e['text']}" for e in timeline]

        last_msg = self.db.last_message_ts()
        if last_msg and now - last_msg > 1800:
            lines.append(f"Your last conversation was {ago(last_msg)}.")
        return "\n".join(lines)

    # --------------------------------------------------------- conversation

    async def reply(
        self, text: str, source: str, image_jpg: bytes | None, speak: bool, stt_ms, sender: str | None = None
    ) -> None:
        t0 = time.perf_counter()
        ms = lambda: round((time.perf_counter() - t0) * 1000)  # noqa: E731
        timings: dict = {"stt_ms": stt_ms}
        names = self.present_names()
        speaker = None

        self.user_busy = True
        try:
            await self.hub.broadcast(
                {"type": "user", "text": text, "source": source, "sender": sender, "ts": time.time()}
            )
            await self.hub.broadcast({"type": "state", "state": "thinking"})
            user_msg: dict = {"role": "user", "content": tag_message(text, source, sender, image=bool(image_jpg))}
            if image_jpg:
                user_msg["images"] = [base64.b64encode(image_jpg).decode()]

            async with self.lock:
                timings["queue_ms"] = ms()
                answering = tag_message("", source, sender).rstrip(":\n")
                messages = [{"role": "system", "content": self.system_prompt(answering)}, *self.history(), user_msg]
                self.db.add_message("user", text, source, names, sender=sender)

                speech_queue: asyncio.Queue = asyncio.Queue()
                speaker = asyncio.create_task(self._speak(speech_queue, t0, timings)) if speak else None
                reply, pending, final = "", "", {}
                await self.hub.broadcast({"type": "assistant_start", "source": "reply", "ts": time.time()})
                async for chunk in self.llm.chat(messages, temperature=config.REPLY_TEMPERATURE):
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        if "ttft_ms" not in timings:
                            timings["ttft_ms"] = ms()
                            await self.hub.broadcast({"type": "state", "state": "answering"})
                        reply += token
                        await self.hub.broadcast({"type": "token", "text": token})
                        if speaker:
                            pending += token
                            while (sentence := pop_sentence(pending))[0] is not None:
                                speech_queue.put_nowait(sentence[0])
                                pending = sentence[1]
                    if chunk.get("done"):
                        final = chunk

            timings["llm_ms"] = ms()
            if final.get("eval_duration"):
                timings["tok_s"] = round(final["eval_count"] / final["eval_duration"] * 1e9, 1)
            timings["prompt_tokens"] = final.get("prompt_eval_count")
            self.db.add_message("assistant", reply, "reply", names)

            if speaker:
                if pending.strip():
                    speech_queue.put_nowait(pending)
                speech_queue.put_nowait(None)
                await self.hub.broadcast({"type": "state", "state": "speaking"})
                await speaker
            timings["total_ms"] = ms()
            await self.hub.broadcast({"type": "done", "timings": timings})
        finally:
            if speaker and not speaker.done():
                speaker.cancel()
            self.user_busy = False
            await self.hub.broadcast({"type": "state", "state": "idle"})

        asyncio.create_task(self._safe(self.extract_memories(text, reply, source, sender), "memory extraction"))

    async def _speak(self, queue: asyncio.Queue, t0: float, timings: dict) -> None:
        while (sentence := await queue.get()) is not None:
            await self.say_audio(sentence)
            timings.setdefault("first_audio_ms", round((time.perf_counter() - t0) * 1000))

    async def say_audio(self, sentence: str) -> None:
        sentence = clean_for_speech(sentence)
        if sentence:
            wav = await asyncio.to_thread(self.tts.synthesize_wav, sentence)
            await self.hub.broadcast({"type": "audio", "wav": base64.b64encode(wav).decode()})

    # ----------------------------------------------------------- initiative

    async def consider_initiative(self, trigger: str, bypass_cooldown: bool = False) -> None:
        """Ask the model whether to speak up. Hard limits are enforced here, not left to the model."""
        now = time.time()
        if self.user_busy:
            return
        if config.INITIATIVE_REQUIRES_FACE and not self.presence.present():
            log.info("initiative skipped (%s): nobody in view", trigger)
            return
        last_user = self.db.last_message_ts("user")
        if last_user and now - last_user < config.CONVERSATION_ACTIVE_S:
            log.info("initiative skipped (%s): conversation active", trigger)
            return
        if not bypass_cooldown:
            # quiet-mode decisions count for the cooldown (keeps background load down), not for the hourly cap
            last = self.db.last_event(["initiative", "initiative_quiet"])
            if last and now - last["ts"] < config.INITIATIVE_COOLDOWN_S:
                log.info("initiative skipped (%s): cooldown", trigger)
                return
        if self.db.count_events_since(["initiative"], now - 3600) >= config.INITIATIVE_MAX_PER_HOUR:
            log.info("initiative skipped (%s): hourly limit", trigger)
            return

        async with self.lock:
            if self.user_busy:
                return
            messages = [
                {"role": "system", "content": self.system_prompt()},
                *self.history(),
                {"role": "user", "content": DECISION_PROMPT.format(trigger=trigger)},
            ]
            raw = await self.llm.complete(messages, schema=DECISION_SCHEMA, max_tokens=250, temperature=0.7)
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("unparseable decision: %r", raw)
            return

        reason = decision.get("reason", "").strip()
        text = decision.get("text", "").strip()
        if decision.get("action") != "speak" or not text:
            await self.emit("decision", f"stayed silent ({trigger}): {reason}", {"trigger": trigger})
            return

        quiet = self.quiet
        label = "would have said (quiet mode)" if quiet else "said"
        await self.emit(
            "initiative_quiet" if quiet else "initiative",
            f'{label}: "{text}"', {"trigger": trigger, "reason": reason, "quiet": quiet},
        )
        if quiet:
            return
        qid = decision.get("question_id") or 0
        if qid and qid in {q["id"] for q in self.curiosity.askable(limit=50)}:
            await self.curiosity.asked(qid)
        self.db.add_message("assistant", text, "initiative", self.present_names())
        await self.hub.broadcast(
            {"type": "assistant", "text": text, "source": "initiative", "reason": reason, "ts": time.time()}
        )
        for sentence in split_sentences(text):
            await self.say_audio(sentence)

    # --------------------------------------------------------------- vision

    async def describe_scene(self, jpg: bytes, prev: dict | None, prev_jpg: bytes | None) -> str:
        names = self.present_names()
        who = f"People you recognize by face right now: {', '.join(names)}. " if names else ""
        task = (
            "Describe what the camera sees in one or two short sentences (max 35 words): who is there, what they "
            f"are doing, and anything notable. {who}Don't guess names for anyone else. No preamble."
        )
        images = [base64.b64encode(jpg).decode()]
        if prev and prev_jpg:
            task = (
                f'Image 1 was taken {ago(prev["ts"])} (at {hhmm(prev["ts"])}); back then it was described as: '
                f'"{prev["text"]}". Image 2 is the camera view now.\n'
                + task.replace("what the camera sees", "image 2")
                + " Then briefly say what changed compared to image 1. Ignore small differences in light,"
                " exposure or camera noise. If nothing meaningful changed, answer only: UNCHANGED"
            )
            images.insert(0, base64.b64encode(prev_jpg).decode())
        messages = [
            {"role": "system", "content": "You write the perception log of a home AI's webcam. Be factual and concise."},
            {"role": "user", "content": task, "images": images},
        ]
        async with self.lock:
            caption = await self.llm.complete(messages, max_tokens=120, temperature=0.2)
        # The model likes "**Change:**" headings; the log should be plain sentences.
        return re.sub(r"\s+", " ", re.sub(r"[*#_`]+", "", caption)).strip()

    # --------------------------------------------------------------- memory

    async def extract_memories(self, user_text: str, reply: str, source: str, sender: str | None) -> None:
        present = self.presence.present()
        known = [t for t in present if t.person_id]
        unknown = [t for t in present if t.person_id is None]
        from_tool = source == "developer"
        if from_tool:
            speaker = (
                f"The message was sent by {sender or 'a developer tool'}, not by a person in the room. "
                "Only keep facts it states about real people or the household; it is not introducing itself as a person."
            )
        elif len(known) == 1 and not unknown:
            speaker = f"The user is most likely {known[0].name}."
        elif known or unknown:
            names = [t.name for t in known] + ["an unknown person"] * len(unknown)
            speaker = f"People in view: {', '.join(names)}; it's unclear who is speaking."
        else:
            speaker = "Nobody is recognized in view; it's unclear who is speaking."

        known_facts = []
        for t in known:
            known_facts += [f"{t.name}: {m['text']}" for m in self.db.memories(t.person_id, 20)]
        known_facts += [f"household: {m['text']}" for m in self.db.memories(None, 20)]

        # The message before this exchange: often the AI's own question that the user is now answering.
        recent = self.db.recent_messages(3, since=time.time() - 900)
        before = ""
        if len(recent) == 3 and recent[0]["role"] == "assistant":
            before = f"Before that, the AI said: {recent[0]['content']}\n"

        open_qs = self.db.open_questions(12)
        questions = "\n".join(f"#{q['id']} {q['text']}" for q in open_qs) or "none"

        prompt = f"""Conversation snippet:
{before}{tag_message(user_text, source, sender)}
AI: {reply}

{speaker}
Already remembered: {"; ".join(known_facts) or "nothing"}
Open questions the AI wants answered:
{questions}

Return JSON:
- introduced_name: the name the user just said they are called ("I'm Anna", "my name is Anna"); otherwise "".
- facts: new durable facts worth remembering for future conversations (interests, projects, plans, preferences, important events, relationships, things in the home). Each has "about" (the person's name, or "household" if it's about the place or unclear) and "fact" (one short sentence in third person, using names). Skip small talk, things only relevant right now, and anything already remembered. Usually this list is empty.
- answered: only the open questions that the user's message actually answers, each with "id" (the #number) and "answer" (one short sentence stating the answer, e.g. "The cat is called Mila."). Leave out every question the message does not answer. Usually empty."""
        async with self.lock:
            raw = await self.llm.complete(
                [{"role": "system", "content": "You maintain the long-term memory of a home AI."},
                 {"role": "user", "content": prompt}],
                schema=MEMORY_SCHEMA, max_tokens=300, temperature=0,
            )
        result = json.loads(raw)

        # Only people talking in the room can introduce themselves; a tool must never claim a face.
        if (name := result.get("introduced_name", "").strip()) and not from_tool:
            await self.learn_name(name)

        for item in result.get("facts", [])[:5]:
            about, fact = item.get("about", "").strip(), item.get("fact", "").strip()
            if not fact:
                continue
            pid = None
            if about and about.lower() not in ("household", "unknown", "user", "the user"):
                pid = self.db.get_or_create_person(about)["id"]
            if self.db.add_memory(fact, pid):
                await self.emit("memory", f"remembered{f' about {about}' if pid else ''}: {fact}")

        by_id = {q["id"]: q for q in open_qs}
        for item in result.get("answered", [])[:3]:
            q, answer = by_id.get(item.get("id")), item.get("answer", "")
            if q and answer_supported(answer, q["text"], user_text):
                await self.curiosity.answered(q["id"], answer)
            elif q:
                log.info("ignored unsupported answer to #%s: %r", q["id"], answer)

    async def learn_name(self, name: str, track_key: str | None = None) -> None:
        """Bind a name to an unknown face: the given track, or the only unknown face in view."""
        if track_key is None:
            person = self.db.person_by_name(name)
            if person is None:
                person = self.db.get_or_create_person(name)
                new = True
            else:
                new = False
            self.claimed = (person["id"], time.time())
            unknown = self.presence.unknown_present()
            if len(unknown) != 1:
                if new:
                    await self.emit("person_named", f"learned the name {name} (face not linked: {len(unknown)} unknown faces in view)")
                return
            track_key = unknown[0].key
        track = self.presence.name_track(track_key, name)
        if track:
            await self.emit("person_named", f"the person in view is {track.name}; face saved")
            await self.broadcast_presence()

    # ------------------------------------------------------------ presence

    async def on_presence(self, kind: str, track: Track) -> None:
        if kind == "arrived":
            if track.person_id is None:
                text = "someone I don't know appeared"
            elif track.prev_seen:
                text = f"{track.name} arrived (last seen {ago(track.prev_seen)})"
            else:
                text = f"{track.name} arrived (first time seen)"
        else:
            who = track.name or "the unknown person"
            text = f"{who} left (was here for {fmt_dur(track.last_seen - track.first_seen)})"
        await self.emit(f"person_{kind}", text, {"key": track.key, "person_id": track.person_id})
        await self.broadcast_presence()

    # ----------------------------------------------------------- lifecycle

    def on_start(self) -> None:
        """Record the start, and detect whether the previous run ended cleanly."""
        now = time.time()
        last = self.db.last_event(["system_start", "system_stop"])
        heartbeat = float(self.db.get_setting("heartbeat", "0"))
        if last is None:
            detail = "This is the very first start."
        elif last["kind"] == "system_stop":
            detail = f"Was off for {fmt_dur(now - last['ts'])} after a clean shutdown at {fmt_ts(last['ts'])}."
        else:
            seen = max(heartbeat, last["ts"])
            self.db.add_event(
                "unclean_shutdown",
                f"previous run ended unexpectedly (crash or power loss); last sign of life at {fmt_ts(seen)}",
                ts=seen,
            )
            detail = f"Was off for about {fmt_dur(now - seen)} after an unexpected shutdown."
        self.db.add_event("system_start", f"system started at {fmt_ts(now)}. {detail}", {"started": now}, ts=now)
        self.db.set_setting("heartbeat", str(now))
        log.info("system start: %s", detail)

    async def on_stop(self) -> None:
        await self.emit("system_stop", f"system shutting down at {fmt_ts(time.time())}")

    async def heartbeat(self) -> None:
        while True:
            await asyncio.sleep(config.HEARTBEAT_S)
            self.db.set_setting("heartbeat", str(time.time()))

    @staticmethod
    async def _safe(coro, what: str) -> None:
        try:
            await coro
        except Exception:
            log.exception("%s failed", what)
