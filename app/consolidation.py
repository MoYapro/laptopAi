"""Memory consolidation: a background job with a bigger model that organizes what Housemate knows.

Runs when the room has been quiet for a while (or on demand via POST /api/consolidate):
  0. back up the database (data/backups/, last BACKUPS_KEEP kept)
  1. catch up: read conversations since the last run, add facts the per-reply extraction missed and
     note which topics were discussed with whom
  2. notes: turn short-term camera notes that recur at different times into facts; expire old notes
  3. reorganize: drop facts about a single moment (code), then separate small model calls: merge duplicate
     or contradicting pairs, correct facts (wrong person, Housemate mixed up with a person),
     sort facts into topics (merging duplicate topics), write person profiles, link related topics
Every change is reversible: edited/removed facts are archived (memories.archived_ts / replaced_by), not deleted.
The memory model runs outside the conversation lock: Ollama serves it next to the chat model
(on the laptop on CPU via MEMORY_NUM_GPU=0). A background run gives way as soon as someone shows up (the running
model call is cancelled) and continues in the next quiet period; a manual run (POST /api/consolidate) doesn't.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

import config
from util import fmt_dur, hhmm, is_momentary, word_similarity

if TYPE_CHECKING:
    from agent import Agent

log = logging.getLogger("consolidation")

# Guards the 9B model doesn't reliably follow on its own:
CLOTHES = re.compile(r"\b(wears?|wearing|hoodie|blanket|shirt|jacket|sweater)\b", re.I)  # not for profiles
MOMENTARY_TOPIC = re.compile(r"\b(current|currently|attire|clothes|clothing|outfit)\b", re.I)
DUPLICATE_CANDIDATE_SIMILARITY = 0.4  # content words two facts share (of the shorter one) to be compared as a pair
MAX_DUPLICATE_PAIRS = 12
MENTIONS_HOUSEMATE = re.compile(r"\b(housemate|the ai|ai's)\b", re.I)
# Without thinking the 9B model calls every pair "same"; with it, it tells them apart. Thinking counts against the
# output limit and takes about 8k tokens for 12 pairs.
PAIRS_THINK_TOKENS = 16000
DATE_OR_TIME = re.compile(r"\b\d{1,2}\.\d{1,2}\b|\b\d{1,2}:\d{2}\b")  # copied from a prompt into a fact
CATCH_ALL_TOPIC = re.compile(r"\b(general|misc|miscellaneous|other|various|context|stuff|things|household)\b", re.I)
TOPIC_OVERLAP_MERGE = 0.6  # topics sharing this much of the smaller one's facts are merged into the bigger one
FILLER_WORDS = set("""the a an and or but of to in on at for with by from as is are was were be been has have had it its
this that these those he she they his her their him them who which what when where while also not no can now
""".split())


def content_words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in FILLER_WORDS}
DUPLICATE_SIMILARITY = 0.6  # a "new" fact this close to a stored one is a duplicate
NOTES_KEEP_S = 48 * 3600  # short-term notes nobody confirmed are forgotten after this
# A note becomes a fact only when seen on separate occasions this far apart: while a scene stays the same, the
# reflection keeps paraphrasing it (and its misreadings) every few minutes, which is no evidence.
NOTE_MIN_SPAN_S = 6 * 3600
NOTE_GROUP_SIMILARITY = 0.5


def fits(text: str, longest: int) -> bool:
    """A rewritten or merged fact: one durable sentence, not much longer than what it replaces."""
    return (bool(text) and not is_momentary(text) and not DATE_OR_TIME.search(text)
            and len(text) <= min(240, longest * 1.25 + 15))


def name_words(name: str) -> set[str]:
    """'Guests' and 'Guest identification' share 'guest'."""
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", name.lower()) if len(w) > 3}


def person_named_in(text: str, people: dict) -> dict | None:
    """'Tom plays …' is about Tom, whatever the model put in "about"."""
    first = re.match(r"\s*([A-Z][\w-]+)", text)
    return people.get(first.group(1).lower()) if first else None

CATCHUP_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"about": {"type": "string"}, "fact": {"type": "string"}},
                "required": ["about", "fact"],
            },
        },
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"topic": {"type": "string"}, "people": {"type": "array", "items": {"type": "string"}}},
                "required": ["topic", "people"],
            },
        },
    },
    "required": ["facts", "topics"],
}

NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {  # keys are written alphabetically: the verdict comes before the fact
                    "a_group": {"type": "integer"},
                    "b_shared": {"type": "string"},
                    "c_remembered": {"type": "string"},
                    "d_verdict": {"type": "string", "enum": ["lasting", "moment", "different", "already_remembered"]},
                    "e_fact": {"type": "string"},
                    "f_about": {"type": "string"},
                },
                "required": ["a_group", "b_shared", "c_remembered", "d_verdict", "e_fact", "f_about"],
            },
        },
    },
    "required": ["groups"],
}

FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                    "about": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "text", "about", "reason"],
            },
        },
    },
    "required": ["corrections"],
}

DUPLICATES_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                # Ollama writes the keys in alphabetical order; the prefixes make the model compare first and
                # decide after (with "keep" first it picked "same" for every pair).
                "properties": {
                    "a_pair": {"type": "integer"},
                    "b_difference": {"type": "string"},
                    "c_relation": {"type": "string", "enum": ["different", "same", "contradict"]},
                    "d_keep": {"type": "string", "enum": ["A", "B"]},
                    "e_text": {"type": "string"},
                },
                "required": ["a_pair", "b_difference", "c_relation", "d_keep", "e_text"],
            },
        },
    },
    "required": ["pairs"],
}

TOPICS_SCHEMA = {
    "type": "object",
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                    "fact_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "summary", "fact_ids"],
            },
        },
        "merges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"from": {"type": "string"}, "into": {"type": "string"}},
                "required": ["from", "into"],
            },
        },
    },
    "required": ["topics", "merges"],
}

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {"profile": {"type": "string"}},
    "required": ["profile"],
}

LINK_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}, "why": {"type": "string"}},
                "required": ["a", "b", "why"],
            },
        },
    },
    "required": ["pairs"],
}

LINK_PROMPT = """Topics in Housemate's memory:
{topics}

Which pairs of these topics belong together (one is part of the other, they are about the same activity, project,
place, animal or interest, or knowing one helps understand the other)? For example "Sim racing" and "Gaming
preferences" belong together. Return JSON: pairs, each with a and b (exact topic names from the list) and why (a few
words). Only clear connections; it's fine to return few or none."""

SYSTEM = (
    "You organize the long-term memory of Housemate, an AI living in a laptop in someone's home. "
    "You are careful and conservative: you never invent facts, and you only change what is clearly wrong, "
    "duplicated or useless."
)

CATCHUP_PROMPT = """Conversations Housemate had since the last review. Each message says how it arrived:
"spoken"/"typed" come from people in the room, "developer tool" messages come from tools testing Housemate (not people).

{conversation}

People Housemate knows: {people}
Facts already remembered:
{known}
Existing topics: {topics}

Return JSON:
- facts: at most 8 durable facts from these conversations that are NOT already remembered (interests, projects, plans, preferences, relationships, important events, things and animals in the home). "about" is the name of the known person the fact is about, or "household" for the place, animals or anyone else; "fact" is one short third-person sentence with names. Only use what people in the room said; developer tool messages are just context. Skip anything about a single moment (what someone is wearing or doing right now). Often empty.
- topics: the topics that were actually talked about, with the names of the known people who talked about them ("people" may be empty). Reuse an existing topic name when it fits; otherwise a short general name like "Sim racing" or "Luna the cat"."""

NOTES_PROMPT = """Housemate notes what it sees on its webcam. Notes are short-term: a single camera image is often
a passing moment or misread. Each group below holds notes about a similar thing, made on separate occasions hours apart.

{groups}

People Housemate knows: {people}
Facts already remembered:
{known}

For each group return:
- a_group: its number
- b_shared: what all its notes show, in a few words
- c_remembered: the remembered fact that already says this, copied, or "none"
- d_verdict, e_fact, f_about:
- "lasting": the notes show the same lasting thing again: an animal or object that belongs to the home, how the
  room is used, a habit of a person. e_fact: one short sentence with names, no times; f_about: the known person it is
  about, or "household".
- "moment": a passing moment (someone sitting, looking down, a thing that moved), a guess about visitors, or something
  that didn't happen.
- "different": the notes don't show the same thing, or contradict each other.
- "already_remembered": c_remembered is not "none".
For anything but "lasting", e_fact and f_about are empty strings.

"""

FACTS_PROMPT = """All facts Housemate remembers (#id [about] text):
{facts}

People Housemate knows: {people}.
Housemate itself is the AI in the laptop: it watches the room through a webcam and talks. It cannot play games, eat,
wear clothes or go anywhere, so a fact saying it does is a mix-up with a person.

Find facts that are wrong as written and correct them. Return JSON: corrections, each with id, the corrected text
(one short sentence), about (a known person's name or "household") and a short reason (max 15 words). Only these
mistakes count:
- attached to the wrong person: a "household" fact that is clearly about a known person, or the other way round
- Housemate mixed up with a person: the fact says Housemate (or "the AI") did something only a person can do,
  like playing a game. Then the person is the subject.
Never put Housemate into a fact about what a person does or has: "Tom wears headphones" stays as it is.
A fact that is fine as it is must not appear. Usually the list is empty or very short."""

DUPLICATES_PROMPT = """Pairs of facts from Housemate's memory that share some words ([about] in front):
{pairs}

For each pair, compare A and B. Return JSON: pairs, each with
- a_pair: its number
- b_difference: in a few words, what one of them says that the other does not ("nothing" if they say the same)
- c_relation:
  "same": they say the same thing in other words, or one is a shorter version of the other ("Tom sim races" /
  "Tom enjoys sim racing on his steering wheel"). They become one sentence.
  "contradict": one shows the other is wrong ("an unknown man was seen" / "that man was Tom"; "Luna's name is
  Mia" / "the cat is called Luna"). The wrong one is dropped.
  "different": each says something the other does not ("Luna sleeps in her hammock" / "Luna shifts her paws").
  Both stay.
- d_keep: for "contradict", the one that is right: what a person said or confirmed beats a guess from the camera,
  otherwise the newer one (B). Else "A".
- e_text: for "same", one short sentence that says it once and keeps every detail of both. Else "".
Sharing a subject (the same person, cat or object) does not make two facts the same."""

TOPICS_PROMPT = """Facts Housemate remembers (#id [about] text):
{facts}

Existing topics (name: summary):
{topics}

Sort the facts into topics. Return JSON:
- topics: name, summary and fact_ids for each topic. Reuse the existing topic names wherever they fit. A topic is
  about one thing (a hobby, a game, an animal, a project, a preference, a visitor), never a catch-all like
  "Household" or a person's name. Each fact belongs to one or two topics. Create a new topic only for facts that no
  existing topic fits. No topics for momentary things (clothes, what someone is doing right now). summary: one
  short sentence (max 25 words) on what Housemate knows about it, not a list of all its facts.
- merges: existing topics that are the same subject under two names: from (the name to drop) and into (the name to
  keep). Usually empty."""

PROFILE_PROMPT = """What Housemate remembers about {name}:
{facts}

Current profile: {profile}

Write {name}'s profile: two or three sentences in third person about who they are and what matters to them: their
main interests, projects, preferences, and the people and animals in their life. Keep what is still true from the
current profile. Not their looks or clothes, nothing they are doing right now. Return JSON: profile."""


class Paused(Exception):
    """Someone became active during a background run; it continues in the next quiet period."""


class Consolidator:
    def __init__(self, agent: "Agent"):
        self.agent, self.db, self.llm = agent, agent.db, agent.llm
        self.running = False
        self.interruptible = False  # background runs give way to people; manual runs don't
        self.last_run = float(self.db.get_setting("consolidated_at", "0"))

    # ---------------------------------------------------------------- triggers

    def room_idle(self) -> bool:
        if self.agent.user_busy or self.agent.presence.present():
            return False
        last_msg = self.db.last_message_ts() or 0
        last_seen = max((p["last_seen"] or 0 for p in self.db.people()), default=0)
        return time.time() - max(last_msg, last_seen) >= config.CONSOLIDATE_IDLE_S

    def someone_active(self) -> bool:
        """Someone in view, a reply being made, or a message just now."""
        last_msg = self.db.last_message_ts() or 0
        return bool(self.agent.user_busy or self.agent.presence.present() or time.time() - last_msg < 120)

    def _check_active(self) -> None:
        if self.interruptible and self.someone_active():
            raise Paused()

    async def loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                if (
                    time.time() - self.last_run >= config.CONSOLIDATE_INTERVAL_S
                    and self.room_idle()
                    and (self.db.memory_count_since(self.last_run) or self.db.messages_since(self.last_run, 1)
                         or self.db.note_count_since(self.last_run))
                ):
                    await self.run("idle")
            except Exception:
                log.exception("consolidation failed")

    # --------------------------------------------------------------------- run

    async def run(self, trigger: str) -> dict:
        if self.running:
            return {"status": "already running"}
        self.running = True
        self.interruptible = trigger != "manual"
        try:
            return await self._run(trigger)
        except Paused:
            log.info("memory review paused: someone is around")
            await self.agent.emit("consolidation", "memory review paused: someone is around (it continues in the "
                                  "next quiet period; what was done so far is kept)")
            return {"status": "paused"}
        except Exception as e:
            log.exception("memory review failed")
            await self.agent.emit("consolidation", f"memory review failed: {e} (changes so far are reversible; "
                                  "the backup from the start of the run is in data/backups)")
            return {"status": "failed", "error": str(e)}
        finally:
            self.running = False

    async def _run(self, trigger: str) -> dict:
        model = await self._pick_model()
        if model is None:
            await self.agent.emit("consolidation", f"memory review skipped: none of {config.MEMORY_MODELS} is in Ollama")
            return {"status": "no model"}
        t0 = time.time()
        stats = {"model": model, "trigger": trigger, "added": 0, "from_notes": 0, "merged_facts": 0, "rewritten": 0,
                 "archived": 0, "topics": 0, "merged_topics": 0, "links": 0, "profiles": 0, "changes": []}
        backup = self._backup()
        log.info("memory review started (%s, %s), backup %s", model, trigger, backup)
        await self.agent.emit("consolidation", f"memory review started ({model})", {"backup": backup})

        await self._catch_up(model, stats)
        await self._promote_notes(model, stats)
        await self._organize(model, stats)
        await self._finish(stats, t0, None)
        return stats

    async def _finish(self, stats: dict, t0: float, note: str | None) -> None:
        self.last_run = time.time()
        self.db.set_setting("consolidated_at", str(self.last_run))
        removed = self.db.delete_empty_topics()
        parts = [
            f"{stats['added']} new facts", f"{stats['from_notes']} from camera notes", f"{stats['merged_facts']} merged",
            f"{stats['rewritten']} rewritten", f"{stats['archived']} archived",
            f"{stats['topics']} topics", f"{stats['links']} topic links", f"{stats['profiles']} profiles",
        ]
        if stats["merged_topics"]:
            parts.append(f"{stats['merged_topics']} topics merged")
        if removed:
            parts.append(f"{removed} empty topics removed")
        text = f"memory review done in {fmt_dur(time.time() - t0)}: " + ", ".join(parts)
        if note:
            text += f" ({note})"
        stats["duration_s"] = round(time.time() - t0)
        await self.agent.emit("consolidation", text, stats)

    async def _pick_model(self) -> str | None:
        for model in config.MEMORY_MODELS:
            if await self.llm.has_model(model):
                return model
        return None

    def _backup(self) -> str:
        folder = config.DATA_DIR / "backups"
        folder.mkdir(exist_ok=True)
        # A run paused and resumed (maybe several times) keeps the backup from its first start.
        since = max(float(self.db.get_setting("consolidated_at", "0")), time.time() - 6 * 3600)
        unfinished = [b for b in folder.glob("laptopai-2*.db") if b.stat().st_mtime > since]
        if unfinished:
            return f"backups/{min(unfinished, key=lambda b: b.stat().st_mtime).name}"
        path = folder / f"laptopai-{datetime.now():%Y%m%d-%H%M%S}.db"
        self.db.backup(path)
        for old in sorted(folder.glob("laptopai-*.db"))[: -config.BACKUPS_KEEP]:
            old.unlink()
        return f"backups/{path.name}"

    async def _ask(self, model: str, prompt: str, schema: dict, max_tokens: int, think: bool | None = None) -> dict:
        options = {"num_ctx": config.MEMORY_NUM_CTX}
        if config.MEMORY_NUM_GPU >= 0:
            options["num_gpu"] = config.MEMORY_NUM_GPU
        for attempt in (1, 2):
            raw = await self._interruptible(self.llm.complete(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                schema=schema, max_tokens=max_tokens, temperature=0.1 if attempt == 1 else 0.3, model=model,
                think=config.MEMORY_THINK if think is None else think, options=options, keep_alive="5m",
            ))
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                if attempt == 2:
                    raise
                log.warning("invalid JSON from %s (%d chars, %s), retrying", model, len(raw), e)

    async def _interruptible(self, call):
        """Run a model call, but cancel it as soon as someone becomes active (a background run can take half an
        hour on CPU). Cancelling closes the HTTP request, which makes Ollama stop generating."""
        try:
            self._check_active()
        except Paused:
            call.close()  # never started
            raise
        task = asyncio.ensure_future(call)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=5)
                if done:
                    return task.result()
                self._check_active()
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    # ------------------------------------------------- step 1: conversations

    async def _catch_up(self, model: str, stats: dict) -> None:
        since = float(self.db.get_setting("consolidated_messages_until", "0"))
        all_messages = self.db.messages_since(since, limit=400)
        # Developer-tool exchanges are tests and setup talk, not life in the room: leave them (and the replies) out.
        messages, skip_reply = [], False
        for m in all_messages:
            if m["role"] == "user":
                skip_reply = m["source"] == "developer"
                if not skip_reply:
                    messages.append(m)
            elif not skip_reply:
                messages.append(m)
        if all_messages and not messages:
            self.db.set_setting("consolidated_messages_until", str(all_messages[-1]["ts"]))
        people = {p["name"].lower(): p for p in self.db.people()}
        if all_messages and messages and messages[-1]["ts"] < all_messages[-1]["ts"]:
            messages_tail_ts = all_messages[-1]["ts"]
        else:
            messages_tail_ts = None
        for start in range(0, len(messages), 25):
            chunk = messages[start:start + 25]
            convo = "\n".join(self._format_message(m) for m in chunk)
            known = "\n".join(
                f"- [{m['person'] or 'household'}] {m['text']}" for m in self.db.active_memories()
            ) or "nothing yet"
            result = await self._ask(model, CATCHUP_PROMPT.format(
                conversation=convo,
                people=", ".join(p["name"] for p in people.values()) or "nobody yet",
                known=known,
                topics=", ".join(t["name"] for t in self.db.topics()) or "none yet",
            ), CATCHUP_SCHEMA, 3000)

            for item in result.get("facts", [])[:8]:
                fact = item.get("fact", "").strip()
                if not fact or is_momentary(fact):
                    continue
                person = people.get(item.get("about", "").strip().lower()) or person_named_in(fact, people)
                if any(word_similarity(fact, m["text"]) >= DUPLICATE_SIMILARITY for m in self.db.active_memories()):
                    continue
                if self.db.add_memory(fact, person["id"] if person else None):
                    stats["added"] += 1
                    stats["changes"].append({"added": fact, "about": person["name"] if person else "household"})
            chunk_ts = chunk[-1]["ts"]
            for item in result.get("topics", [])[:10]:
                name = item.get("topic", "").strip()
                if not name:
                    continue
                tid = self.db.upsert_topic(name)
                for pname in item.get("people", []):
                    if person := people.get(str(pname).strip().lower()):
                        self.db.touch_topic_person(tid, person["id"], chunk_ts)
            self.db.set_setting("consolidated_messages_until", str(chunk_ts))
        if messages_tail_ts:
            self.db.set_setting("consolidated_messages_until", str(messages_tail_ts))

    def _note_groups(self, notes: list[dict]) -> list[list[dict]]:
        """Each group: a note and the later notes sharing most of its content words (overlap coefficient). Only
        groups seen on separate occasions (NOTE_MIN_SPAN_S apart) count. Not transitive: chains through shared
        room words ("window", "chair") would join unrelated notes."""
        words = {n["id"]: content_words(n["text"]) for n in notes}
        taken: set[int] = set()
        groups = []
        for seed in notes:
            if seed["id"] in taken:
                continue
            a = words[seed["id"]]
            group = [seed] + [
                n for n in notes if n["ts"] > seed["ts"] and n["id"] not in taken
                and len(a & words[n["id"]]) / (min(len(a), len(words[n["id"]])) or 1) >= NOTE_GROUP_SIMILARITY
            ]
            if group[-1]["ts"] - seed["ts"] >= NOTE_MIN_SPAN_S:
                groups.append(group)
                taken |= {n["id"] for n in group}
        return groups

    async def _promote_notes(self, model: str, stats: dict) -> None:
        """Short-term camera notes become facts only when similar notes come back on separate occasions (code
        finds those groups, the model judges them); notes nobody confirmed expire."""
        now = time.time()
        groups = self._note_groups(self.db.notes(since=now - NOTES_KEEP_S))[:8]
        if groups:
            people = self._people()
            known = self.db.active_memories()

            def shown(g: list[dict]) -> list[dict]:  # the first, the last and a few in between
                return g if len(g) <= 6 else [g[0], *g[1:-1][:: max(1, (len(g) - 2) // 4)][:4], g[-1]]

            result = await self._ask(model, NOTES_PROMPT.format(
                groups="\n".join(
                    f"Group {k}:\n" + "\n".join(
                        f"  {datetime.fromtimestamp(n['ts']).strftime('%a %H:%M')}: {n['text']}" for n in shown(g))
                    for k, g in enumerate(groups, 1)),
                people=", ".join(p["name"] for p in people.values()) or "nobody yet",
                known="\n".join(f"- [{m['person'] or 'household'}] {m['text']}" for m in known) or "nothing yet",
            ), NOTES_SCHEMA, 1500)
            for item in result.get("groups", []):
                k, verdict = item.get("a_group"), item.get("d_verdict")
                if not isinstance(k, int) or not 1 <= k <= len(groups):
                    continue
                g, fact = groups[k - 1], str(item.get("e_fact", "")).strip()
                log.info("camera notes group %d: %s (remembered: %s) -> %s %s", k, item.get("b_shared"),
                         item.get("c_remembered"), verdict, fact)
                if verdict == "already_remembered":
                    self.db.keep_notes([n["id"] for n in g])
                if verdict != "lasting" or not fits(fact, max(len(n["text"]) for n in g)):
                    continue
                self.db.keep_notes([n["id"] for n in g])
                if any(word_similarity(fact, m["text"]) >= DUPLICATE_SIMILARITY for m in known):
                    continue
                pid = self._person_for(people, str(item.get("f_about", "")), fact, None)
                if self.db.add_memory(fact, pid):
                    known.append({"person": None, "text": fact})
                    stats["from_notes"] += 1
                    stats["changes"].append({"added": fact, "from_notes": [n["text"] for n in g],
                                             "about": next((p["name"] for p in people.values() if p["id"] == pid),
                                                           "household")})
        if expired := self.db.expire_notes(now - NOTES_KEEP_S):
            log.info("forgot %d unconfirmed camera notes", expired)

    @staticmethod
    def _format_message(m: dict) -> str:  # noqa: C901
        if m["role"] == "assistant":
            who = "Housemate" + {"initiative": " (on its own)", "reminder": " (reminder)"}.get(m["source"], "")
        elif m["source"] == "developer":
            who = f"{m['sender'] or 'a developer tool'} (developer tool)"
        else:
            in_view = ", ".join(json.loads(m["people"] or "[]")) or "nobody recognized"
            who = f"{'spoken' if m['source'] == 'voice' else 'typed'} by someone in the room (in view: {in_view})"
        return f"{hhmm(m['ts'])} {who}: {m['content']}"

    # ---------------------------------------------------- step 2: organizing

    async def _organize(self, model: str, stats: dict) -> None:
        """Three small questions instead of one big one: the 9B model loses track (and the JSON) otherwise."""
        if not self.db.active_memories():
            return
        self._drop_momentary(stats)
        await self._merge_duplicates(model, stats)
        await self._clean_facts(model, stats)
        await self._sort_topics(model, stats)
        await self._write_profiles(model, stats)
        await self._link_topics(model, stats)

    def _people(self) -> dict:
        return {p["name"].lower(): p for p in self.db.people()}

    @staticmethod
    def _person_for(people: dict, about: str, text: str, current: int | None) -> int | None:
        # A fact about a person stays with a person: "household" never detaches it.
        person = people.get(about.strip().lower()) or person_named_in(text, people)
        return person["id"] if person else current

    def _drop_momentary(self, stats: dict) -> None:
        """Facts about a single moment ("Tom is wearing …", "… currently …") are no memories. The extraction
        steps no longer store them; this clears out the ones stored before."""
        for m in self.db.active_memories():
            if is_momentary(m["text"]):
                self.db.archive_memory(m["id"], "about a single moment")
                stats["archived"] += 1
                stats["changes"].append({"archived": m["text"], "reason": "about a single moment"})

    async def _merge_duplicates(self, model: str, stats: dict) -> None:
        """Code picks pairs of similar facts; the model only answers "same thing?" for each pair, which a 9B model
        does far more reliably than finding duplicate groups among all facts."""
        memories = self.db.active_memories()
        names = {p["name"].lower() for p in self.db.people()}
        words = {m["id"]: content_words(m["text"]) - names for m in memories}
        candidates = []
        for i, a in enumerate(memories):
            for b in memories[i + 1:]:
                if a["person_id"] and b["person_id"] and a["person_id"] != b["person_id"]:
                    continue
                wa, wb = words[a["id"]], words[b["id"]]
                if len(wa) < 2 or len(wb) < 2:
                    continue
                if (sim := len(wa & wb) / min(len(wa), len(wb))) >= DUPLICATE_CANDIDATE_SIMILARITY:
                    candidates.append((sim, a, b))
        pairs = [(a, b) for _, a, b in sorted(candidates, key=lambda c: -c[0])[:MAX_DUPLICATE_PAIRS]]
        if not pairs:
            return
        schema = json.loads(json.dumps(DUPLICATES_SCHEMA))
        schema["properties"]["pairs"]["maxItems"] = len(pairs)
        pairs = [(a, b) if a["ts"] <= b["ts"] else (b, a) for a, b in pairs]  # A is always the older one
        result = await self._ask(model, DUPLICATES_PROMPT.format(pairs="\n".join(
            f"{n}. A (older): [{a['person'] or 'household'}] {a['text']}\n"
            f"   B (newer): [{b['person'] or 'household'}] {b['text']}"
            for n, (a, b) in enumerate(pairs, 1)
        )), schema, PAIRS_THINK_TOKENS, think=True)

        people = self._people()
        name_of = {p["id"]: p["name"] for p in people.values()}
        done: set[int] = set()  # a fact merged once waits for the next run before it can merge again
        dropped: set[int] = set()  # archived as wrong: can't be the "right" side of a contradiction any more
        merged_into: dict[int, int] = {}  # a merged fact lives on in the new one
        for item in result.get("pairs", []):
            n, relation = item.get("a_pair"), item.get("c_relation")
            if relation not in ("same", "contradict") or not isinstance(n, int) or not 1 <= n <= len(pairs):
                continue
            a, b = pairs[n - 1]
            log.info("pair %d %s (keep %s; %s): %r / %r", n, relation, item.get("d_keep"), item.get("b_difference"),
                     a["text"], b["text"])
            if relation == "contradict":
                # Either side may already be merged into a new fact this run; then that new fact is what's
                # kept or dropped.
                keep, drop = (a, b) if item.get("d_keep") != "B" else (b, a)
                keep_id, drop_id = merged_into.get(keep["id"], keep["id"]), merged_into.get(drop["id"], drop["id"])
                if keep_id in dropped or drop_id in dropped or keep_id == drop_id:
                    continue
                drop_text = drop["text"] if drop_id == drop["id"] else self.db.memory_text(drop_id)
                reason = f"contradicted by #{keep_id}: {keep['text']}"[:200]
                self.db.archive_memory(drop_id, reason)
                done.add(drop["id"])
                dropped.add(drop_id)
                stats["archived"] += 1
                stats["changes"].append({"archived": drop_text, "reason": reason})
                continue
            if a["id"] in done or b["id"] in done:
                continue
            text = item.get("e_text", "").strip()
            if a["id"] in done or b["id"] in done or not fits(text, max(len(a["text"]), len(b["text"]))):
                continue
            newer, older = (a, b) if a["ts"] >= b["ts"] else (b, a)
            pid = self._person_for(people, "", text, newer["person_id"] or older["person_id"])
            new_id = self.db.replace_memory(newer["id"], text, pid, "merged duplicates")
            self.db.archive_memory(older["id"], f"merged into #{new_id}", replaced_by=new_id)
            done.update((a["id"], b["id"]))
            merged_into.update({a["id"]: new_id, b["id"]: new_id})
            stats["merged_facts"] += 2
            stats["changes"].append({"merged": [older["text"], newer["text"]], "into": text,
                                     "about": name_of.get(pid, "household")})

    async def _clean_facts(self, model: str, stats: dict) -> None:
        memories = {m["id"]: m for m in self.db.active_memories()}
        people = self._people()
        name_of = {p["id"]: p["name"] for p in people.values()}
        result = await self._ask(model, FACTS_PROMPT.format(
            facts="\n".join(f"#{m['id']} [{m['person'] or 'household'}] {m['text']}" for m in memories.values()),
            people=", ".join(p["name"] for p in people.values()) or "nobody yet",
        ), FACTS_SCHEMA, 1500)

        done: set[int] = set()  # every fact is changed at most once per run

        for ch in result.get("corrections", []):
            old = memories.get(ch.get("id"))
            if old is None or old["id"] in done:
                continue
            text = ch.get("text", "").strip()
            pid = self._person_for(people, ch.get("about", ""), text, old["person_id"])
            if not fits(text, len(old["text"])) or (text == old["text"] and pid == old["person_id"]):
                continue
            if MENTIONS_HOUSEMATE.search(text) and not MENTIONS_HOUSEMATE.search(old["text"]):
                log.info("correction of #%s skipped: it writes Housemate into a fact about a person: %r", old["id"], text)
                continue
            reason = (ch.get("reason", "").strip() or "consolidation")[:200]
            self.db.replace_memory(old["id"], text, pid, reason)
            done.add(old["id"])
            stats["rewritten"] += 1
            stats["changes"].append({"from": old["text"], "to": text, "about": name_of.get(pid, "household"),
                                     "reason": reason})

    async def _sort_topics(self, model: str, stats: dict) -> None:
        memories = {m["id"]: m for m in self.db.active_memories()}
        topics = self.db.topics()
        existing = {t["name"].lower(): t for t in topics}
        result = await self._ask(model, TOPICS_PROMPT.format(
            facts="\n".join(f"#{m['id']} [{m['person'] or 'household'}] {m['text']}" for m in memories.values()),
            topics="\n".join(f"- {t['name']}: {t['summary'] or '(no summary yet)'}" for t in topics) or "none yet",
        ), TOPICS_SCHEMA, 2500)

        # merges first, and only between topics that are recognizably the same subject
        for m in result.get("merges", []):
            src, dst = existing.get(str(m.get("from", "")).strip().lower()), existing.get(str(m.get("into", "")).strip().lower())
            if not src or not dst or src["id"] == dst["id"]:
                continue
            if not (name_words(src["name"]) & name_words(dst["name"])
                    or word_similarity(src["summary"] or "", dst["summary"] or "") >= 0.4):
                log.info("topic merge %r -> %r skipped: different subjects", src["name"], dst["name"])
                continue
            self.db.merge_topic(src["id"], dst["id"])
            existing.pop(src["name"].lower())
            stats["merged_topics"] += 1
            stats["changes"].append({"merged_topic": src["name"], "into": dst["name"]})

        fact_topics: dict[int, list[int]] = {}
        for t in result.get("topics", [])[:20]:
            name = t.get("name", "").strip()
            fids = [f for f in dict.fromkeys(t.get("fact_ids", [])) if f in memories]
            if not name or (not fids and name.lower() not in existing) or MOMENTARY_TOPIC.search(name):
                continue
            if name.lower() not in existing and (CATCH_ALL_TOPIC.search(name) or re.search(r"&| and ", name)):
                log.info("new catch-all topic %r skipped", name)
                continue
            summary = re.split(r"(?<=[.!?])\s+", t.get("summary", "").strip())[0]
            summary = " ".join(summary.split()[:30])
            if is_momentary(summary):
                summary = ""
            tid = self.db.upsert_topic(name, summary or None)
            stats["topics"] += 1
            for fid in fids:
                fact_topics.setdefault(fid, []).append(tid)
                if memories[fid]["person_id"]:
                    self.db.ensure_topic_person(tid, memories[fid]["person_id"])
        for fid, tids in fact_topics.items():
            self.db.set_memory_topics(fid, tids[:2])
        self._tidy_topics(stats)

    def _tidy_topics(self, stats: dict) -> None:
        """Rules the model doesn't follow reliably: no summaries about a single moment, and topics that hold
        (nearly) the same facts are one topic."""
        for t in self.db.topics():
            if t["summary"] and is_momentary(t["summary"]):
                self.db.upsert_topic(t["name"], "")
                t["summary"] = ""
            if not t["facts"] and not t["summary"]:
                self.db.delete_topic(t["id"])
                stats["changes"].append({"removed_topic": t["name"], "reason": "no facts and no summary"})
        # a topic without facts whose summary another topic already covers is that topic
        for t in [t for t in self.db.topics() if not t["facts"]]:
            others = [o for o in self.db.topics() if o["id"] != t["id"] and o["facts"]]
            best = max(others, key=lambda o: word_similarity(t["summary"], o["summary"] or ""), default=None)
            if best and word_similarity(t["summary"], best["summary"] or "") >= 0.4:
                self.db.merge_topic(t["id"], best["id"])
                stats["merged_topics"] += 1
                stats["changes"].append({"merged_topic": t["name"], "into": best["name"], "reason": "same summary"})
        merged = True
        while merged:  # one merge at a time; the topic list changes with every merge
            merged = False
            topics = sorted(self.db.topics(), key=lambda t: -len(t["facts"]))
            for i, big in enumerate(topics):
                for small in topics[i + 1:]:
                    fa, fb = {f["id"] for f in big["facts"]}, {f["id"] for f in small["facts"]}
                    if fb and len(fa & fb) / len(fb) >= TOPIC_OVERLAP_MERGE:
                        self.db.merge_topic(small["id"], big["id"])
                        stats["merged_topics"] += 1
                        stats["changes"].append({"merged_topic": small["name"], "into": big["name"],
                                                 "reason": "same facts"})
                        merged = True
                        break
                if merged:
                    break

    async def _write_profiles(self, model: str, stats: dict) -> None:
        for person in self.db.people():
            facts = self.db.memories(person["id"], 60)
            if not facts:
                continue
            result = await self._ask(model, PROFILE_PROMPT.format(
                name=person["name"],
                facts="\n".join(f"- {f['text']}" for f in facts),
                profile=person.get("profile") or "none yet",
            ), PROFILE_SCHEMA, 400)
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", result.get("profile", "").strip())
                         if s and not is_momentary(s) and not CLOTHES.search(s)]
            if sentences:
                self.db.set_profile(person["id"], " ".join(sentences[:3]))
                stats["profiles"] += 1

    async def _link_topics(self, model: str, stats: dict) -> None:
        topics = self.db.topics()
        if len(topics) < 2:
            return
        by_name = {t["name"].lower(): t["id"] for t in topics}
        result = await self._ask(model, LINK_PROMPT.format(
            topics="\n".join(f"- {t['name']}: {t['summary'] or ''}" for t in topics)
        ), LINK_SCHEMA, 1500)
        for pair in result.get("pairs", [])[:30]:
            a, b = by_name.get(pair.get("a", "").strip().lower()), by_name.get(pair.get("b", "").strip().lower())
            if a and b and a != b:
                self.db.link_topics(a, b)
                stats["links"] += 1
