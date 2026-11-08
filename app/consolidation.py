"""Memory consolidation: a background job with a bigger model that organizes what Housemate knows.

Runs when the room has been quiet for a while (or on demand via POST /api/consolidate):
  0. back up the database (data/backups/, last BACKUPS_KEEP kept)
  1. catch up: read conversations since the last run, add facts the per-reply extraction missed and
     note which topics were discussed with whom
  2. reorganize: rewrite or archive facts (duplicates, wrong person, outdated, trivial), sort facts into
     topics, merge duplicate topics, link related topics, link topics to people, write person profiles
Every change is reversible: edited/removed facts are archived (memories.archived_ts / replaced_by), not deleted.
The memory model runs outside the conversation lock: Ollama serves it next to the chat model
(on the laptop on CPU via MEMORY_NUM_GPU=0), and the job stops between steps when someone shows up.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING

import config
from util import fmt_dur, hhmm, word_similarity

if TYPE_CHECKING:
    from agent import Agent

log = logging.getLogger("consolidation")

# Guards the 9B model doesn't reliably follow on its own:
MOMENTARY = re.compile(r"\b(currently|right now|at the moment|is wearing|is sitting|today)\b", re.I)
MERGE_MIN_SIMILARITY = 0.5  # only near-identical facts may be merged; different facts stay separate
DUPLICATE_SIMILARITY = 0.6  # a "new" fact this close to a stored one is a duplicate


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

ORGANIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "action": {"type": "string", "enum": ["rewrite", "archive"]},
                    "about": {"type": "string"},
                    "text": {"type": "string"},
                    "replaces": {"type": "array", "items": {"type": "integer"}},
                    "duplicate_of": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "action", "about", "text", "replaces", "duplicate_of", "reason"],
            },
        },
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "summary": {"type": "string"},
                    "fact_ids": {"type": "array", "items": {"type": "integer"}},
                    "people": {"type": "array", "items": {"type": "string"}},
                    "related": {"type": "array", "items": {"type": "string"}},
                    "merged_from": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "summary", "fact_ids", "people", "related", "merged_from"],
            },
        },
        "profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"person": {"type": "string"}, "profile": {"type": "string"}},
                "required": ["person", "profile"],
            },
        },
    },
    "required": ["changes", "topics", "profiles"],
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

ORGANIZE_PROMPT = """All facts Housemate currently remembers (#id [about] text):
{facts}

People Housemate knows: {people}
Existing topics (name: summary; people; related):
{topics}

Reorganize this memory. Return JSON:
- changes: only facts that need fixing. Each has id, action, about, text, replaces, duplicate_of and a short reason (max 15 words):
  - "rewrite": the corrected "text" (one short sentence, one fact) and "about". Use it to attach a fact to the right person (if a "household" fact is clearly about a known person, set about to that person's name), to correct a fact that a newer one updates, or to merge facts that say THE SAME THING into one sentence: then list the other ids in "replaces" (they are removed). Never merge different facts into one long sentence. Never move a fact about a person to "household".
  - "archive": the fact is trivial, only about one moment, or contradicted by a newer fact; or it duplicates another fact, then put that fact's id in "duplicate_of" (else 0). Copy "about" and "text" from the original; "replaces" is empty.
  "about" is always a known person's name or "household". Leave correct facts alone. Usually only a few changes are needed.
- topics: group the facts into topics (a fact may belong to several). Each topic: a short name, a one-sentence summary of what Housemate knows about it, fact_ids (ids of the facts above that belong to it), people (known people connected to it), related (names of other topics in your list that belong together with it), merged_from (names of existing topics that are duplicates of this one, else empty). Keep existing topic names when they still fit. Prefer a handful of meaningful topics over many tiny ones.
- profiles: for each known person with facts, a 1-2 sentence profile in third person: who they are and what matters to them. No momentary details (clothes, what they are doing right now)."""


class Consolidator:
    def __init__(self, agent: "Agent"):
        self.agent, self.db, self.llm = agent, agent.db, agent.llm
        self.running = False
        self.last_run = float(self.db.get_setting("consolidated_at", "0"))

    # ---------------------------------------------------------------- triggers

    def room_idle(self) -> bool:
        if self.agent.user_busy or self.agent.presence.present():
            return False
        last_msg = self.db.last_message_ts() or 0
        last_seen = max((p["last_seen"] or 0 for p in self.db.people()), default=0)
        return time.time() - max(last_msg, last_seen) >= config.CONSOLIDATE_IDLE_S

    async def loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                if (
                    time.time() - self.last_run >= config.CONSOLIDATE_INTERVAL_S
                    and self.room_idle()
                    and (self.db.memory_count_since(self.last_run) or self.db.messages_since(self.last_run, 1))
                ):
                    await self.run("idle")
            except Exception:
                log.exception("consolidation failed")

    # --------------------------------------------------------------------- run

    async def run(self, trigger: str) -> dict:
        if self.running:
            return {"status": "already running"}
        self.running = True
        try:
            return await self._run(trigger)
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
        stats = {"model": model, "trigger": trigger, "added": 0, "rewritten": 0, "archived": 0,
                 "topics": 0, "merged_topics": 0, "links": 0, "profiles": 0, "changes": []}
        backup = self._backup()
        log.info("memory review started (%s, %s), backup %s", model, trigger, backup)
        await self.agent.emit("consolidation", f"memory review started ({model})", {"backup": backup})

        await self._catch_up(model, stats)
        if trigger != "manual" and not self.room_idle():
            await self._finish(stats, t0, "paused after catch-up: someone is around")
            return stats
        await self._organize(model, stats)
        await self._finish(stats, t0, None)
        return stats

    async def _finish(self, stats: dict, t0: float, note: str | None) -> None:
        self.last_run = time.time()
        self.db.set_setting("consolidated_at", str(self.last_run))
        removed = self.db.delete_empty_topics()
        parts = [
            f"{stats['added']} new facts", f"{stats['rewritten']} rewritten", f"{stats['archived']} archived",
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
        path = folder / f"laptopai-{datetime.now():%Y%m%d-%H%M%S}.db"
        self.db.backup(path)
        for old in sorted(folder.glob("laptopai-*.db"))[: -config.BACKUPS_KEEP]:
            old.unlink()
        return f"backups/{path.name}"

    async def _ask(self, model: str, prompt: str, schema: dict, max_tokens: int) -> dict:
        options = {"num_ctx": config.MEMORY_NUM_CTX}
        if config.MEMORY_NUM_GPU >= 0:
            options["num_gpu"] = config.MEMORY_NUM_GPU
        raw = await self.llm.complete(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            schema=schema, max_tokens=max_tokens, temperature=0.1, model=model,
            think=config.MEMORY_THINK, options=options, keep_alive="5m",
        )
        return json.loads(raw)

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
                if not fact or MOMENTARY.search(fact):
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

    @staticmethod
    def _format_message(m: dict) -> str:  # noqa: C901
        if m["role"] == "assistant":
            who = "Housemate" + (" (on its own)" if m["source"] == "initiative" else "")
        elif m["source"] == "developer":
            who = f"{m['sender'] or 'a developer tool'} (developer tool)"
        else:
            in_view = ", ".join(json.loads(m["people"] or "[]")) or "nobody recognized"
            who = f"{'spoken' if m['source'] == 'voice' else 'typed'} by someone in the room (in view: {in_view})"
        return f"{hhmm(m['ts'])} {who}: {m['content']}"

    # ---------------------------------------------------- step 2: organizing

    async def _organize(self, model: str, stats: dict) -> None:
        memories = {m["id"]: m for m in self.db.active_memories()}
        if not memories:
            return
        people = {p["name"].lower(): p for p in self.db.people()}
        topics = self.db.topics()
        topic_lines = "\n".join(
            f"- {t['name']}: {t['summary'] or '(no summary yet)'}; people: "
            f"{', '.join(p['name'] for p in t['people']) or '-'}; related: {', '.join(r['name'] for r in t['related']) or '-'}"
            for t in topics
        ) or "none yet"
        result = await self._ask(model, ORGANIZE_PROMPT.format(
            facts="\n".join(f"#{m['id']} [{m['person'] or 'household'}] {m['text']}" for m in memories.values()),
            people=", ".join(p["name"] for p in people.values()) or "nobody yet",
            topics=topic_lines,
        ), ORGANIZE_SCHEMA, 6000)

        # --- fact changes (bounded, reversible)
        max_archive = max(3, int(len(memories) * config.CONSOLIDATE_MAX_ARCHIVE_FRACTION))
        renamed: dict[int, int | None] = {}  # old id -> new id (rewritten) or None (archived)
        name_of = {p["id"]: p["name"] for p in people.values()}

        def archive(mid: int, reason: str, replaced_by: int | None = None) -> bool:
            if stats["archived"] >= max_archive:
                log.info("archive limit reached, keeping #%s", mid)
                return False
            self.db.archive_memory(mid, reason, replaced_by)
            renamed[mid] = None
            stats["archived"] += 1
            stats["changes"].append({"archived": memories[mid]["text"], "reason": reason})
            return True

        for ch in result.get("changes", []):
            mid, action = ch.get("id"), ch.get("action")
            reason = (ch.get("reason", "").strip() or "consolidation")[:200]
            old = memories.get(mid)
            if old is None or mid in renamed:
                continue
            if action == "archive":
                dup = memories.get(ch.get("duplicate_of") or 0)
                if dup and dup["id"] not in renamed and old["person_id"] and not dup["person_id"]:
                    # Keep the version that is attached to a person; drop the household copy instead.
                    archive(dup["id"], f"duplicate of #{mid}, which is attached to {name_of.get(old['person_id'])}")
                    continue
                archive(mid, reason)
            elif action == "rewrite":
                text = ch.get("text", "").strip() or old["text"]
                if MOMENTARY.search(text):
                    continue
                person = people.get(ch.get("about", "").strip().lower()) or person_named_in(text, people)
                # A fact about a person stays with a person: "household" never detaches it.
                pid = person["id"] if person else old["person_id"]
                if len(text) > min(240, len(old["text"]) * 1.25 + 15):
                    log.info("rewrite of #%s grows the fact too much (merging different facts?), skipped: %r", mid, text)
                    continue
                if text == old["text"] and pid == old["person_id"]:
                    continue
                new_id = self.db.replace_memory(mid, text, pid, reason)
                renamed[mid] = new_id
                stats["rewritten"] += 1
                stats["changes"].append({"from": old["text"], "to": text, "about": name_of.get(pid, "household"),
                                         "reason": reason})
                for rid in ch.get("replaces", []):
                    other = memories.get(rid)
                    # merge only facts about the same person (or household facts into anything)
                    if (other and rid != mid and rid not in renamed and other["person_id"] in (None, pid)
                            and word_similarity(other["text"], old["text"]) >= MERGE_MIN_SIMILARITY):
                        archive(rid, f"merged into #{new_id}", replaced_by=new_id)

        # --- topics
        existing = {t["name"].lower(): t["id"] for t in topics}
        new_topics = [t for t in result.get("topics", []) if t.get("name", "").strip()][:20]
        name_to_id: dict[str, int] = {}
        for t in new_topics:  # pass 1: create/update all topics
            name_to_id[t["name"].strip().lower()] = self.db.upsert_topic(t["name"], t.get("summary", ""))
            stats["topics"] += 1

        merged_into: dict[int, int] = {}  # pass 2: merges (a merged topic is deleted; references are redirected)

        def resolve(tid: int | None) -> int | None:
            seen = set()
            while tid in merged_into and tid not in seen:
                seen.add(tid)
                tid = merged_into[tid]
            return tid

        for t in new_topics:
            dst = resolve(name_to_id[t["name"].strip().lower()])
            for src in t.get("merged_from", []):
                src_id = resolve(existing.get(str(src).strip().lower()) or name_to_id.get(str(src).strip().lower()))
                if src_id and dst and src_id != dst:
                    self.db.merge_topic(src_id, dst)
                    merged_into[src_id] = dst
                    stats["merged_topics"] += 1

        fact_topics: dict[int, list[int]] = {}  # pass 3: facts, people, related topics
        for t in new_topics:
            tid = resolve(name_to_id[t["name"].strip().lower()])
            for fid in t.get("fact_ids", []):
                fid = renamed.get(fid, fid)
                if fid is not None and (fid in memories or fid in renamed.values()):
                    fact_topics.setdefault(fid, []).append(tid)
            for pname in t.get("people", []):
                if person := people.get(str(pname).strip().lower()):
                    self.db.ensure_topic_person(tid, person["id"])
            for rel in t.get("related", []):
                key = str(rel).strip().lower()
                other = resolve(name_to_id.get(key) or existing.get(key))
                if other and other != tid:
                    self.db.link_topics(tid, other)
                    stats["links"] += 1
        for fid, tids in fact_topics.items():
            self.db.set_memory_topics(fid, tids)

        # --- profiles: short and about who someone is, not what they wore today
        for p in result.get("profiles", []):
            if (person := people.get(p.get("person", "").strip().lower())) and p.get("profile", "").strip():
                sentences = [x for x in re.split(r"(?<=[.!?])\s+", p["profile"].strip()) if x and not MOMENTARY.search(x)]
                if sentences:
                    self.db.set_profile(person["id"], " ".join(sentences[:3]))
                    stats["profiles"] += 1

        # --- related topics: a small focused question works much better than a field in the big one
        await self._link_topics(model, stats)

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
