"""SQLite store: timeline events, chat messages, people, face embeddings, memories, settings."""

import json
import sqlite3
import threading
import time

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, data TEXT
);
CREATE INDEX IF NOT EXISTS events_ts ON events (ts);
CREATE INDEX IF NOT EXISTS events_kind_ts ON events (kind, ts);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
    source TEXT, people TEXT
);
CREATE INDEX IF NOT EXISTS messages_ts ON messages (ts);

CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, created REAL NOT NULL, last_seen REAL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY, person_id INTEGER NOT NULL REFERENCES people (id) ON DELETE CASCADE,
    ts REAL NOT NULL, vec BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL,
    person_id INTEGER REFERENCES people (id) ON DELETE CASCADE,  -- NULL = household / general
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);

-- things the AI wants to find out about its surroundings and the people in them
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY, ts REAL NOT NULL, text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',  -- open | answered | dropped
    asked_count INTEGER NOT NULL DEFAULT 0, last_asked REAL,
    answer TEXT, answered_ts REAL
);

-- organized knowledge (built by the consolidation job, see consolidation.py)
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE, summary TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL, updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_topics (
    memory_id INTEGER NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
    topic_id INTEGER NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, topic_id)
);
CREATE TABLE IF NOT EXISTS topic_links (  -- undirected: a < b
    a INTEGER NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    b INTEGER NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    ts REAL NOT NULL, PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS topic_people (  -- who it talked about a topic with, and when
    topic_id INTEGER NOT NULL REFERENCES topics (id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES people (id) ON DELETE CASCADE,
    first_ts REAL NOT NULL, last_ts REAL NOT NULL, count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (topic_id, person_id)
);
"""


class DB:
    def __init__(self, path):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            # migrations for databases created by earlier versions
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(messages)")}
            if "sender" not in cols:
                self._conn.execute("ALTER TABLE messages ADD COLUMN sender TEXT")
            # embeddings from different face models can't be compared; tag each with its model
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(face_embeddings)")}
            if "model" not in cols:
                self._conn.execute("ALTER TABLE face_embeddings ADD COLUMN model TEXT NOT NULL DEFAULT 'sface'")
            # memories: soft delete / audit trail for the consolidation job
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(memories)")}
            for col, typ in [("archived_ts", "REAL"), ("archive_reason", "TEXT"), ("replaced_by", "INTEGER")]:
                if col not in cols:
                    self._conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {typ}")
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(people)")}
            for col, typ in [("profile", "TEXT"), ("profile_ts", "REAL")]:
                if col not in cols:
                    self._conn.execute(f"ALTER TABLE people ADD COLUMN {col} {typ}")
            # quiet-mode initiatives used to share the "initiative" kind (and so ate into the hourly cap)
            self._conn.execute(
                "UPDATE events SET kind = 'initiative_quiet' WHERE kind = 'initiative' AND data LIKE '%\"quiet\": true%'"
            )

    def _q(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _insert(self, sql: str, args=()) -> int:
        with self._lock:
            return self._conn.execute(sql, args).lastrowid

    # ------------------------------------------------------------- events

    def add_event(self, kind: str, text: str, data: dict | None = None, ts: float | None = None) -> dict:
        ts = ts or time.time()
        eid = self._insert(
            "INSERT INTO events (ts, kind, text, data) VALUES (?, ?, ?, ?)",
            (ts, kind, text, json.dumps(data) if data else None),
        )
        return {"id": eid, "ts": ts, "kind": kind, "text": text, "data": data}

    @staticmethod
    def _event(row) -> dict:
        return {
            "id": row["id"], "ts": row["ts"], "kind": row["kind"], "text": row["text"],
            "data": json.loads(row["data"]) if row["data"] else None,
        }

    def recent_events(self, limit: int, kinds: list[str] | None = None, exclude: list[str] | None = None) -> list[dict]:
        sql, args = "SELECT * FROM events", []
        conds = []
        if kinds:
            conds.append(f"kind IN ({','.join('?' * len(kinds))})")
            args += kinds
        if exclude:
            conds.append(f"kind NOT IN ({','.join('?' * len(exclude))})")
            args += exclude
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        rows = self._q(sql + " ORDER BY ts DESC LIMIT ?", (*args, limit))
        return [self._event(r) for r in reversed(rows)]

    def last_event(self, kinds: list[str]) -> dict | None:
        events = self.recent_events(1, kinds=kinds)
        return events[0] if events else None

    def count_events_since(self, kinds: list[str], since: float) -> int:
        rows = self._q(
            f"SELECT COUNT(*) AS n FROM events WHERE kind IN ({','.join('?' * len(kinds))}) AND ts >= ?",
            (*kinds, since),
        )
        return rows[0]["n"]

    # ----------------------------------------------------------- messages

    def add_message(
        self, role: str, content: str, source: str, people: list[str] | None = None, sender: str | None = None
    ) -> dict:
        ts = time.time()
        self._insert(
            "INSERT INTO messages (ts, role, content, source, people, sender) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, role, content, source, json.dumps(people or []), sender),
        )
        return {"ts": ts, "role": role, "content": content, "source": source, "sender": sender}

    def recent_messages(self, limit: int, since: float = 0) -> list[dict]:
        rows = self._q(
            "SELECT ts, role, content, source, sender FROM messages WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
            (since, limit),
        )
        return [dict(r) for r in reversed(rows)]

    def last_message_ts(self, role: str | None = None) -> float | None:
        if role:
            rows = self._q("SELECT MAX(ts) AS ts FROM messages WHERE role = ?", (role,))
        else:
            rows = self._q("SELECT MAX(ts) AS ts FROM messages")
        return rows[0]["ts"]

    # ------------------------------------------------------------- people

    def person_by_name(self, name: str) -> dict | None:
        rows = self._q("SELECT * FROM people WHERE name = ?", (name.strip(),))
        return dict(rows[0]) if rows else None

    def person(self, pid: int) -> dict | None:
        rows = self._q("SELECT * FROM people WHERE id = ?", (pid,))
        return dict(rows[0]) if rows else None

    def get_or_create_person(self, name: str) -> dict:
        name = name.strip()
        if existing := self.person_by_name(name):
            return existing
        pid = self._insert("INSERT INTO people (name, created) VALUES (?, ?)", (name, time.time()))
        return self.person(pid)

    def people(self) -> list[dict]:
        return [dict(r) for r in self._q("SELECT * FROM people ORDER BY name")]

    def touch_person(self, pid: int, ts: float) -> None:
        self._q("UPDATE people SET last_seen = ? WHERE id = ?", (ts, pid))

    def add_embedding(self, pid: int, vec: np.ndarray, model: str) -> None:
        self._insert(
            "INSERT INTO face_embeddings (person_id, ts, vec, model) VALUES (?, ?, ?, ?)",
            (pid, time.time(), vec.astype(np.float32).tobytes(), model),
        )

    def embeddings(self, model: str) -> list[tuple[int, np.ndarray]]:
        rows = self._q("SELECT person_id, vec FROM face_embeddings WHERE model = ?", (model,))
        return [(r["person_id"], np.frombuffer(r["vec"], dtype=np.float32)) for r in rows]

    # ----------------------------------------------------------- memories

    def active_memories(self, limit: int = 300) -> list[dict]:
        rows = self._q(
            "SELECT m.id, m.ts, m.person_id, p.name AS person, m.text FROM memories m "
            "LEFT JOIN people p ON p.id = m.person_id WHERE m.archived_ts IS NULL ORDER BY m.ts LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in rows]

    def memory_count_since(self, ts: float) -> int:
        return self._q("SELECT COUNT(*) AS n FROM memories WHERE ts > ? OR archived_ts > ?", (ts, ts))[0]["n"]

    def archive_memory(self, mid: int, reason: str, replaced_by: int | None = None) -> None:
        self._q(
            "UPDATE memories SET archived_ts = ?, archive_reason = ?, replaced_by = ? WHERE id = ? AND archived_ts IS NULL",
            (time.time(), reason, replaced_by, mid),
        )

    def replace_memory(self, mid: int, text: str, pid: int | None, reason: str) -> int:
        """Reversible edit: the old row is archived and points to the new one."""
        new_id = self._insert(
            "INSERT INTO memories (ts, person_id, text) SELECT ts, ?, ? FROM memories WHERE id = ?", (pid, text, mid)
        )
        self.archive_memory(mid, reason, replaced_by=new_id)
        return new_id

    def add_memory(self, text: str, pid: int | None) -> bool:
        """Store a fact unless an identical one exists. Returns True if stored."""
        text = text.strip()
        dup = self._q(
            "SELECT 1 FROM memories WHERE person_id IS ? AND lower(text) = lower(?) AND archived_ts IS NULL",
            (pid, text),
        )
        if dup or not text:
            return False
        self._insert("INSERT INTO memories (ts, person_id, text) VALUES (?, ?, ?)", (time.time(), pid, text))
        return True

    def memories(self, pid: int | None, limit: int) -> list[dict]:
        rows = self._q(
            "SELECT id, ts, text FROM memories WHERE person_id IS ? AND archived_ts IS NULL ORDER BY ts DESC LIMIT ?",
            (pid, limit),
        )
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------- topics

    def upsert_topic(self, name: str, summary: str | None = None) -> int:
        name = name.strip()
        now = time.time()
        rows = self._q("SELECT id FROM topics WHERE name = ?", (name,))
        if rows:
            if summary is not None:
                self._q("UPDATE topics SET summary = ?, updated = ? WHERE id = ?", (summary.strip(), now, rows[0]["id"]))
            return rows[0]["id"]
        return self._insert(
            "INSERT INTO topics (name, summary, created, updated) VALUES (?, ?, ?, ?)", (name, (summary or "").strip(), now, now)
        )

    def set_memory_topics(self, mid: int, topic_ids: list[int]) -> None:
        self._q("DELETE FROM memory_topics WHERE memory_id = ?", (mid,))
        for tid in set(topic_ids):
            self._q("INSERT OR IGNORE INTO memory_topics (memory_id, topic_id) VALUES (?, ?)", (mid, tid))

    def link_topics(self, a: int, b: int) -> None:
        if a != b:
            self._q("INSERT OR IGNORE INTO topic_links (a, b, ts) VALUES (?, ?, ?)", (min(a, b), max(a, b), time.time()))

    def touch_topic_person(self, tid: int, pid: int, ts: float | None = None) -> None:
        ts = ts or time.time()
        self._q(
            "INSERT INTO topic_people (topic_id, person_id, first_ts, last_ts, count) VALUES (?, ?, ?, ?, 1) "
            "ON CONFLICT (topic_id, person_id) DO UPDATE SET last_ts = max(last_ts, excluded.last_ts), count = count + 1",
            (tid, pid, ts, ts),
        )

    def ensure_topic_person(self, tid: int, pid: int) -> None:
        """Link a person to a topic without counting it as a conversation (count stays 0 if new)."""
        now = time.time()
        self._q(
            "INSERT OR IGNORE INTO topic_people (topic_id, person_id, first_ts, last_ts, count) VALUES (?, ?, ?, ?, 0)",
            (tid, pid, now, now),
        )

    def merge_topic(self, src: int, dst: int) -> None:
        """Move facts, people and links of topic src onto dst, then delete src."""
        with self._lock:
            c = self._conn
            c.execute("INSERT OR IGNORE INTO memory_topics (memory_id, topic_id) "
                      "SELECT memory_id, ? FROM memory_topics WHERE topic_id = ?", (dst, src))
            c.execute(
                "INSERT INTO topic_people (topic_id, person_id, first_ts, last_ts, count) "
                "SELECT ?, person_id, first_ts, last_ts, count FROM topic_people WHERE topic_id = ? "
                "ON CONFLICT (topic_id, person_id) DO UPDATE SET first_ts = min(first_ts, excluded.first_ts), "
                "last_ts = max(last_ts, excluded.last_ts), count = count + excluded.count",
                (dst, src),
            )
            for a, b in c.execute("SELECT a, b FROM topic_links WHERE a = ? OR b = ?", (src, src)).fetchall():
                other = b if a == src else a
                if other != dst:
                    c.execute("INSERT OR IGNORE INTO topic_links (a, b, ts) VALUES (?, ?, ?)",
                              (min(other, dst), max(other, dst), time.time()))
            c.execute("DELETE FROM topics WHERE id = ?", (src,))

    def delete_empty_topics(self) -> int:
        """Topics with no active facts and nobody linked are leftovers of earlier runs."""
        rows = self._q(
            "SELECT t.id FROM topics t WHERE NOT EXISTS (SELECT 1 FROM memory_topics mt JOIN memories m ON m.id = mt.memory_id "
            "WHERE mt.topic_id = t.id AND m.archived_ts IS NULL) AND NOT EXISTS (SELECT 1 FROM topic_people tp WHERE tp.topic_id = t.id)"
        )
        for r in rows:
            self._q("DELETE FROM topics WHERE id = ?", (r["id"],))
        return len(rows)

    def topics(self) -> list[dict]:
        """All topics with their facts, people and related topics (for context, the memory page, consolidation)."""
        out = []
        for t in self._q("SELECT * FROM topics ORDER BY updated DESC"):
            tid = t["id"]
            facts = self._q(
                "SELECT m.id, m.text, p.name AS person FROM memory_topics mt JOIN memories m ON m.id = mt.memory_id "
                "LEFT JOIN people p ON p.id = m.person_id WHERE mt.topic_id = ? AND m.archived_ts IS NULL ORDER BY m.ts",
                (tid,),
            )
            people = self._q(
                "SELECT p.id, p.name, tp.last_ts, tp.count FROM topic_people tp JOIN people p ON p.id = tp.person_id "
                "WHERE tp.topic_id = ? ORDER BY tp.last_ts DESC",
                (tid,),
            )
            related = self._q(
                "SELECT t2.id, t2.name FROM topic_links l JOIN topics t2 ON t2.id = CASE WHEN l.a = ? THEN l.b ELSE l.a END "
                "WHERE l.a = ? OR l.b = ? ORDER BY t2.name",
                (tid, tid, tid),
            )
            out.append({
                "id": tid, "name": t["name"], "summary": t["summary"], "updated": t["updated"],
                "facts": [dict(f) for f in facts], "people": [dict(p) for p in people],
                "related": [dict(r) for r in related],
            })
        return out

    def topics_for_person(self, pid: int, limit: int = 6) -> list[dict]:
        rows = self._q(
            "SELECT t.id, t.name, t.summary, tp.last_ts, tp.count FROM topic_people tp JOIN topics t ON t.id = tp.topic_id "
            "WHERE tp.person_id = ? ORDER BY tp.last_ts DESC LIMIT ?",
            (pid, limit),
        )
        return [dict(r) for r in rows]

    def related_topic_names(self, tid: int) -> list[str]:
        rows = self._q(
            "SELECT t2.name FROM topic_links l JOIN topics t2 ON t2.id = CASE WHEN l.a = ? THEN l.b ELSE l.a END "
            "WHERE l.a = ? OR l.b = ?",
            (tid, tid, tid),
        )
        return [r["name"] for r in rows]

    def set_profile(self, pid: int, profile: str) -> None:
        self._q("UPDATE people SET profile = ?, profile_ts = ? WHERE id = ?", (profile.strip(), time.time(), pid))

    def messages_since(self, ts: float, limit: int = 200) -> list[dict]:
        rows = self._q(
            "SELECT ts, role, content, source, sender, people FROM messages WHERE ts > ? ORDER BY ts LIMIT ?", (ts, limit)
        )
        return [dict(r) for r in rows]

    def backup(self, path) -> None:
        dest = sqlite3.connect(path)
        with self._lock:
            self._conn.backup(dest)
        dest.close()

    # ---------------------------------------------------------- questions

    def add_question(self, text: str) -> dict:
        ts = time.time()
        qid = self._insert("INSERT INTO questions (ts, text) VALUES (?, ?)", (ts, text.strip()))
        return {"id": qid, "ts": ts, "text": text.strip(), "asked_count": 0, "last_asked": None}

    def open_questions(self, limit: int = 50) -> list[dict]:
        rows = self._q("SELECT * FROM questions WHERE status = 'open' ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def question(self, qid: int) -> dict | None:
        rows = self._q("SELECT * FROM questions WHERE id = ?", (qid,))
        return dict(rows[0]) if rows else None

    def all_questions(self, limit: int) -> list[dict]:
        rows = self._q("SELECT * FROM questions ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def mark_asked(self, qid: int) -> None:
        self._q(
            "UPDATE questions SET asked_count = asked_count + 1, last_asked = ? WHERE id = ?", (time.time(), qid)
        )

    def set_question_status(self, qid: int, status: str, answer: str | None = None) -> None:
        self._q(
            "UPDATE questions SET status = ?, answer = ?, answered_ts = ? WHERE id = ?",
            (status, answer, time.time() if answer else None, qid),
        )

    # ----------------------------------------------------------- settings

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        rows = self._q("SELECT value FROM settings WHERE key = ?", (key,))
        return rows[0]["value"] if rows else default

    def set_setting(self, key: str, value: str) -> None:
        self._q(
            "INSERT INTO settings (key, value) VALUES (?, ?) "
            "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
