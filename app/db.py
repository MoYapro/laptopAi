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

    def touch_person(self, pid: int, ts: float) -> None:
        self._q("UPDATE people SET last_seen = ? WHERE id = ?", (ts, pid))

    def add_embedding(self, pid: int, vec: np.ndarray) -> None:
        self._insert(
            "INSERT INTO face_embeddings (person_id, ts, vec) VALUES (?, ?, ?)",
            (pid, time.time(), vec.astype(np.float32).tobytes()),
        )

    def embeddings(self) -> list[tuple[int, np.ndarray]]:
        rows = self._q("SELECT person_id, vec FROM face_embeddings")
        return [(r["person_id"], np.frombuffer(r["vec"], dtype=np.float32)) for r in rows]

    # ----------------------------------------------------------- memories

    def add_memory(self, text: str, pid: int | None) -> bool:
        """Store a fact unless an identical one exists. Returns True if stored."""
        text = text.strip()
        dup = self._q(
            "SELECT 1 FROM memories WHERE person_id IS ? AND lower(text) = lower(?)", (pid, text)
        )
        if dup or not text:
            return False
        self._insert("INSERT INTO memories (ts, person_id, text) VALUES (?, ?, ?)", (time.time(), pid, text))
        return True

    def memories(self, pid: int | None, limit: int) -> list[dict]:
        rows = self._q(
            "SELECT ts, text FROM memories WHERE person_id IS ? ORDER BY ts DESC LIMIT ?", (pid, limit)
        )
        return [dict(r) for r in reversed(rows)]

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
