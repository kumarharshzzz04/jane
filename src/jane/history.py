"""Persistent chat history in SQLite — simple, local, upgradeable."""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    title TEXT NOT NULL DEFAULT 'session'
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


@dataclass
class HistoryRecord:
    role: str
    content: str
    ts: float


class HistoryStore:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Streamlit runs script threads per-rerun: allow cross-thread use + lock.
        self._lock = threading.Lock()
        self._con = sqlite3.connect(str(db_path), check_same_thread=False)
        self._con.executescript(_SCHEMA)
        self._con.commit()

    def close(self) -> None:
        self._con.close()

    # sessions ----------------------------------------------------------

    def new_session(self, title: str = "session") -> int:
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO sessions (created_at, title) VALUES (?, ?)",
                (time.time(), title[:120]),
            )
            self._con.commit()
            return int(cur.lastrowid)

    def list_sessions(self, limit: int = 50) -> list[tuple[int, str, float]]:
        with self._lock:
            rows = self._con.execute(
                "SELECT id, title, created_at FROM sessions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(int(r[0]), str(r[1]), float(r[2])) for r in rows]

    def rename_session(self, session_id: int, title: str) -> None:
        with self._lock:
            self._con.execute("UPDATE sessions SET title = ? WHERE id = ?",
                              (title[:120], session_id))
            self._con.commit()

    # messages -----------------------------------------------------------

    def add_message(self, session_id: int, role: str, content: str) -> None:
        with self._lock:
            self._con.execute(
                "INSERT INTO messages (session_id, ts, role, content) VALUES (?, ?, ?, ?)",
                (session_id, time.time(), role, content),
            )
            self._con.commit()

    def get_messages(self, session_id: int, limit: Optional[int] = None) -> list[HistoryRecord]:
        with self._lock:
            q = "SELECT role, content, ts FROM messages WHERE session_id = ? ORDER BY id"
            if limit:
                q += " DESC LIMIT " + str(int(limit))
                rows = self._con.execute(q, (session_id,)).fetchall()
                rows.reverse()
            else:
                rows = self._con.execute(q, (session_id,)).fetchall()
        return [HistoryRecord(role=r[0], content=r[1], ts=float(r[2])) for r in rows]

    def delete_session(self, session_id: int) -> None:
        with self._lock:
            self._con.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._con.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._con.commit()
