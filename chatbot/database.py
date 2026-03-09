"""
database.py — SQLite persistence layer for the AI Chatbot
==========================================================
Tables
------
messages
    id          INTEGER  PRIMARY KEY AUTOINCREMENT
    session_id  TEXT     NOT NULL          -- IP or custom session token
    role        TEXT     NOT NULL          -- 'user' | 'bot'
    content     TEXT     NOT NULL          -- raw message text
    model       TEXT                       -- 'nlp' | 'llama3.2' | 'gemma3' | NULL
    intent      TEXT                       -- top NLP intent tag, or NULL
    timestamp   TEXT     NOT NULL          -- ISO-8601 UTC

Usage
-----
    from database import db
    db.log_message(session_id, role, content, model, intent)
    history = db.get_history(session_id, limit=50)
    db.clear_history(session_id)
    stats   = db.get_stats()
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

# ── Config ───────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent / ".env")

# DB_PATH can be overridden in .env:  DB_PATH=chatbot.db  or  DB_PATH=/data/chat.db
_db_env = os.getenv("DB_PATH", "chatbot.db")
DB_PATH = Path(_db_env) if Path(_db_env).is_absolute() else Path(__file__).parent / _db_env

# ── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    role        TEXT    NOT NULL CHECK(role IN ('user', 'bot')),
    content     TEXT    NOT NULL,
    model       TEXT,
    intent      TEXT,
    timestamp   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages (session_id);

CREATE INDEX IF NOT EXISTS idx_messages_timestamp
    ON messages (timestamp);
"""

# ── Database class ────────────────────────────────────────────────────────────

class ChatDatabase:
    """Thread-safe SQLite wrapper using per-thread connections."""

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._local   = threading.local()
        self._init_db()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return (or create) a per-thread connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
            conn.execute("PRAGMA foreign_keys=ON;")
            self._local.conn = conn
        return self._local.conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._connect()

    def _init_db(self):
        """Create tables and indexes (idempotent)."""
        with self._conn:
            self._conn.executescript(_SCHEMA)

    @staticmethod
    def _now_utc() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Public API ────────────────────────────────────────────────────────────

    def log_message(
        self,
        session_id: str,
        role: str,          # 'user' | 'bot'
        content: str,
        model: str  = None, # 'nlp' | 'llama3.2' | 'gemma3'
        intent: str = None, # top intent tag from NLP classifier
    ) -> int:
        """Insert one message row; return its new row id."""
        sql = """
            INSERT INTO messages (session_id, role, content, model, intent, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cur = self._conn.execute(sql, (
            session_id, role, content, model, intent, self._now_utc()
        ))
        self._conn.commit()
        return cur.lastrowid

    def get_history(
        self,
        session_id: str,
        limit: int = 100,
    ) -> list[dict]:
        """Return the last `limit` messages for a session (oldest first)."""
        sql = """
            SELECT id, role, content, model, intent, timestamp
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """
        rows = self._conn.execute(sql, (session_id, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def clear_history(self, session_id: str) -> int:
        """Delete all messages for a session; return rows deleted."""
        cur = self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def get_stats(self) -> dict:
        """Return aggregate stats across all sessions."""
        row = self._conn.execute("""
            SELECT
                COUNT(*)                                    AS total_messages,
                COUNT(DISTINCT session_id)                  AS unique_sessions,
                SUM(CASE WHEN role='user' THEN 1 ELSE 0 END) AS user_messages,
                SUM(CASE WHEN role='bot'  THEN 1 ELSE 0 END) AS bot_messages
            FROM messages
        """).fetchone()
        top_intents = self._conn.execute("""
            SELECT intent, COUNT(*) AS cnt
            FROM   messages
            WHERE  role = 'user' AND intent IS NOT NULL
            GROUP  BY intent
            ORDER  BY cnt DESC
            LIMIT  5
        """).fetchall()
        return {
            "total_messages":   row["total_messages"],
            "unique_sessions":  row["unique_sessions"],
            "user_messages":    row["user_messages"],
            "bot_messages":     row["bot_messages"],
            "top_intents":      [dict(r) for r in top_intents],
        }

    def get_all_sessions(self) -> list[dict]:
        """Return a summary row per session (latest activity, message count)."""
        rows = self._conn.execute("""
            SELECT
                session_id,
                COUNT(*) AS message_count,
                MAX(timestamp) AS last_active
            FROM messages
            GROUP BY session_id
            ORDER BY last_active DESC
        """).fetchall()
        return [dict(r) for r in rows]


# ── Singleton ─────────────────────────────────────────────────────────────────

db = ChatDatabase()
