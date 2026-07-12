from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
import uuid


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationRepository:
    def __init__(self, database_file: str):
        self.database_file = str(database_file)

    def _connect(self):
        path = Path(self.database_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_file, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS anonymous_sessions (
                    id TEXT PRIMARY KEY,
                    session_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES anonymous_sessions(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL UNIQUE,
                    thread_id TEXT NOT NULL REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL CHECK(status IN ('processing', 'completed', 'failed')),
                    trace_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_threads_owner_updated
                    ON chat_threads(owner_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_thread_sequence
                    ON chat_messages(thread_id, sequence);
            """)

    def get_or_create_owner(self, session_hash: str) -> str:
        now = _now()
        with self._connect() as db:
            row = db.execute(
                "SELECT id FROM anonymous_sessions WHERE session_hash = ?", (session_hash,)
            ).fetchone()
            if row:
                db.execute("UPDATE anonymous_sessions SET last_seen_at = ? WHERE id = ?", (now, row["id"]))
                return row["id"]
            owner_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO anonymous_sessions(id, session_hash, created_at, last_seen_at) VALUES (?, ?, ?, ?)",
                (owner_id, session_hash, now, now),
            )
            return owner_id

    def create_thread(self, owner_id: str, title: str) -> dict:
        thread_id, now = str(uuid.uuid4()), _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO chat_threads(thread_id, owner_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, owner_id, title, now, now),
            )
        return self.get_thread(owner_id, thread_id)

    def get_thread(self, owner_id: str, thread_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT thread_id, title, created_at, updated_at FROM chat_threads WHERE thread_id = ? AND owner_id = ?",
                (thread_id, owner_id),
            ).fetchone()
        return dict(row) if row else None

    def list_threads(self, owner_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT thread_id, title, created_at, updated_at FROM chat_threads WHERE owner_id = ? ORDER BY updated_at DESC",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_thread(self, owner_id: str, thread_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute(
                "DELETE FROM chat_threads WHERE thread_id = ? AND owner_id = ?", (thread_id, owner_id)
            )
        return cursor.rowcount > 0

    def create_message(self, thread_id: str, role: str, content: str, status: str,
                       sources: list | None = None, trace_id: str | None = None) -> dict:
        message_id, now = str(uuid.uuid4()), _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO chat_messages(message_id, thread_id, role, content, sources_json, status, trace_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (message_id, thread_id, role, content, json.dumps(sources or [], ensure_ascii=False), status, trace_id, now),
            )
            db.execute("UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
        return {"message_id": message_id, "thread_id": thread_id, "role": role,
                "content": content, "sources": sources or [], "status": status,
                "trace_id": trace_id, "created_at": now}

    def update_message_status(self, message_id: str, status: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE chat_messages SET status = ? WHERE message_id = ?", (status, message_id))

    def list_messages(self, owner_id: str, thread_id: str, completed_only: bool = False) -> list[dict] | None:
        if not self.get_thread(owner_id, thread_id):
            return None
        condition = " AND m.status = 'completed'" if completed_only else ""
        with self._connect() as db:
            rows = db.execute(
                "SELECT m.message_id, m.thread_id, m.role, m.content, m.sources_json, m.status, m.trace_id, m.created_at "
                "FROM chat_messages m WHERE m.thread_id = ?" + condition + " ORDER BY m.sequence",
                (thread_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json") or "[]")
            messages.append(item)
        return messages
