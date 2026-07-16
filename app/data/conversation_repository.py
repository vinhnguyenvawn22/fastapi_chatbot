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

    @staticmethod
    def _columns(db, table: str) -> set[str]:
        return {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}

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
                    status TEXT NOT NULL DEFAULT 'active'
                        CHECK(status IN ('active', 'deleted')),
                    deleted_at TEXT,
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
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    request_id TEXT,
                    status TEXT NOT NULL CHECK(status IN ('processing', 'completed', 'failed')),
                    trace_id TEXT,
                    created_at TEXT NOT NULL
                );
            """)

            thread_columns = self._columns(db, "chat_threads")
            if "status" not in thread_columns:
                db.execute("ALTER TABLE chat_threads ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
            if "deleted_at" not in thread_columns:
                db.execute("ALTER TABLE chat_threads ADD COLUMN deleted_at TEXT")

            message_columns = self._columns(db, "chat_messages")
            if "metadata_json" not in message_columns:
                db.execute(
                    "ALTER TABLE chat_messages ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "request_id" not in message_columns:
                db.execute("ALTER TABLE chat_messages ADD COLUMN request_id TEXT")

            db.executescript("""
                CREATE TABLE IF NOT EXISTS chat_requests (
                    owner_id TEXT NOT NULL REFERENCES anonymous_sessions(id) ON DELETE CASCADE,
                    request_id TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    thread_id TEXT NOT NULL REFERENCES chat_threads(thread_id),
                    user_message_id TEXT NOT NULL,
                    assistant_message_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('processing', 'completed', 'failed')),
                    response_json TEXT,
                    error_detail TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_threads_owner_updated
                    ON chat_threads(owner_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_thread_sequence
                    ON chat_messages(thread_id, sequence);
                CREATE INDEX IF NOT EXISTS idx_messages_request
                    ON chat_messages(thread_id, request_id);
            """)

    def get_or_create_owner(self, session_hash: str) -> str:
        now = _now()
        with self._connect() as db:
            row = db.execute(
                "SELECT id FROM anonymous_sessions WHERE session_hash = ?", (session_hash,)
            ).fetchone()
            if row:
                db.execute(
                    "UPDATE anonymous_sessions SET last_seen_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                return row["id"]
            owner_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO anonymous_sessions(id, session_hash, created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?)",
                (owner_id, session_hash, now, now),
            )
            return owner_id

    def create_thread(self, owner_id: str, title: str) -> dict:
        thread_id, now = str(uuid.uuid4()), _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO chat_threads(thread_id, owner_id, title, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'active', ?, ?)",
                (thread_id, owner_id, title, now, now),
            )
        return self.get_thread(owner_id, thread_id)

    def get_thread(
        self, owner_id: str, thread_id: str, include_deleted: bool = False
    ) -> dict | None:
        deleted_filter = "" if include_deleted else " AND status = 'active'"
        with self._connect() as db:
            row = db.execute(
                "SELECT thread_id, title, status, deleted_at, created_at, updated_at "
                "FROM chat_threads WHERE thread_id = ? AND owner_id = ?" + deleted_filter,
                (thread_id, owner_id),
            ).fetchone()
        return dict(row) if row else None

    def get_thread_detail(self, owner_id: str, thread_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT t.thread_id, t.title, t.created_at, t.updated_at, "
                "COUNT(m.message_id) AS message_count, "
                "(SELECT content FROM chat_messages latest "
                " WHERE latest.thread_id = t.thread_id ORDER BY latest.sequence DESC LIMIT 1) "
                "AS last_message "
                "FROM chat_threads t LEFT JOIN chat_messages m ON m.thread_id = t.thread_id "
                "WHERE t.thread_id = ? AND t.owner_id = ? AND t.status = 'active' "
                "GROUP BY t.thread_id, t.title, t.created_at, t.updated_at",
                (thread_id, owner_id),
            ).fetchone()
        return dict(row) if row else None

    def list_threads(self, owner_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT thread_id, title, created_at, updated_at FROM chat_threads "
                "WHERE owner_id = ? AND status = 'active' ORDER BY updated_at DESC",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_thread(self, owner_id: str, thread_id: str) -> bool:
        now = _now()
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE chat_threads SET status = 'deleted', deleted_at = ?, updated_at = ? "
                "WHERE thread_id = ? AND owner_id = ? AND status = 'active'",
                (now, now, thread_id, owner_id),
            )
        return cursor.rowcount > 0

    def claim_chat_request(
        self,
        owner_id: str,
        request_id: str,
        request_fingerprint: str,
        original_question: str,
        title: str,
        thread_id: str | None = None,
    ) -> dict:
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM chat_requests WHERE owner_id = ? AND request_id = ?",
                (owner_id, request_id),
            ).fetchone()
            if existing:
                result = dict(existing)
                active_thread = db.execute(
                    "SELECT 1 FROM chat_threads WHERE thread_id = ? AND owner_id = ? "
                    "AND status = 'active'",
                    (result["thread_id"], owner_id),
                ).fetchone()
                if not active_thread:
                    return {"claim_status": "thread_not_found"}
                result["claim_status"] = "conflict" if (
                    result["request_fingerprint"] != request_fingerprint
                ) else "existing"
                return result

            if thread_id:
                thread = db.execute(
                    "SELECT thread_id FROM chat_threads "
                    "WHERE thread_id = ? AND owner_id = ? AND status = 'active'",
                    (thread_id, owner_id),
                ).fetchone()
                if not thread:
                    return {"claim_status": "thread_not_found"}
            else:
                thread_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO chat_threads(thread_id, owner_id, title, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'active', ?, ?)",
                    (thread_id, owner_id, title, now, now),
                )

            user_message_id = str(uuid.uuid4())
            assistant_message_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO chat_requests(owner_id, request_id, request_fingerprint, thread_id, "
                "user_message_id, assistant_message_id, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'processing', ?, ?)",
                (
                    owner_id, request_id, request_fingerprint, thread_id,
                    user_message_id, assistant_message_id, now, now,
                ),
            )
            db.execute(
                "INSERT INTO chat_messages(message_id, thread_id, role, content, sources_json, "
                "metadata_json, request_id, status, trace_id, created_at) "
                "VALUES (?, ?, 'user', ?, '[]', '{}', ?, 'processing', NULL, ?)",
                (user_message_id, thread_id, original_question, request_id, now),
            )
            db.execute(
                "UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id)
            )
            return {
                "claim_status": "claimed",
                "status": "processing",
                "thread_id": thread_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
            }

    def get_chat_request(self, owner_id: str, request_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM chat_requests WHERE owner_id = ? AND request_id = ?",
                (owner_id, request_id),
            ).fetchone()
        return dict(row) if row else None

    def complete_chat_request(
        self,
        owner_id: str,
        request_id: str,
        answer: str,
        sources: list,
        trace_id: str | None,
        metadata: dict,
        response: dict,
    ) -> dict:
        now = _now()
        sources_json = json.dumps(sources, ensure_ascii=False)
        metadata_json = json.dumps(metadata, ensure_ascii=False)
        response_json = json.dumps(response, ensure_ascii=False)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            request_row = db.execute(
                "SELECT * FROM chat_requests WHERE owner_id = ? AND request_id = ?",
                (owner_id, request_id),
            ).fetchone()
            if not request_row or request_row["status"] != "processing":
                raise RuntimeError("Chat request is no longer processing")
            db.execute(
                "UPDATE chat_messages SET status = 'completed', metadata_json = ? "
                "WHERE message_id = ?",
                (metadata_json, request_row["user_message_id"]),
            )
            db.execute(
                "INSERT INTO chat_messages(message_id, thread_id, role, content, sources_json, "
                "metadata_json, request_id, status, trace_id, created_at) "
                "VALUES (?, ?, 'assistant', ?, ?, '{}', ?, 'completed', ?, ?)",
                (
                    request_row["assistant_message_id"], request_row["thread_id"], answer,
                    sources_json, request_id, trace_id, now,
                ),
            )
            db.execute(
                "UPDATE chat_requests SET status = 'completed', response_json = ?, updated_at = ? "
                "WHERE owner_id = ? AND request_id = ?",
                (response_json, now, owner_id, request_id),
            )
            db.execute(
                "UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?",
                (now, request_row["thread_id"]),
            )
        return response

    def fail_chat_request(self, owner_id: str, request_id: str, detail: str) -> None:
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT user_message_id FROM chat_requests "
                "WHERE owner_id = ? AND request_id = ? AND status = 'processing'",
                (owner_id, request_id),
            ).fetchone()
            if not row:
                return
            db.execute(
                "UPDATE chat_messages SET status = 'failed' WHERE message_id = ?",
                (row["user_message_id"],),
            )
            db.execute(
                "UPDATE chat_requests SET status = 'failed', error_detail = ?, updated_at = ? "
                "WHERE owner_id = ? AND request_id = ?",
                (detail, now, owner_id, request_id),
            )

    def create_message(
        self, thread_id: str, role: str, content: str, status: str,
        sources: list | None = None, trace_id: str | None = None,
        metadata: dict | None = None, request_id: str | None = None,
        message_id: str | None = None,
    ) -> dict:
        message_id, now = message_id or str(uuid.uuid4()), _now()
        with self._connect() as db:
            db.execute(
                "INSERT INTO chat_messages(message_id, thread_id, role, content, sources_json, "
                "metadata_json, request_id, status, trace_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id, thread_id, role, content,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False), request_id,
                    status, trace_id, now,
                ),
            )
            db.execute("UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?", (now, thread_id))
        return {
            "message_id": message_id, "thread_id": thread_id, "role": role,
            "content": content, "sources": sources or [], "metadata": metadata or {},
            "status": status, "trace_id": trace_id, "created_at": now,
        }

    def update_message_status(self, message_id: str, status: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE chat_messages SET status = ? WHERE message_id = ?", (status, message_id))

    def update_message_metadata(self, message_id: str, metadata: dict) -> None:
        with self._connect() as db:
            db.execute(
                "UPDATE chat_messages SET metadata_json = ? WHERE message_id = ?",
                (json.dumps(metadata, ensure_ascii=False), message_id),
            )

    def list_messages(
        self, owner_id: str, thread_id: str, completed_only: bool = False
    ) -> list[dict] | None:
        if not self.get_thread(owner_id, thread_id):
            return None
        condition = " AND m.status = 'completed'" if completed_only else ""
        with self._connect() as db:
            rows = db.execute(
                "SELECT m.message_id, m.thread_id, m.role, m.content, m.sources_json, "
                "m.metadata_json, m.status, m.trace_id, m.created_at "
                "FROM chat_messages m WHERE m.thread_id = ?" + condition + " ORDER BY m.sequence",
                (thread_id,),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json") or "[]")
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
            messages.append(item)
        return messages
