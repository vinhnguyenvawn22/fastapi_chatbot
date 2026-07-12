from hashlib import sha256
import secrets
import uuid

from fastapi import HTTPException
from pydantic import BaseModel

from app.core.config import CHAT_HISTORY_MAX_CHARS, CHAT_HISTORY_MAX_MESSAGES
from app.data.contextualizer import contextualize_question, limit_history
from app.data.conversation_context import (
    ConversationContext,
    reset_conversation_context,
    set_conversation_context,
)
from app.data.langchain_pipeline import GEMINI_UNAVAILABLE_ANSWER


class _ControllerRequest(BaseModel):
    question: str


SYSTEM_FAILURE_MARKERS = (
    "Loi khi goi Gemini API",
    "He thong AI dang ban",
    "He thong AI tam thoi vuot gioi han",
    GEMINI_UNAVAILABLE_ANSWER,
)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def session_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def validate_uuid(value: str, label: str = "thread_id") -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"{label} khong hop le") from exc


class ConversationService:
    def __init__(self, repository):
        self.repository = repository

    def owner_for_token(self, token: str) -> str:
        return self.repository.get_or_create_owner(session_hash(token))

    def create_thread(self, owner_id: str, title: str = "Cuoc tro chuyen moi") -> dict:
        return self.repository.create_thread(owner_id, (title.strip() or "Cuoc tro chuyen moi")[:120])

    def require_thread(self, owner_id: str, thread_id: str) -> dict:
        thread_id = validate_uuid(thread_id)
        thread = self.repository.get_thread(owner_id, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Khong tim thay cuoc tro chuyen")
        return thread

    async def chat(self, owner_id: str, request, handler) -> dict:
        original = request.question.strip()
        if request.thread_id:
            thread = self.require_thread(owner_id, request.thread_id)
        else:
            thread = self.create_thread(owner_id, original)

        all_history = self.repository.list_messages(owner_id, thread["thread_id"], completed_only=True) or []
        history = limit_history(all_history, CHAT_HISTORY_MAX_MESSAGES, CHAT_HISTORY_MAX_CHARS)
        user_message = self.repository.create_message(
            thread["thread_id"], "user", original, "processing"
        )
        standalone, rewrite_debug = await contextualize_question(original, history)
        context = ConversationContext(
            thread_id=thread["thread_id"], original_question=original,
            standalone_question=standalone, history=history, rewrite_debug=rewrite_debug,
        )
        token = set_conversation_context(context)
        try:
            result = await handler(_ControllerRequest(question=standalone))
            answer = str(result.get("answer") or "")
            if not answer or any(marker in answer for marker in SYSTEM_FAILURE_MARKERS):
                self.repository.update_message_status(user_message["message_id"], "failed")
                raise HTTPException(status_code=503, detail="He thong AI tam thoi khong the tra loi")
            assistant = self.repository.create_message(
                thread["thread_id"], "assistant", answer, "completed",
                sources=result.get("sources") or [], trace_id=result.get("trace_id"),
            )
            self.repository.update_message_status(user_message["message_id"], "completed")
            return {
                **result,
                "question": original,
                "thread_id": thread["thread_id"],
                "user_message_id": user_message["message_id"],
                "assistant_message_id": assistant["message_id"],
            }
        except HTTPException:
            self.repository.update_message_status(user_message["message_id"], "failed")
            raise
        except Exception:
            self.repository.update_message_status(user_message["message_id"], "failed")
            raise
        finally:
            reset_conversation_context(token)
