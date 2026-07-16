import asyncio
from hashlib import sha256
import json
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
        normalized_title = " ".join(title.split()) or "Cuoc tro chuyen moi"
        return self.repository.create_thread(owner_id, normalized_title[:120])

    def require_thread(self, owner_id: str, thread_id: str) -> dict:
        thread_id = validate_uuid(thread_id)
        thread = self.repository.get_thread(owner_id, thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Khong tim thay cuoc tro chuyen")
        return thread

    async def chat(self, owner_id: str, request, handler) -> dict:
        original = request.question.strip()
        request_id = request.request_id.strip()
        if not request_id:
            raise HTTPException(status_code=422, detail="request_id khong duoc de trong")
        requested_thread_id = (
            validate_uuid(request.thread_id) if request.thread_id else None
        )
        fingerprint = sha256(json.dumps(
            {"question": original, "thread_id": requested_thread_id},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        title = (" ".join(original.split()) or "Cuoc tro chuyen moi")[:120]
        claim = self.repository.claim_chat_request(
            owner_id, request_id, fingerprint, original, title, requested_thread_id
        )
        if claim["claim_status"] == "thread_not_found":
            raise HTTPException(status_code=404, detail="Khong tim thay cuoc tro chuyen")
        if claim["claim_status"] == "conflict":
            raise HTTPException(
                status_code=409,
                detail="request_id da duoc dung cho mot noi dung khac",
            )
        if claim["claim_status"] == "existing":
            return await self._existing_response(owner_id, request_id, claim)

        thread_id = claim["thread_id"]
        user_message_id = claim["user_message_id"]
        assistant_message_id = claim["assistant_message_id"]
        all_history = self.repository.list_messages(
            owner_id, thread_id, completed_only=True
        ) or []
        history = limit_history(all_history, CHAT_HISTORY_MAX_MESSAGES, CHAT_HISTORY_MAX_CHARS)
        try:
            standalone, rewrite_debug = await contextualize_question(original, history)
            history_chars = sum(len(str(item.get("content") or "")) for item in history)
            metadata = {
                "original_question": original,
                "standalone_question": standalone,
                "rewrite_debug": rewrite_debug,
                "history_message_count": len(history),
            }
            self.repository.update_message_metadata(user_message_id, metadata)
            context = ConversationContext(
                thread_id=thread_id,
                original_question=original,
                standalone_question=standalone,
                history=history,
                rewrite_debug=rewrite_debug,
                history_message_count=len(history),
                history_chars=history_chars,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
            )
            token = set_conversation_context(context)
            try:
                result = await handler(_ControllerRequest(question=standalone))
            finally:
                reset_conversation_context(token)
            answer = str(result.get("answer") or "")
            if not answer or any(marker in answer for marker in SYSTEM_FAILURE_MARKERS):
                raise HTTPException(status_code=503, detail="He thong AI tam thoi khong the tra loi")
            response = {
                **result,
                "question": original,
                "thread_id": thread_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
            }
            return self.repository.complete_chat_request(
                owner_id, request_id, answer, result.get("sources") or [],
                result.get("trace_id"), metadata, response,
            )
        except HTTPException as exc:
            self.repository.fail_chat_request(owner_id, request_id, str(exc.detail))
            raise
        except Exception as exc:
            self.repository.fail_chat_request(owner_id, request_id, str(exc))
            raise

    async def _existing_response(self, owner_id: str, request_id: str, row: dict) -> dict:
        for _ in range(600):
            if row["status"] == "completed":
                return json.loads(row["response_json"])
            if row["status"] == "failed":
                raise HTTPException(
                    status_code=503,
                    detail=row.get("error_detail") or "Request truoc da xu ly that bai",
                )
            await asyncio.sleep(0.05)
            row = self.repository.get_chat_request(owner_id, request_id)
            if not row:
                break
        raise HTTPException(status_code=409, detail="Request dang duoc xu ly")
