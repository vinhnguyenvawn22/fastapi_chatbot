import asyncio
import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from fastapi.testclient import TestClient

import app.routers.chat_router as chat_router
import app.services.conversation_service as conversation_service
from app.data.conversation_context import get_conversation_context
from app.data.conversation_context import (
    ConversationContext, reset_conversation_context, set_conversation_context,
)
import app.data.contextualizer as contextualizer
from app.data.contextualizer import _needs_rewrite, limit_history
from app.data.conversation_repository import ConversationRepository
from app.main import app
from app.controller.chatbot_controller import _new_trace


def test_contextualizer_rewrites_questions_with_at_most_seven_words():
    history = [{"role": "user", "content": "Cau hoi truoc"}]

    assert _needs_rewrite(
        "Ôn tập và Thi thử trên UNETI Online khác nhau thế nào?",
        history,
    ) is False
    assert _needs_rewrite("Phúc khảo ở đâu?", history) is True
    assert _needs_rewrite("có mất phí không", history) is True
    assert _needs_rewrite("Còn cái đó thì sao?", history) is True


def test_contextualizer_expands_short_followup_with_recent_topic(monkeypatch):
    history = [
        {
            "role": "user",
            "content": "Tôi muốn chấm lại bài thi thì làm thế nào",
        },
        {
            "role": "assistant",
            "content": "Bạn có thể gửi yêu cầu phúc khảo trực tuyến.",
        },
    ]
    observed = {}

    def fake_ask_gemini(prompt):
        observed["prompt"] = prompt
        return '{"question":"Phúc khảo có mất phí không?"}'

    monkeypatch.setattr(contextualizer, "ask_gemini", fake_ask_gemini)
    standalone, debug = asyncio.run(
        contextualizer.contextualize_question("có mất phí không", history)
    )

    assert standalone == "Phúc khảo có mất phí không?"
    assert debug["llm_called"] is True
    assert debug["reason"] == "rewritten"
    assert "Câu hỏi hiện tại ngắn" in observed["prompt"]


def _client_with_repository(tmp_path):
    repository = ConversationRepository(str(tmp_path / "chat.sqlite3"))
    repository.initialize()
    app.state.conversation_repository = repository
    return TestClient(app), repository


def _fake_result(request, answer="Tra loi test", sources=None):
    return {
        "question": request.question,
        "answer": answer,
        "source": None,
        "sources": sources or [],
        "intent": "internal_document",
        "trace_id": "00000000-0000-0000-0000-000000000001",
    }


def test_session_thread_history_and_sources(monkeypatch, tmp_path):
    client, repository = _client_with_repository(tmp_path)
    source = {"title": "Dieu 1", "doc_name": "quy-dinh.pdf"}

    async def fake_handle(request):
        return _fake_result(request, sources=[source])

    monkeypatch.setattr(chat_router, "handle_chat", fake_handle)
    response = client.post(
        "/api/chat/", json={"question": "Cau hoi dau tien", "request_id": "req-first"}
    )

    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    data = response.json()
    assert data["thread_id"]
    assert data["user_message_id"] != data["assistant_message_id"]

    history = client.get(f'/api/chat/threads/{data["thread_id"]}/messages')
    assert history.status_code == 200
    messages = history.json()
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["sources"][0]["title"] == source["title"]
    assert messages[1]["sources"][0]["doc_name"] == source["doc_name"]
    assert all(item["status"] == "completed" for item in messages)

    reopened = ConversationRepository(repository.database_file)
    owner = reopened.get_or_create_owner(
        conversation_service.session_hash(client.cookies.get("chat_session"))
    )
    assert len(reopened.list_messages(owner, data["thread_id"])) == 2


def test_second_turn_uses_standalone_question_and_history(monkeypatch, tmp_path):
    client, _ = _client_with_repository(tmp_path)
    observed = []

    async def fake_contextualize(question, history):
        if not history:
            return question, {"llm_called": False, "fallback": False}
        assert [item["content"] for item in history] == ["Quy trinh cap lai mat khau?", "Tra loi test"]
        return "Giay to can co khi cap lai mat khau?", {"llm_called": True, "fallback": False}

    async def fake_handle(request):
        context = get_conversation_context()
        observed.append((request.question, context.original_question, len(context.history)))
        return _fake_result(request)

    monkeypatch.setattr(conversation_service, "contextualize_question", fake_contextualize)
    monkeypatch.setattr(chat_router, "handle_chat", fake_handle)
    first = client.post(
        "/api/chat/",
        json={"question": "Quy trinh cap lai mat khau?", "request_id": "req-turn-1"},
    ).json()
    second = client.post(
        "/api/chat/",
        json={
            "question": "Can giay to gi?",
            "thread_id": first["thread_id"],
            "request_id": "req-turn-2",
        },
    )

    assert second.status_code == 200
    assert second.json()["question"] == "Can giay to gi?"
    assert observed[-1] == ("Giay to can co khi cap lai mat khau?", "Can giay to gi?", 2)


def test_local_endpoint_defers_contextualization_in_conversation_service(
    monkeypatch,
    tmp_path,
):
    client, _ = _client_with_repository(tmp_path)
    observed = []

    async def fail_if_called_before_handler(_question, _history):
        raise AssertionError("contextualizer must be deferred for local endpoint")

    async def fake_handle(request):
        context = get_conversation_context()
        observed.append((
            request.question,
            context.original_question,
            context.rewrite_debug.get("reason"),
        ))
        return _fake_result(request)

    fake_handle.defer_contextualization = True
    monkeypatch.setattr(
        conversation_service,
        "contextualize_question",
        fail_if_called_before_handler,
    )
    monkeypatch.setattr(chat_router, "handle_local_documents_chat", fake_handle)

    first = client.post(
        "/api/chat/local-documents",
        json={"question": "Lich thi o dau?", "request_id": "local-defer-1"},
    ).json()
    second = client.post(
        "/api/chat/local-documents",
        json={
            "question": "Con cai do thi sao?",
            "thread_id": first["thread_id"],
            "request_id": "local-defer-2",
        },
    )

    assert second.status_code == 200
    assert observed[-1] == (
        "Con cai do thi sao?",
        "Con cai do thi sao?",
        "deferred_until_after_original_retrieval",
    )


def test_two_sessions_cannot_access_same_thread(monkeypatch, tmp_path):
    owner, _ = _client_with_repository(tmp_path)
    other = TestClient(app)

    async def fake_handle(request):
        return _fake_result(request)

    monkeypatch.setattr(chat_router, "handle_chat", fake_handle)
    thread_id = owner.post(
        "/api/chat/", json={"question": "Noi dung rieng", "request_id": "req-private"}
    ).json()["thread_id"]

    assert other.get(f"/api/chat/threads/{thread_id}").status_code == 404
    assert other.get(f"/api/chat/threads/{thread_id}/messages").status_code == 404
    assert other.delete(f"/api/chat/threads/{thread_id}").status_code == 404
    assert other.post(
        "/api/chat/",
        json={"question": "Doc thu", "thread_id": thread_id, "request_id": "req-other"},
    ).status_code == 404


def test_pipeline_failure_marks_user_message_failed(monkeypatch, tmp_path):
    client, repository = _client_with_repository(tmp_path)
    client.get("/")

    async def broken_handler(request):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(chat_router, "handle_chat", broken_handler)
    try:
        client.post(
            "/api/chat/", json={"question": "Cau hoi loi", "request_id": "req-failure"}
        )
    except RuntimeError:
        pass

    token = client.cookies.get("chat_session")
    owner_id = repository.get_or_create_owner(conversation_service.session_hash(token))
    threads = repository.list_threads(owner_id)
    messages = repository.list_messages(owner_id, threads[0]["thread_id"])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["status"] == "failed"


def test_schema_initialization_is_idempotent(tmp_path):
    repository = ConversationRepository(str(tmp_path / "persistent.sqlite3"))
    repository.initialize()
    owner_id = repository.get_or_create_owner("hash-value")
    thread = repository.create_thread(owner_id, "Persistent")
    repository.initialize()

    reopened = ConversationRepository(repository.database_file)
    reopened.initialize()
    assert reopened.get_thread(owner_id, thread["thread_id"])["title"] == "Persistent"


def test_history_limit_prefers_recent_complete_messages():
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "middle"},
        {"role": "user", "content": "new"},
    ]

    assert limit_history(history, max_messages=2, max_chars=20) == history[-2:]
    assert limit_history(history, max_messages=10, max_chars=4) == [history[-1]]


def test_trace_records_conversation_questions_without_session_secret():
    token = set_conversation_context(ConversationContext(
        thread_id="00000000-0000-0000-0000-000000000010",
        original_question="Can giay to gi?",
        standalone_question="Can giay to gi khi cap lai mat khau?",
        rewrite_debug={"llm_called": True, "fallback": False},
        history_message_count=2,
        history_chars=42,
        user_message_id="00000000-0000-0000-0000-000000000011",
        assistant_message_id="00000000-0000-0000-0000-000000000012",
    ))
    try:
        payload = _new_trace("Can giay to gi khi cap lai mat khau?").payload
    finally:
        reset_conversation_context(token)

    assert payload["original_question"] == "Can giay to gi?"
    assert payload["standalone_question"].endswith("cap lai mat khau?")
    assert payload["history_message_count"] == 2
    assert payload["history_chars"] == 42
    assert payload["rewrite_debug"]["llm_called"] is True
    assert payload["user_message_id"].endswith("11")
    assert payload["assistant_message_id"].endswith("12")
    serialized = str(payload).lower()
    assert "session_hash" not in serialized
    assert "chat_session" not in serialized


def test_duplicate_request_replays_response_without_calling_handler_twice(monkeypatch, tmp_path):
    client, repository = _client_with_repository(tmp_path)
    calls = 0

    async def fake_handle(request):
        nonlocal calls
        calls += 1
        return _fake_result(request)

    monkeypatch.setattr(chat_router, "handle_chat", fake_handle)
    payload = {"question": "Khong gui trung", "request_id": "req-idempotent"}
    first = client.post("/api/chat/", json=payload)
    second = client.post("/api/chat/", json=payload)

    assert first.status_code == second.status_code == 200
    assert second.json() == first.json()
    assert calls == 1

    owner_id = repository.get_or_create_owner(
        conversation_service.session_hash(client.cookies.get("chat_session"))
    )
    messages = repository.list_messages(owner_id, first.json()["thread_id"])
    assert len(messages) == 2


def test_soft_delete_hides_thread_but_keeps_database_row(monkeypatch, tmp_path):
    client, repository = _client_with_repository(tmp_path)

    async def fake_handle(request):
        return _fake_result(request)

    monkeypatch.setattr(chat_router, "handle_chat", fake_handle)
    result = client.post(
        "/api/chat/",
        json={"question": "Thread se xoa", "request_id": "req-delete"},
    ).json()
    thread_id = result["thread_id"]

    assert client.delete(f"/api/chat/threads/{thread_id}").status_code == 204
    assert all(item["thread_id"] != thread_id for item in client.get("/api/chat/threads").json())
    assert client.get(f"/api/chat/threads/{thread_id}").status_code == 404
    assert client.get(f"/api/chat/threads/{thread_id}/messages").status_code == 404
    assert client.post(
        "/api/chat/",
        json={
            "question": "Khong duoc chat tiep",
            "thread_id": thread_id,
            "request_id": "req-deleted-thread",
        },
    ).status_code == 404
    assert client.post(
        "/api/chat/",
        json={"question": "Thread se xoa", "request_id": "req-delete"},
    ).status_code == 404

    owner_id = repository.get_or_create_owner(
        conversation_service.session_hash(client.cookies.get("chat_session"))
    )
    deleted = repository.get_thread(owner_id, thread_id, include_deleted=True)
    assert deleted["status"] == "deleted"
    assert deleted["deleted_at"]


def test_thread_detail_and_rewrite_metadata_are_persisted(monkeypatch, tmp_path):
    client, repository = _client_with_repository(tmp_path)

    async def fake_contextualize(question, history):
        return "Cau hoi doc lap", {
            "history_present": bool(history),
            "llm_called": True,
            "fallback": False,
            "reason": "rewritten",
        }

    async def fake_handle(request):
        return _fake_result(request)

    monkeypatch.setattr(conversation_service, "contextualize_question", fake_contextualize)
    monkeypatch.setattr(chat_router, "handle_chat", fake_handle)
    result = client.post(
        "/api/chat/",
        json={
            "question": "  Cau hoi\n  nhieu dong  ",
            "request_id": "req-metadata",
        },
    ).json()

    detail = client.get(f'/api/chat/threads/{result["thread_id"]}')
    assert detail.status_code == 200
    assert detail.json()["title"] == "Cau hoi nhieu dong"
    assert detail.json()["message_count"] == 2
    assert detail.json()["last_message"] == "Tra loi test"

    owner_id = repository.get_or_create_owner(
        conversation_service.session_hash(client.cookies.get("chat_session"))
    )
    user_message = repository.list_messages(owner_id, result["thread_id"])[0]
    assert user_message["metadata"] == {
        "original_question": "Cau hoi\n  nhieu dong",
        "standalone_question": "Cau hoi doc lap",
        "rewrite_debug": {
            "history_present": False,
            "llm_called": True,
            "fallback": False,
            "reason": "rewritten",
        },
        "history_message_count": 0,
    }
