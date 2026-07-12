import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from fastapi.testclient import TestClient

import app.routers.chat_router as chat_router
import app.services.conversation_service as conversation_service
from app.data.conversation_context import get_conversation_context
from app.data.conversation_context import (
    ConversationContext, reset_conversation_context, set_conversation_context,
)
from app.data.contextualizer import limit_history
from app.data.conversation_repository import ConversationRepository
from app.main import app
from app.controller.chatbot_controller import _new_trace


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
    response = client.post("/api/chat/", json={"question": "Cau hoi dau tien"})

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
    first = client.post("/api/chat/", json={"question": "Quy trinh cap lai mat khau?"}).json()
    second = client.post(
        "/api/chat/",
        json={"question": "Can giay to gi?", "thread_id": first["thread_id"]},
    )

    assert second.status_code == 200
    assert second.json()["question"] == "Can giay to gi?"
    assert observed[-1] == ("Giay to can co khi cap lai mat khau?", "Can giay to gi?", 2)


def test_two_sessions_cannot_access_same_thread(monkeypatch, tmp_path):
    owner, _ = _client_with_repository(tmp_path)
    other = TestClient(app)

    async def fake_handle(request):
        return _fake_result(request)

    monkeypatch.setattr(chat_router, "handle_chat", fake_handle)
    thread_id = owner.post("/api/chat/", json={"question": "Noi dung rieng"}).json()["thread_id"]

    assert other.get(f"/api/chat/threads/{thread_id}/messages").status_code == 404
    assert other.delete(f"/api/chat/threads/{thread_id}").status_code == 404
    assert other.post(
        "/api/chat/", json={"question": "Doc thu", "thread_id": thread_id}
    ).status_code == 404


def test_pipeline_failure_marks_user_message_failed(monkeypatch, tmp_path):
    client, repository = _client_with_repository(tmp_path)
    client.get("/")

    async def broken_handler(request):
        raise RuntimeError("pipeline failed")

    monkeypatch.setattr(chat_router, "handle_chat", broken_handler)
    try:
        client.post("/api/chat/", json={"question": "Cau hoi loi"})
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
    ))
    try:
        payload = _new_trace("Can giay to gi khi cap lai mat khau?").payload
    finally:
        reset_conversation_context(token)

    assert payload["original_question"] == "Can giay to gi?"
    assert payload["standalone_question"].endswith("cap lai mat khau?")
    serialized = str(payload).lower()
    assert "session_hash" not in serialized
    assert "chat_session" not in serialized
