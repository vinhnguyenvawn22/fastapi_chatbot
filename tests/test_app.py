import os

# Set key giả trước khi import app để tránh lỗi thiếu GEMINI_API_KEY khi chạy test.
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from fastapi.testclient import TestClient

from app.main import app
import app.routers.chat_router as chat_router


client = TestClient(app)


def test_home_page_returns_html():
    """
    Kiểm tra web chatbot tại GET /
    Mục tiêu: đảm bảo giao diện chat_ui.html đã được mount đúng.
    """
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "<html" in response.text.lower() or "chatbot" in response.text.lower()


def test_docs_page_available():
    """
    Kiểm tra Swagger UI.
    Mục tiêu: đảm bảo FastAPI mở được tài liệu API tại /docs.
    """
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_openapi_json_available():
    """
    Kiểm tra OpenAPI schema.
    Mục tiêu: đảm bảo FastAPI sinh được file openapi.json.
    """
    response = client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert "paths" in data


def test_chat_api_with_mock(monkeypatch):
    """
    Kiểm tra endpoint POST /api/chat/ nhưng không gọi Gemini thật.
    Mục tiêu: test router/schema/response hoạt động đúng.
    """

    async def fake_handle_chat(request):
        return {
            "question": request.question,
            "answer": "Đây là câu trả lời test.",
            "source": "test-document.pdf"
        }

    monkeypatch.setattr(chat_router, "handle_chat", fake_handle_chat)

    response = client.post(
        "/api/chat/",
        json={"question": "Phòng Tổ hợp STUDIO ở đâu?"}
    )

    assert response.status_code == 200

    data = response.json()
    assert data["question"] == "Phòng Tổ hợp STUDIO ở đâu?"
    assert data["answer"] == "Đây là câu trả lời test."
    assert data["source"] == "test-document.pdf"