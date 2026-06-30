import os


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from fastapi.testclient import TestClient

from app.main import app
import app.routers.website_router as website_router


client = TestClient(app)


def test_website_search_api_with_mock(monkeypatch):
    """Kiem tra endpoint POST /api/website/search khong goi website/Gemini that."""

    async def fake_search_website_knowledge(request):
        return {
            "query": request.query,
            "answer": "Thong tin test tren website UNETI.",
            "selected_count": 1,
            "sources": [
                {
                    "title": "Thong bao tuyen sinh",
                    "url": "https://uneti.edu.vn/thong-bao-tuyen-sinh/",
                    "source_type": "website_uneti",
                    "score": 88.5,
                    "confidence": 1,
                    "confidence_percent": 100,
                    "confidence_label": "Cao",
                }
            ],
            "trace": [
                {
                    "name": "website_search",
                    "selected_count": 1,
                }
            ],
        }

    monkeypatch.setattr(
        website_router,
        "search_website_knowledge",
        fake_search_website_knowledge,
    )

    response = client.post(
        "/api/website/search",
        json={"query": "tin tuyen sinh moi nhat", "top_k": 1},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "tin tuyen sinh moi nhat"
    assert data["answer"] == "Thong tin test tren website UNETI."
    assert data["selected_count"] == 1
    assert data["sources"][0]["source_type"] == "website_uneti"
    assert data["trace"][0]["name"] == "website_search"
