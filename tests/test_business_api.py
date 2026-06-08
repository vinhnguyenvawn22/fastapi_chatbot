import os


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from fastapi.testclient import TestClient

from app.main import app
import app.routers.business_router as business_router


client = TestClient(app)


def test_business_search_api_with_mock(monkeypatch):
    """Kiem tra endpoint POST /api/nghiep-vu/search khong goi retrieval that."""

    async def fake_search_business_knowledge(request):
        return {
            "query": request.query,
            "intent": "internal_document",
            "candidate_count": 2,
            "selected_count": 1,
            "has_confident_evidence": True,
            "evidence_reason": "keyword_score_passed",
            "sources": [
                {
                    "title": "Quy dinh email",
                    "doc_name": "email.pdf",
                    "score": 9.5,
                    "confidence": 1,
                    "confidence_percent": 100,
                    "confidence_label": "Cao",
                }
            ],
            "trace": [
                {
                    "name": "retrieval",
                    "candidate_count": 2,
                }
            ],
        }

    monkeypatch.setattr(
        business_router,
        "search_business_knowledge",
        fake_search_business_knowledge,
    )

    response = client.post(
        "/api/nghiep-vu/search",
        json={"query": "quy dinh email", "top_k": 1},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "quy dinh email"
    assert data["candidate_count"] == 2
    assert data["selected_count"] == 1
    assert data["has_confident_evidence"] is True
    assert data["sources"][0]["doc_name"] == "email.pdf"
    assert data["trace"][0]["name"] == "retrieval"
