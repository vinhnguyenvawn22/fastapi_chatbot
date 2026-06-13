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


def test_business_ask_api_matches_mapping():
    """Kiem tra endpoint POST /api/nghiep-vu/ask tra loi tu FAQ mapping."""

    response = client.post(
        "/api/nghiep-vu/ask",
        json={"query": "toi vao muc tin tuc thong bao o dau", "top_k": 2},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["query"] == "toi vao muc tin tuc thong bao o dau"
    assert data["matched"] is True
    assert data["answer"]
    assert data["confidence"] >= 0.58
    assert data["matched_question"] == 'Làm thế nào để tôi truy cập vào mục "Tin tức – Thông báo"?'
    assert data["file_id"] == "PCNTT_FILE_01"
    assert data["source_file"] == "2026.03.03.ChatbotAI_CBGV_SV_V4"
    assert data["source_location"] == "Mục I -> 2"
    assert data["keywords"]
    assert len(data["candidates"]) == 2


def test_business_ask_api_returns_fallback_when_not_confident():
    """Kiem tra endpoint khong bia cau tra loi khi mapping khong du tin cay."""

    response = client.post(
        "/api/nghiep-vu/ask",
        json={"query": "thoi tiet hom nay ra sao", "top_k": 2},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["matched"] is False
    assert data["answer"] is None
    assert data["matched_question"] is None
    assert data["fallback_suggestion"] == "fallback_to_rag_or_chat"
    assert len(data["candidates"]) == 2
