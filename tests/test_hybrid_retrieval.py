import asyncio

import pytest

import app.data.elasticsearch_client as retrieval
import app.data.query_expander as query_expander
import app.data.reranker as reranker


@pytest.fixture(autouse=True)
def clear_model_caches():
    query_expander.clear_expansion_cache()
    reranker.clear_rerank_cache()
    yield
    query_expander.clear_expansion_cache()
    reranker.clear_rerank_cache()


def _chunk(name: str, index: int, **extra):
    return {
        "doc_name": name,
        "title": f"Dieu {index}",
        "content": f"Noi dung {name}",
        "chunk_index": index,
        **extra,
    }


def test_query_expansion_skips_specific_and_long_queries(monkeypatch):
    monkeypatch.setattr(
        query_expander,
        "ask_gemini",
        lambda prompt: (_ for _ in ()).throw(AssertionError("Gemini must not be called")),
    )

    assert query_expander.expand_query("Điều 12 quy định gì")[1]["reason"] == "specific_metadata_query"
    assert query_expander.expand_query(
        "tôi muốn biết quy định sử dụng phòng học như thế nào"
    )[1]["reason"] == "query_too_long"


def test_query_expansion_keeps_original_and_limits_variants(monkeypatch):
    monkeypatch.setattr(
        query_expander,
        "ask_gemini",
        lambda prompt: '{"queries":["đăng ký học phần","thời gian đăng ký","biến thể dư"]}',
    )

    queries, debug = query_expander.expand_query("đăng ký môn")

    assert queries[0] == "đăng ký môn"
    assert len(queries) == query_expander.QUERY_EXPANSION_MAX_VARIANTS
    assert debug["used"] is True


def test_query_expansion_accepts_json_code_fence(monkeypatch):
    monkeypatch.setattr(
        query_expander,
        "ask_gemini",
        lambda prompt: '```json\n{"queries":["đào tạo trực tuyến","học từ xa"]}\n```',
    )

    queries, debug = query_expander.expand_query("đào tạo từ xa")

    assert queries == ["đào tạo từ xa", "đào tạo trực tuyến", "học từ xa"]
    assert debug["reason"] == "expanded"


def test_query_expansion_rejects_invalid_json_and_wrong_shape(monkeypatch):
    for response in ('```json\n{\n```', '["không đúng cấu trúc"]', '{"query":"sai key"}'):
        monkeypatch.setattr(query_expander, "ask_gemini", lambda prompt, value=response: value)
        queries, debug = query_expander.expand_query("tốt nghiệp")

        assert queries == ["tốt nghiệp"]
        assert debug["used"] is False
        assert debug["reason"] == "invalid_llm_response"


def test_query_expansion_rejects_gemini_error_messages(monkeypatch):
    responses = [
        "He thong AI tam thoi vuot gioi han su dung. Vui long thu lai sau it phut.",
        "429 RESOURCE_EXHAUSTED",
        "503 UNAVAILABLE",
        "Loi khi goi Gemini API. Vui long thu lai sau.",
    ]
    for response in responses:
        monkeypatch.setattr(query_expander, "ask_gemini", lambda prompt, value=response: value)
        queries, debug = query_expander.expand_query("tốt nghiệp")

        assert queries == ["tốt nghiệp"]
        assert debug["used"] is False
        assert debug["reason"] == "llm_error"


def test_query_expansion_removes_duplicates_and_non_strings(monkeypatch):
    monkeypatch.setattr(
        query_expander,
        "ask_gemini",
        lambda prompt: (
            '{"queries":["đăng ký môn","ĐĂNG KÝ MÔN",null,12,'
            '"đăng ký học phần","đăng ký học phần"]}'
        ),
    )

    queries, debug = query_expander.expand_query("đăng ký môn")

    assert queries == ["đăng ký môn", "đăng ký học phần"]
    assert debug["used"] is True


def test_rrf_deduplicates_chunks():
    duplicate = _chunk("a.pdf", 1)
    results = retrieval._merge_with_rrf(
        [[duplicate, _chunk("b.pdf", 1)], [dict(duplicate)]],
        limit=10,
    )

    assert len(results) == 2
    assert results[0]["doc_name"] == "a.pdf"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_rrf_preserves_ann_similarity_across_nested_fusion():
    ann_chunk = _chunk(
        "distance-learning.pdf",
        1,
        score=0.6482,
        distance=0.3518,
    )

    first_fusion = retrieval._merge_with_rrf([[ann_chunk]], limit=10)
    assert first_fusion[0]["vector_score"] == 0.6482

    second_fusion = retrieval._merge_with_rrf([first_fusion], limit=10)

    assert second_fusion[0]["vector_score"] == 0.6482
    assert second_fusion[0]["distance"] == 0.3518
    assert second_fusion[0]["score"] == second_fusion[0]["rrf_score"]


def test_rrf_keeps_strongest_vector_and_keyword_scores():
    weak = _chunk(
        "same.pdf",
        1,
        score=0.4,
        distance=0.6,
        keyword_score=4.0,
    )
    strong = _chunk(
        "same.pdf",
        1,
        score=0.8,
        distance=0.2,
        keyword_score=12.0,
    )

    results = retrieval._merge_with_rrf([[weak], [strong]], limit=10)

    assert results[0]["vector_score"] == 0.8
    assert results[0]["distance"] == 0.2
    assert results[0]["keyword_score"] == 12.0


def test_cross_encoder_reranks_and_falls_back(monkeypatch):
    class FakeModel:
        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: FakeModel())
    ranked, debug = reranker.rerank_chunks(
        "cau hoi", [_chunk("a.pdf", 1), _chunk("b.pdf", 2)]
    )
    assert ranked[0]["doc_name"] == "b.pdf"
    assert debug["used"] is True

    reranker.clear_rerank_cache()
    monkeypatch.setattr(
        reranker,
        "get_cross_encoder",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    original = [_chunk("a.pdf", 1), _chunk("b.pdf", 2)]
    fallback, debug = reranker.rerank_chunks("cau hoi", original)
    assert [item["doc_name"] for item in fallback] == ["a.pdf", "b.pdf"]
    assert debug["reason"] == "model_error"


def test_hybrid_pipeline_falls_back_when_ann_fails(monkeypatch):
    bm25_doc = _chunk("bm25.pdf", 1, keyword_score=8.0, score=8.0)
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("test",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(retrieval, "_search_metadata_documents", lambda question, limit: ([], {}))
    monkeypatch.setattr(
        retrieval,
        "expand_query",
        lambda question: ([question], {"used": False, "queries": [question]}),
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedding offline")),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_keyword_documents",
        lambda *args, **kwargs: [bm25_doc],
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_chunks",
        lambda question, chunks: (chunks, {"used": False, "reason": "mock"}),
    )

    debug = {}
    results = asyncio.run(retrieval.search_documents("quy dinh", debug=debug))

    assert results[0]["doc_name"] == "bm25.pdf"
    assert debug["vector_errors"]
    assert debug["bm25_results_count"] == 1
