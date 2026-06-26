import asyncio
import os


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

import app.data.elasticsearch_client as retrieval
import app.data.ambiguity_analyzer as ambiguity
import app.data.hyde as hyde
import app.data.langchain_pipeline as langchain_pipeline
import app.data.query_expansion as query_expansion
import app.data.reranker as reranker
from rank_bm25 import BM25Okapi


def _doc(name: str, chunk_index: int, **scores):
    return {
        "doc_name": name,
        "relative_path": name,
        "title": f"Chunk {chunk_index}",
        "chunk_index": chunk_index,
        "content": f"Content from {name}",
        **scores,
    }


def test_query_expansion_skips_specific_and_long_queries():
    assert query_expansion.should_expand_query("van ban so 880")[0] is False
    assert query_expansion.should_expand_query("Điều 15 quy định gì")[0] is False
    assert query_expansion.should_expand_query("Mục 2 áp dụng thế nào")[0] is False
    assert query_expansion.should_expand_query(
        "quy định sử dụng thiết bị trong phòng học như thế nào"
    )[0] is False


def test_query_expansion_keeps_original_when_gemini_fails(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "expand_query_with_gemini",
        lambda question: (_ for _ in ()).throw(RuntimeError("Gemini unavailable")),
    )

    queries, debug = query_expansion.build_query_variants("quên mật khẩu")

    assert queries == ["quên mật khẩu"]
    assert debug["attempted"] is True
    assert debug["reason"] == "gemini_error_fallback_original"
    assert "Gemini unavailable" in debug["error"]


def test_query_expansion_keeps_original_and_limits_variants(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "expand_query_with_gemini",
        lambda question: [
            "khôi phục mật khẩu",
            "cấp lại mật khẩu",
            "đặt lại thông tin đăng nhập",
            "biến thể vượt giới hạn",
        ],
    )

    queries, debug = query_expansion.build_query_variants("quên mật khẩu")

    assert queries[0] == "quên mật khẩu"
    assert len(queries) == 1 + query_expansion.QUERY_EXPANSION_MAX_VARIANTS
    assert len(queries) <= 3
    assert debug["variant_count"] == query_expansion.QUERY_EXPANSION_MAX_VARIANTS


def test_rrf_deduplicates_same_chunk():
    shared = _doc("a.pdf", 1)
    other = _doc("b.pdf", 1)

    results = retrieval._merge_with_rrf(
        [
            [{**shared, "bm25_score": 10, "keyword_score": 10}],
            [{**shared, "score": 0.8, "distance": 0.2}, other],
        ],
        limit=10,
    )

    assert len(results) == 2
    assert results[0]["doc_name"] == "a.pdf"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
    assert results[0]["bm25_score"] == 10
    assert results[0]["vector_score"] == 0.8


def test_bm25_prioritizes_document_number_and_department(monkeypatch):
    weak = {
        **_doc("general.pdf", 1),
        "content": "Quy định sử dụng thiết bị chung.",
        "phong_ban": "Phòng Đào tạo",
    }
    strong = {
        **_doc("qd-880.pdf", 1),
        "content": "Quy định khai thác và bảo trì thiết bị phòng học.",
        "so_van_ban": "880",
        "so_van_ban_ngan": "880",
        "phong_ban": "Phòng Công nghệ thông tin",
    }
    unrelated = {
        **_doc("unrelated.pdf", 1),
        "content": "Hướng dẫn đăng ký học phần.",
        "phong_ban": "Phòng Công tác sinh viên",
    }
    chunks = [weak, strong, unrelated]
    corpus = [retrieval._bm25_document_tokens(doc) for doc in chunks]
    retrieval._INDEX_CACHE["bm25"] = BM25Okapi(corpus)
    monkeypatch.setattr(
        retrieval,
        "_load_document_index",
        lambda: (chunks, {}, len(chunks)),
    )

    results = retrieval._search_bm25_documents(
        "văn bản 880 phòng công nghệ thông tin",
        limit=2,
    )

    assert results[0]["doc_name"] == "qd-880.pdf"
    assert results[0]["bm25_score"] >= results[-1]["bm25_score"]


def test_cross_encoder_failure_falls_back_to_rrf(monkeypatch):
    docs = [
        _doc("a.pdf", 1, rrf_score=0.03),
        _doc("b.pdf", 1, rrf_score=0.02),
    ]
    monkeypatch.setattr(
        reranker,
        "get_cross_encoder",
        lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    results, debug = reranker.rerank_documents("test query", docs, final_top_k=2)

    assert [doc["doc_name"] for doc in results] == ["a.pdf", "b.pdf"]
    assert debug["reason"] == "cross_encoder_error_rrf_fallback"
    assert "model unavailable" in debug["error"]


def test_cross_encoder_success_reorders_candidates(monkeypatch):
    class FakeCrossEncoder:
        def predict(self, pairs):
            return [0.1, 0.9]

    monkeypatch.setattr(reranker, "get_cross_encoder", lambda: FakeCrossEncoder())
    docs = [
        _doc("a.pdf", 1, rrf_score=0.04),
        _doc("b.pdf", 1, rrf_score=0.03),
    ]

    results, debug = reranker.rerank_documents("test query", docs, final_top_k=2)

    assert [doc["doc_name"] for doc in results] == ["b.pdf", "a.pdf"]
    assert debug["reason"] == "cross_encoder_success"
    assert results[0]["rerank_score"] == 0.9


def test_hybrid_pipeline_traces_all_steps_and_ann_fallback(monkeypatch):
    bm25_doc = _doc("bm25.pdf", 1, bm25_score=12, keyword_score=12, score=12)
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda question, limit, source_type_filter=None: [bm25_doc],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("embedding failed")),
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_error_rrf_fallback"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "quên mật khẩu",
            debug=debug,
            ambiguity_decision={
                "action": ambiguity.DIRECT_RETRIEVAL,
                "reason": "test",
            },
        )
    )

    assert results[0]["doc_name"] == "bm25.pdf"
    assert debug["expanded_queries"] == ["quên mật khẩu"]
    assert len(debug["bm25_results"]) == 1
    assert len(debug["ann_results"]) == 1
    assert len(debug["vector_errors"]) == 1
    assert debug["rrf_results"]
    assert debug["reranking"]["reason"] == "cross_encoder_error_rrf_fallback"
    assert debug["final_sources"][0]["doc_name"] == "bm25.pdf"


def test_bm25_failure_falls_back_to_ann(monkeypatch):
    ann_doc = _doc(
        "ann.pdf",
        1,
        score=0.85,
        distance=0.15,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("BM25 failed")),
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: [ann_doc],
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(retrieval.search_documents("quy định email", debug=debug))

    assert results[0]["doc_name"] == "ann.pdf"
    assert debug["bm25_errors"] == [
        {"query": "quy định email", "error": "BM25 failed"}
    ]
    assert debug["ann_results"][0]["results"]


def test_langchain_emits_separate_hybrid_trace_steps():
    names = []

    def callback(name, output, input_data=None):
        names.append(name)

    langchain_pipeline._trace_hybrid_retrieval(
        {"trace_callback": callback},
        {
            "ambiguity": {
                "action": ambiguity.HYDE_RETRIEVAL,
                "topic": "camera",
            },
            "hyde": {"status": "success"},
            "bm25_results": [],
            "ann_results": [],
            "rrf_results": [],
            "reranking": {"reason": "cross_encoder_success"},
        },
    )

    assert names == [
        "ambiguity_detection",
        "probe_retrieval",
        "probe_evidence_decision",
        "hyde_generation",
        "grounded_hyde_generation",
        "bm25_retrieval",
        "ann_retrieval",
        "rrf_fusion",
        "cross_encoder_rerank",
    ]


def test_missing_exact_document_number_never_falls_back(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {"so_van_ban": "999"}),
    )
    debug = {}
    results = asyncio.run(retrieval.search_documents("văn bản số 999", debug=debug))

    assert results == []
    assert debug["reranking"]["reason"] == "no_exact_document"


def test_rule_ambiguity_routes_known_topic_directly():
    decision = ambiguity._rule_decision("camera ai quản lý")

    assert decision.action == ambiguity.DIRECT_RETRIEVAL
    assert decision.topic == "camera"


def test_rule_ambiguity_routes_specific_query_direct():
    decision = ambiguity._rule_decision("văn bản số 877 quy định gì")

    assert decision.action == ambiguity.DIRECT_RETRIEVAL


def test_rule_ambiguity_routes_article_and_section_directly():
    article = ambiguity._rule_decision("Điều 16 quy định trách nhiệm gì?")
    section = ambiguity._rule_decision("Mục 2 áp dụng cho đối tượng nào?")

    assert article.action == ambiguity.DIRECT_RETRIEVAL
    assert section.action == ambiguity.DIRECT_RETRIEVAL


def test_rule_ambiguity_routes_garbled_query_to_probe():
    decision = ambiguity._rule_decision("xtet đầu ra ta4 kiểu gì")

    assert decision.action == ambiguity.PROBE_RETRIEVAL
    assert decision.reason == "garbled_query_requires_probe"


def test_rule_ambiguity_does_not_use_query_length():
    decision = ambiguity._rule_decision(
        "nội dung hoàn toàn mới chưa thuộc danh sách chủ đề viết tay hiện tại"
    )

    assert decision.action == ambiguity.PROBE_RETRIEVAL
    assert decision.reason == "unknown_topic_requires_probe"


def test_hyde_cache_avoids_repeated_gemini_calls(monkeypatch):
    class FakeResponse:
        text = (
            "Quy định quản lý hệ thống camera giám sát mô tả trách nhiệm "
            "quản lý, vận hành, khai thác và bảo vệ dữ liệu hình ảnh."
        )

    calls = []

    def fake_generate_content(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    hyde.clear_hyde_cache()
    monkeypatch.setattr(hyde._client.models, "generate_content", fake_generate_content)

    first = hyde.generate_hyde_document("camera ai quản lý")
    second = hyde.generate_hyde_document("camera ai quản lý")

    assert first["status"] == "success"
    assert second["cache_hit"] is True
    assert len(calls) == 1


def test_ambiguity_cache_avoids_repeated_llm_calls(monkeypatch):
    calls = []
    ambiguity.clear_ambiguity_cache()
    monkeypatch.setattr(ambiguity, "_rule_decision", lambda question: None)

    def fake_llm(question):
        calls.append(question)
        return ambiguity.AmbiguityDecision(
            ambiguity.DIRECT_RETRIEVAL,
            "custom_topic",
            0.8,
            "mock_llm",
            analyzer="llm",
        )

    monkeypatch.setattr(ambiguity, "_analyze_with_llm", fake_llm)

    first = ambiguity.analyze_ambiguity("câu hỏi chưa có rule")
    second = ambiguity.analyze_ambiguity("câu hỏi chưa có rule")

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1


def test_hyde_need_clarification_stops_before_retrieval(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "generate_hyde_document",
        lambda question: {
            "text": "",
            "attempted": True,
            "status": "need_clarification",
            "text_hash": None,
            "char_count": 0,
            "word_count": 0,
            "error": None,
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("BM25 must not run")
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ANN must not run")
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "camera gì đó",
            debug=debug,
            ambiguity_decision={"action": ambiguity.HYDE_RETRIEVAL},
        )
    )

    assert results == []
    assert debug["fallback_reason"] == "hyde_requested_clarification"


def test_hyde_error_falls_back_to_direct_retrieval(monkeypatch):
    bm25_doc = _doc("direct.pdf", 1, bm25_score=6, keyword_score=6, score=6)
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "generate_hyde_document",
        lambda question: {
            "text": "",
            "attempted": True,
            "status": "error_direct_fallback",
            "text_hash": None,
            "char_count": 0,
            "word_count": 0,
            "error": "Gemini unavailable",
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [bm25_doc],
    )
    monkeypatch.setattr(retrieval, "search_similar_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs,
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "camera ai quản lý",
            debug=debug,
            ambiguity_decision={"action": ambiguity.HYDE_RETRIEVAL},
        )
    )

    assert results[0]["doc_name"] == "direct.pdf"
    assert debug["fallback_reason"] == "hyde_error_direct_retrieval"


def test_hyde_ann_is_fused_but_original_question_is_used_for_rerank(monkeypatch):
    bm25_doc = _doc("original.pdf", 1, bm25_score=5, keyword_score=5, score=5)
    hyde_doc = _doc("hyde.pdf", 1, score=0.9, distance=0.1)
    ann_queries = []
    rerank_questions = []

    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda question, limit, source_type_filter=None: [bm25_doc],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_hyde_document",
        lambda question: {
            "text": "hypothetical camera administrative document",
            "attempted": True,
            "status": "success",
            "text_hash": "hash",
            "char_count": 43,
            "word_count": 4,
            "error": None,
            "cache_hit": False,
        },
    )

    def fake_ann(query, **kwargs):
        ann_queries.append(query)
        return [hyde_doc] if query.startswith("hypothetical") else []

    def fake_rerank(question, docs, final_top_k=None):
        rerank_questions.append(question)
        return docs, {"reason": "cross_encoder_success"}

    monkeypatch.setattr(retrieval, "search_similar_chunks", fake_ann)
    monkeypatch.setattr(retrieval, "rerank_documents", fake_rerank)

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "camera ai quản lý",
            debug=debug,
            ambiguity_decision={
                "action": ambiguity.HYDE_RETRIEVAL,
                "topic": "camera",
                "confidence": 0.8,
            },
        )
    )

    assert ann_queries == [
        "camera ai quản lý",
        "hypothetical camera administrative document",
    ]
    assert rerank_questions == ["camera ai quản lý"]
    assert debug["ann_hyde_results"]
    assert all("hypothetical" not in doc.get("content", "") for doc in results)
    assert "text" not in debug["hyde"]


def test_probe_unknown_topic_uses_grounded_hyde(monkeypatch):
    bm25_doc = _doc(
        "advisor.pdf",
        1,
        title="Tiêu chuẩn của cố vấn học tập",
        content="Cố vấn học tập phải đáp ứng các tiêu chuẩn theo quy định.",
        bm25_score=8,
        keyword_score=8,
        score=8,
    )
    grounded_doc = _doc(
        "advisor.pdf",
        2,
        title="Nhiệm vụ của cố vấn học tập",
        score=0.88,
        distance=0.12,
    )
    ann_queries = []

    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(bm25_doc)],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda question, evidence: {
            "text": "Tiêu chuẩn, nhiệm vụ và trách nhiệm của cố vấn học tập.",
            "attempted": True,
            "status": "success",
            "text_hash": "grounded-hash",
            "char_count": 58,
            "word_count": 10,
            "error": None,
            "cache_hit": False,
            "grounding_hash": "evidence-hash",
            "grounding_source_count": len(evidence),
        },
    )

    def fake_ann(query, **kwargs):
        ann_queries.append(query)
        if query.startswith("Tiêu chuẩn, nhiệm vụ"):
            return [dict(grounded_doc)]
        return [dict(bm25_doc, score=0.7, distance=0.3)]

    monkeypatch.setattr(retrieval, "search_similar_chunks", fake_ann)
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:2],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "Tiêu chuẩn của cố vấn học tập",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert len(ann_queries) == 2
    assert debug["probe_retrieval"]["has_confident_evidence"] is True
    assert debug["grounded_hyde"]["status"] == "success"
    assert debug["ann_grounded_hyde_results"]
    assert results


def test_probe_garbled_query_can_use_strong_document_evidence(monkeypatch):
    evidence = _doc(
        "hoc-phan.pdf",
        1,
        title="Đăng ký học phần",
        bm25_score=3,
        keyword_score=3,
        score=0.8,
        distance=0.2,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(evidence)],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: [dict(evidence)],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda question, docs: {
            "text": "Quy trình đăng ký học phần trong hệ thống đào tạo.",
            "attempted": True,
            "status": "success",
            "text_hash": "hash",
            "char_count": 49,
            "word_count": 9,
            "error": None,
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "dk hp sao",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results
    assert debug["probe_retrieval"]["has_confident_evidence"] is True
    assert debug["grounded_hyde"]["attempted"] is True


def test_probe_without_evidence_requests_clarification(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(retrieval, "search_similar_chunks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Grounded HyDE must not run without evidence")
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "xyz abc không xác định",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results == []
    assert debug["fallback_reason"] == "probe_insufficient_evidence"
    assert debug["probe_retrieval"]["decision"] == "clarification_needed"


def test_cached_empty_probe_keeps_clarification_decision(monkeypatch):
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: [])

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "xyz abc không xác định",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results == []
    assert debug["cache_hit"] is True
    assert debug["fallback_reason"] == "probe_insufficient_evidence"


def test_grounded_hyde_error_falls_back_to_original_retrieval(monkeypatch):
    original = _doc(
        "advisor.pdf",
        1,
        title="Cố vấn học tập",
        bm25_score=5,
        keyword_score=5,
        score=0.75,
        distance=0.25,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(original)],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: [dict(original)],
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda *args, **kwargs: {
            "text": "",
            "attempted": True,
            "status": "error_direct_fallback",
            "text_hash": None,
            "char_count": 0,
            "word_count": 0,
            "error": "Gemini unavailable",
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "cố vấn hỗ trợ gì",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results
    assert debug["fallback_reason"] == "grounded_hyde_error_original_retrieval"


def test_probe_embedding_error_can_fall_back_to_bm25(monkeypatch):
    bm25_doc = _doc(
        "advisor.pdf",
        1,
        title="Tiêu chuẩn cố vấn học tập",
        bm25_score=8,
        keyword_score=8,
        score=8,
    )
    monkeypatch.setattr(retrieval, "_current_document_signature", lambda: ("sig",))
    monkeypatch.setattr(retrieval, "_get_search_cache", lambda key: None)
    monkeypatch.setattr(retrieval, "_set_search_cache", lambda key, value: None)
    monkeypatch.setattr(
        retrieval,
        "_search_metadata_documents",
        lambda question, limit: ([], {}),
    )
    monkeypatch.setattr(
        retrieval,
        "_search_bm25_documents",
        lambda *args, **kwargs: [dict(bm25_doc)],
    )
    monkeypatch.setattr(
        retrieval,
        "search_similar_chunks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("embedding unavailable")
        ),
    )
    monkeypatch.setattr(
        retrieval,
        "generate_grounded_hyde_document",
        lambda *args, **kwargs: {
            "text": "Tiêu chuẩn của cố vấn học tập.",
            "attempted": True,
            "status": "success",
            "text_hash": "hash",
            "char_count": 34,
            "word_count": 6,
            "error": None,
            "cache_hit": False,
        },
    )
    monkeypatch.setattr(
        retrieval,
        "rerank_documents",
        lambda question, docs, final_top_k=None: (
            docs[:1],
            {"reason": "cross_encoder_success"},
        ),
    )

    debug = {}
    results = asyncio.run(
        retrieval.search_documents(
            "tiêu chuẩn cố vấn",
            debug=debug,
            ambiguity_decision={"action": ambiguity.PROBE_RETRIEVAL},
        )
    )

    assert results[0]["doc_name"] == "advisor.pdf"
    assert len(debug["vector_errors"]) == 2
