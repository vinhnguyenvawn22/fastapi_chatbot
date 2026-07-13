import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.controller.chatbot_controller import (
    _allow_aggregate_direct_business_answer,
    _should_prefer_business_over_internal,
)
from app.data.business_knowledge import clear_business_knowledge_cache, search_business_sources
from app.data.query_context import analyze_query_context


def test_explicit_history_inference_and_default_audience_priority():
    explicit = analyze_query_context("Giang vien xem lich day o dau?", [])
    assert explicit["audience_hint"] == "cbgv"
    assert explicit["audience_source"] == "explicit_question"

    history = [
        {"role": "user", "content": "Toi la giang vien"},
        {"role": "assistant", "content": "Da hieu"},
    ]
    inherited = analyze_query_context("Xem lich o dau?", history)
    assert inherited["audience_hint"] == "cbgv"
    assert inherited["audience_source"] == "conversation_history"

    inferred = analyze_query_context("Xem diem o dau?", [])
    assert inferred["audience_hint"] == "sv"
    assert inferred["audience_source"] == "business_inference"

    defaulted = analyze_query_context("Huong dan toi", [])
    assert defaulted["audience_hint"] == "sv"
    assert defaulted["audience_source"] == "default_student"


def test_current_question_overrides_history_and_mixed_is_preserved():
    history = [{"role": "user", "content": "Toi la sinh vien"}]
    corrected = analyze_query_context("Khong, toi la giang vien", history)
    assert corrected["audience_hint"] == "cbgv"
    assert corrected["audience_source"] == "explicit_question"

    mixed = analyze_query_context("Sinh vien va giang vien xem lich o dau?", [])
    assert mixed["audience_hint"] == "mixed"


def test_information_need_is_separate_from_audience():
    procedure = analyze_query_context("Sinh vien xem diem o dau?", [])
    policy = analyze_query_context("Quy dinh ve diem hoc phan", [])
    mixed = analyze_query_context("Dieu kien phuc khao va nop yeu cau o dau?", [])

    assert procedure["information_need"] == "procedure_ui"
    assert policy["information_need"] == "policy_document"
    assert mixed["information_need"] == "mixed"


def test_business_mapping_handles_student_procedure_and_rejects_generic_query():
    clear_business_knowledge_cache()
    debug = {}
    docs = search_business_sources(
        "xem diem o dau",
        debug=debug,
        query_context=analyze_query_context("xem diem o dau", []),
    )
    assert docs
    assert debug["mapping_selected"] is True
    assert "WEB SUPPORT SV" in docs[0]["doc_name"].upper()
    assert debug["audience_source"] == "business_inference"
    assert debug["information_need"] == "procedure_ui"
    assert debug["mapping_gate_decisions"][0]["score_components"]

    vague_debug = {}
    vague_docs = search_business_sources(
        "diem",
        debug=vague_debug,
        query_context=analyze_query_context("diem", []),
    )
    assert vague_docs == []
    assert vague_debug["mapping_ambiguous"] is True
    assert vague_debug["mapping_rejected_reason"] == "overly_generic_query"


def test_teacher_schedule_does_not_accept_unrelated_business_mapping():
    clear_business_knowledge_cache()
    question = "giang vien xem lich day o dau"
    debug = {}
    docs = search_business_sources(
        question,
        debug=debug,
        query_context=analyze_query_context(question, []),
    )

    assert docs
    assert "CBGV" in docs[0]["doc_name"].upper()
    assert debug["mapping_selected"] is False
    assert all(
        decision["decision"] == "reject"
        for decision in debug["mapping_gate_decisions"]
    )


def test_business_cache_isolated_by_audience_and_information_need():
    clear_business_knowledge_cache()
    student = {
        "audience_hint": "sv", "audience_source": "default_student",
        "information_need": "unknown",
    }
    teacher = {
        "audience_hint": "cbgv", "audience_source": "explicit_question",
        "information_need": "unknown",
    }
    first, second, third = {}, {}, {}
    search_business_sources("diem", debug=first, query_context=student)
    search_business_sources("diem", debug=second, query_context=teacher)
    search_business_sources("diem", debug=third, query_context=student)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert third["cache_hit"] is True


def test_policy_need_blocks_business_priority_and_direct_answer():
    business_state = {
        "retrieval_debug": {
            "mapping_selected": True,
            "retrieval_method": "keyword",
            "mapping_gate_score": 95,
            "information_need": "policy_document",
            "audience_source": "default_student",
        }
    }
    business_doc = {"doc_name": "support-sv.docx", "relative_path": "support-sv.docx"}
    internal_doc = {
        "doc_name": "quy-che.pdf", "relative_path": "quy-che.pdf",
        "metadata_matched": True,
    }

    preferred, debug = _should_prefer_business_over_internal(
        "quy dinh ve diem hoc phan", business_state, [business_doc], [internal_doc]
    )
    direct_allowed, reason = _allow_aggregate_direct_business_answer(
        business_state, [internal_doc]
    )

    assert preferred is False
    assert debug["information_need"] == "policy_document"
    assert direct_allowed is False
    assert reason in {"policy_document", "internal_metadata_matched"}
