import os

os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.controller.chatbot_controller import (
    _allow_aggregate_direct_business_answer,
    _should_prefer_business_over_internal,
)
from app.data.business_knowledge import clear_business_knowledge_cache, search_business_sources
from app.data.query_context import analyze_query_context
from app.data.query_analyzer import QueryIntent, classify_query


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


def test_procedure_ui_terms_cover_student_and_teacher_web_support():
    student = analyze_query_context("cach xem diem danh theo hoc ky", [])
    regrade = analyze_query_context("toi muon cham lai bai thi thi lam the nao", [])
    teacher = analyze_query_context("giang vien xem khoi luong coi thi o dau", [])
    student_policy = analyze_query_context("quy dinh diem chuyen can the nao", [])
    teacher_policy = analyze_query_context("quy dinh khoi luong giang day cua giang vien", [])

    assert student["audience_hint"] == "sv"
    assert student["information_need"] == "procedure_ui"
    assert regrade["audience_hint"] == "sv"
    assert regrade["information_need"] == "procedure_ui"
    assert teacher["audience_hint"] == "cbgv"
    assert teacher["information_need"] == "procedure_ui"
    assert student_policy["information_need"] == "policy_document"
    assert teacher_policy["information_need"] == "policy_document"


def test_web_support_terms_do_not_fall_to_default_document_classification():
    student = classify_query("cach xem diem danh theo hoc ky")
    regrade = classify_query("toi muon cham lai bai thi thi lam the nao")
    teacher = classify_query("giang vien xem khoi luong coi thi o dau")
    policy = classify_query("quy dinh diem chuyen can the nao")

    assert student.reason == "business_support_terms"
    assert regrade.reason == "business_support_terms"
    assert teacher.reason == "business_support_terms"
    assert policy.reason == "document_terms"


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


def test_teacher_schedule_uses_related_cbgv_mapping():
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
    assert debug["mapping_selected"] is True
    assert debug["mapping_question"] == "Giảng viên xem lớp học phần giảng viên ở đâu?"
    assert debug["retrieval_method"] in {"location", "keyword", "vector"}


def test_teacher_workload_lookup_keeps_mapping_with_query_context():
    clear_business_knowledge_cache()
    question = "cach truy cap chuc nang tra cuu khoi luong cong tac giang vien"
    debug = {}
    docs = search_business_sources(
        question,
        debug=debug,
        query_context=analyze_query_context(question, []),
    )

    assert docs
    assert debug["mapping_selected"] is True
    assert debug["mapping_question"] == (
        "Làm thế nào để truy cập chức năng Tra cứu khối lượng công tác giảng viên?"
    )
    assert debug["retrieval_method"] in {"location", "keyword", "vector"}
    assert "WEB SUPPORT CBGV" in docs[0]["doc_name"].upper()


def test_student_attendance_lookup_uses_web_support_mapping():
    clear_business_knowledge_cache()
    question = "cach xem diem danh theo hoc ky"
    debug = {}
    docs = search_business_sources(
        question,
        debug=debug,
        query_context=analyze_query_context(question, []),
    )

    assert docs
    assert debug["mapping_selected"] is True
    assert debug["mapping_question"] == "Làm thế nào để xem điểm danh theo học kỳ?"
    assert debug["retrieval_method"] in {"location", "keyword", "vector"}
    assert "WEB SUPPORT SV" in docs[0]["doc_name"].upper()
    assert "diem danh" in docs[0]["content"].lower() or "Điểm danh" in docs[0]["content"]


def test_student_regrade_exam_lookup_uses_web_support_mapping():
    clear_business_knowledge_cache()
    question = "toi muon cham lai bai thi thi lam the nao"
    debug = {}
    docs = search_business_sources(
        question,
        debug=debug,
        query_context=analyze_query_context(question, []),
    )

    assert docs
    assert debug["mapping_selected"] is True
    assert debug["mapping_question"] in {
        "Tôi muốn chấm lại bài thi thì làm thế nào?",
        "Làm thế nào để gửi yêu cầu phúc khảo/chấm lại bài thi?",
    }
    assert debug["retrieval_method"] in {"location", "keyword", "vector"}
    assert "WEB SUPPORT SV" in docs[0]["doc_name"].upper()
    assert "phuc khao" in docs[0]["content"].lower() or "Phúc khảo" in docs[0]["content"]


def test_business_priority_prefers_web_support_for_ui_procedures():
    business_doc = {
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
    }
    business_state = {
        "retrieval_debug": {
            "mapping_selected": True,
            "retrieval_method": "keyword",
            "mapping_gate_score": 60,
            "information_need": "procedure_ui",
            "audience_hint": "sv",
            "audience_source": "business_inference",
        }
    }
    preferred, debug = _should_prefer_business_over_internal(
        "cach xem diem danh theo hoc ky",
        business_state,
        [business_doc],
        [{"doc_name": "quy-che.docx", "relative_path": "quy-che.docx"}],
    )

    assert preferred is True
    assert debug["reason"] == "procedure_ui_web_support_business_source"


def test_business_priority_prefers_teacher_web_support_for_ui_procedures():
    business_doc = {
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT CBGV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT CBGV.docx",
    }
    business_state = {
        "retrieval_debug": {
            "mapping_selected": True,
            "retrieval_method": "keyword",
            "mapping_gate_score": 60,
            "information_need": "procedure_ui",
            "audience_hint": "cbgv",
            "audience_source": "explicit_question",
        }
    }
    preferred, debug = _should_prefer_business_over_internal(
        "giang vien xem khoi luong coi thi o dau",
        business_state,
        [business_doc],
        [{"doc_name": "quy-che.docx", "relative_path": "quy-che.docx"}],
    )

    assert preferred is True
    assert debug["has_web_support_source"] is True


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
        "quy dinh ve diem hoc phan", business_state, [internal_doc]
    )

    assert preferred is False
    assert debug["information_need"] == "policy_document"
    assert direct_allowed is False
    assert reason in {"policy_document", "internal_metadata_matched"}


def test_policy_document_questions_do_not_force_business_priority():
    policy_questions = [
        "quy dinh diem chuyen can the nao",
        "sinh vien vang bao nhieu thi bi cam thi",
        "theo quy che diem chuyen can duoc tinh ra sao",
        "dieu nao quy dinh ve diem chuyen can",
        "quy dinh phuc khao diem thi the nao",
        "thoi han phuc khao theo quy che",
        "van ban nao quy dinh phuc khao",
        "quy dinh khoi luong giang day cua giang vien",
        "theo quy che giang vien phai day bao nhieu gio",
        "quy dinh ve coi thi cham thi cua giang vien",
        "van ban nao quy dinh che do lam viec cua giang vien",
    ]
    business_state = {
        "retrieval_debug": {
            "mapping_selected": True,
            "retrieval_method": "keyword",
            "mapping_gate_score": 60,
            "information_need": "policy_document",
            "audience_hint": "sv",
        }
    }
    business_doc = {
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
    }

    for question in policy_questions:
        preferred, debug = _should_prefer_business_over_internal(
            question,
            business_state,
            [business_doc],
            [{"doc_name": "quy-che.docx", "relative_path": "quy-che.docx"}],
        )
        assert preferred is False
        assert debug["reason"] in {
            "academic_policy_terms_prefer_internal_or_merge",
            "policy_document",
            "document_intent_terms_prefer_internal_or_merge",
        }


def test_policy_with_website_news_term_still_routes_to_internal_document():
    policy = classify_query("quy che tuyen sinh dai hoc nam nay")
    news = classify_query("thong tin tuyen sinh dai hoc nam nay")

    assert policy.intent == QueryIntent.INTERNAL_DOCUMENT
    assert policy.reason == "document_terms"
    assert news.intent == QueryIntent.WEBSITE_UNETI


def test_attendance_exam_question_is_policy_not_procedure():
    context = analyze_query_context("nghi hoc khong phep co bi cam thi khong", [])

    assert context["information_need"] == "policy_document"
    assert "cam thi" in context["information_need_signals"]["policy"]


def test_exam_retake_registration_is_procedure_ui_with_dang_ki_variant():
    context = analyze_query_context("huong dan dang ki thi lai", [])

    assert context["information_need"] == "procedure_ui"
    assert context["audience_hint"] == "sv"
