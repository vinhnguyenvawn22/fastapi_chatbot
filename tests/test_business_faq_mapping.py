import os
from pathlib import Path


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.data.business_knowledge import (
    BUSINESS_FAQ_SOURCE_TYPE,
    BUSINESS_SOURCE_TYPE,
    _build_business_faq_rows,
    _score_business_faq,
    build_business_faq_answer,
    clear_business_knowledge_cache,
    search_business_sources,
)


def _mapping_rows():
    root = Path("documents/nghiep_vu").resolve()
    return _build_business_faq_rows(root / "PCNTT_MAPPING_FILE.docx", root)


def _top_faq(query: str, rows: list[dict]):
    ranked = sorted(
        rows,
        key=lambda row: _score_business_faq(query, row),
        reverse=True,
    )
    return ranked[0]


def test_business_faq_mapping_matches_student_grade_question():
    top_doc = _top_faq("sinh vien xem diem o dau", _mapping_rows())

    assert top_doc["source_type"] == BUSINESS_FAQ_SOURCE_TYPE
    assert top_doc["file_id"] == "PCNTT_FILE_02"
    assert "support.uneti.edu.vn" in top_doc["faq_answer"]


def test_business_faq_mapping_matches_lecturer_exam_workload_question():
    top_doc = _top_faq(
        "giang vien xem khoi luong coi thi cham thi o dau",
        _mapping_rows(),
    )

    assert top_doc["source_type"] == BUSINESS_FAQ_SOURCE_TYPE
    assert top_doc["file_id"] == "PCNTT_FILE_03"


def test_mapping_guided_password_search_returns_original_file():
    debug = {}

    docs = search_business_sources(
        "Em quen mat khau email truong thi lay lai nhu the nao?",
        debug=debug,
    )

    assert docs
    assert docs[0]["source_type"] == BUSINESS_SOURCE_TYPE
    assert docs[0]["doc_name"] == "2026.03.03.ChatbotAI_CBGV_SV_V4.docx"
    assert "faq_answer" not in docs[0]
    assert debug["file_id"] == "PCNTT_FILE_01"
    assert debug["retrieval_method"] in {"location", "keyword", "vector"}
    assert "content" not in str(debug["final_sources"])


def test_mapping_guided_lms_search_returns_original_file():
    debug = {}

    docs = search_business_sources(
        "Khong dang nhap duoc LMS thi lam the nao?",
        debug=debug,
    )

    assert docs
    assert docs[0]["doc_name"] == "2026.03.03.ChatbotAI_CBGV_SV_V4.docx"
    assert debug["file_id"] == "PCNTT_FILE_01"


def test_mapping_guided_projector_search_returns_lecturer_source():
    docs = search_business_sources("May chieu trong phong hoc bi hong thi bao o dau?")

    assert docs
    assert docs[0]["doc_name"] == "2026.03.25.AI_HDSD TREN WEB SUPPORT CBGV.docx"
    assert docs[0]["source_type"] == BUSINESS_SOURCE_TYPE
    assert docs[0]["faq_location"] is None


def test_mapping_guided_grade_and_schedule_search_returns_student_source():
    docs = search_business_sources("Em xem diem va thoi khoa bieu o dau?")

    assert docs
    assert docs[0]["doc_name"] == "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx"


def test_mapping_match_without_original_file_returns_no_source(monkeypatch):
    mapping = {
        "source_type": BUSINESS_FAQ_SOURCE_TYPE,
        "title": "Reset password",
        "faq_question": "Reset password",
        "faq_answer": "Summary must not be returned",
        "faq_keywords": "reset password",
        "faq_location": "Section 1",
        "file_id": "MISSING_FILE",
        "doc_name": "missing.docx",
        "relative_path": "missing.docx",
        "source_relative_path": None,
        "source_file_found": False,
        "source_root": "nghiep_vu",
        "chunk_index": 1,
    }
    monkeypatch.setattr(
        "app.data.business_knowledge._load_business_index",
        lambda: ([mapping], {}, 1),
    )
    clear_business_knowledge_cache()
    debug = {}

    docs = search_business_sources("Reset password", debug=debug)

    assert docs == []
    assert debug["mapping_selected"] is True
    assert debug["source_file_found"] is False


def test_missing_location_falls_back_to_keyword_in_original_file(monkeypatch):
    mapping = {
        "source_type": BUSINESS_FAQ_SOURCE_TYPE,
        "title": "Reset password",
        "faq_question": "Reset password",
        "faq_answer": "Summary must not be returned",
        "faq_keywords": "reset password account",
        "faq_location": "Section 9.9",
        "file_id": "SOURCE_01",
        "doc_name": "source.docx",
        "relative_path": "source.docx",
        "source_relative_path": "source.docx",
        "source_file_found": True,
        "source_root": "nghiep_vu",
        "chunk_index": 1,
    }
    source = {
        "source_type": BUSINESS_SOURCE_TYPE,
        "title": "Account recovery",
        "content": "Reset password for the user account from the account recovery screen.",
        "doc_name": "source.docx",
        "relative_path": "source.docx",
        "source_root": "nghiep_vu",
        "chunk_index": 3,
    }
    monkeypatch.setattr(
        "app.data.business_knowledge._load_business_index",
        lambda: ([mapping, source], {}, 2),
    )
    clear_business_knowledge_cache()
    debug = {}

    docs = search_business_sources("Reset password", debug=debug)

    assert docs
    assert debug["retrieval_method"] == "keyword"
    assert debug["matched_location"] is None
    assert docs[0]["title"] == "Account recovery"
    assert "faq_answer" not in docs[0]


def test_mapping_summary_is_never_returned_as_direct_answer():
    docs = [{
        "source_type": BUSINESS_FAQ_SOURCE_TYPE,
        "faq_answer": "This is only a summary.",
    }]

    assert build_business_faq_answer(docs) is None
