import os
from pathlib import Path


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.data.business_knowledge import (
    BUSINESS_FAQ_SOURCE_TYPE,
    _build_business_faq_rows,
    _score_business_faq,
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
    rows = _mapping_rows()

    top_doc = _top_faq("sinh vien xem diem o dau", rows)

    assert top_doc["source_type"] == BUSINESS_FAQ_SOURCE_TYPE
    assert top_doc["file_id"] == "PCNTT_FILE_02"
    assert "kết quả học tập" in top_doc["faq_question"].lower()
    assert "support.uneti.edu.vn" in top_doc["faq_answer"]


def test_business_faq_mapping_matches_lecturer_exam_workload_question():
    rows = _mapping_rows()

    top_doc = _top_faq("giang vien xem khoi luong coi thi cham thi o dau", rows)

    assert top_doc["source_type"] == BUSINESS_FAQ_SOURCE_TYPE
    assert top_doc["file_id"] == "PCNTT_FILE_03"
    assert "coi thi" in top_doc["faq_question"].lower()
    assert "Khối lượng coi, chấm thi" in top_doc["faq_answer"]
