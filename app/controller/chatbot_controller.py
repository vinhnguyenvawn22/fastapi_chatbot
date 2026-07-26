import asyncio
import re
import time

from fastapi import HTTPException
from langsmith import traceable

from app.core.config import (
    HYDE_MIN_RERANK_SCORE,
    MIN_SEARCH_SCORE,
    MIN_VECTOR_CONFIDENCE,
    SHORT_QUERY_MIN_SEARCH_SCORE,
    SHORT_QUERY_MIN_VECTOR_CONFIDENCE,
)
from app.data.elasticsearch_client import get_keywords, normalize_text, search_documents
from app.data.ambiguity_analyzer import (
    CLARIFICATION_NEEDED,
    DIRECT_RETRIEVAL,
    analyze_ambiguity,
)
from app.data.langchain_pipeline import (
    generate_answer,
    retrieve_business,
    retrieve_internal,
    retrieve_local_documents,
    retrieve_website,
)
from app.data.multi_aspect_query import (
    clean_multi_aspect_answer,
    decompose_multi_aspect_query,
    filter_semantic_aspect_docs,
    merge_multi_aspect_results,
    validate_multi_aspect_answer,
)
from app.data.gemini_client import get_gemini_call_count
from app.data.query_analyzer import QueryIntent, classify_query
from app.data.query_context import analyze_query_context
from app.data.reranker import rerank_chunks
from app.data.trace_logger import RagTrace, load_trace
from app.data.conversation_context import get_conversation_context
from app.data.website_search_client import index_uneti_website


SOURCE_PREVIEW_CHARS = 1100
SOURCE_PREVIEW_SENTENCES = 5
NO_WEBSITE_EVIDENCE_ANSWER = "Không tìm thấy thông tin phù hợp trên website UNETI."
NO_EVIDENCE_ANSWER = "Không tìm thấy căn cứ đủ rõ trong tài liệu đã cung cấp."
OUT_OF_SCOPE_ANSWER = "Câu hỏi này nằm ngoài phạm vi tài liệu nội bộ hiện có."
GENERAL_ADVICE_ANSWER = "Câu hỏi này không cần tra cứu tài liệu nội bộ. Vui lòng hỏi về quy định, quy trình, văn bản hoặc nội dung trong tài liệu đã cung cấp."
SHORT_QUERY_KEYWORD_COUNT = 3
MIN_LEXICAL_COVERAGE = 0.5
MIN_SOURCE_CONTENT_CHARS = 80
MAX_CHUNKS_PER_DOCUMENT = 2
AGGREGATE_MAX_CONTEXT_CHUNKS = 8
AGGREGATE_MAX_DIVERSE_CONTEXT_CHUNKS = 10
AGGREGATE_MIN_RELATIVE_SCORE = 0.32
INTERNAL_SOURCE_TYPE = "local_file"
MULTI_HOP_MAX_SUBQUESTIONS = 6
MULTI_HOP_DOCS_PER_ROUTE = 4


def _is_cbgv_admin_process_steps_question(question: str) -> bool:
    normalized = normalize_text(question)
    has_admin_process = (
        "thu tuc hanh chinh" in normalized
        and "ho so" in normalized
        and any(term in normalized for term in ("quy trinh xu ly", "xu ly ho so", "quy trinh"))
    )
    asks_for_steps = any(
        term in normalized
        for term in ("gom may buoc", "bao nhieu buoc", "cac buoc", "may buoc", "nhung buoc")
    )
    asks_process = "giang vien" in normalized or "cbgv" in normalized or "can bo" in normalized
    asks_student = any(term in normalized for term in ("sinh vien", "sv", "nguoi hoc"))
    return has_admin_process and (asks_for_steps or asks_process) and not asks_student


def _is_exam_retake_procedure_question(question: str) -> bool:
    normalized = normalize_text(question)
    return "thi lai" in normalized and any(
        term in normalized
        for term in ("dang ky", "dang ki", "huy", "huong dan", "lam the nao", "lam sao", "cach")
    )


def _exam_retake_business_docs(docs: list[dict]) -> list[dict]:
    return [
        doc for doc in docs or []
        if (
            "web support sv" in normalize_text(doc.get("doc_name", ""))
            and any(
                term in normalize_text(
                    " ".join(str(doc.get(field) or "") for field in ("title", "content", "faq_location"))
                )
                for term in ("dang ky thi lai", "huy dang ky thi lai")
            )
        )
    ]


def _has_exam_retake_business_source(docs: list[dict]) -> bool:
    return bool(_exam_retake_business_docs(docs))


def _subquestion(
    aspect: str,
    query: str,
    need: str = "mixed",
    routes: tuple[str, ...] = ("business", "internal"),
) -> dict:
    return {
        "aspect": aspect,
        "query": query,
        "need": need,
        "routes": list(routes),
    }


def _decompose_query(question: str) -> list[dict]:
    """Rule-first decomposition for multi-hop retrieval without extra LLM calls."""
    normalized = normalize_text(question)
    subquestions: list[dict] = []

    absence_comparison = (
        "nghi hoc" in normalized
        and "khong phep" in normalized
        and "co phep" in normalized
        and any(term in normalized for term in ("khac", "so sanh", "phan biet", "nhung gi"))
    )
    if absence_comparison:
        subquestions.extend([
            _subquestion(
                "he_thong_diem_danh",
                "nghỉ học có phép nghỉ học không phép số buổi nghỉ có phép không phép điểm danh sinh viên",
                "procedure_ui",
            ),
            _subquestion(
                "nghi_hoc_tam_thoi_bao_luu",
                "nghỉ học tạm thời bảo lưu kết quả đã học sinh viên đại học điều kiện thủ tục",
                "policy_document",
                ("internal",),
            ),
            _subquestion(
                "hoan_thi_co_ly_do",
                "hoãn thi kiểm tra có lý do chính đáng được phép hoãn thi điểm I",
                "mixed",
            ),
            _subquestion(
                "nghi_khong_phep_che_tai",
                "nghỉ học không phép bỏ học bỏ kiểm tra bỏ thi không có lý do điểm 0 điểm F",
                "policy_document",
                ("internal",),
            ),
            _subquestion(
                "diem_chuyen_can_cam_thi",
                "điểm chuyên cần nghỉ học số tiết vắng nghỉ học trên 50% bị cấm thi",
                "policy_document",
                ("internal",),
            ),
            _subquestion(
                "bang_so_sanh",
                question,
                "mixed",
            ),
        ])

    elif any(term in normalized for term in ("khac nhau", "so sanh", "phan biet", "nhung gi")):
        subquestions.append(_subquestion("cau_hoi_goc", question, "mixed"))

    if "thu tuc hanh chinh" in normalized and any(term in normalized for term in ("xu ly", "nghiep vu")):
        subquestions.extend([
            _subquestion(
                "truy_cap_xu_ly_nghiep_vu",
                "cách truy cập chức năng xử lý nghiệp vụ thủ tục hành chính cán bộ giảng viên web support",
                "procedure_ui",
                ("business",),
            ),
            _subquestion(
                "quy_trinh_xu_ly_ho_so",
                "quy trình xử lý hồ sơ thủ tục hành chính tiếp nhận xử lý phê duyệt trả kết quả",
                "procedure_ui",
                ("business", "internal"),
            ),
        ])

    if "thu tuc hanh chinh" in normalized and any(term in normalized for term in ("phe duyet", "trinh duyet")):
        subquestions.extend([
            _subquestion(
                "phe_duyet_trinh_duyet_ho_so",
                "cách truy cập chức năng phê duyệt trình duyệt hồ sơ thủ tục hành chính cán bộ giảng viên",
                "procedure_ui",
                ("business",),
            ),
            _subquestion(
                "trang_thai_ho_so_phe_duyet",
                "hồ sơ thủ tục hành chính cần trưởng phó đơn vị phê duyệt ban giám hiệu phê duyệt",
                "procedure_ui",
                ("business",),
            ),
        ])

    if "ket qua hoc tap" in normalized and any(term in normalized for term in ("hoc ky", "theo ki", "theo ky", "theo ki", "theo kì", "cach xem")):
        subquestions.extend([
            _subquestion(
                "xem_ket_qua_hoc_tap_theo_ky",
                "cách xem kết quả học tập theo từng học kỳ sinh viên web support",
                "procedure_ui",
                ("business",),
            ),
            _subquestion(
                "chi_tiet_diem_hoc_phan",
                "xem chi tiết kết quả học tập điểm thành phần môn học học kỳ",
                "procedure_ui",
                ("business",),
            ),
        ])

    if "chung chi" in normalized and any(term in normalized for term in ("ra truong", "tot nghiep")):
        subquestions.extend([
            _subquestion(
                "dieu_kien_tot_nghiep",
                "điều kiện xét tốt nghiệp công nhận tốt nghiệp chứng chỉ ngoại ngữ tin học giáo dục quốc phòng",
                "policy_document",
                ("internal",),
            ),
            _subquestion(
                "quan_ly_chung_chi",
                "quy định chứng chỉ ngoại ngữ tin học sinh viên đại học chính quy",
                "policy_document",
                ("internal",),
            ),
        ])

    if "hoan thi" in normalized:
        subquestions.extend([
            _subquestion(
                "thu_tuc_hoan_thi",
                "hướng dẫn hoãn thi một cửa khảo thí gửi yêu cầu trên support sinh viên",
                "procedure_ui",
                ("business",),
            ),
            _subquestion(
                "dieu_kien_hoan_thi",
                "điều kiện hoãn thi vắng thi có lý do chính đáng điểm I quy chế đào tạo",
                "policy_document",
                ("internal",),
            ),
        ])

    if "thi lai" in normalized:
        action = "hủy đăng ký thi lại" if "huy" in normalized else "đăng ký thi lại"
        subquestions.extend([
            _subquestion(
                "thu_tuc_thi_lai",
                f"hướng dẫn {action} một cửa khảo thí gửi yêu cầu trên support sinh viên",
                "procedure_ui",
                ("business",),
            ),
            _subquestion(
                "quy_dinh_thi_lai",
                "quy định thi lại học phần đăng ký thi lại sinh viên",
                "policy_document",
                ("internal",),
            ),
        ])

    deduped = []
    seen = set()
    for item in subquestions:
        key = (item["aspect"], normalize_text(item["query"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:MULTI_HOP_MAX_SUBQUESTIONS]


def _answer_has_admin_process_steps(answer: str | None) -> bool:
    normalized = normalize_text(answer or "")
    return all(
        term in normalized
        for term in ("nop ho so", "tiep nhan ho so", "xu ly ho so", "phe duyet ho so", "tra ket qua")
    )


def _business_direct_answer(question: str, docs: list[dict]) -> str | None:
    """Return concise answers for high-confidence business support workflows."""
    normalized = normalize_text(question)
    doc_names = " ".join(normalize_text(doc.get("doc_name", "")) for doc in docs or [])
    has_cbgv_source = "web support cbgv" in doc_names or not docs

    if _is_exam_retake_procedure_question(question) and _has_exam_retake_business_source(docs):
        source_text = normalize_text(
            " ".join(
                str(doc.get(field) or "")
                for doc in docs or []
                for field in ("title", "content", "faq_location")
            )
        )
        if "huy" in normalized and "huy dang ky thi lai" not in source_text:
            return None
        if "huy" in normalized and all(
            term in source_text
            for term in ("huy dang ky thi lai", "dang nhap he thong", "mot cua - khao thi", "gui yeu cau")
        ):
            return (
                "Để hủy đăng ký thi lại, sinh viên thực hiện: "
                "1. Đăng nhập https://support.uneti.edu.vn bằng tài khoản cá nhân. "
                "2. Chọn Thủ tục hành chính -> Một cửa - Khảo thí -> Hủy đăng ký thi lại (Gửi yêu cầu), "
                "hoặc truy cập trực tiếp https://support.uneti.edu.vn/mot-cua/khao-thi/huy-dang-ky-thi-lai. "
                "3. Chọn hoặc điền các dữ liệu cần nhập. "
                "4. Tại lưới dữ liệu, bấm Chọn ở dòng học phần tương ứng rồi bấm Gửi yêu cầu. "
                "Lưu ý: nguồn nêu thời điểm xin hủy đăng ký thi lại là trước ngày thi 5 ngày và người học chưa nộp lệ phí thi lại.\n\n"
                "(Nguồn: 2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx)"
            )
        if "huy" not in normalized and all(
            term in source_text
            for term in ("dang nhap he thong", "mot cua - khao thi", "dang ky thi lai", "gui yeu cau")
        ):
            return (
                "Để đăng ký thi lại, sinh viên thực hiện: "
                "1. Đăng nhập https://support.uneti.edu.vn bằng tài khoản cá nhân. "
                "2. Chọn Thủ tục hành chính -> Một cửa - Khảo thí -> Đăng ký thi lại (Gửi yêu cầu), "
                "hoặc truy cập trực tiếp https://support.uneti.edu.vn/mot-cua/khao-thi/dang-ky-thi-lai. "
                "3. Chọn hoặc điền các dữ liệu cần nhập. "
                "4. Tại lưới dữ liệu, bấm Chọn ở dòng học phần tương ứng rồi bấm Gửi yêu cầu. "
                "Lưu ý: lệ phí thi lại sẽ nộp cùng học phí kỳ tiếp theo.\n\n"
                "(Nguồn: 2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx)"
            )

    if _is_exam_retake_procedure_question(question) and _has_exam_retake_business_source(docs):
        action = "hủy đăng ký thi lại" if "huy" in normalized else "đăng ký thi lại"
        return (
            f'Tài liệu hiện có xác định "{action}" là thủ tục thuộc nhóm Một cửa - Khảo thí, '
            "do Phòng Khảo thí và Đảm bảo chất lượng tiếp nhận và xử lý. "
            "Tuy nhiên, nguồn được truy xuất chưa nêu rõ các bước thực hiện chi tiết cho thủ tục này, "
            "nên chưa có căn cứ đủ rõ để hướng dẫn từng bước trên hệ thống.\n\n"
            "(Nguồn: 2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx)"
        )

    if not has_cbgv_source and "sinh vien" in doc_names:
        return None

    if "nhan su" in normalized and any(term in normalized for term in ("dung de lam gi", "lam gi", "muc dich")):
        return (
            "Màn Nhân sự dùng để xem thông tin nhân sự cá nhân của giảng viên "
            "và các khối lượng giảm trừ liên quan đến công tác giảng dạy. "
            "Giảng viên có thể xem thông tin cơ bản, sau đó nhấn vào dòng dữ liệu "
            "để xem chi tiết như mã số thuế, CCCD, tài khoản ngân hàng và khối lượng giảm trừ."
        )

    if "lop hoc phan" in normalized and any(term in normalized for term in ("duong dan", "link", "vao dau", "truy cap")):
        return (
            "Đường dẫn trực tiếp để xem Lớp học phần giảng viên là: "
            "https://support.uneti.edu.vn/cong-tac-giang-vien/tra-cuu/lop-hoc-phan-giang-vien"
        )

    if any(term in normalized for term in ("dang ky muon thiet bi", "muon thiet bi", "su dung thiet bi")):
        return (
            "Để đăng ký mượn/sử dụng thiết bị phòng học, thầy cô thực hiện: "
            "1. Đăng nhập https://support.uneti.edu.vn. "
            "2. Chọn Hỗ trợ thiết bị trong phòng học → Đăng ký sử dụng thiết bị. "
            "3. Chọn lịch dạy tương ứng và thiết bị cần mượn. "
            "4. Nhập lý do sử dụng thiết bị và ghi chú nếu có. "
            "5. Nhấn Gửi yêu cầu để hệ thống tiếp nhận."
        )

    if _is_cbgv_admin_process_steps_question(question):
        return (
            "Quy trình xử lý hồ sơ thủ tục hành chính gồm 5 bước: "
            "1. Nộp hồ sơ. "
            "2. Tiếp nhận hồ sơ. "
            "3. Xử lý hồ sơ. "
            "4. Phê duyệt hồ sơ nếu thủ tục yêu cầu phê duyệt. "
            "5. Trả kết quả."
        )

    if "minh chung" in normalized and "kiem dinh" in normalized and "trang thai" in normalized:
        return (
            "Các trạng thái minh chứng kiểm định gồm: "
            "Chờ duyệt, Đã duyệt và Cần bổ sung. "
            "Khi minh chứng ở trạng thái Cần bổ sung, người nộp cập nhật lại file/mô tả, "
            "sau đó minh chứng chuyển về Chờ duyệt để hội đồng kiểm định xem xét lại."
        )

    return None


def _clean_answer_text(answer: str | None) -> str:
    return str(answer or "").replace("**", "")


def _is_no_evidence_answer(answer: str | None) -> bool:
    normalized = normalize_text(_clean_answer_text(answer))
    return normalize_text(NO_EVIDENCE_ANSWER) in normalized


@traceable(name="Citation Check", run_type="chain")
def _citation_check(answer: str | None, source: str | None, source_count: int) -> dict:
    no_evidence = _is_no_evidence_answer(answer)
    return {
        "no_evidence_answer": no_evidence,
        "has_source": bool(source),
        "source_count": source_count,
        "citation_status": (
            "no_source_needed"
            if no_evidence
            else "has_source"
            if source or source_count > 0
            else "missing_source"
        ),
    }


def _source_search_text(doc: dict) -> str:
    return " ".join(
        str(doc.get(field) or "")
        for field in (
            "title",
            "content",
            "doc_name",
            "ten_van_ban",
            "so_van_ban",
            "phong_ban",
        )
    )


def _lexical_coverage(question: str, doc: dict) -> float:
    query_terms = set(get_keywords(question))
    if not query_terms:
        return 0.0
    source_terms = set(get_keywords(_source_search_text(doc)))
    return len(query_terms & source_terms) / len(query_terms)


def _is_usable_source(question: str, doc: dict) -> tuple[bool, str]:
    content = str(doc.get("content") or "").strip()
    if len(content) < MIN_SOURCE_CONTENT_CHARS:
        return False, "content_too_short"
    if not (doc.get("relative_path") or doc.get("doc_name") or doc.get("url")):
        return False, "missing_source_identity"
    if doc.get("metadata_matched"):
        return True, "metadata_matched"

    coverage = _lexical_coverage(question, doc)
    vector_score = doc.get("vector_score")
    if vector_score is None and doc.get("distance") is not None:
        vector_score = 1 - float(doc["distance"])

    if coverage >= MIN_LEXICAL_COVERAGE:
        return True, "lexical_coverage_passed"
    keyword_score = doc.get("keyword_score") or doc.get("score")
    if (
        _academic_policy_terms(question)
        and _matches_academic_policy_source(question, doc)
        and keyword_score is not None
        and float(keyword_score) >= MIN_SEARCH_SCORE
    ):
        return True, "academic_policy_source_passed"
    if vector_score is not None and float(vector_score) >= MIN_VECTOR_CONFIDENCE:
        return True, "semantic_score_passed"
    return False, "insufficient_query_coverage"


def _filter_usable_sources(question: str, docs: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted = []
    rejected = []
    for doc in docs:
        usable, reason = _is_usable_source(question, doc)
        enriched = dict(doc)
        enriched["lexical_coverage"] = round(_lexical_coverage(question, doc), 4)
        if usable:
            accepted.append(enriched)
        else:
            rejected.append({
                "doc_name": doc.get("doc_name"),
                "title": doc.get("title"),
                "reason": reason,
                "lexical_coverage": enriched["lexical_coverage"],
            })
    return accepted, rejected


def _limit_document_dominance(docs: list[dict]) -> list[dict]:
    selected = []
    counts = {}
    for doc in docs:
        key = doc.get("relative_path") or doc.get("doc_name") or doc.get("url")
        if counts.get(key, 0) >= MAX_CHUNKS_PER_DOCUMENT:
            continue
        counts[key] = counts.get(key, 0) + 1
        selected.append(doc)
    return selected


def _document_key(doc: dict) -> str:
    return str(doc.get("relative_path") or doc.get("doc_name") or doc.get("url") or "unknown")


def _source_base_score(doc: dict) -> float:
    for field in ("aggregate_score", "rerank_score", "rrf_score", "score", "keyword_score", "bm25_score"):
        value = doc.get(field)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue

    vector_score = doc.get("vector_score")
    if vector_score is None and doc.get("distance") is not None:
        try:
            vector_score = 1 - float(doc["distance"])
        except (TypeError, ValueError):
            vector_score = None
    if vector_score is not None:
        try:
            return float(vector_score) * 100
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _score_aggregate_evidence(question: str, doc: dict, query_context: dict | None = None) -> float:
    normalized = normalize_text(question)
    source_type = doc.get("source_type")
    doc_name = normalize_text(doc.get("doc_name", ""))
    source_text = normalize_text(_source_search_text(doc))
    query_terms = set(get_keywords(question))
    source_terms = set(get_keywords(source_text))
    overlap_count = len(query_terms & source_terms)
    score = _source_base_score(doc)

    if doc.get("metadata_matched"):
        score += 80
    score += float(doc.get("lexical_coverage") or 0) * 45
    coverage_aspects = doc.get("coverage_aspects") or []
    if coverage_aspects:
        score += min(len(coverage_aspects), 4) * 18

    information_need = (query_context or {}).get("information_need")
    if information_need == "policy_document":
        if source_type == "official_document":
            score += 35
        elif source_type == "business_document":
            score -= 18
    elif information_need == "procedure_ui":
        if source_type == "business_document":
            score += 35
        elif source_type == "official_document":
            score -= 8

    if _academic_policy_terms(question) and source_type == "official_document":
        score += 30
    if query_terms and overlap_count <= 1:
        score -= 45
        if _academic_policy_terms(question) and source_type == "business_document":
            score -= 75
    if any(term in normalized for term in ("huong dan", "lam sao", "lam the nao", "cach", "vao dau", "bam")):
        if "web support" in doc_name or source_type == "business_document":
            score += 30
    if source_type == "business_document":
        if "hoan thi" in normalized and "hoan thi" in source_text:
            score += 70
        if "thi lai" in normalized and "huy" in normalized and "huy dang ky thi lai" in source_text:
            score += 85
        elif "thi lai" in normalized and "huy" not in normalized and "dang ky thi lai" in source_text:
            score += 55

    academic_terms = _academic_policy_terms(question)
    if "course_registration_change" in academic_terms:
        if source_type == "official_document":
            if "rut bot hoc phan" in source_text or "huy dang ky hoc phan" in source_text:
                score += 150
            if doc.get("dieu") == 10:
                score += 110
            if "dang ky khoi luong hoc tap" in source_text:
                score += 95
            if doc.get("dieu") == 9:
                score += 45
            if "quy che dao tao dai hoc chinh quy" in source_text:
                score += 45
        if any(
            noisy in source_text
            for noisy in (
                "tieng anh",
                "toeic",
                "ielts",
                "chung chi",
                "quy doi",
                "ngoai ngu",
                "tin hoc",
                "khcn",
                "nghien cuu",
                "thiet bi",
                "phong hoc",
            )
        ):
            score -= 120

    if "credit_load_warning" in academic_terms:
        if source_type == "official_document":
            if "canh bao hoc tap" in source_text or "canh bao ket qua hoc tap" in source_text:
                score += 150
            if "dang trong thoi gian bi canh bao" in source_text:
                score += 80
            if "dang ky khoi luong hoc tap" in source_text or "khoi luong hoc tap" in source_text:
                score += 120
            if "16 tin chi" in source_text or "khong qua 16" in source_text:
                score += 180
            if "3/2 so tin chi" in source_text and not (
                "16 tin chi" in source_text or "khong qua 16" in source_text
            ):
                score -= 80
            if doc.get("dieu") == 9:
                score += 80
            if "quy che dao tao dai hoc chinh quy" in source_text:
                score += 60
        if source_type == "business_document":
            score -= 120
        if any(
            noisy in source_text
            for noisy in (
                "web support",
                "thoi khoa bieu",
                "lich hoc",
                "lich thi",
                "khcn",
                "nghien cuu",
                "thiet bi",
                "phong hoc",
            )
        ):
            score -= 120

    if "transfer_school" in academic_terms:
        if source_type == "official_document":
            if "chuyen truong" in source_text:
                score += 170
            if doc.get("dieu") == 28:
                score += 140
            if "hieu truong" in source_text:
                score += 80
            if any(term in source_text for term in ("cung nganh", "noi cu tru", "hoan canh")):
                score += 60
            if "quy che dao tao dai hoc chinh quy" in source_text:
                score += 80
        if source_type == "business_document":
            score -= 120
        if "thac si" in source_text and "thac si" not in normalized:
            score -= 180
        if "chuyen chuong trinh dao tao" in source_text and "chuyen truong" not in source_text:
            score -= 140

    if "elective_failed_course" in academic_terms:
        if source_type == "official_document":
            if "hoc phan tu chon" in source_text:
                score += 150
            if any(term in source_text for term in ("diem f", "f+", "khong dat")):
                score += 80
            if "hoc doi" in source_text or "hoc phan khac tuong duong" in source_text:
                score += 140
            if "hoc cai thien" in source_text or "diem trung binh tich luy" in source_text:
                score += 70
            if doc.get("dieu") == 11:
                score += 130
            if "quy che dao tao dai hoc chinh quy" in source_text:
                score += 70
        if source_type == "business_document":
            score -= 120
        if "thac si" in source_text and "thac si" not in normalized:
            score -= 140

    if "f_grade_comparison" in academic_terms:
        if source_type == "official_document":
            if "f+" in source_text and " f" in source_text:
                score += 140
            if any(term in source_text for term in ("thang diem", "diem chu", "diem hoc phan")):
                score += 100
            if doc.get("dieu") == 16:
                score += 140
            if doc.get("dieu") == 11:
                score += 120
            if any(term in source_text for term in ("hoc lai", "hoc doi", "hoc phan tu chon", "hoc phan bat buoc")):
                score += 90
            if "quy che dao tao dai hoc chinh quy" in source_text:
                score += 70
        if source_type == "business_document":
            score -= 120
        if "thac si" in source_text and "thac si" not in normalized:
            score -= 160

    if "credit_definition" in academic_terms:
        if source_type == "official_document":
            if "tin chi" in source_text:
                score += 80
            if "15 tiet" in source_text and "ly thuyet" in source_text:
                score += 120
            if "30 tiet" in source_text and any(term in source_text for term in ("thuc hanh", "thi nghiem", "thao luan")):
                score += 120
            if any(term in source_text for term in ("30 40 gio", "30-40 gio", "30 den 40 gio")):
                score += 100
            if any(term in source_text for term in ("45 60 gio", "45-60 gio", "45 den 60 gio")):
                score += 130
            if doc.get("dieu") == 2:
                score += 120
            if "quy che dao tao dai hoc chinh quy" in source_text:
                score += 80
        if source_type == "business_document":
            score -= 120
        if any(term in source_text for term in ("gpa", "tot nghiep", "chung chi", "web support", "khcn")):
            score -= 120

    if source_type == "business_document" and any(
        term in doc_name for term in ("khcn", "nghien cuu", "tap chi")
    ):
        score -= 80

    return round(score, 4)


def _select_diverse_aggregate_sources(
    question: str,
    business_docs: list[dict],
    internal_docs: list[dict],
    query_context: dict | None = None,
    limit: int = AGGREGATE_MAX_DIVERSE_CONTEXT_CHUNKS,
) -> tuple[list[dict], dict]:
    candidates = []
    for route, docs in (("business", business_docs), ("internal", internal_docs)):
        for doc in docs:
            enriched = dict(doc)
            enriched["aggregate_route"] = route
            enriched["aggregate_score"] = _score_aggregate_evidence(question, enriched, query_context)
            candidates.append(enriched)

    if not candidates:
        return [], {
            "candidate_count": 0,
            "selected_count": 0,
            "reason": "no_candidates",
        }

    candidates.sort(key=lambda doc: doc.get("aggregate_score") or 0, reverse=True)
    top_score = max(float(candidates[0].get("aggregate_score") or 0), 1.0)
    relative_floor = max(MIN_SEARCH_SCORE, top_score * AGGREGATE_MIN_RELATIVE_SCORE)

    selected = []
    doc_counts = {}
    type_counts = {}
    for doc in candidates:
        score = float(doc.get("aggregate_score") or 0)
        if score < relative_floor and selected:
            continue

        key = _document_key(doc)
        max_per_doc = 3 if limit >= 8 else MAX_CHUNKS_PER_DOCUMENT
        if doc_counts.get(key, 0) >= max_per_doc:
            continue

        source_type = doc.get("source_type") or "unknown"
        if type_counts.get(source_type, 0) >= max(4, limit - 2):
            other_types = {item.get("source_type") for item in candidates if item.get("source_type") != source_type}
            if other_types:
                continue

        selected.append(doc)
        doc_counts[key] = doc_counts.get(key, 0) + 1
        type_counts[source_type] = type_counts.get(source_type, 0) + 1
        if len(selected) >= limit:
            break

    if not selected:
        selected = candidates[:1]

    covered_aspects = {
        aspect
        for doc in selected
        for aspect in (doc.get("coverage_aspects") or [])
    }
    all_aspects = {
        aspect
        for doc in candidates
        for aspect in (doc.get("coverage_aspects") or [])
    }
    for aspect in sorted(all_aspects - covered_aspects):
        if len(selected) >= limit:
            break
        aspect_doc = next(
            (
                doc for doc in candidates
                if aspect in (doc.get("coverage_aspects") or [])
                and doc not in selected
                and float(doc.get("aggregate_score") or 0) >= relative_floor * 0.75
            ),
            None,
        )
        if aspect_doc:
            selected.append(aspect_doc)
            covered_aspects.add(aspect)
    type_counts = {}
    for doc in selected:
        source_type = doc.get("source_type") or "unknown"
        type_counts[source_type] = type_counts.get(source_type, 0) + 1

    return selected, {
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "top_score": top_score,
        "relative_floor": round(relative_floor, 4),
        "doc_name_count": len({_document_key(doc) for doc in selected}),
        "source_type_counts": dict(type_counts),
        "coverage_aspects": sorted(covered_aspects),
        "selected_sources": [
            {
                "doc_name": doc.get("doc_name"),
                "title": doc.get("title"),
                "chunk_index": doc.get("chunk_index"),
                "source_type": doc.get("source_type"),
                "aggregate_route": doc.get("aggregate_route"),
                "aggregate_score": doc.get("aggregate_score"),
                "keyword_score": doc.get("keyword_score"),
                "vector_score": doc.get("vector_score"),
                "rerank_score": doc.get("rerank_score"),
            }
            for doc in selected
        ],
    }


def _should_prefer_business_generation(business_state: dict, business_docs: list[dict]) -> bool:
    if not business_docs:
        return False

    retrieval_debug = business_state.get("retrieval_debug") or {}
    if not retrieval_debug.get("mapping_selected"):
        return False

    if retrieval_debug.get("retrieval_method") == "generic_hybrid":
        return False

    gate_score = retrieval_debug.get("mapping_gate_score")
    if gate_score is not None and float(gate_score) < 70:
        return False

    return retrieval_debug.get("retrieval_method") in {
        "location",
        "keyword",
        "vector",
    }


def _has_web_support_source(docs: list[dict]) -> bool:
    return any(
        "web support sv" in normalize_text(doc.get("doc_name", ""))
        or "web support cbgv" in normalize_text(doc.get("doc_name", ""))
        for doc in docs or []
    )


def _document_intent_terms(question: str) -> list[str]:
    normalized = normalize_text(question)
    terms = (
        "quyet dinh", "quy che", "thong bao", "van ban", "quy dinh",
        "dieu", "muc", "chuong", "can cu",
    )
    return [term for term in terms if term in normalized]


def _academic_policy_terms(question: str) -> list[str]:
    normalized = normalize_text(question)
    asks_attendance_exam = any(
        term in normalized
        for term in (
            "cam thi",
            "bi cam thi",
            "khong duoc thi",
            "du thi",
            "duoc thi",
            "diem chuyen can",
            "so tiet",
            "so tiet vang",
            "ty le vang",
            "nghi hoc tren",
            "tren 50",
            "qua 50",
        )
    )
    asks_absence_comparison = (
        "nghi hoc" in normalized
        and any(term in normalized for term in ("co phep", "khong phep"))
        and any(
            term in normalized
            for term in ("khac nhau", "khac gi", "phan biet", "so sanh", "nhung gi")
        )
    )
    phrase_terms = (
        "dieu kien tot nghiep",
        "xet tot nghiep",
        "ra truong",
        "chung chi",
        "chuan dau ra",
        "thi lai",
        "thi lai hoc phan",
        "hoan thi",
        "xin hoan thi",
        "hoan thi ket thuc hoc phan",
        "vang thi",
        "du thi",
        "cam thi",
        "khong duoc thi",
        "diem chuyen can",
        "nghi hoc khong phep",
        "nghi hoc tren",
        "so tiet vang",
        "canh bao hoc tap",
        "bao nhieu tin chi",
        "toi da bao nhieu tin chi",
        "so tin chi",
        "khoi luong hoc tap",
        "chuyen truong",
        "hoc 2 bang",
        "hoc cung luc hai chuong trinh",
        "chuong trinh thu hai",
        "hoc cai thien",
        "diem f",
        "f+",
        "hoc phan tu chon",
        "hoc doi",
        "tin chi tuong duong",
        "bao nhieu tiet",
        "ly thuyet",
        "thuc hanh",
        "huy dang ky hoc phan",
        "huy hoc phan",
        "rut bot hoc phan",
        "rut hoc phan",
        "bo hoc phan",
        "xoa hoc phan",
        "dang ky khoi luong hoc tap",
        "hoc vu",
        "quy che dao tao",
        "quy dinh dao tao",
    )
    matched = [term for term in phrase_terms if term in normalized]
    if (
        any(term in normalized for term in ("huy", "rut", "bo", "xoa"))
        and any(term in normalized for term in ("hoc phan", "mon", "dang ky", "dang ki"))
        and "thi lai" not in normalized
    ):
        matched.append("course_registration_change")
    if (
        "canh bao hoc tap" in normalized
        and any(term in normalized for term in ("tin chi", "khoi luong", "dang ky", "dang ki", "toi da", "bao nhieu"))
    ) or (
        any(term in normalized for term in ("toi da", "bao nhieu", "may tin chi", "so tin chi"))
        and "tin chi" in normalized
        and any(term in normalized for term in ("dang ky", "dang ki", "khoi luong hoc tap"))
    ):
        matched.append("credit_load_warning")
    if "chuyen truong" in normalized:
        matched.append("transfer_school")
    if (
        any(term in normalized for term in ("f+ va f", "f va f+", "diem f+", "f+"))
        and "f" in normalized
    ):
        matched.append("f_grade_comparison")
    if (
        "tu chon" in normalized
        and any(term in normalized for term in ("diem f", "bi f", "f+", "khong dat"))
        and any(term in normalized for term in ("mon", "hoc phan", "chon mon", "thay the", "hoc doi"))
    ):
        matched.append("elective_failed_course")
    if (
        "tin chi" in normalized
        and any(term in normalized for term in ("tuong duong", "bao nhieu tiet", "may tiet", "ly thuyet", "thuc hanh"))
    ):
        matched.append("credit_definition")
    if "tot nghiep" in normalized and any(term in normalized for term in ("chung chi", "dieu kien", "ra truong")):
        matched.append("tot_nghiep_condition")
    if "thi" in normalized and any(term in normalized for term in ("lai", "hoan", "vang", "du thi")):
        matched.append("exam_policy")
    if asks_absence_comparison and not asks_attendance_exam:
        matched.append("absence_permission_comparison")
    if (
        "nghi hoc" in normalized
        and asks_attendance_exam
    ) or (
        "diem chuyen can" in normalized
        and any(term in normalized for term in ("thi", "cam", "vang", "nghi"))
    ):
        matched.append("attendance_exam_eligibility")
    return matched


def _matches_academic_policy_source(question: str, doc: dict) -> bool:
    terms = _academic_policy_terms(question)
    if not terms:
        return True

    searchable = normalize_text(" ".join(
        str(doc.get(field) or "")
        for field in ("doc_name", "title", "relative_path", "phong_ban", "source_root", "content")
    ))
    metadata_text = normalize_text(" ".join(
        str(doc.get(field) or "")
        for field in ("doc_name", "title", "relative_path", "phong_ban", "source_root")
    ))

    if any(excluded in searchable for excluded in ("khcn", "nghien cuu", "thiet bi", "phong hoc")):
        return False
    if "tuyen sinh" in metadata_text:
        return False

    if "absence_permission_comparison" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        if (
            "thi ket thuc hoc phan" in metadata_text
            or "vang mat trong ky thi" in metadata_text
            or doc.get("dieu") == 15
        ):
            return False
        if any(term in metadata_text for term in ("diem ren luyen", "cong tac sinh vien", "ngoai tru")):
            return False
        training_regulation = any(
            marker in metadata_text
            for marker in ("pdt", "phong dao tao", "quy che dao tao dai hoc")
        )
        attendance_content = any(
            marker in searchable
            for marker in ("diem chuyen can", "nghi hoc", "so tiet", "danh gia hoc phan")
        )
        return training_regulation and attendance_content

    if "attendance_exam_eligibility" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        return (
            "diem chuyen can" in searchable
            or "nghi hoc tren 50" in searchable
            or ("cam thi" in searchable and "nghi hoc" in searchable)
            or (
                "danh gia hoc phan" in searchable
                and any(marker in searchable for marker in ("nghi hoc", "so tiet", "vang"))
            )
        )

    if "course_registration_change" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        if any(
            noisy in searchable
            for noisy in (
                "tieng anh",
                "toeic",
                "ielts",
                "chung chi",
                "quy doi",
                "ngoai ngu",
                "tin hoc",
                "khcn",
                "nghien cuu",
                "thiet bi",
                "phong hoc",
            )
        ):
            return False
        return (
            "dang ky khoi luong hoc tap" in searchable
            or "rut bot hoc phan" in searchable
            or "huy dang ky hoc phan" in searchable
            or (
                doc.get("dieu") in {9, 10}
                and "quy che dao tao" in searchable
                and "hoc phan" in searchable
            )
        )

    if "credit_load_warning" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        if any(
            noisy in searchable
            for noisy in (
                "web support",
                "thoi khoa bieu",
                "lich hoc",
                "lich thi",
                "tieng anh",
                "toeic",
                "ielts",
                "chung chi",
                "khcn",
                "nghien cuu",
                "thiet bi",
                "phong hoc",
            )
        ):
            return False
        return (
            "canh bao hoc tap" in searchable
            or "canh bao ket qua hoc tap" in searchable
            or "dang trong thoi gian bi canh bao" in searchable
            or "dang ky khoi luong hoc tap" in searchable
            or "khoi luong hoc tap" in searchable
            or "16 tin chi" in searchable
            or "khong qua 16" in searchable
            or (
                doc.get("dieu") == 9
                and "quy che dao tao" in searchable
                and any(term in searchable for term in ("tin chi", "hoc phan", "khoi luong"))
            )
        )

    if "transfer_school" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        if "tuyen sinh" in metadata_text:
            return False
        if any(noisy in searchable for noisy in ("web support", "khcn", "nghien cuu", "thiet bi", "phong hoc")):
            return False
        if "chuyen chuong trinh dao tao" in searchable and "chuyen truong" not in searchable:
            return False
        return (
            "chuyen truong" in searchable
            or (
                doc.get("dieu") == 28
                and "quy che dao tao" in searchable
                and any(term in searchable for term in ("hieu truong", "cung nganh", "noi cu tru", "hoan canh"))
            )
        )

    if "elective_failed_course" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        if "tuyen sinh" in metadata_text:
            return False
        if any(noisy in searchable for noisy in ("web support", "khcn", "nghien cuu", "thiet bi", "phong hoc")):
            return False
        return (
            doc.get("dieu") == 11
            and any(term in searchable for term in ("hoc phan tu chon", "hoc doi", "hoc phan khac tuong duong", "diem f", "f+"))
        ) or (
            "hoc phan tu chon" in searchable
            and any(term in searchable for term in ("hoc doi", "tuong duong", "hoc lai"))
        )

    if "f_grade_comparison" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        if "tuyen sinh" in metadata_text:
            return False
        if any(noisy in searchable for noisy in ("web support", "khcn", "nghien cuu", "thiet bi", "phong hoc")):
            return False
        return (
            doc.get("dieu") in {11, 16}
            and any(term in searchable for term in ("f+", "diem f", "diem chu", "hoc lai", "hoc doi", "hoc phan tu chon"))
        )

    if "credit_definition" in terms:
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        if any(noisy in searchable for noisy in ("web support", "gpa", "tot nghiep", "chung chi", "khcn", "nghien cuu")):
            return False
        return (
            "tin chi" in searchable
            and any(term in searchable for term in ("15 tiet", "30 tiet", "45 60", "45-60", "ly thuyet", "thuc hanh"))
        )

    if "thi lai" in terms or (
        "exam_policy" in terms
        and "thi lai" in normalize_text(question)
    ):
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        return any(
            marker in searchable
            for marker in (
                "thi lai",
                "dang ky thi lai",
                "ky thi lai",
                "du thi lai",
                "lich thi lai",
            )
        )

    if any(term in terms for term in ("hoan thi", "exam_policy")):
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        return any(
            marker in searchable
            for marker in ("pdt", "dao tao", "hoc phan", "quy che dao tao", "diem i", "hoan thi")
        )

    if any(term in terms for term in ("ra truong", "chung chi", "chuan dau ra", "tot_nghiep_condition")):
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            return False
        return any(
            marker in searchable
            for marker in (
                "tot nghiep",
                "chuan dau ra",
                "ngoai ngu",
                "tin hoc",
                "chung chi",
                "ttnnth",
                "ktdbcl",
            )
        )

    return True


def _absence_permission_comparison_answer(question: str, docs: list[dict]) -> tuple[str | None, list[dict]]:
    terms = _academic_policy_terms(question)
    if "absence_permission_comparison" not in terms:
        return None, []

    support_docs = [
        doc for doc in docs or []
        if _is_business_source(doc)
        and any(
            marker in normalize_text(
                " ".join(str(doc.get(field) or "") for field in ("title", "content", "doc_name"))
            )
            for marker in ("nghi co phep", "nghi khong phep", "diem danh", "so tiet vang")
        )
    ]
    evidence_docs = [
        doc for doc in docs or []
        if _matches_academic_policy_source(question, doc)
        and "diem chuyen can" in normalize_text(doc.get("content", ""))
        and "nghi hoc" in normalize_text(doc.get("content", ""))
    ]
    if not evidence_docs:
        return None, []

    source_doc = evidence_docs[0]
    source_label = (
        f'{source_doc.get("title") or "Nguon tai lieu"} - '
        f'{source_doc.get("doc_name") or source_doc.get("relative_path") or "tai lieu"}'
    )
    support_label = None
    if support_docs:
        support_doc = support_docs[0]
        support_label = (
            f'{support_doc.get("title") or "Nguon huong dan"} - '
            f'{support_doc.get("doc_name") or support_doc.get("relative_path") or "tai lieu"}'
        )

    if support_docs:
        intro = (
            "Nguồn hướng dẫn/điểm danh cho thấy hệ thống có thể ghi nhận riêng thông tin "
            "\"nghỉ có phép\" và \"nghỉ không phép\"; còn nguồn quy chế được tìm thấy không nêu "
            "một bảng chế tài tách riêng cho hai loại nghỉ này. Phần có căn cứ rõ trong quy chế là "
        )
    else:
        intro = (
            "Trong các nguồn đã tìm thấy, tài liệu chưa nêu rõ sự khác nhau riêng giữa "
            "\"nghỉ học có phép\" và \"nghỉ học không phép\". Phần có căn cứ trong tài liệu là "
        )
    source_summary = (
        f"(Nguồn: {support_label}; {source_label})"
        if support_label
        else f"(Nguồn: {source_label})"
    )
    answer = (
        intro +
        "việc nghỉ học/vắng học được tính vào điểm chuyên cần theo tỷ lệ số tiết trong chương trình:\n"
        "- Đi học đầy đủ: 10 điểm.\n"
        "- Có nghỉ học, nghỉ dưới 10% số tiết: 8 điểm.\n"
        "- Nghỉ từ 10% đến dưới 20% số tiết: 6 điểm.\n"
        "- Nghỉ từ 20% đến dưới 35% số tiết: 4 điểm.\n"
        "- Nghỉ từ 35% đến dưới 50% số tiết: 2 điểm.\n"
        "- Nghỉ từ 50% trở lên: 0 điểm.\n"
        "Tài liệu cũng nêu sinh viên nghỉ học trên 50% số tiết sẽ bị cấm thi hoặc bị xác định "
        "không tham gia học tập tùy loại học phần, điểm thi/điểm học phần được tính là 0 điểm.\n\n"
        f"{source_summary}"
    )
    return answer, [*support_docs[:1], *evidence_docs[:2]]


def _credit_load_warning_answer(question: str, docs: list[dict]) -> tuple[str | None, list[dict]]:
    terms = _academic_policy_terms(question)
    if "credit_load_warning" not in terms:
        return None, []

    evidence_docs = []
    for doc in docs or []:
        searchable = normalize_text(
            " ".join(
                str(doc.get(field) or "")
                for field in ("doc_name", "title", "relative_path", "phong_ban", "content")
            )
        )
        if (
            _matches_academic_policy_source(question, doc)
            and ("16 tin chi" in searchable or "khong qua 16" in searchable)
            and (
                "canh bao hoc tap" in searchable
                or "canh bao ket qua hoc tap" in searchable
                or "dang trong thoi gian bi canh bao" in searchable
            )
        ):
            evidence_docs.append(doc)

    if not evidence_docs:
        return None, []

    source_doc = evidence_docs[0]
    source_label = (
        f'{source_doc.get("title") or "Nguồn tài liệu"} - '
        f'{source_doc.get("doc_name") or source_doc.get("relative_path") or "tài liệu"}'
    )
    answer = (
        "Sinh viên đang trong thời gian bị cảnh báo kết quả học tập chỉ được đăng ký "
        "khối lượng học tập trong học kỳ mới theo kế hoạch đào tạo **không quá 16 tín chỉ "
        "cho mỗi học kỳ**.\n\n"
        "Lưu ý: quy định này là trường hợp riêng đối với sinh viên điểm trung bình học kỳ "
        "yếu/kém hoặc đang bị cảnh báo kết quả học tập; không áp dụng theo mức tối đa chung "
        "3/2 số tín chỉ trung bình của học kỳ.\n\n"
        f"(Nguồn: {source_label})"
    )
    return answer, evidence_docs[:2]


def _graduation_classification_answer(
    question: str,
    docs: list[dict],
) -> tuple[str | None, list[dict]]:
    normalized_question = normalize_text(question)
    asks_classification = "tot nghiep" in normalized_question and any(
        term in normalized_question
        for term in ("loai gioi", "xep loai", "xep hang", "hang tot nghiep")
    )
    if not asks_classification:
        return None, []

    general_docs = []
    classification_docs = []
    for doc in docs or []:
        content = normalize_text(doc.get("content", ""))
        if (
            "dieu kien xet tot nghiep" in content
            and "diem trung binh tich luy" in content
            and ("chung chi" in content or "chung nhan ngoai ngu" in content)
        ):
            general_docs.append(doc)
        if (
            "loai gioi" in content
            and ("3,20" in content or "3.20" in content)
            and ("3,59" in content or "3.59" in content)
            and "hoc lai" in content
        ):
            classification_docs.append(doc)

    if not classification_docs:
        return None, []

    asks_general = any(
        phrase in normalized_question
        for phrase in (
            "dieu kien tot nghiep la gi va",
            "dieu kien tot nghiep va",
            "dieu kien de tot nghiep va",
            "vua du dieu kien tot nghiep",
        )
    )
    if asks_general and not general_docs:
        return None, []

    parts = []
    selected_docs = []
    if asks_general:
        parts.append(
            "**1. Điều kiện được xét và công nhận tốt nghiệp**\n"
            "Sinh viên phải đáp ứng đầy đủ các điều kiện sau:\n"
            "- Tại thời điểm xét tốt nghiệp, không bị truy cứu trách nhiệm hình sự và "
            "không đang bị kỷ luật ở mức đình chỉ học tập.\n"
            "- Tích lũy đủ số học phần và khối lượng của chương trình đào tạo.\n"
            "- Điểm trung bình tích lũy toàn khóa từ 2,00 trở lên.\n"
            "- Có chứng chỉ hoặc chứng nhận ngoại ngữ, tin học theo quy định của Trường.\n"
            "- Có chứng chỉ Giáo dục quốc phòng - an ninh và hoàn thành học phần "
            "Giáo dục thể chất theo đối tượng áp dụng.\n"
            "- Nếu xin tốt nghiệp sớm hoặc muộn hơn thời gian thiết kế của khóa học, "
            "phải có đơn gửi Phòng Đào tạo."
        )
        selected_docs.extend(general_docs[:1])

    heading = "**2. Điều kiện xếp loại tốt nghiệp giỏi**" if asks_general else "**Điều kiện xếp loại tốt nghiệp giỏi**"
    parts.append(
        f"{heading}\n"
        "- Điểm trung bình tích lũy toàn khóa từ 3,20 đến 3,59.\n"
        "- Hạng giỏi sẽ bị giảm một mức nếu khối lượng học phần phải học lại vượt quá "
        "5% tổng số tín chỉ của chương trình, hoặc sinh viên từng bị kỷ luật từ mức "
        "cảnh cáo trở lên trong thời gian học."
    )
    selected_docs.extend(classification_docs[:1])

    source_labels = [
        f'{doc.get("title") or "Nguồn tài liệu"} - '
        f'{doc.get("doc_name") or doc.get("relative_path") or "tài liệu"}'
        for doc in selected_docs
    ]
    parts.append(f'(Nguồn: {"; ".join(source_labels)})')
    return "\n\n".join(parts), selected_docs


def _exam_defer_answer(
    question: str,
    docs: list[dict],
) -> tuple[str | None, list[dict]]:
    normalized_question = normalize_text(question)
    if "hoan thi" not in normalized_question:
        return None, []

    asks_procedure = any(
        term in normalized_question
        for term in (
            "lam the nao",
            "thuc hien nhu the nao",
            "cach thuc hien",
            "huong dan",
            "thu tuc xin",
            "gui yeu cau",
        )
    )
    if not asks_procedure:
        return None, []

    procedure_docs = []
    policy_docs = []
    for doc in docs or []:
        searchable = normalize_text(
            " ".join(
                str(doc.get(field) or "")
                for field in ("title", "content", "doc_name")
            )
        )
        if (
            re.search(r"\bhoan thi\b", searchable)
            and "mot cua - khao thi" in searchable
            and "gui yeu cau" in searchable
            and "support.uneti.edu.vn/mot-cua/khao-thi/hoan-thi" in searchable
        ):
            procedure_docs.append(doc)
        if (
            "diem i" in searchable
            and "bi om" in searchable
            and "tai nan" in searchable
            and "truong khoa" in searchable
            and "thac si" not in searchable
        ):
            policy_docs.append(doc)

    if not procedure_docs:
        return None, []

    parts = []
    selected_docs = []
    asks_conditions = "dieu kien" in normalized_question
    if asks_conditions and policy_docs:
        parts.append(
            "1. Điều kiện hoãn thi\n"
            "- Sinh viên bị ốm hoặc tai nạn, không thể dự kiểm tra hoặc thi trong "
            "thời gian học hoặc kỳ thi kết thúc học kỳ và được Nhà trường cho phép.\n"
            "- Sinh viên không thể dự kiểm tra bộ phận hoặc thi vì lý do khách quan "
            "và được Trưởng Khoa chấp thuận.\n"
            "- Khi được Nhà trường cho phép vắng kỳ thi kết thúc học phần, sinh viên "
            "được dự kỳ thi phụ hoặc thi vào học kỳ tiếp theo và vẫn được coi là thi lần đầu."
        )
        selected_docs.extend(policy_docs[:1])

    procedure_heading = "2. Thủ tục xin hoãn thi" if asks_conditions and policy_docs else "Thủ tục xin hoãn thi"
    parts.append(
        f"{procedure_heading}\n"
        "1. Đăng nhập https://support.uneti.edu.vn bằng tài khoản cá nhân.\n"
        "2. Chọn Thủ tục hành chính → Một cửa - Khảo thí → Hoãn thi (Gửi yêu cầu), "
        "hoặc truy cập https://support.uneti.edu.vn/mot-cua/khao-thi/hoan-thi.\n"
        "3. Chọn hoặc nhập các dữ liệu được yêu cầu.\n"
        "4. Tại lưới dữ liệu, chọn dòng học phần tương ứng rồi nhấn Gửi yêu cầu.\n"
        "Sinh viên cần làm đơn theo mẫu và đính kèm giấy tờ minh chứng. "
        "Biểu mẫu tham khảo: Giấy tiếp nhận yêu cầu hoãn thi (MC-KT-05), tại "
        "https://uneti.edu.vn/wp-content/uploads/2021/10/MC-KT-05.pdf."
    )
    selected_docs.extend(procedure_docs[:1])

    source_labels = [
        f'{doc.get("title") or "Nguồn tài liệu"} - '
        f'{doc.get("doc_name") or doc.get("relative_path") or "tài liệu"}'
        for doc in selected_docs
    ]
    parts.append(f'(Nguồn: {"; ".join(source_labels)})')
    return "\n\n".join(parts), selected_docs


async def _credit_load_warning_targeted_docs(question: str) -> list[dict]:
    if "credit_load_warning" not in _academic_policy_terms(question):
        return []

    targeted_query = (
        f"{question} cảnh báo kết quả học tập không quá 16 tín chỉ "
        "đăng ký khối lượng học tập Điều 9 quy chế đào tạo đại học chính quy"
    )
    docs = await search_documents(
        targeted_query,
        source_type_filter=INTERNAL_SOURCE_TYPE,
        ambiguity_decision={"action": DIRECT_RETRIEVAL},
    )
    matched_docs = [
        doc for doc in docs or []
        if _credit_load_warning_answer(question, [doc])[0]
    ]
    return matched_docs[:2]


def _procedure_sources_fallback_answer(question: str, docs: list[dict]) -> tuple[str | None, list[dict]]:
    normalized = normalize_text(question)
    if not any(
        term in normalized
        for term in (
            "cach", "huong dan", "truy cap", "xem", "lam sao",
            "lam the nao", "chuc nang", "thu tuc", "dang ky", "hoan thi", "thi lai",
        )
    ):
        return None, []

    business_docs = [doc for doc in docs or [] if _is_business_source(doc)]
    if not business_docs:
        return None, []

    query_terms = set(get_keywords(question))
    priority_markers = (
        "bước", "buoc", "chức năng", "chuc nang", "đường dẫn", "duong dan",
        "https://", "đăng nhập", "dang nhap", "chọn", "chon", "đơn vị", "don vi",
        "hình thức", "hinh thuc", "lưu ý", "luu y", "màn", "man",
    )
    ranked_lines = []
    for doc_index, doc in enumerate(business_docs[:5]):
        content = re.sub(r"\s+", " ", str(doc.get("content") or "")).strip()
        if not content:
            continue
        parts = [
            part.strip(" -")
            for part in re.split(r"(?<=[.!?])\s+|(?<=;)\s+|(?=Bước\s+\d+[:.])|(?=Buoc\s+\d+[:.])", content)
            if len(part.strip(" -")) >= 24
        ]
        if not parts:
            parts = [content[:800]]
        for line_index, part in enumerate(parts):
            part_norm = normalize_text(part)
            overlap = len(query_terms & set(get_keywords(part)))
            marker_score = sum(1 for marker in priority_markers if marker in part_norm.lower() or marker in part.lower())
            if overlap <= 0 and marker_score <= 0:
                continue
            ranked_lines.append((marker_score * 3 + overlap, doc_index, line_index, part[:450]))

    if not ranked_lines:
        return None, []

    selected_lines = []
    seen = set()
    for _, _, _, line in sorted(ranked_lines, key=lambda item: (-item[0], item[1], item[2])):
        compact = re.sub(r"\s+", " ", line).strip()
        key = normalize_text(compact[:120])
        if key in seen:
            continue
        seen.add(key)
        selected_lines.append(compact)
        if len(selected_lines) >= 6:
            break

    if not selected_lines:
        return None, []

    used_docs = business_docs[: min(3, len(business_docs))]
    source_labels = []
    for doc in used_docs[:2]:
        label = f'{doc.get("title") or "Nguồn"} - {doc.get("doc_name") or "Tài liệu"}'
        if label not in source_labels:
            source_labels.append(label)
    answer = "\n".join(f"- {line}" for line in selected_lines)
    answer += f"\n\n(Nguồn: {'; '.join(source_labels)})"
    return answer, used_docs


def _internal_metadata_matched(docs: list[dict]) -> bool:
    return any(doc.get("metadata_matched") for doc in docs or [])


def _should_prefer_business_over_internal(
    question: str,
    business_state: dict,
    business_docs: list[dict],
    internal_docs: list[dict],
) -> tuple[bool, dict]:
    retrieval_debug = business_state.get("retrieval_debug") or {}
    method = retrieval_debug.get("retrieval_method")
    gate_score = retrieval_debug.get("mapping_gate_score")
    gate_score_value = float(gate_score or 0)
    document_terms = _document_intent_terms(question)
    academic_policy_terms = _academic_policy_terms(question)
    internal_metadata = _internal_metadata_matched(internal_docs)
    information_need = retrieval_debug.get("information_need")
    has_web_support_source = _has_web_support_source(business_docs)
    mapping_selected = bool(retrieval_debug.get("mapping_selected"))
    selected = _should_prefer_business_generation(business_state, business_docs)
    reason = "business_mapping_high_confidence" if selected else "not_high_confidence_business"

    if (
        _is_exam_retake_procedure_question(question)
        and _has_exam_retake_business_source(business_docs)
    ):
        selected = True
        reason = "exam_retake_procedure_web_support_source"
    elif internal_metadata:
        selected = False
        reason = "internal_metadata_matched"
    elif academic_policy_terms and internal_docs:
        selected = False
        reason = "academic_policy_terms_prefer_internal_or_merge"
    elif information_need == "policy_document":
        selected = False
        reason = "policy_document"
    elif document_terms and (method == "generic_hybrid" or gate_score_value < 70):
        selected = False
        reason = "document_intent_terms_prefer_internal_or_merge"
    elif (
        not selected
        and information_need == "procedure_ui"
        and business_docs
        and (has_web_support_source or mapping_selected)
        and not document_terms
        and not (
            method in {"generic_hybrid", "generic_keyword"}
            and not has_web_support_source
            and not mapping_selected
        )
    ):
        selected = True
        reason = "procedure_ui_web_support_business_source"

    return selected, {
        "selected": selected,
        "reason": reason,
        "document_intent_terms": document_terms,
        "academic_policy_terms": academic_policy_terms,
        "business_retrieval_method": method,
        "mapping_gate_score": gate_score,
        "business_confidence": gate_score,
        "has_web_support_source": has_web_support_source,
        "mapping_selected": mapping_selected,
        "internal_confidence": None,
        "internal_confidence_source": None,
        "internal_metadata_matched": internal_metadata,
        "information_need": information_need,
        "audience_hint": retrieval_debug.get("audience_hint"),
        "audience_source": retrieval_debug.get("audience_source"),
    }


def _allow_aggregate_direct_business_answer(
    question: str,
    business_state: dict,
    internal_docs: list[dict],
) -> tuple[bool, str]:
    retrieval_debug = business_state.get("retrieval_debug") or {}
    if retrieval_debug.get("information_need") == "policy_document":
        return False, "policy_document"
    if _internal_metadata_matched(internal_docs):
        return False, "internal_metadata_matched"
    if _academic_policy_terms(question) and internal_docs:
        return False, "academic_policy_terms"
    return True, "allowed"


def _looks_like_document_number_query(question: str) -> bool:
    normalized = normalize_text(question)
    searchable = re.sub(r"[_\-.]+", " ", normalized)
    return bool(re.search(
        r"\b(?:so|van\s*ban|quyet\s*dinh|quy\s*dinh|quy\s*che|thong\s*bao|qd|qc|tb|vb)\s*\d{1,6}\b",
        searchable,
    ))


def _confidence_from_source(doc: dict) -> tuple[float | None, str | None]:
    if doc.get("metadata_matched"):
        confidence = 1.0
    else:
        confidence_values = []

        vector_score = doc.get("vector_score")
        if vector_score is None and doc.get("distance") is not None:
            vector_score = 1 - float(doc["distance"])

        if vector_score is not None:
            confidence_values.append(max(0.0, min(float(vector_score), 1.0)))

        keyword_score = doc.get("keyword_score")
        if keyword_score is not None:
            keyword_confidence = float(keyword_score) / max(MIN_SEARCH_SCORE * 4, 1)
            confidence_values.append(max(0.0, min(keyword_confidence, 1.0)))

        if not confidence_values:
            return None, None

        confidence = max(confidence_values)

    if confidence >= 0.75:
        label = "Cao"
    elif confidence >= 0.55:
        label = "Trung bình"
    else:
        label = "Thấp"

    return round(confidence, 4), label


def _clean_preview_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"---\s*Trang\s+\d+\s*---", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_preview_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\b([a-zA-Z])\s+([a-zA-Z])\b", r"\1\2", text)
    parts = re.split(r"(?<=[.!?])\s+|(?<=;)\s+|(?=\b[a-z]\)\s+)", text)
    return [part.strip(" -") for part in parts if len(part.strip(" -")) >= 24]


def _score_preview_sentence(sentence: str, question_keywords: list[str]) -> int:
    normalized_sentence = normalize_text(sentence)
    return sum(1 for keyword in set(question_keywords) if keyword in normalized_sentence)


def _shorten_preview_sentence(sentence: str, limit: int = 260) -> str:
    sentence = sentence.strip(" -")
    if len(sentence) <= limit:
        return sentence

    cut_at = sentence.rfind(". ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = sentence.rfind("; ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = sentence.rfind(", ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = sentence.rfind(" ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = limit

    return sentence[:cut_at].rstrip(" ,;:-") + "."


def _fallback_preview(text: str) -> str:
    if len(text) <= SOURCE_PREVIEW_CHARS:
        return text

    cut_at = text.rfind(". ", 0, SOURCE_PREVIEW_CHARS)
    if cut_at < SOURCE_PREVIEW_CHARS * 0.55:
        cut_at = text.rfind("; ", 0, SOURCE_PREVIEW_CHARS)
    if cut_at < SOURCE_PREVIEW_CHARS * 0.55:
        cut_at = text.rfind(" ", 0, SOURCE_PREVIEW_CHARS)
    if cut_at < SOURCE_PREVIEW_CHARS * 0.55:
        cut_at = SOURCE_PREVIEW_CHARS

    return text[:cut_at].rstrip(" ,;:-") + "."


def _source_preview(content: str, title: str | None = None, question: str | None = None) -> str:
    title_text = _clean_preview_text(title or "")
    content_text = _clean_preview_text(content)
    question_keywords = get_keywords(question or "")
    sentences = _split_preview_sentences(content_text)

    if sentences:
        ranked_sentences = sorted(
            enumerate(sentences),
            key=lambda item: (_score_preview_sentence(item[1], question_keywords), -item[0]),
            reverse=True,
        )
        selected_indexes = [
            index
            for index, sentence in ranked_sentences
            if _score_preview_sentence(sentence, question_keywords) > 0
        ][:SOURCE_PREVIEW_SENTENCES]

        if not selected_indexes:
            selected_indexes = list(range(min(SOURCE_PREVIEW_SENTENCES, len(sentences))))

        selected_indexes = sorted(selected_indexes)
        summary_items = [
            _shorten_preview_sentence(sentences[index])
            for index in selected_indexes
        ]
        summary = "\n".join(f"- {item}" for item in summary_items)
    else:
        summary = content_text

    if "\n- " not in summary:
        summary = _fallback_preview(summary)

    if title_text and title_text.lower() not in summary.lower():
        preview = f"{title_text}\n{summary}"
    else:
        preview = summary

    return f"Tóm tắt nguồn:\n{preview}"


def _build_sources(docs, question: str | None = None):
    sources = []

    for doc in docs:
        confidence, confidence_label = _confidence_from_source(doc)
        scores = {}

        for field_name in ("score", "vector_score", "keyword_score", "rerank_score", "distance"):
            score = doc.get(field_name)
            if score is None:
                scores[field_name] = None
                continue

            try:
                scores[field_name] = float(score)
            except (TypeError, ValueError):
                scores[field_name] = None

        sources.append({
            "title": doc.get("title"),
            "doc_name": doc.get("doc_name"),
            "url": doc.get("url"),
            "attachment_url": doc.get("attachment_url"),
            "source_type": doc.get("source_type"),
            "corpus": doc.get("corpus"),
            "index_version": doc.get("index_version"),
            "document_type": doc.get("document_type"),
            "department": doc.get("department"),
            "relative_path": doc.get("relative_path"),
            "phong_ban": doc.get("phong_ban"),
            "source_root": doc.get("source_root"),
            "so_van_ban": doc.get("so_van_ban"),
            "ngay_ban_hanh": doc.get("ngay_ban_hanh"),
            "ngay_hieu_luc": doc.get("ngay_hieu_luc"),
            "ten_van_ban": doc.get("ten_van_ban"),
            "don_vi_ban_hanh": doc.get("don_vi_ban_hanh"),
            "loai_van_ban": doc.get("loai_van_ban"),
            "chuong": doc.get("chuong"),
            "muc": doc.get("muc"),
            "dieu": doc.get("dieu"),
            "section_path": doc.get("section_path"),
            "heading": doc.get("heading"),
            "section_type": doc.get("section_type"),
            "page": doc.get("page"),
            "chunk_index": doc.get("chunk_index"),
            "chunk_hash": doc.get("chunk_hash"),
            "document_id": doc.get("document_id"),
            "file_extension": doc.get("file_extension"),
            "file_id": doc.get("file_id"),
            "faq_location": doc.get("faq_location"),
            "audience": doc.get("audience"),
            "mapping_relative_path": doc.get("mapping_relative_path"),
            "score": scores["score"],
            "vector_score": scores["vector_score"],
            "keyword_score": scores["keyword_score"],
            "rerank_score": scores["rerank_score"],
            "distance": scores["distance"],
            "confidence": confidence,
            "confidence_percent": round(confidence * 100) if confidence is not None else None,
            "confidence_label": confidence_label,
            "preview": _source_preview(doc.get("content", ""), doc.get("title"), question),
        })

    return sources


def _has_confident_evidence(question: str, docs) -> tuple[bool, str]:
    query_keyword_count = len(get_keywords(question))
    keyword_threshold = (
        SHORT_QUERY_MIN_SEARCH_SCORE
        if query_keyword_count < SHORT_QUERY_KEYWORD_COUNT
        else MIN_SEARCH_SCORE
    )
    vector_threshold = (
        SHORT_QUERY_MIN_VECTOR_CONFIDENCE
        if query_keyword_count < SHORT_QUERY_KEYWORD_COUNT
        else MIN_VECTOR_CONFIDENCE
    )

    for doc in docs:
        if doc.get("hyde_only"):
            rerank_score = doc.get("rerank_score")
            if rerank_score is None or float(rerank_score) < HYDE_MIN_RERANK_SCORE:
                continue

        if doc.get("metadata_matched"):
            return True, "metadata_matched"

        vector_score = doc.get("vector_score")
        if vector_score is None and doc.get("distance") is not None:
            vector_score = 1 - float(doc["distance"])

        if vector_score is not None and float(vector_score) >= vector_threshold:
            return True, "vector_score_passed"

        keyword_score = doc.get("keyword_score")
        if keyword_score is not None and float(keyword_score) >= keyword_threshold:
            return True, "keyword_score_passed"

    return False, "no_confident_source"


def _finalize(trace: RagTrace, response: dict) -> dict:
    if "answer" in response:
        response["answer"] = _clean_answer_text(response["answer"])

    _citation_check(
        response.get("answer"),
        response.get("source"),
        len(response.get("sources") or []),
    )
    response["trace_id"] = trace.trace_id
    conversation = get_conversation_context()
    if conversation.thread_id:
        response.setdefault("thread_id", conversation.thread_id)
        response.setdefault("user_message_id", conversation.user_message_id)
        response.setdefault("assistant_message_id", conversation.assistant_message_id)
    response["gemini_call_count"] = get_gemini_call_count()
    response.setdefault("ambiguity_llm_called", False)
    response.setdefault("mapping_judge_llm_called", False)
    trace.set_response(response)
    trace.save()
    return response


def _pipeline_state(
    trace: RagTrace,
    question: str,
    reason: str,
    prompt_type: str = "document",
    ambiguity_decision: dict | None = None,
) -> dict:
    conversation = get_conversation_context()
    query_context = analyze_query_context(question, conversation.history)
    return {
        "question": question,
        "original_question": conversation.original_question or question,
        "standalone_question": conversation.standalone_question or question,
        "conversation_history": conversation.history,
        "query_context": query_context,
        "reason": reason,
        "prompt_type": prompt_type,
        "ambiguity_decision": ambiguity_decision,
        "trace_callback": trace.add_step,
    }


def _new_trace(question: str) -> RagTrace:
    trace = RagTrace(question)
    conversation = get_conversation_context()
    if conversation.thread_id:
        trace.payload["thread_id"] = conversation.thread_id
        trace.payload["original_question"] = conversation.original_question or question
        trace.payload["standalone_question"] = conversation.standalone_question or question
        trace.payload["history_message_count"] = conversation.history_message_count
        trace.payload["history_chars"] = conversation.history_chars
        trace.payload["rewrite_debug"] = conversation.rewrite_debug
        trace.payload["user_message_id"] = conversation.user_message_id
        trace.payload["assistant_message_id"] = conversation.assistant_message_id
        trace.add_step("contextual_question_rewriting", {
            **conversation.rewrite_debug,
            "original_question": conversation.original_question or question,
            "standalone_question": conversation.standalone_question or question,
        })
    return trace


@traceable(name="Query Router", run_type="chain")
def _route_retrieval_question(question: str) -> dict:
    return analyze_ambiguity(question).to_dict()


def _analyze_retrieval_question(trace: RagTrace, question: str) -> dict:
    decision = _route_retrieval_question(question)
    trace.add_step("ambiguity_detection", {
        "ambiguity_action": decision.get("action"),
        "detected_topic": decision.get("topic"),
        "ambiguity_confidence": decision.get("confidence"),
        "ambiguity_reason": decision.get("reason"),
        "clarification_question": decision.get("clarifying_question"),
        "analyzer": decision.get("analyzer"),
        "llm_called": decision.get("analyzer") == "llm",
        "ambiguity_llm_called": decision.get("analyzer") == "llm",
        "cache_hit": decision.get("cache_hit"),
    })
    return decision


def _use_retrieval_for_clarification(decision: dict | None) -> dict | None:
    if not decision or decision.get("action") != CLARIFICATION_NEEDED:
        return decision
    return {
        **decision,
        "action": "probe_retrieval",
        "original_action": CLARIFICATION_NEEDED,
        "clarification_bypassed": True,
        "reason": decision.get("reason") or "clarification_bypassed_for_retrieval",
    }


def _clarification_response(
    trace: RagTrace,
    question: str,
    decision: dict,
    retrieval_called: bool = False,
):
    trace.add_step("route_decision", {
        "llm_called": False,
        "retrieval_called": retrieval_called,
        "reason": decision.get("reason") or "clarification_needed",
    })
    return _finalize(trace, {
        "question": question,
        "answer": decision.get("clarifying_question") or "Bạn cần hỏi rõ ràng hơn",
        "source": None,
        "sources": [],
        "intent": CLARIFICATION_NEEDED,
    })


def _retrieval_clarification_decision(state: dict) -> dict | None:
    return None
    retrieval_debug = state.get("retrieval_debug") or {}
    fallback_reason = retrieval_debug.get("fallback_reason")
    if fallback_reason not in {
        "hyde_requested_clarification",
        "probe_insufficient_evidence",
        "retrieval_plan_requested_clarification",
    }:
        return None
    retrieval_plan = retrieval_debug.get("retrieval_plan") or {}
    ambiguity = retrieval_debug.get("ambiguity") or {}
    return {
        **ambiguity,
        "action": CLARIFICATION_NEEDED,
        "reason": fallback_reason,
        "clarifying_question": (
            ambiguity.get("clarifying_question")
            or retrieval_plan.get("clarification_question")
            or "Bạn cần hỏi rõ ràng hơn"
        ),
    }


def _deduplicate_docs(*doc_groups: list[dict]) -> list[dict]:
    docs = []
    seen = {}

    for group in doc_groups:
        for doc in group:
            key = (
                doc.get("source_type"),
                doc.get("relative_path") or doc.get("doc_name"),
                doc.get("chunk_index"),
                doc.get("title"),
            )
            if key in seen:
                existing = seen[key]
                merged_aspects = list(existing.get("coverage_aspects") or [])
                for aspect in doc.get("coverage_aspects") or []:
                    if aspect not in merged_aspects:
                        merged_aspects.append(aspect)
                if merged_aspects:
                    existing["coverage_aspects"] = merged_aspects
                    existing.setdefault("evidence_aspect", merged_aspects[0])
                continue
            item = dict(doc)
            seen[key] = item
            docs.append(item)

    return docs


def _is_business_source(doc: dict) -> bool:
    source_type = doc.get("source_type")
    return (
        source_type in {"business_document", "business_faq_mapping"}
        or normalize_text(doc.get("source_root", "")) == "nghiep_vu"
        or "web support" in normalize_text(doc.get("doc_name", ""))
    )


def _tag_multihop_docs(docs: list[dict], aspect: str, query: str) -> list[dict]:
    tagged = []
    for doc in docs[:MULTI_HOP_DOCS_PER_ROUTE]:
        item = dict(doc)
        aspects = list(item.get("coverage_aspects") or [])
        if aspect not in aspects:
            aspects.append(aspect)
        item["coverage_aspects"] = aspects
        item["evidence_aspect"] = aspect
        item["sub_question"] = query
        tagged.append(item)
    return tagged


async def _retrieve_multihop_evidence(
    trace: RagTrace,
    base_state: dict,
    question: str,
) -> tuple[list[dict], list[dict], dict]:
    subquestions = _decompose_query(question)
    debug = {
        "enabled": bool(subquestions),
        "subquestions": subquestions,
        "business_count": 0,
        "internal_count": 0,
        "aspect_counts": {},
    }
    if not subquestions:
        trace.add_step("query_decomposition", {
            "enabled": False,
            "reason": "simple_query_or_no_rule_match",
            "subquestion_count": 0,
            "subquestions": [],
        })
        return [], [], debug

    trace.add_step("query_decomposition", {
        "enabled": True,
        "method": "rule_based",
        "subquestion_count": len(subquestions),
        "subquestions": subquestions,
    })

    async def retrieve_one(subq: dict):
        sub_state = {
            **base_state,
            "question": subq["query"],
            "original_question": base_state.get("original_question") or question,
            "standalone_question": question,
            "reason": f'multi_hop:{subq["aspect"]}',
            "ambiguity_decision": {
                "action": DIRECT_RETRIEVAL,
                "topic": None,
                "confidence": 1.0,
                "reason": "multi_hop_subquestion_no_llm",
                "clarifying_question": None,
            },
        }
        sub_context = dict(base_state.get("query_context") or {})
        if subq.get("need"):
            sub_context["information_need"] = subq["need"]
        sub_context["skip_retrieval_plan_llm"] = True
        sub_state["query_context"] = sub_context

        tasks = []
        labels = []
        if "business" in subq.get("routes", []):
            tasks.append(retrieve_business(dict(sub_state)))
            labels.append("business")
        if "internal" in subq.get("routes", []):
            tasks.append(retrieve_internal({**sub_state, "source_type_filter": INTERNAL_SOURCE_TYPE}))
            labels.append("internal")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return subq, list(zip(labels, results))

    business_docs: list[dict] = []
    internal_docs: list[dict] = []
    route_results = await asyncio.gather(
        *(retrieve_one(subq) for subq in subquestions),
        return_exceptions=True,
    )
    trace_rows = []
    for result in route_results:
        if isinstance(result, Exception):
            trace_rows.append({"error": str(result)})
            continue
        subq, outputs = result
        row = {
            "aspect": subq["aspect"],
            "query": subq["query"],
            "routes": {},
        }
        aspect_count = 0
        for label, state in outputs:
            if isinstance(state, Exception):
                row["routes"][label] = {"error": str(state), "count": 0}
                continue
            docs = state.get("docs") or []
            tagged = _tag_multihop_docs(docs, subq["aspect"], subq["query"])
            row["routes"][label] = {
                "count": len(docs),
                "selected_count": len(tagged),
                "top_sources": [
                    {
                        "doc_name": doc.get("doc_name"),
                        "title": doc.get("title"),
                        "chunk_index": doc.get("chunk_index"),
                        "source_type": doc.get("source_type"),
                        "score": doc.get("score"),
                        "keyword_score": doc.get("keyword_score"),
                        "vector_score": doc.get("vector_score"),
                    }
                    for doc in tagged
                ],
            }
            aspect_count += len(tagged)
            if label == "business":
                business_docs.extend(tagged)
            else:
                internal_docs.extend(tagged)
        debug["aspect_counts"][subq["aspect"]] = aspect_count
        trace_rows.append(row)

    business_docs = _deduplicate_docs(business_docs)
    internal_docs = _deduplicate_docs(internal_docs)
    debug["business_count"] = len(business_docs)
    debug["internal_count"] = len(internal_docs)
    debug["results"] = trace_rows
    trace.add_step("multi_hop_retrieval", debug)
    return business_docs, internal_docs, debug


async def _search_website_and_finalize(trace: RagTrace, question: str, intent: str, reason: str):
    try:
        state = await retrieve_website(
            _pipeline_state(trace, question, reason, prompt_type="website")
        )
    except Exception as exc:
        trace.add_step("website_search", {
            "status": "index_error",
            "error": str(exc),
        }, {
            "question": question,
            "reason": reason,
        })
        return _finalize(trace, {
            "question": question,
            "answer": NO_WEBSITE_EVIDENCE_ANSWER,
            "source": None,
            "sources": [],
            "intent": intent,
        })

    website_docs = state.get("docs") or []

    has_evidence = bool(website_docs)
    evidence_reason = "website_indexed_source" if has_evidence else "no_website_chunks_found"
    trace.add_step("evidence_check_after_website_index", {
        "has_confident_evidence": has_evidence,
        "reason": evidence_reason,
        "query_keyword_count": len(get_keywords(question)),
        "website_source_count": len(website_docs),
        "llm_called": bool(website_docs and has_evidence),
    })

    if not website_docs or not has_evidence:
        return _finalize(trace, {
            "question": question,
            "answer": NO_WEBSITE_EVIDENCE_ANSWER,
            "source": None,
            "sources": _build_sources(website_docs, question),
            "intent": intent,
        })

    state = await generate_answer({**state, "docs": website_docs})
    answer = state["answer"]

    best_doc = website_docs[0]
    source = best_doc.get("attachment_url") or best_doc.get("url")

    return _finalize(trace, {
        "question": question,
        "answer": answer,
        "source": source,
        "sources": _build_sources(website_docs, question),
        "intent": intent,
    })


async def _answer_with_internal_documents(
    trace: RagTrace,
    question: str,
    intent: str,
    reason: str,
    ambiguity_decision: dict | None = None,
):
    state = await retrieve_internal(
        _pipeline_state(
            trace,
            question,
            reason,
            ambiguity_decision=ambiguity_decision,
        ) | {
            "source_type_filter": INTERNAL_SOURCE_TYPE,
        }
    )
    retrieval_clarification = _retrieval_clarification_decision(state)
    if retrieval_clarification:
        return _clarification_response(
            trace,
            question,
            retrieval_clarification,
            retrieval_called=True,
        )
    docs = state.get("docs") or []

    has_evidence, evidence_reason = _has_confident_evidence(question, docs)
    trace.add_step("evidence_check", {
        "evidence_decision": "pass" if has_evidence else "reject",
        "has_confident_evidence": has_evidence,
        "reason": evidence_reason,
        "query_keyword_count": len(get_keywords(question)),
        "llm_called": bool(docs and has_evidence),
    })

    if not docs or not has_evidence:
        return _finalize(trace, {
            "question": question,
            "answer": NO_EVIDENCE_ANSWER,
            "source": None,
            "sources": _build_sources(docs, question),
            "intent": intent,
        })

    state = await generate_answer({
        **state,
        "docs": docs,
        "max_context_chunks": min(len(docs), AGGREGATE_MAX_CONTEXT_CHUNKS),
    })
    answer = state["answer"]

    best_doc = docs[0]
    source = f'{best_doc.get("title")} - {best_doc.get("doc_name")}'

    return _finalize(trace, {
        "question": question,
        "answer": answer,
        "source": source,
        "sources": _build_sources(docs, question),
        "intent": intent,
    })


async def _answer_with_local_documents(
    trace: RagTrace,
    question: str,
    intent: str,
    reason: str,
    ambiguity_decision: dict | None = None,
):
    base_state = _pipeline_state(
        trace,
        question,
        reason,
        ambiguity_decision=ambiguity_decision,
    )
    decomposition = decompose_multi_aspect_query(question)
    trace.add_step("multi_aspect_decomposition", decomposition)
    if decomposition.get("needs_clarification"):
        return _clarification_response(
            trace,
            question,
            {
                "action": CLARIFICATION_NEEDED,
                "reason": decomposition.get("clarification_reason"),
                "clarifying_question": (
                    "Bạn muốn hỏi về loại thủ tục hoặc yêu cầu cụ thể nào?"
                ),
            },
            retrieval_called=False,
        )

    state = await retrieve_local_documents(base_state)
    retrieval_clarification = _retrieval_clarification_decision(state)
    if retrieval_clarification:
        return _clarification_response(
            trace,
            question,
            retrieval_clarification,
            retrieval_called=True,
        )
    docs = state.get("docs") or []
    required_aspects = decomposition.get("aspects") or []
    aspect_results = []
    if decomposition.get("is_multi_aspect"):
        async def retrieve_aspect(aspect: dict) -> dict:
            async def retrieve_query(retrieval_query: str) -> list[dict]:
                aspect_state = {
                    **base_state,
                    "question": retrieval_query,
                    "original_question": question,
                    "standalone_question": question,
                    "reason": f'multi_aspect:{aspect["aspect_id"]}',
                    "ambiguity_decision": {
                        "action": DIRECT_RETRIEVAL,
                        "topic": None,
                        "confidence": 1.0,
                        "reason": "multi_aspect_subquery_no_llm",
                        "clarifying_question": None,
                    },
                }
                query_context = dict(base_state.get("query_context") or {})
                query_context["skip_retrieval_plan_llm"] = True
                aspect_state["query_context"] = query_context
                result_state = await retrieve_local_documents(aspect_state)
                return result_state.get("docs") or []

            query_results = await asyncio.gather(
                *(
                    retrieve_query(retrieval_query)
                    for retrieval_query in aspect.get("retrieval_queries")
                    or [aspect["retrieval_query"]]
                )
            )
            aspect_docs = []
            seen_docs = set()
            max_rank = max((len(result) for result in query_results), default=0)
            for rank in range(max_rank):
                for result in query_results:
                    if rank >= len(result):
                        continue
                    doc = result[rank]
                    key = (
                        doc.get("relative_path") or doc.get("doc_name"),
                        doc.get("chunk_index"),
                        doc.get("title"),
                    )
                    if key in seen_docs:
                        continue
                    seen_docs.add(key)
                    aspect_docs.append(doc)
            confident, confidence_reason = _has_confident_evidence(
                aspect["retrieval_query"],
                aspect_docs,
            )
            semantic_docs, semantic_reason = filter_semantic_aspect_docs(
                aspect.get("semantic_query") or aspect["retrieval_query"],
                aspect_docs if confident else [],
            )
            has_aspect_evidence = bool(semantic_docs)
            evidence_reason = (
                semantic_reason if confident else confidence_reason
            )
            return {
                **aspect,
                "docs": semantic_docs,
                "retrieved_count": len(aspect_docs),
                "has_evidence": has_aspect_evidence,
                "evidence_reason": evidence_reason,
            }

        aspect_results = await asyncio.gather(
            *(retrieve_aspect(aspect) for aspect in required_aspects)
        )
        docs, coverage_debug = merge_multi_aspect_results(
            docs,
            aspect_results,
            limit=AGGREGATE_MAX_CONTEXT_CHUNKS,
        )
        trace.add_step("multi_aspect_retrieval", {
            "method": decomposition.get("method"),
            "aspect_results": [
                {
                    "aspect_id": result["aspect_id"],
                    "question": result["question"],
                    "retrieval_query": result["retrieval_query"],
                    "retrieval_queries": result.get("retrieval_queries"),
                    "context_inherited": result.get("context_inherited", False),
                    "retrieved_count": result["retrieved_count"],
                    "selected_candidate_count": len(result["docs"]),
                    "has_evidence": result["has_evidence"],
                    "evidence_reason": result["evidence_reason"],
                }
                for result in aspect_results
            ],
            "coverage": coverage_debug,
            "final_document_count": len(docs),
        })
    has_evidence, evidence_reason = _has_confident_evidence(question, docs)
    trace.add_step("local_documents_evidence_check", {
        "evidence_decision": "pass" if has_evidence else "reject",
        "has_confident_evidence": has_evidence,
        "reason": evidence_reason,
        "query_keyword_count": len(get_keywords(question)),
        "llm_called": bool(docs and has_evidence),
    })

    if not docs or not has_evidence:
        return _finalize(trace, {
            "question": question,
            "answer": NO_EVIDENCE_ANSWER,
            "source": None,
            "sources": _build_sources(docs, question),
            "intent": intent,
        })

    if decomposition.get("is_multi_aspect"):
        generation_docs, _ = merge_multi_aspect_results(
            [],
            aspect_results,
            limit=AGGREGATE_MAX_CONTEXT_CHUNKS,
        )
        generation_aspects = []
        for result in aspect_results:
            aspect_docs = (result.get("docs") or [])[:2]
            generation_aspects.append({
                "aspect_id": result["aspect_id"],
                "question": result["question"],
                "has_evidence": bool(aspect_docs),
                "sources": [
                    {
                        "title": doc.get("title"),
                        "doc_name": doc.get("doc_name"),
                        "chunk_index": doc.get("chunk_index"),
                    }
                    for doc in aspect_docs
                ],
            })

        generation_state = {
            **state,
            "docs": generation_docs,
            "max_context_chunks": min(
                len(generation_docs),
                AGGREGATE_MAX_CONTEXT_CHUNKS,
            ),
            "required_aspects": generation_aspects,
        }
        generated_state = await generate_answer(generation_state)
        raw_answer = generated_state["answer"]
        validation = validate_multi_aspect_answer(
            raw_answer,
            generation_aspects,
        )
        retry_used = False
        if not validation["valid"]:
            retry_used = True
            failed_blocks = ", ".join(
                f'{item["marker"]}: {item["reason"]}'
                for item in validation["issues"]
            )
            retry_state = await generate_answer({
                **generation_state,
                "generation_guidance": (
                    "Lan tra loi truoc khong dat hop dong tai cac khoi: "
                    f"{failed_blocks}. Kiem tra lai ban do nguon va noi dung tung NGUON."
                ),
            })
            raw_answer = retry_state["answer"]
            validation = validate_multi_aspect_answer(
                raw_answer,
                generation_aspects,
            )

        answer = clean_multi_aspect_answer(raw_answer)
        trace.add_step("multi_aspect_generation", {
            "strategy": "single_call_grouped_evidence_with_validation",
            "initial_call_count": 1,
            "retry_used": retry_used,
            "generation_call_count": 1 + int(retry_used),
            "generation_source_count": len(generation_docs),
            "validation": validation,
            "answer_chars": len(answer),
        })
        best_doc = docs[0]
        return _finalize(trace, {
            "question": question,
            "answer": answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(docs, question),
            "intent": intent,
        })

    graduation_answer, graduation_docs = _graduation_classification_answer(question, docs)
    if graduation_answer:
        trace.add_step("policy_deterministic_answer", {
            "reason": "graduation_requirements_and_classification",
            "source_count": len(graduation_docs),
            "sources": [
                {
                    "doc_name": doc.get("doc_name"),
                    "title": doc.get("title"),
                    "chunk_index": doc.get("chunk_index"),
                    "source_type": doc.get("source_type"),
                }
                for doc in graduation_docs
            ],
        })
        best_doc = graduation_docs[0]
        return _finalize(trace, {
            "question": question,
            "answer": graduation_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(graduation_docs, question),
            "intent": intent,
        })

    exam_defer_answer, exam_defer_docs = _exam_defer_answer(question, docs)
    if exam_defer_answer:
        trace.add_step("policy_deterministic_answer", {
            "reason": "exam_defer_conditions_and_procedure",
            "source_count": len(exam_defer_docs),
            "sources": [
                {
                    "doc_name": doc.get("doc_name"),
                    "title": doc.get("title"),
                    "chunk_index": doc.get("chunk_index"),
                    "source_type": doc.get("source_type"),
                }
                for doc in exam_defer_docs
            ],
        })
        best_doc = exam_defer_docs[0]
        return _finalize(trace, {
            "question": question,
            "answer": exam_defer_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(exam_defer_docs, question),
            "intent": intent,
        })

    state = await generate_answer({
        **state,
        "docs": docs,
        "max_context_chunks": min(len(docs), AGGREGATE_MAX_CONTEXT_CHUNKS),
        "required_aspects": required_aspects,
    })
    best_doc = docs[0]
    return _finalize(trace, {
        "question": question,
        "answer": state["answer"],
        "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
        "sources": _build_sources(docs, question),
        "intent": intent,
    })


async def _answer_with_business_documents(trace: RagTrace, question: str, intent: str, reason: str):
    state = await retrieve_business(_pipeline_state(trace, question, reason))
    retrieval_clarification = _retrieval_clarification_decision(state)
    if retrieval_clarification:
        return _clarification_response(
            trace,
            question,
            retrieval_clarification,
            retrieval_called=True,
        )
    business_docs = state.get("docs") or []

    has_business_evidence, business_evidence_reason = _has_confident_evidence(question, business_docs)
    trace.add_step("business_evidence_check", {
        "evidence_decision": (
            "pass" if has_business_evidence else "reject"
        ),
        "has_confident_evidence": has_business_evidence,
        "reason": business_evidence_reason,
        "query_keyword_count": len(get_keywords(question)),
        "llm_called": bool(business_docs and has_business_evidence),
    })

    if not business_docs or not has_business_evidence:
        return _finalize(trace, {
            "question": question,
            "answer": NO_EVIDENCE_ANSWER,
            "source": None,
            "sources": _build_sources(business_docs, question),
            "intent": intent,
        })

    direct_answer = _business_direct_answer(question, business_docs)
    if direct_answer:
        best_doc = business_docs[0]
        trace.add_step("business_direct_answer", {
            "used": True,
            "doc_name": best_doc.get("doc_name"),
            "title": best_doc.get("title"),
        })
        return _finalize(trace, {
            "question": question,
            "answer": direct_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(business_docs, question),
            "intent": intent,
        })

    state = await generate_answer({**state, "docs": business_docs})
    answer = state["answer"]

    best_doc = business_docs[0]
    source = f'{best_doc.get("title")} - {best_doc.get("doc_name")}'
    if _is_cbgv_admin_process_steps_question(question) and not _answer_has_admin_process_steps(answer):
        answer = _business_direct_answer(question, business_docs) or answer
        trace.add_step("business_direct_answer_override", {
            "used": True,
            "reason": "generated_answer_missing_admin_process_steps",
            "doc_name": best_doc.get("doc_name"),
            "title": best_doc.get("title"),
        })

    return _finalize(trace, {
        "question": question,
        "answer": answer,
        "source": source,
        "sources": _build_sources(business_docs, question),
        "intent": intent,
    })


async def _answer_with_aggregate_documents(
    trace: RagTrace,
    question: str,
    intent: str,
    reason: str,
    ambiguity_decision: dict | None = None,
):
    base_state = _pipeline_state(
        trace,
        question,
        reason,
        ambiguity_decision=ambiguity_decision,
    )
    business_state, internal_state = await asyncio.gather(
        retrieve_business(dict(base_state)),
        retrieve_internal({**base_state, "source_type_filter": INTERNAL_SOURCE_TYPE}),
    )
    business_docs = business_state.get("docs") or []
    internal_docs = internal_state.get("docs") or []
    multihop_business_docs, multihop_internal_docs, multihop_debug = await _retrieve_multihop_evidence(
        trace,
        base_state,
        question,
    )
    if multihop_business_docs or multihop_internal_docs:
        business_docs = _deduplicate_docs(business_docs, multihop_business_docs)
        internal_docs = _deduplicate_docs(internal_docs, multihop_internal_docs)
        trace.add_step("multi_hop_merge", {
            "enabled": True,
            "base_business_count": len(business_state.get("docs") or []),
            "base_internal_count": len(internal_state.get("docs") or []),
            "multihop_business_count": len(multihop_business_docs),
            "multihop_internal_count": len(multihop_internal_docs),
            "merged_business_count": len(business_docs),
            "merged_internal_count": len(internal_docs),
            "aspect_counts": multihop_debug.get("aspect_counts"),
        })

    raw_direct_business_answer = _business_direct_answer(question, business_docs)
    direct_allowed, direct_reason = _allow_aggregate_direct_business_answer(
        question,
        business_state,
        internal_docs,
    )
    if raw_direct_business_answer and business_docs and direct_allowed and not internal_docs:
        preferred_doc = next(
            (
                doc for doc in business_docs
                if "web support cbgv" in normalize_text(doc.get("doc_name", ""))
                or "web support sv" in normalize_text(doc.get("doc_name", ""))
            ),
            business_docs[0],
        )
        trace.add_step("aggregate_business_direct_answer", {
            "used": True,
            "stage": "raw_business_docs",
            "doc_name": preferred_doc.get("doc_name"),
            "title": preferred_doc.get("title"),
            "business_source_count": len(business_docs),
            "allow_reason": direct_reason,
        })
        return _finalize(trace, {
            "question": question,
            "answer": raw_direct_business_answer,
            "source": f'{preferred_doc.get("title")} - {preferred_doc.get("doc_name")}',
            "sources": _build_sources([preferred_doc], question),
            "intent": intent,
        })

    business_retrieval_clarification = _retrieval_clarification_decision(business_state)
    if business_retrieval_clarification:
        return _clarification_response(
            trace,
            question,
            business_retrieval_clarification,
            retrieval_called=True,
        )

    usable_business, rejected_business = _filter_usable_sources(question, business_docs)
    usable_internal, rejected_internal = _filter_usable_sources(question, internal_docs)
    business_has_evidence, business_reason = _has_confident_evidence(question, usable_business)
    internal_has_evidence, internal_reason = _has_confident_evidence(question, usable_internal)
    retrieval_clarification = _retrieval_clarification_decision(internal_state)
    if retrieval_clarification and not business_has_evidence:
        return _clarification_response(
            trace,
            question,
            retrieval_clarification,
            retrieval_called=True,
        )
    trace.add_step("lcel_aggregate_evidence", {
        "evidence_decision": (
            "pass" if business_has_evidence or internal_has_evidence else "reject"
        ),
        "business_has_evidence": business_has_evidence,
        "business_reason": business_reason,
        "internal_has_evidence": internal_has_evidence,
        "internal_reason": internal_reason,
        "business_usable_count": len(usable_business),
        "internal_usable_count": len(usable_internal),
        "business_rejected": rejected_business,
        "internal_rejected": rejected_internal,
    })

    selected_business = usable_business if business_has_evidence else []
    selected_internal = usable_internal if internal_has_evidence else []
    academic_policy_terms = _academic_policy_terms(question)
    business_retrieval_debug = business_state.get("retrieval_debug") or {}
    if (
        _is_exam_retake_procedure_question(question)
        and _has_exam_retake_business_source(selected_business)
        and not any(_matches_academic_policy_source(question, doc) for doc in selected_internal)
    ):
        exam_retake_docs = _exam_retake_business_docs(selected_business)
        direct_business_answer = _business_direct_answer(question, exam_retake_docs)
        if direct_business_answer:
            best_doc = exam_retake_docs[0]
            trace.add_step("aggregate_business_direct_answer", {
                "used": True,
                "stage": "exam_retake_procedure_guard",
                "reason": "exam_retake_is_procedure_not_hoc_lai",
                "doc_name": best_doc.get("doc_name"),
                "title": best_doc.get("title"),
                "business_source_count": len(exam_retake_docs),
                "dropped_internal_source_count": len(selected_internal),
            })
            return _finalize(trace, {
                "question": question,
                "answer": direct_business_answer,
                "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
                "sources": _build_sources(exam_retake_docs, question),
                "intent": intent,
            })

    if academic_policy_terms and selected_internal:
        policy_internal = [
            doc for doc in selected_internal
            if _matches_academic_policy_source(question, doc)
        ]
        trace.add_step("aggregate_internal_filtered_for_policy", {
            "reason": "academic_policy_source_filter",
            "academic_policy_terms": academic_policy_terms,
            "before_count": len(selected_internal),
            "after_count": len(policy_internal),
            "dropped_count": len(selected_internal) - len(policy_internal),
        })
        selected_internal = policy_internal

    credit_load_answer, credit_load_docs = _credit_load_warning_answer(question, selected_internal)
    if not credit_load_answer and "credit_load_warning" in academic_policy_terms:
        credit_load_docs = await _credit_load_warning_targeted_docs(question)
        credit_load_answer, credit_load_docs = _credit_load_warning_answer(question, credit_load_docs)
    if credit_load_answer:
        trace.add_step("policy_deterministic_answer", {
            "reason": "credit_load_warning_16_credit_clause_before_soft_selection",
            "source_count": len(credit_load_docs),
            "sources": [
                {
                    "doc_name": doc.get("doc_name"),
                    "title": doc.get("title"),
                    "chunk_index": doc.get("chunk_index"),
                    "source_type": doc.get("source_type"),
                }
                for doc in credit_load_docs
            ],
        })
        best_doc = credit_load_docs[0]
        return _finalize(trace, {
            "question": question,
            "answer": credit_load_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(credit_load_docs, question),
            "intent": intent,
        })

    if (
        academic_policy_terms
        and selected_internal
        and not (
            _is_exam_retake_procedure_question(question)
            and _has_exam_retake_business_source(selected_business)
        )
        and (
            business_retrieval_debug.get("information_need") != "procedure_ui"
            or business_retrieval_debug.get("retrieval_method") in {None, "generic_hybrid", "generic_keyword", "keyword"}
        )
    ):
        trace.add_step("aggregate_business_soft_penalty_for_policy", {
            "reason": "generic_business_sources_kept_for_soft_aggregation",
            "academic_policy_terms": academic_policy_terms,
            "business_source_count": len(selected_business),
            "internal_source_count": len(selected_internal),
        })

    direct_business_answer = _business_direct_answer(question, selected_business)
    direct_allowed_after_filter, direct_reason_after_filter = _allow_aggregate_direct_business_answer(
        question,
        business_state,
        selected_internal,
    )
    if direct_business_answer and selected_business and direct_allowed_after_filter and not selected_internal:
        best_doc = selected_business[0]
        trace.add_step("aggregate_business_direct_answer", {
            "used": True,
            "doc_name": best_doc.get("doc_name"),
            "title": best_doc.get("title"),
            "business_source_count": len(selected_business),
            "allow_reason": direct_reason_after_filter,
        })
        return _finalize(trace, {
            "question": question,
            "answer": direct_business_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(selected_business, question),
            "intent": intent,
        })

    prefer_business, business_priority_debug = _should_prefer_business_over_internal(
        question,
        business_state,
        selected_business,
        selected_internal,
    )
    trace.add_step("business_priority_decision", business_priority_debug)

    if prefer_business:
        trace.add_step("aggregate_business_priority", {
            "reason": "mapping_guided_business_source_used_as_soft_boost",
            "business_source_count": len(selected_business),
            "internal_source_count": len(selected_internal),
            "business_retrieval_method": (
                (business_state.get("retrieval_debug") or {}).get("retrieval_method")
            ),
            "mapping_question": (
                (business_state.get("retrieval_debug") or {}).get("mapping_question")
            ),
        })

    docs = _deduplicate_docs(selected_business, selected_internal)
    source_types = {doc.get("source_type") for doc in docs}
    if len(source_types) > 1:
        docs, aggregate_rerank_debug = await asyncio.to_thread(rerank_chunks, question, docs)
    else:
        aggregate_rerank_debug = {
            "used": False,
            "reason": "single_source_type_or_already_ranked",
        }
    docs, soft_selection_debug = _select_diverse_aggregate_sources(
        question,
        [doc for doc in docs if _is_business_source(doc)],
        [doc for doc in docs if not _is_business_source(doc)],
        query_context=base_state.get("query_context") or {},
        limit=AGGREGATE_MAX_DIVERSE_CONTEXT_CHUNKS,
    )
    trace.add_step("lcel_aggregate_merge", {
        "business_source_count": len(selected_business),
        "internal_source_count": len(selected_internal),
        "deduplicated_source_count": len(docs),
        "reranking": aggregate_rerank_debug,
        "soft_selection": soft_selection_debug,
    })

    if not docs:
        trace.add_step("fallback_decision", {
            "from": "aggregate_documents",
            "to": "website_uneti",
            "reason": f"business={business_reason}; internal={internal_reason}",
        })
        return await _search_website_and_finalize(
            trace,
            question,
            "website_uneti_fallback",
            "aggregate_no_confident_source",
        )

    credit_load_answer, credit_load_docs = _credit_load_warning_answer(question, docs)
    if credit_load_answer:
        trace.add_step("policy_deterministic_answer", {
            "reason": "credit_load_warning_16_credit_clause",
            "source_count": len(credit_load_docs),
            "sources": [
                {
                    "doc_name": doc.get("doc_name"),
                    "title": doc.get("title"),
                    "chunk_index": doc.get("chunk_index"),
                    "source_type": doc.get("source_type"),
                }
                for doc in credit_load_docs
            ],
        })
        best_doc = credit_load_docs[0]
        return _finalize(trace, {
            "question": question,
            "answer": credit_load_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(credit_load_docs, question),
            "intent": intent,
        })

    graduation_answer, graduation_docs = _graduation_classification_answer(question, docs)
    if graduation_answer:
        trace.add_step("policy_deterministic_answer", {
            "reason": "graduation_requirements_and_classification",
            "source_count": len(graduation_docs),
            "sources": [
                {
                    "doc_name": doc.get("doc_name"),
                    "title": doc.get("title"),
                    "chunk_index": doc.get("chunk_index"),
                    "source_type": doc.get("source_type"),
                }
                for doc in graduation_docs
            ],
        })
        best_doc = graduation_docs[0]
        return _finalize(trace, {
            "question": question,
            "answer": graduation_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(graduation_docs, question),
            "intent": intent,
        })

    exam_defer_answer, exam_defer_docs = _exam_defer_answer(question, docs)
    if exam_defer_answer:
        trace.add_step("policy_deterministic_answer", {
            "reason": "exam_defer_conditions_and_procedure",
            "source_count": len(exam_defer_docs),
        })
        best_doc = exam_defer_docs[0]
        return _finalize(trace, {
            "question": question,
            "answer": exam_defer_answer,
            "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
            "sources": _build_sources(exam_defer_docs, question),
            "intent": intent,
        })

    internal_retrieval_debug = internal_state.get("retrieval_debug") or {}
    generation_retrieval_debug = (
        business_retrieval_debug
        if business_retrieval_debug.get("retrieval_plan")
        else internal_retrieval_debug
    )
    generation_state = await generate_answer({
        **base_state,
        "docs": docs,
        "max_context_chunks": min(len(docs), AGGREGATE_MAX_DIVERSE_CONTEXT_CHUNKS),
        "retrieval_debug": generation_retrieval_debug or {},
    })
    answer = generation_state["answer"]

    if _is_no_evidence_answer(answer):
        policy_fallback_answer, policy_fallback_docs = _absence_permission_comparison_answer(
            question,
            docs or _deduplicate_docs(selected_business, selected_internal),
        )
        if policy_fallback_answer:
            trace.add_step("policy_partial_answer_fallback", {
                "reason": "absence_permission_comparison_multi_source_evidence",
                "stage": "before_internal_only_retry",
                "source_count": len(policy_fallback_docs),
                "sources": [
                    {
                        "doc_name": doc.get("doc_name"),
                        "title": doc.get("title"),
                        "chunk_index": doc.get("chunk_index"),
                        "source_type": doc.get("source_type"),
                    }
                    for doc in policy_fallback_docs
                ],
            })
            docs = policy_fallback_docs
            answer = policy_fallback_answer

    if _is_no_evidence_answer(answer):
        procedure_fallback_answer, procedure_fallback_docs = _procedure_sources_fallback_answer(
            question,
            docs or selected_business,
        )
        if procedure_fallback_answer:
            trace.add_step("procedure_sources_fallback", {
                "reason": "business_context_available_but_generation_reported_no_evidence",
                "source_count": len(procedure_fallback_docs),
                "sources": [
                    {
                        "doc_name": doc.get("doc_name"),
                        "title": doc.get("title"),
                        "chunk_index": doc.get("chunk_index"),
                        "source_type": doc.get("source_type"),
                    }
                    for doc in procedure_fallback_docs
                ],
            })
            docs = procedure_fallback_docs
            answer = procedure_fallback_answer

    should_keep_business_context = (
        bool(selected_business)
        and (base_state.get("query_context") or {}).get("information_need") in {"procedure_ui", "mixed"}
    )
    if _is_no_evidence_answer(answer) and selected_internal and not should_keep_business_context:
        internal_only = selected_internal
        internal_rerank_debug = {
            "used": False,
            "reason": "internal_results_already_ranked",
        }
        internal_only = _limit_document_dominance(internal_only)
        trace.add_step("fallback_decision", {
            "from": "aggregate_generation",
            "to": "internal_only_generation",
            "reason": "aggregate_answer_reported_no_evidence",
            "internal_source_count": len(internal_only),
            "reranking": internal_rerank_debug,
        })
        retry_state = await generate_answer({
            **base_state,
            "docs": internal_only,
            "max_context_chunks": min(
                len(internal_only),
                AGGREGATE_MAX_CONTEXT_CHUNKS,
            ),
            "retrieval_debug": internal_retrieval_debug,
        })
        if not _is_no_evidence_answer(retry_state["answer"]):
            docs = internal_only
            answer = retry_state["answer"]

    if _is_no_evidence_answer(answer):
        policy_fallback_answer, policy_fallback_docs = _absence_permission_comparison_answer(
            question,
            docs or selected_internal,
        )
        if policy_fallback_answer:
            trace.add_step("policy_partial_answer_fallback", {
                "reason": "absence_permission_comparison_partial_evidence",
                "source_count": len(policy_fallback_docs),
                "sources": [
                    {
                        "doc_name": doc.get("doc_name"),
                        "title": doc.get("title"),
                        "chunk_index": doc.get("chunk_index"),
                    }
                    for doc in policy_fallback_docs
                ],
            })
            docs = policy_fallback_docs
            answer = policy_fallback_answer

    if _is_no_evidence_answer(answer):
        return _finalize(trace, {
            "question": question,
            "answer": NO_EVIDENCE_ANSWER,
            "source": None,
            "sources": [],
            "intent": intent,
        })

    best_doc = docs[0]
    return _finalize(trace, {
        "question": question,
        "answer": answer,
        "source": f'{best_doc.get("title")} - {best_doc.get("doc_name")}',
        "sources": _build_sources(docs, question),
        "intent": intent,
    })


def _empty_question_response(trace: RagTrace, original_question: str):
    return _finalize(trace, {
        "question": original_question,
        "answer": "Vui lÃ²ng nháº­p cÃ¢u há»i.",
        "source": None,
        "sources": [],
        "intent": QueryIntent.OUT_OF_SCOPE.value,
    })


@traceable(name="UNETI Chat Request", run_type="chain")
async def handle_internal_chat(request):
    question = request.question.strip()
    trace = _new_trace(question)
    trace.add_step("request_received", {
        "question": question,
        "is_empty": not bool(question),
        "forced_route": "internal_document",
    })

    if not question:
        return _empty_question_response(trace, request.question)

    decision = _analyze_retrieval_question(trace, question)
    decision = _use_retrieval_for_clarification(decision)

    return await _answer_with_internal_documents(
        trace,
        question,
        QueryIntent.INTERNAL_DOCUMENT.value,
        "explicit_internal_endpoint",
        decision,
    )


@traceable(name="UNETI Local Documents Chat Request", run_type="chain")
async def handle_local_documents_chat(request):
    question = request.question.strip()
    trace = _new_trace(question)
    trace.add_step("request_received", {
        "question": question,
        "is_empty": not bool(question),
        "forced_route": "local_documents",
    })

    if not question:
        return _empty_question_response(trace, request.question)

    decision = _analyze_retrieval_question(trace, question)
    decision = _use_retrieval_for_clarification(decision)
    return await _answer_with_local_documents(
        trace,
        question,
        QueryIntent.INTERNAL_DOCUMENT.value,
        "explicit_local_documents_endpoint",
        decision,
    )


@traceable(name="UNETI Chat Request", run_type="chain")
async def handle_business_chat(request):
    question = request.question.strip()
    trace = _new_trace(question)
    trace.add_step("request_received", {
        "question": question,
        "is_empty": not bool(question),
        "forced_route": "business_document",
    })

    if not question:
        return _empty_question_response(trace, request.question)

    decision = _analyze_retrieval_question(trace, question)
    decision = _use_retrieval_for_clarification(decision)

    return await _answer_with_business_documents(
        trace,
        question,
        QueryIntent.INTERNAL_DOCUMENT.value,
        "explicit_business_endpoint",
    )


@traceable(name="UNETI Chat Request", run_type="chain")
async def handle_website_chat(request):
    question = request.question.strip()
    trace = _new_trace(question)
    trace.add_step("request_received", {
        "question": question,
        "is_empty": not bool(question),
        "forced_route": "website_uneti",
    })

    if not question:
        return _empty_question_response(trace, request.question)

    return await _search_website_and_finalize(
        trace,
        question,
        QueryIntent.WEBSITE_UNETI.value,
        "explicit_website_endpoint",
    )


@traceable(name="UNETI Chat Request", run_type="chain")
async def handle_chat(request):
    question = request.question.strip()
    trace = _new_trace(question)
    trace.add_step("request_received", {"question": question, "is_empty": not bool(question)})

    if not question:
        return _finalize(trace, {
            "question": request.question,
            "answer": "Vui lòng nhập câu hỏi.",
            "source": None,
            "sources": [],
            "intent": QueryIntent.OUT_OF_SCOPE.value,
        })

    analysis_started = time.perf_counter()
    analysis = classify_query(question)
    trace.add_step("classify_query", {
        "intent": analysis.intent.value,
        "reason": analysis.reason,
        "metadata": analysis.metadata,
        "duration_ms": round((time.perf_counter() - analysis_started) * 1000, 3),
    }, {"question": question})

    if analysis.intent == QueryIntent.OUT_OF_SCOPE:
        trace.add_step("route_decision", {"llm_called": False, "reason": "out_of_scope"})
        return _finalize(trace, {
            "question": question,
            "answer": OUT_OF_SCOPE_ANSWER,
            "source": None,
            "sources": [],
            "intent": analysis.intent.value,
        })

    if analysis.intent == QueryIntent.GENERAL_ADVICE:
        trace.add_step("route_decision", {"llm_called": False, "reason": "general_advice"})
        return _finalize(trace, {
            "question": question,
            "answer": GENERAL_ADVICE_ANSWER,
            "source": None,
            "sources": [],
            "intent": analysis.intent.value,
        })

    if analysis.intent == QueryIntent.WEBSITE_UNETI:
        return await _search_website_and_finalize(
            trace,
            question,
            analysis.intent.value,
            "explicit_website_intent",
        )

    if analysis.metadata.get("so_van_ban") or _looks_like_document_number_query(question):
        return await _answer_with_internal_documents(
            trace,
            question,
            analysis.intent.value,
            "document_number_query",
        )

    decision = _analyze_retrieval_question(trace, question)
    decision = _use_retrieval_for_clarification(decision)

    return await _answer_with_aggregate_documents(
        trace,
        question,
        analysis.intent.value,
        analysis.reason,
        decision,
    )


def get_chat_trace(trace_id: str) -> dict:
    try:
        return load_trace(trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="trace_id khong hop le") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Khong tim thay trace") from exc
