from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
import unicodedata

from app.controller.document_controller import build_document_chunks, list_documents
from app.core.config import (
    ANN_TOP_K,
    BM25_B,
    BM25_K1,
    BM25_METADATA_BOOST,
    BM25_MIN_SCORE,
    BM25_TOP_K,
    CROSS_ENCODER_FINAL_TOP_K,
    CROSS_ENCODER_TOP_N,
    DOCUMENT_INDEX_CACHE_ENABLED,
    DOCUMENT_INDEX_CACHE_FILE,
    GROUNDED_HYDE_ANN_TOP_K,
    MIN_SEARCH_SCORE,
    PROBE_BM25_MIN_SCORE,
    PROBE_EVIDENCE_TOP_K,
    PROBE_MIN_EVIDENCE_SIGNALS,
    PROBE_MIN_TITLE_OVERLAP,
    PROBE_RRF_MIN_SCORE,
    PROBE_RRF_SCORE_GAP,
    PROBE_TOP_K,
    PROBE_VECTOR_MIN_SCORE,
    RRF_CANDIDATE_TOP_K,
    RRF_K,
    RETRIEVAL_CACHE_MAX_ITEMS,
    RETRIEVAL_CACHE_TTL_SECONDS,
    SEARCH_TOP_K,
)
from app.data.ambiguity_analyzer import (
    CLARIFICATION_NEEDED,
    DIRECT_RETRIEVAL,
    HYDE_RETRIEVAL,
    PROBE_RETRIEVAL,
)
from app.data.hyde import (
    generate_grounded_hyde_document,
    generate_hyde_document,
)
from app.data.query_analyzer import extract_metadata_constraints, normalize_date
from app.data.reranker import rerank_documents
from app.data.vector_store import search_similar_chunks
from rank_bm25 import BM25Okapi


STOP_WORDS = {
    "là", "gì", "của", "và", "có", "không", "như", "nào",
    "được", "trong", "về", "cho", "các", "những", "một", "này",
    "tôi", "em", "anh", "chị", "hỏi", "muốn", "biết", "thì",
    "khi", "cần", "chú", "ý", "lưu",
}
QUERY_EXPANSION = {
    "cap bang": [
        "dieu kien xet tot nghiep",
        "cong nhan tot nghiep",
        "cap bang tot nghiep",
    ],
    "ket qua thi": [
        "ket qua hoc tap",
        "diem thi",
        "tra cuu ket qua thi",
        "web support sinh vien",
    ],
    "le phi thi lai": [
        "dang ky thi lai",
        "le phi thi lai",
        "thanh toan hoc phi hoc ky tiep theo",
        "web support sinh vien",
    ],
    "hoc lai": [
        "dang ky hoc lai",
        "dang ky hoc phan truc tuyen",
        "website dang ky hoc tap",
        "hoc phan chua dat",
    ],
    "chuyen truong": [
        "chuyen truong",
        "dieu kien chuyen truong",
        "hieu truong",
        "cung nganh",
        "noi cu tru",
        "quy che dao tao dai hoc chinh quy",
        "dieu 28",
    ],
    "toi muon chuyen truong": [
        "chuyen truong",
        "dieu kien chuyen truong",
        "hieu truong",
        "cung nganh",
        "noi cu tru",
        "quy che dao tao dai hoc chinh quy",
        "dieu 28",
    ],
    "hoc 2 bang": [
        "hoc cung luc hai chuong trinh",
        "chuong trinh thu hai",
        "nam thu hai",
        "tot nghiep chuong trinh thu nhat",
        "quy che dao tao dai hoc chinh quy",
        "dieu 29",
    ],
    "chuong trinh hai": [
        "hoc cung luc hai chuong trinh",
        "chuong trinh thu hai",
        "don dang ky hoc chuong trinh thu hai",
        "phu luc 1",
        "phong chinh tri va cong tac sinh vien",
        "phong dao tao",
    ],
    "chuong trinh thu hai": [
        "hoc cung luc hai chuong trinh",
        "don dang ky hoc chuong trinh thu hai",
        "phu luc 1",
        "phong chinh tri va cong tac sinh vien",
        "phong dao tao",
    ],
    "f mon tu chon": [
        "hoc phan tu chon bi diem F F+",
        "hoc doi hoc phan khac tuong duong",
        "hoc cai thien diem trung binh tich luy",
        "quy che dao tao dai hoc chinh quy",
        "dieu 11",
    ],
    "f+ va f": [
        "diem chu F+ F",
        "thang diem",
        "diem hoc phan khong dat",
        "hoc lai hoc doi hoc phan tuong duong",
        "quy che dao tao dai hoc chinh quy",
        "dieu 16",
        "dieu 11",
    ],
    "diem f+ va f": [
        "diem chu F+ F",
        "thang diem",
        "diem hoc phan khong dat",
        "hoc lai hoc doi hoc phan tuong duong",
        "quy che dao tao dai hoc chinh quy",
        "dieu 16",
        "dieu 11",
    ],
    "mot tin chi": [
        "tin chi 15 tiet ly thuyet 30 tiet thuc hanh thi nghiem thao luan",
        "30 40 gio thuc tap tai co so",
        "45 60 gio lam tieu luan bai tap lon do an khoa luan",
        "quy che dao tao dai hoc chinh quy",
        "dieu 2",
    ],
    "tin chi tuong duong": [
        "tin chi 15 tiet ly thuyet 30 tiet thuc hanh thi nghiem thao luan",
        "30 40 gio thuc tap tai co so",
        "45 60 gio lam tieu luan bai tap lon do an khoa luan",
        "quy che dao tao dai hoc chinh quy",
        "dieu 2",
    ],
    "canh bao hoc tap": [
        "cảnh báo học tập",
        "khối lượng học tập",
        "đăng ký khối lượng học tập",
        "không quá 16 tín chỉ",
        "quy chế đào tạo đại học chính quy",
        "điều 9",
    ],
    "toi da bao nhieu tin chi": [
        "khối lượng học tập",
        "đăng ký khối lượng học tập",
        "số tín chỉ đăng ký tối đa",
        "quy chế đào tạo đại học chính quy",
        "điều 9",
    ],
    "dang ky bao nhieu tin chi": [
        "khối lượng học tập",
        "đăng ký khối lượng học tập",
        "số tín chỉ đăng ký",
        "quy chế đào tạo đại học chính quy",
        "điều 9",
    ],
    "sử dụng phòng học": [
        "quy định chung khi khai thác sử dụng phòng học",
        "quy định thực hiện 5S trong phòng học",
        "bảo quản bảo trì thiết bị trong phòng học",
        "trách nhiệm người học khi sử dụng phòng học",
    ],
    "phòng học cần chú ý": [
        "quy định chung khi khai thác sử dụng phòng học",
        "quy định thực hiện 5S trong phòng học",
        "bảo quản bảo trì thiết bị trong phòng học",
        "trách nhiệm người học khi sử dụng phòng học",
    ],
    "lưu ý khi sử dụng phòng học": [
        "quy định chung khi khai thác sử dụng phòng học",
        "quy định thực hiện 5S trong phòng học",
        "bảo quản bảo trì thiết bị trong phòng học",
        "trách nhiệm người học khi sử dụng phòng học",
    ],
    "bao nhiêu tín chỉ": [
        "khối lượng kiến thức toàn khóa",
        "chương trình cử nhân",
        "chương trình kỹ sư",
        "120 tín chỉ",
        "150 tín chỉ",
    ],
    "đăng ký môn": ["đăng ký học phần", "thời gian đăng ký"],
    "bỏ môn": ["rút bớt học phần", "hủy đăng ký học phần"],
    "hủy môn": ["rút bớt học phần", "hủy đăng ký học phần"],
    "hủy học phần": [
        "hủy đăng ký học phần",
        "rút bớt học phần",
        "đăng ký khối lượng học tập",
        "quy chế đào tạo đại học chính quy",
        "điều 10",
        "điều 9",
    ],
    "hủy học phần đã đăng ký": [
        "hủy đăng ký học phần",
        "rút bớt học phần",
        "đăng ký khối lượng học tập",
        "quy chế đào tạo đại học chính quy",
        "điều 10",
        "điều 9",
    ],
    "hủy đăng ký học phần": [
        "rút bớt học phần",
        "đăng ký khối lượng học tập",
        "quy chế đào tạo đại học chính quy",
        "điều 10",
        "điều 9",
    ],
    "rút học phần": [
        "rút bớt học phần",
        "hủy đăng ký học phần",
        "đăng ký khối lượng học tập",
        "quy chế đào tạo đại học chính quy",
        "điều 10",
        "điều 9",
    ],
    "rút bớt học phần": [
        "hủy đăng ký học phần",
        "đăng ký khối lượng học tập",
        "quy chế đào tạo đại học chính quy",
        "điều 10",
        "điều 9",
    ],
    "trượt": ["học lại", "điểm F", "không đạt", "cải thiện điểm"],
    "rớt": ["học lại", "điểm F", "không đạt"],
    "thi lại": ["kỳ thi phụ", "đánh giá lại học phần"],
    "qua môn": ["điểm học phần đạt", "đánh giá học phần", "điểm D trở lên"],
    "cách tính điểm": [
        "điểm trung bình học kỳ",
        "điểm trung bình tích lũy",
        "thang điểm 4",
        "thang điểm 10",
    ],
    "gpa": [
        "điểm trung bình tích lũy",
        "điểm trung bình học tập",
        "điểm trung bình chung tích lũy",
        "tính điểm trung bình",
        "điểm hệ 4",
    ],
    "điểm giả định": [
        "điểm dự kiến",
        "điểm mong muốn",
        "dự kiến kết quả học tập",
        "nhập điểm tổng kết dự kiến",
    ],
    "gpa dự kiến": [
        "điểm tích lũy dự kiến",
        "dự kiến kết quả học tập",
        "nhập điểm mong muốn",
        "ước tính điểm trung bình tích lũy",
    ],
    "ra trường": ["điều kiện xét tốt nghiệp", "công nhận tốt nghiệp"],
    "tốt nghiệp": [
        "điều kiện xét tốt nghiệp",
        "công nhận tốt nghiệp",
        "hạng tốt nghiệp",
    ],
    "bằng giỏi": ["hạng tốt nghiệp giỏi", "xếp loại tốt nghiệp"],
    "bằng khá": ["hạng tốt nghiệp khá", "xếp loại tốt nghiệp"],
    "đuổi học": ["buộc thôi học", "cảnh báo học vụ", "xử lý học vụ"],
    "nghỉ học": ["nghỉ học tạm thời", "bảo lưu kết quả", "thôi học"],
    "chuyển ngành": [
        "chuyển ngành đào tạo",
        "chuyển chương trình",
        "học cùng lúc hai chương trình",
    ],
    "một tín chỉ": [
        "tín chỉ được sử dụng để tính khối lượng học tập",
        "một tín chỉ được quy định bằng",
        "15 tiết học lý thuyết",
        "quy chế đào tạo",
    ],
    "bao nhiêu tiết": [
        "tiết học lý thuyết",
        "tiết thảo luận",
        "thực hành môn học",
        "tiết thí nghiệm",
    ],
    "học 2 ngành": ["học cùng lúc hai chương trình", "đào tạo song ngành"],
    "hoãn thi": [
        "vắng mặt dự thi có lý do chính đáng",
        "đơn xin hoãn thi",
        "lý do bất khả kháng",
        "điểm I",
        "kỳ thi phụ",
    ],
    "nghỉ thi": [
        "vắng mặt dự thi",
        "không tham gia kỳ thi",
        "lý do chính đáng",
        "nhận điểm 0",
    ],
    "ốm không thi được": [
        "vắng mặt dự thi có lý do chính đáng",
        "đơn xin hoãn thi",
        "chứng từ y tế",
        "lý do bất khả kháng",
    ],
    "cấm thi": [
        "không đủ điều kiện dự thi",
        "điểm đánh giá quá trình",
        "nghỉ học quá số buổi",
        "không hoàn thành học phí",
    ],
    "không được thi": [
        "không đủ điều kiện dự thi",
        "không có tên trong danh sách dự thi",
        "điểm chuyên cần",
    ],
    "nợ học phí thi": [
        "không đủ điều kiện dự thi",
        "nghĩa vụ học phí",
        "cấm dự thi",
    ],
    "đình chỉ thi": [
        "đình chỉ làm bài",
        "vi phạm quy chế thi",
        "hủy kết quả thi",
        "lập biên bản",
        "nhận điểm 0",
    ],
    "bắt phao": [
        "vi phạm quy chế thi",
        "đình chỉ làm bài",
        "sử dụng tài liệu trái phép",
        "nhận điểm 0",
    ],
    "quay cóp": [
        "vi phạm quy chế thi",
        "khiển trách",
        "cảnh cáo",
        "đình chỉ thi",
        "trao đổi bài",
    ],
    "dùng điện thoại lúc thi": [
        "vi phạm quy chế thi",
        "đình chỉ làm bài",
        "mang vật dụng cấm",
        "sử dụng thiết bị",
    ],
    "quên pass email": [
        "quên mật khẩu email",
        "xử lý vấn đề Email/LMS",
        "thủ tục hành chính",
        "một cửa đào tạo",
    ],
    "quên mật khẩu email": [
        "xử lý vấn đề Email/LMS",
        "thủ tục hành chính",
        "một cửa đào tạo",
    ],
    "mất pass email": [
        "quên mật khẩu email",
        "xử lý vấn đề Email/LMS",
        "một cửa đào tạo",
    ],
    "không đăng nhập được email": [
        "không đăng nhập được",
        "xử lý vấn đề Email/LMS",
        "một cửa đào tạo",
    ],
}

_INDEX_CACHE = {
    "signature": None,
    "chunks": [],
    "doc_freq": Counter(),
    "total_docs": 0,
    "skipped_files": [],
    "bm25": None,
    "bm25_corpus": [],
}
_SEARCH_CACHE = OrderedDict()
DOCUMENT_INDEX_CACHE_VERSION = 2
METADATA_EXACT_SCORE = 100.0
SEARCHABLE_METADATA_FIELDS = (
    "so_van_ban",
    "so_van_ban_ngan",
    "ten_van_ban",
    "doc_name",
    "relative_path",
    "loai_van_ban",
    "don_vi_ban_hanh",
    "ngay_ban_hanh",
    "ngay_hieu_luc",
    "phong_ban",
)


def normalize_text(text: str = ""):
    """Chuẩn hóa text về chữ thường, bỏ dấu tiếng Việt để so khớp keyword ổn định hơn."""
    text = str(text or "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return " ".join(text.lower().split())


def _has_normalized_phrase(normalized_text: str, phrase: str) -> bool:
    return bool(re.search(
        rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])",
        normalized_text,
    ))


NORMALIZED_STOP_WORDS = {normalize_text(word) for word in STOP_WORDS}


def clear_document_index_cache():
    """Xóa cache keyword index trong RAM sau khi tài liệu được upload hoặc cập nhật."""
    _INDEX_CACHE["signature"] = None
    _INDEX_CACHE["chunks"] = []
    _INDEX_CACHE["doc_freq"] = Counter()
    _INDEX_CACHE["total_docs"] = 0
    _INDEX_CACHE["skipped_files"] = []
    _INDEX_CACHE["bm25"] = None
    _INDEX_CACHE["bm25_corpus"] = []
    _SEARCH_CACHE.clear()


def apply_uneti_query_expansion(query: str) -> str:
    """Mở rộng câu hỏi bằng các cụm từ đồng nghĩa/quy ước nội bộ để keyword search dễ trúng hơn."""
    normalized_query = normalize_text(query)
    expanded_terms = [query]

    for key, terms in QUERY_EXPANSION.items():
        if normalize_text(key) in normalized_query:
            expanded_terms.extend(terms)

    return " ".join(dict.fromkeys(expanded_terms))


def _is_exam_regrade_query(query: str) -> bool:
    normalized_query = normalize_text(query)
    return any(
        phrase in normalized_query
        for phrase in (
            "phuc khao",
            "cham lai bai thi",
            "cham lai bai",
            "xem xet lai diem thi",
            "khieu nai diem thi",
            "diem thi sai",
        )
    )


def _is_projected_grade_ui_query(query: str) -> bool:
    """Detect requests to simulate future grades, not policy GPA calculations."""
    normalized_query = normalize_text(query)
    has_grade_target = any(
        _has_normalized_phrase(normalized_query, phrase)
        for phrase in (
            "gpa", "diem trung binh", "diem tich luy",
            "ket qua hoc tap", "xep loai hoc luc",
        )
    )
    has_projection = any(
        _has_normalized_phrase(normalized_query, phrase)
        for phrase in (
            "gia dinh", "du kien", "mong muon", "uoc tinh",
            "neu dat", "thu diem", "mo phong",
        )
    )
    has_interaction = any(
        _has_normalized_phrase(normalized_query, phrase)
        for phrase in (
            "nhap diem", "co cho nao", "xem", "tinh", "tra cuu",
            "chuc nang", "man hinh", "o dau", "vao dau",
        )
    )
    return has_grade_target and has_projection and has_interaction


def _academic_policy_retrieval_query(query: str) -> str:
    normalized_query = normalize_text(query)
    profile = _policy_query_profile(query)
    expanded_terms = [
        query
        if profile in {
            "credit_load_warning",
            "academic_warning_conditions",
            "course_retake_conditions",
        }
        else apply_uneti_query_expansion(query)
    ]

    if _is_exam_regrade_query(query):
        expanded_terms.append(
            "phuc khao ket qua bai thi man phuc khao mot cua khao thi "
            "gui yeu cau phuc khao sinh vien support uneti"
        )

    if _is_projected_grade_ui_query(query):
        expanded_terms.append(
            "du kien ket qua hoc tap nhap diem du kien diem tong ket du kien "
            "diem mong muon tinh diem tich luy du kien he 10 he 4 "
            "xep loai hoc luc tra cuu support uneti"
        )

    if _has_normalized_phrase(normalized_query, "thi lai"):
        expanded_terms.append(
            "dang ky thi lai ky thi lai du thi lai lich thi lai hoc phan "
            "khao thi thi ket thuc hoc phan"
        )

    if _has_normalized_phrase(normalized_query, "hoan thi"):
        expanded_terms.append(
            "vang mat du thi co ly do chinh dang don xin hoan thi diem I "
            "ky thi phu danh gia hoc phan quy che dao tao dai hoc chinh quy"
        )

    if profile == "attendance_exam_eligibility":
        expanded_terms.append(
            "diem chuyen can nghi hoc tren 50 phan tram so tiet trong chuong trinh "
            "bi cam thi ca ky thi chinh va ky thi phu diem thi tinh la 0 diem "
            "danh gia hoc phan quy che dao tao dai hoc chinh quy"
        )
    elif profile == "absence_permission_comparison":
        expanded_terms.append(
            "nghi hoc co phep nghi hoc khong phep so tiet vang diem chuyen can "
            "danh gia hoc phan hoc tap tren lop quy che dao tao dai hoc chinh quy dieu 13"
        )
    elif profile == "credit_load_warning":
        expanded_terms.append(
            "canh bao hoc tap dang ky khoi luong hoc tap khoi luong hoc tap "
            "so tin chi dang ky toi da khong qua 16 tin chi quy che dao tao "
            "dai hoc chinh quy dieu 9"
        )
    elif profile == "academic_warning_conditions":
        expanded_terms.append(
            "canh bao ket qua hoc tap tin chi khong dat diem F F+ "
            "diem trung binh hoc ky diem trung binh tich luy"
        )
    elif profile == "course_retake_conditions":
        expanded_terms.append(
            "hoc phan bat buoc diem F F+ phai dang ky hoc lai "
            "hoc phan tu chon hoc doi hoc phan tuong duong"
        )

    if profile == "graduation_classification":
        expanded_terms.append(
            "hang tot nghiep xep loai tot nghiep loai gioi diem trung binh tich luy "
            "toan khoa hoc lai vuot qua 5 phan tram ky luat canh cao quy che dao tao "
            "dai hoc chinh quy"
        )
        if "dieu kien tot nghiep la gi" in normalized_query:
            expanded_terms.append(
                "dieu kien xet tot nghiep cong nhan tot nghiep tich luy du hoc phan "
                "chung chi ngoai ngu tin hoc chuan dau ra dieu 24"
            )
    elif any(term in normalized_query for term in ("ra truong", "tot nghiep", "chung chi", "chuan dau ra")):
        expanded_terms.append(
            "dieu kien xet tot nghiep cong nhan tot nghiep chung chi ngoai ngu "
            "chung chi tin hoc chuan dau ra quy che dao tao dai hoc chinh quy dieu 24"
        )

    if (
        ("gpa" in normalized_query or "diem trung binh" in normalized_query)
        and not _is_projected_grade_ui_query(query)
    ):
        expanded_terms.append(
            "diem trung binh hoc ky diem trung binh tich luy diem trung binh chung "
            "tich luy tinh diem trung binh thang diem 4 quy che dao tao dai hoc "
            "chinh quy dieu 20"
        )

    if profile == "course_registration_change":
        expanded_terms.append(
            "huy dang ky hoc phan rut bot hoc phan dang ky khoi luong hoc tap "
            "thoi gian dang ky rut hoc phan quy che dao tao dai hoc chinh quy dieu 10 dieu 9"
        )
    elif profile == "transfer_school":
        expanded_terms.append(
            "chuyen truong dieu kien chuyen truong hieu truong dong y cung nganh "
            "noi cu tru hoan canh kho khan khong phai chuyen chuong trinh dao tao "
            "quy che dao tao dai hoc chinh quy dieu 28"
        )
    elif profile == "elective_failed_course":
        expanded_terms.append(
            "hoc phan tu chon bi diem F F+ hoc lai hoc doi sang hoc phan khac "
            "tuong duong cai thien diem trung binh tich luy quy che dao tao dai hoc "
            "chinh quy dieu 11"
        )
    elif profile == "f_grade_comparison":
        expanded_terms.append(
            "diem chu F+ F thang diem diem hoc phan khong dat quy doi diem chu "
            "hoc phan bat buoc hoc lai hoc phan tu chon hoc doi hoc phan tuong duong "
            "quy che dao tao dai hoc chinh quy dieu 16 dieu 11"
        )
    elif profile == "credit_definition":
        expanded_terms.append(
            "tin chi 15 tiet ly thuyet 30 tiet thuc hanh thi nghiem thao luan "
            "30 40 gio thuc tap tai co so 45 60 gio lam tieu luan bai tap lon "
            "do an khoa luan quy che dao tao dai hoc chinh quy dieu 2"
        )

    return " ".join(dict.fromkeys(expanded_terms))


def _policy_query_profile(query: str) -> str | None:
    normalized_query = normalize_text(query)
    asks_attendance_exam = any(
        term in normalized_query
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
        "nghi hoc" in normalized_query
        and any(term in normalized_query for term in ("co phep", "khong phep"))
        and any(
            term in normalized_query
            for term in ("khac nhau", "khac gi", "phan biet", "so sanh", "nhung gi")
        )
    )

    if _is_projected_grade_ui_query(query):
        return "projected_grade_ui"
    if _has_normalized_phrase(normalized_query, "thi lai"):
        return "exam_retake"
    if _has_normalized_phrase(normalized_query, "hoan thi"):
        return "exam_defer"
    if asks_absence_comparison and not asks_attendance_exam:
        return "absence_permission_comparison"
    if (
        "diem chuyen can" in normalized_query
        or "cam thi" in normalized_query
        or "khong duoc thi" in normalized_query
        or "du thi" in normalized_query
    ) and (
        "nghi hoc" in normalized_query
        or "vang" in normalized_query
        or "chuyen can" in normalized_query
        or "so tiet" in normalized_query
    ):
        return "attendance_exam_eligibility"
    if "tot nghiep" in normalized_query and any(
        term in normalized_query
        for term in ("loai gioi", "loai xuat sac", "xep loai", "xep hang", "hang tot nghiep")
    ):
        return "graduation_classification"
    if any(term in normalized_query for term in ("ra truong", "tot nghiep", "chung chi", "chuan dau ra")):
        return "graduation_requirements"
    if "gpa" in normalized_query or "diem trung binh" in normalized_query:
        return "grade_average"
    if "canh bao hoc tap" in normalized_query and any(
        term in normalized_query
        for term in ("khi nao", "dieu kien", "truong hop", "vi sao")
    ):
        return "academic_warning_conditions"
    if "hoc lai" in normalized_query and any(
        term in normalized_query
        for term in ("khi nao", "dieu kien", "truong hop", "diem f", "khong dat")
    ):
        return "course_retake_conditions"
    if "chuyen truong" in normalized_query:
        return "transfer_school"
    if (
        any(term in normalized_query for term in ("f+ va f", "f va f+", "diem f+", "f+"))
        and "f" in normalized_query
    ):
        return "f_grade_comparison"
    if (
        "tu chon" in normalized_query
        and any(term in normalized_query for term in ("diem f", "bi f", "f+", "khong dat"))
        and any(term in normalized_query for term in ("mon", "hoc phan", "chon mon", "thay the", "hoc doi"))
    ):
        return "elective_failed_course"
    if (
        "tin chi" in normalized_query
        and any(term in normalized_query for term in ("tuong duong", "bao nhieu tiet", "may tiet", "ly thuyet", "thuc hanh"))
    ):
        return "credit_definition"
    if (
        "canh bao hoc tap" in normalized_query
        and any(term in normalized_query for term in ("tin chi", "khoi luong", "dang ky", "dang ki", "toi da", "bao nhieu"))
    ) or (
        any(term in normalized_query for term in ("toi da", "bao nhieu", "may tin chi", "so tin chi"))
        and "tin chi" in normalized_query
        and any(term in normalized_query for term in ("dang ky", "dang ki", "khoi luong hoc tap"))
    ):
        return "credit_load_warning"
    registration_change = (
        (
            any(
                _has_normalized_phrase(normalized_query, term)
                for term in ("huy", "rut", "xoa")
            )
            and any(
                _has_normalized_phrase(normalized_query, term)
                for term in ("hoc phan", "mon", "dang ky", "dang ki")
            )
        )
        or any(
            phrase in normalized_query
            for phrase in ("bo hoc phan", "bo mon", "bo dang ky", "bo dang ki")
        )
    )
    if registration_change and "thi lai" not in normalized_query:
        return "course_registration_change"
    if (
        "nghi hoc" in normalized_query
        and asks_attendance_exam
    ):
        return "attendance_exam_eligibility"
    return None


def _effective_final_top_k(question: str, source_type_filter: str | None) -> int:
    base_top_k = min(CROSS_ENCODER_FINAL_TOP_K, SEARCH_TOP_K)
    if source_type_filter in {"official_document", "local_file"} and _policy_query_profile(question):
        return max(base_top_k, 8)
    return base_top_k


def _policy_rerank_pool_size(question: str, final_top_k: int) -> int:
    if _policy_query_profile(question):
        return max(final_top_k, 16)
    return final_top_k


def _policy_result_priority(question: str, doc: dict) -> int:
    profile = _policy_query_profile(question)
    if not profile:
        return 0

    searchable = normalize_text(" ".join(
        str(doc.get(field) or "")
        for field in ("doc_name", "title", "relative_path", "phong_ban", "content")
    ))
    metadata_text = normalize_text(" ".join(
        str(doc.get(field) or "")
        for field in ("doc_name", "title", "relative_path", "phong_ban")
    ))
    score = 0

    if "quy che dao tao dai hoc chinh quy" in searchable or "quy che dao tao dai hoc" in searchable:
        score += 60
    if "pdt" in searchable or "phong dao tao" in searchable:
        score += 20
    if "thac si" in searchable and "thac si" not in normalize_text(question):
        score -= 45
    if any(term in searchable for term in ("cntt", "khcn", "thiet bi", "phong hoc", "tuyen sinh")):
        score -= 35

    if profile == "attendance_exam_eligibility":
        if "nghi hoc tren 50" in searchable and "cam thi" in searchable:
            score += 120
        if "diem chuyen can" in searchable:
            score += 60
        if "danh gia hoc phan" in searchable or "diem hoc phan" in searchable:
            score += 35
        if doc.get("dieu") == 13:
            score += 35
        if "quy che cong tac sinh vien" in searchable:
            score -= 40
    elif profile == "projected_grade_ui":
        if "du kien ket qua hoc tap" in searchable:
            score += 180
        if "nhap diem mong muon" in searchable or "diem tong ket du kien" in searchable:
            score += 120
        if "support uneti" in searchable or "web support sv" in metadata_text:
            score += 80
        if doc.get("document_type") == "business_document":
            score += 50
        if doc.get("dieu") == 20:
            score -= 80
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score -= 50
    elif profile == "grade_average":
        if "tinh diem trung binh" in searchable:
            score += 120
        if "diem trung binh tich luy" in searchable or "diem trung binh hoc ky" in searchable:
            score += 90
        if doc.get("dieu") == 20:
            score += 60
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score += 50
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            score -= 70
    elif profile == "transfer_school":
        if "chuyen truong" in searchable:
            score += 170
        if doc.get("dieu") == 28:
            score += 140
        if "hieu truong" in searchable:
            score += 80
        if any(term in searchable for term in ("cung nganh", "noi cu tru", "hoan canh")):
            score += 60
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score += 80
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            score -= 180
        if "chuyen chuong trinh dao tao" in searchable and "chuyen truong" not in searchable:
            score -= 140
    elif profile == "elective_failed_course":
        if "hoc phan tu chon" in searchable:
            score += 150
        if any(term in searchable for term in ("diem f", " f ", "f+", "khong dat")):
            score += 80
        if "hoc doi" in searchable or "hoc phan khac tuong duong" in searchable:
            score += 140
        if "hoc cai thien" in searchable or "diem trung binh tich luy" in searchable:
            score += 70
        if doc.get("dieu") == 11:
            score += 130
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score += 70
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            score -= 140
    elif profile == "f_grade_comparison":
        if "f+" in searchable and " f" in searchable:
            score += 140
        if any(term in searchable for term in ("thang diem", "diem chu", "diem hoc phan")):
            score += 100
        if doc.get("dieu") == 16:
            score += 140
        if doc.get("dieu") == 11:
            score += 120
        if any(term in searchable for term in ("hoc lai", "hoc doi", "hoc phan tu chon", "hoc phan bat buoc")):
            score += 90
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score += 70
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            score -= 160
    elif profile == "credit_definition":
        if "tin chi" in searchable:
            score += 80
        if "15 tiet" in searchable and "ly thuyet" in searchable:
            score += 120
        if "30 tiet" in searchable and any(term in searchable for term in ("thuc hanh", "thi nghiem", "thao luan")):
            score += 120
        if any(term in searchable for term in ("30 40 gio", "30-40 gio", "30 den 40 gio")):
            score += 100
        if any(term in searchable for term in ("45 60 gio", "45-60 gio", "45 den 60 gio")):
            score += 130
        if doc.get("dieu") == 2:
            score += 120
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score += 80
        if any(term in searchable for term in ("gpa", "tot nghiep", "chung chi", "web support", "khcn")):
            score -= 120
    elif profile == "course_registration_change":
        if "rut bot hoc phan" in searchable or "huy dang ky hoc phan" in searchable:
            score += 170
        if doc.get("dieu") == 10:
            score += 130
        if "dang ky khoi luong hoc tap" in searchable:
            score += 110
        if doc.get("dieu") == 9:
            score += 45
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score += 60
        if any(term in searchable for term in ("tieng anh", "toeic", "ielts", "chung chi", "quy doi", "ngoai ngu", "tin hoc")):
            score -= 110
        if any(term in searchable for term in ("khcn", "nghien cuu", "thiet bi", "phong hoc")):
            score -= 90
        if "thi ket thuc hoc phan" in searchable and "diem chuyen can" not in searchable:
            score -= 45
    elif profile == "credit_load_warning":
        if "canh bao hoc tap" in searchable or "canh bao ket qua hoc tap" in searchable:
            score += 150
        if "dang trong thoi gian bi canh bao" in searchable:
            score += 80
        if "dang ky khoi luong hoc tap" in searchable or "khoi luong hoc tap" in searchable:
            score += 120
        if "khong qua 16" in searchable or "16 tin chi" in searchable:
            score += 180
        if "3/2 so tin chi" in searchable and not (
            "khong qua 16" in searchable or "16 tin chi" in searchable
        ):
            score -= 80
        if doc.get("dieu") == 9:
            score += 80
        if "quy che dao tao dai hoc chinh quy" in searchable:
            score += 60
        if any(term in metadata_text for term in ("thoi khoa bieu", "lich hoc", "lich thi", "web support")):
            score -= 130
        if any(term in metadata_text for term in ("khcn", "nghien cuu", "tap chi", "thiet bi", "phong hoc")):
            score -= 110
    elif profile == "academic_warning_conditions":
        if "canh bao ket qua hoc tap" in searchable:
            score += 130
        if any(
            term in searchable
            for term in (
                "tin chi khong dat", "diem f", "diem trung binh hoc ky",
                "diem trung binh tich luy",
            )
        ):
            score += 80
        if "dang ky khoi luong hoc tap" in searchable:
            score -= 50
    elif profile == "course_retake_conditions":
        if "hoc phan bat buoc" in searchable and any(
            term in searchable for term in ("diem f", "f+")
        ):
            score += 120
        if "hoc phan tu chon" in searchable and any(
            term in searchable for term in ("hoc doi", "tuong duong")
        ):
            score += 90
        if "dang ky hoc lai" in searchable:
            score += 50
    elif profile == "absence_permission_comparison":
        if "diem chuyen can" in searchable:
            score += 85
        if "nghi hoc" in searchable and any(term in searchable for term in ("so tiet", "vang", "co phep", "khong phep")):
            score += 65
        if "danh gia hoc phan" in searchable:
            score += 45
        if doc.get("dieu") == 13:
            score += 45
        if "thi ket thuc hoc phan" in metadata_text or doc.get("dieu") == 15:
            score -= 90
        if "vang mat trong ky thi" in metadata_text:
            score -= 100
        if any(term in metadata_text for term in ("diem ren luyen", "cong tac sinh vien", "ngoai tru")):
            score -= 70
    elif profile == "exam_retake":
        if any(term in searchable for term in ("dang ky thi lai", "thi lai", "du thi lai", "ky thi lai")):
            score += 100
        if "khao thi" in searchable or "thi ket thuc hoc phan" in searchable:
            score += 25
        if any(
            term in metadata_text
            for term in (
                "dang ky hoc lai",
                "hoc cai thien",
                "chuan dau ra",
                "diem ren luyen",
                "cong tac sinh vien",
                "khcn",
                "nghien cuu",
                "de an tot nghiep",
                "xay dung",
                "tham dinh",
            )
        ):
            score -= 90
    elif profile == "exam_defer":
        if "diem i" in searchable or "duoc phep hoan thi" in searchable:
            score += 100
        if "thi ket thuc hoc phan" in searchable:
            score += 35
    elif profile == "graduation_classification":
        if "hang tot nghiep" in searchable or "xep loai tot nghiep" in searchable:
            score += 150
        if "loai gioi" in searchable and any(term in searchable for term in ("3,20", "3.20")):
            score += 180
        if any(term in searchable for term in ("hoc lai vuot qua 5", "ky luat", "giam di mot muc")):
            score += 90
        if doc.get("dieu") == 25:
            score += 150
        if doc.get("dieu") == 24:
            score += (
                150
                if "dieu kien tot nghiep la gi" in normalize_text(question)
                else 45
            )
        if "thac si" in searchable and "thac si" not in normalize_text(question):
            score -= 160
    elif profile == "graduation_requirements":
        if "dieu kien xet tot nghiep" in searchable or "cong nhan tot nghiep" in searchable:
            score += 100
        if "chung chi" in searchable and any(term in searchable for term in ("ngoai ngu", "tin hoc")):
            score += 45
        if doc.get("dieu") == 24:
            score += 30

    return score


def _prioritize_policy_results(question: str, docs: list[dict]) -> list[dict]:
    if not _policy_query_profile(question):
        return docs

    return sorted(
        docs,
        key=lambda doc: (
            _policy_result_priority(question, doc),
            doc.get("metadata_matched", False),
            float(doc.get("rerank_score", float("-inf"))),
            float(doc.get("keyword_score") or 0),
            float(doc.get("vector_score") or 0),
            float(doc.get("rrf_score") or 0),
        ),
        reverse=True,
    )


def _rank_local_documents(question: str, docs: list[dict], limit: int) -> list[dict]:
    normalized_question = normalize_text(question)
    profile = _local_query_profile(question)

    # Existing policy-specific ranking contains detailed legal rules. Keep that
    # ordering intact so business-document tuning cannot regress those queries.
    if _policy_query_profile(question) and profile != "business":
        return _deduplicate_local_results(docs, limit)

    rerank_values = [float(doc.get("rerank_score") or 0) for doc in docs]
    rrf_values = [float(doc.get("rrf_score") or 0) for doc in docs]
    rerank_is_confident = max(rerank_values, default=float("-inf")) >= 0

    def normalized(value: float, values: list[float]) -> float:
        low, high = min(values, default=0), max(values, default=0)
        return (value - low) / (high - low) if high > low else 1.0

    query_tokens = _bm25_tokens(question)
    business_anchor_phrases = (
        "hoan thi",
        "video huong dan",
        "su co thiet bi",
        "du lieu giang day",
        "lop hoc phan",
        "thong ke mot cua",
        "phieu khao sat",
        "khao sat noi bo",
        "ket qua hoc tap",
        "khoi luong cong tac",
        "tong hop thanh toan",
        "bao hong",
        "file excel",
    )
    query_anchors = [
        phrase
        for phrase in business_anchor_phrases
        if phrase in normalized_question
    ]
    if _is_exam_regrade_query(question):
        query_anchors.append("phuc khao")
    generic_query_tokens = {
        "toi", "lam", "nao", "the", "gi", "co", "duoc", "sau", "khi",
        "nhung", "cac", "mot", "de", "va", "tren", "trong", "cho",
    }
    section_query_tokens = {
        token for token in query_tokens if token not in generic_query_tokens
    }

    ranked = []
    for position, doc in enumerate(docs):
        document_type = normalize_text(doc.get("document_type", ""))
        domain_match = 0.0
        if profile == "business" and document_type == "business_document":
            domain_match = 1.0
        elif profile == "legal" and document_type in {"regulation", "decision"}:
            domain_match = 1.0

        searchable = normalize_text(" ".join(
            str(doc.get(field) or "")
            for field in ("title", "heading", "section_path", "content")
        ))
        section_text = normalize_text(" ".join(
            str(doc.get(field) or "")
            for field in ("title", "heading", "section_path")
        ))
        document_name_text = normalize_text(" ".join(
            str(doc.get(field) or "")
            for field in ("doc_name", "ten_van_ban")
        ))
        section_tokens = set(_bm25_tokens(section_text))
        section_match = (
            len(section_query_tokens & section_tokens) / len(section_query_tokens)
            if section_query_tokens
            else 0.0
        )
        anchor_match = (
            sum(phrase in searchable for phrase in query_anchors)
            / len(query_anchors)
            if query_anchors
            else 0.0
        )
        phrase_count = max(len(query_tokens) - 1, 0)
        phrase_matches = sum(
            1
            for index in range(phrase_count)
            if " ".join(query_tokens[index:index + 2]) in searchable
        )
        exact_phrase_bonus = (
            min(phrase_matches / phrase_count, 1.0)
            if phrase_count
            else float(normalized_question in searchable)
        )
        document_phrase_match = (
            sum(
                " ".join(query_tokens[index:index + 2]) in document_name_text
                for index in range(phrase_count)
            ) / phrase_count
            if phrase_count
            else 0.0
        )
        procedure_marker_count = sum(
            marker in searchable
            for marker in (
                "buoc 1",
                "dang nhap",
                "gui yeu cau",
                "truy cap truc tiep",
                "support.uneti.edu.vn",
            )
        )
        semantic_section_match = 0.0
        if _is_exam_regrade_query(question):
            if "phuc khao" in section_text:
                semantic_section_match = 1.0
            elif "thi lai" in section_text:
                semantic_section_match = -1.0
        raw_rerank_score = float(doc.get("rerank_score") or 0)
        rerank_score = (
            normalized(raw_rerank_score, rerank_values)
            if rerank_is_confident
            else 1 / (1 + math.exp(-raw_rerank_score))
        )
        rrf_score = normalized(float(doc.get("rrf_score") or 0), rrf_values)
        if rerank_is_confident:
            final_score = (
                0.55 * rerank_score
                + 0.12 * rrf_score
                + 0.10 * domain_match
                + 0.07 * section_match
                + 0.13 * anchor_match
                + 0.03 * exact_phrase_bonus
            )
        else:
            # Negative scores mean the cross-encoder considers every candidate
            # weak. In that case, domain and section signals are more reliable
            # than tiny differences between several irrelevant passages.
            final_score = (
                0.35 * rerank_score
                + 0.15 * rrf_score
                + 0.15 * domain_match
                + 0.10 * section_match
                + 0.20 * anchor_match
                + 0.05 * exact_phrase_bonus
            )
        if (
            "thiet bi giang duong" in normalized_question
            and "tru thiet bi giang duong" in searchable
        ):
            final_score -= 0.25
        if profile == "legal":
            final_score += 0.50 * document_phrase_match
        if profile == "business" and document_type == "business_document":
            final_score += 0.45
            if procedure_marker_count >= 2:
                final_score += 0.20
        final_score += 0.55 * semantic_section_match
        if (
            "thac si" in document_name_text
            and "thac si" not in normalized_question
        ):
            final_score -= 0.85
        if doc.get("metadata_matched"):
            final_score += 1.0
        ranked.append({
            **doc,
            "metadata_boost": round(0.10 * domain_match, 6),
            "section_match": round(section_match, 6),
            "business_anchor_match": round(anchor_match, 6),
            "document_phrase_match": round(document_phrase_match, 6),
            "exact_phrase_bonus": round(exact_phrase_bonus, 6),
            "procedure_marker_count": procedure_marker_count,
            "semantic_section_match": semantic_section_match,
            "local_final_score": round(final_score, 6),
            "_original_position": position,
        })

    ranked.sort(
        key=lambda item: (
            item["local_final_score"],
            -item["_original_position"],
        ),
        reverse=True,
    )
    return _deduplicate_local_results(ranked, limit)


def _deduplicate_local_results(docs: list[dict], limit: int) -> list[dict]:
    selected = []
    seen_chunks = set()
    document_counts = Counter()
    for raw_doc in docs:
        doc = dict(raw_doc)
        doc.pop("_original_position", None)
        chunk_identity = doc.get("chunk_hash") or _chunk_key(doc)
        if chunk_identity in seen_chunks:
            continue
        document_identity = (
            doc.get("document_id")
            or doc.get("content_hash")
            or doc.get("relative_path")
            or doc.get("doc_name")
        )
        if document_counts[document_identity] >= 3:
            continue
        seen_chunks.add(chunk_identity)
        document_counts[document_identity] += 1
        selected.append(doc)
        if len(selected) >= limit:
            break
    return selected


def _local_query_profile(question: str) -> str:
    normalized_question = normalize_text(question)
    if re.search(
        r"\b(?:dieu|khoan|muc|chuong)\s+(?:\d+|[ivxlcdm]+)\b",
        normalized_question,
    ) or any(
        term in normalized_question
        for term in ("quy dinh", "quy che", "quyet dinh", "che tai")
    ):
        return "legal"

    if _is_exam_regrade_query(question):
        return "business"
    if _is_projected_grade_ui_query(question):
        return "business"

    business_terms = (
        "web support",
        "uneti online",
        "trang support",
        "trang chu",
        "man hinh",
        "module",
        "thao tac",
        "xem chi tiet",
        "nhan nut",
        "click",
        "thanh menu",
        "truy cap",
        "duong link",
        "xuat excel",
        "dieu kien loc",
        "bo loc",
        "huy loc",
        "thong ke mot cua",
        "phieu khao sat",
        "khao sat",
        "khao sat noi bo",
        "tin tuc",
        "bai viet",
        "bai dang",
        "bo cai dat",
        "tai lieu huong dan",
        "video huong dan",
        "phan he",
        "ket qua hoc tap",
        "lich hoc",
        "lich thi",
        "danh gia thu tuc",
        "thu tuc mot cua",
        "thong ke thu tuc",
        "cong tac giang vien",
        "thanh toan du kien",
        "thanh toan",
        "le phi thi lai",
        "tra cuu",
        "ket qua thi",
        "su co thiet bi",
        "gui yeu cau",
        "du lieu giang day",
        "lop hoc phan",
        "dong bo",
        "file excel",
        "bao cao mot cua",
        "bao hong",
        "khoi luong cong tac",
        "tong hop thanh toan",
        "lam the nao",
        "thuc hien nhu the nao",
        "thu tuc xin",
        "cach lam",
        "cach thuc hien",
        "huong dan",
        "nop don",
        "gui yeu cau",
    )
    if any(term in normalized_question for term in business_terms):
        return "business"
    return "neutral"


def _local_rrf_branch_weights(question: str) -> dict[str, float]:
    profile = _local_query_profile(question)
    if profile == "business":
        return {
            "bm25_original": 0.8,
            "bm25_business": 1.0,
            "ann_original": 1.2,
            "ann_business": 1.2,
            "ann_hyde": 1.2,
            "ann_grounded_hyde": 1.2,
        }
    if profile == "legal":
        return {
            "bm25_original": 1.3,
            "ann_original": 0.8,
            "ann_hyde": 0.8,
            "ann_grounded_hyde": 0.8,
        }
    return {}


def get_keywords(text: str):
    """Tách text thành các keyword đã chuẩn hóa, bỏ stop words và từ quá ngắn."""
    normalized = normalize_text(text)
    words = [
        word.strip(".,;:!?()[]{}\"'")
        for word in normalized.split()
    ]

    return [
        word
        for word in words
        if len(word) >= 3 and word not in NORMALIZED_STOP_WORDS
    ]


def _bm25_tokens(text: str) -> list[str]:
    """Tokenize Vietnamese while preserving legal references and numbers."""
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    return [
        token
        for token in tokens
        if token.isdigit() or len(token) >= 2 or token in {"i", "v", "x"}
    ]


def _document_signature(files):
    """Tạo chữ ký từ danh sách file để biết cache keyword index còn hợp lệ hay không."""
    return tuple(
        (file.get("relative_path") or file["file_name"], file["file_size_kb"], file.get("updated_at"))
        for file in files
    )


def _document_index_cache_path() -> Path:
    return Path(DOCUMENT_INDEX_CACHE_FILE).resolve()


def _json_safe_signature(signature):
    return json.loads(json.dumps(signature))


def _serialize_index_chunk(chunk: dict) -> dict:
    serialized = dict(chunk)
    token_counts = serialized.get("_token_counts")
    token_set = serialized.get("_token_set")

    if isinstance(token_counts, Counter):
        serialized["_token_counts"] = dict(token_counts)
    if isinstance(token_set, set):
        serialized["_token_set"] = sorted(token_set)

    return serialized


def _deserialize_index_chunk(chunk: dict) -> dict:
    deserialized = dict(chunk)
    deserialized["_token_counts"] = Counter(deserialized.get("_token_counts") or {})
    deserialized["_token_set"] = set(deserialized.get("_token_set") or [])
    return deserialized


def _restore_document_index_cache(signature, chunks, doc_freq, total_docs, skipped_files, bm25_corpus):
    _INDEX_CACHE["signature"] = signature
    _INDEX_CACHE["chunks"] = chunks
    _INDEX_CACHE["doc_freq"] = doc_freq
    _INDEX_CACHE["total_docs"] = total_docs
    _INDEX_CACHE["skipped_files"] = skipped_files
    _INDEX_CACHE["bm25_corpus"] = bm25_corpus
    _INDEX_CACHE["bm25"] = (
        BM25Okapi(bm25_corpus, k1=BM25_K1, b=BM25_B)
        if any(bm25_corpus)
        else None
    )


def _load_document_index_from_disk(signature):
    if not DOCUMENT_INDEX_CACHE_ENABLED:
        return None

    cache_path = _document_index_cache_path()
    if not cache_path.is_file():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return None

    if payload.get("version") != DOCUMENT_INDEX_CACHE_VERSION:
        return None
    if payload.get("signature") != _json_safe_signature(signature):
        return None

    chunks = [
        _deserialize_index_chunk(chunk)
        for chunk in payload.get("chunks", [])
        if isinstance(chunk, dict)
    ]
    doc_freq = Counter(payload.get("doc_freq") or {})
    total_docs = int(payload.get("total_docs") or len(chunks))
    skipped_files = payload.get("skipped_files") or []
    bm25_corpus = payload.get("bm25_corpus") or []

    if len(bm25_corpus) != len(chunks):
        bm25_corpus = [_bm25_document_tokens(chunk) for chunk in chunks]

    return chunks, doc_freq, total_docs, skipped_files, bm25_corpus


def _write_document_index_to_disk(signature, chunks, doc_freq, total_docs, skipped_files, bm25_corpus):
    if not DOCUMENT_INDEX_CACHE_ENABLED:
        return

    cache_path = _document_index_cache_path()
    payload = {
        "version": DOCUMENT_INDEX_CACHE_VERSION,
        "signature": _json_safe_signature(signature),
        "chunks": [_serialize_index_chunk(chunk) for chunk in chunks],
        "doc_freq": dict(doc_freq),
        "total_docs": total_docs,
        "skipped_files": skipped_files,
        "bm25_corpus": bm25_corpus,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)
        temp_path.replace(cache_path)
    except Exception:
        return


def _current_document_signature():
    return _document_signature(list_documents())


def _search_cache_key(
    question: str,
    source_type_filter: str | None,
    signature,
    retrieval_action: str = DIRECT_RETRIEVAL,
    final_top_k: int | None = None,
    retrieval_query: str | None = None,
    retrieval_filters: tuple = (),
):
    return (
        normalize_text(question),
        normalize_text(retrieval_query or question),
        source_type_filter or "",
        retrieval_action,
        signature,
        final_top_k or SEARCH_TOP_K,
        retrieval_filters,
    )


def _get_search_cache(key):
    cached = _SEARCH_CACHE.get(key)
    if not cached:
        return None

    created_at, results = cached
    if time.monotonic() - created_at > RETRIEVAL_CACHE_TTL_SECONDS:
        _SEARCH_CACHE.pop(key, None)
        return None

    _SEARCH_CACHE.move_to_end(key)
    return deepcopy(results)


def _set_search_cache(key, results):
    _SEARCH_CACHE[key] = (time.monotonic(), deepcopy(results))
    _SEARCH_CACHE.move_to_end(key)

    while len(_SEARCH_CACHE) > RETRIEVAL_CACHE_MAX_ITEMS:
        _SEARCH_CACHE.popitem(last=False)


def _filter_results_by_source_type(results: list[dict], source_type_filter: str | None) -> list[dict]:
    if not source_type_filter:
        return results
    return [doc for doc in results if doc.get("source_type") == source_type_filter]


def _normalized_filter_values(values) -> set[str]:
    if not values:
        return set()
    if isinstance(values, str):
        values = {values}
    return {normalize_text(value) for value in values if value is not None}


def _matches_retrieval_filters(
    chunk: dict,
    *,
    source_type_filter: str | None = None,
    corpus_filter: str | None = None,
    rag_enabled_filter: bool | None = None,
    exclude_document_names=None,
    exclude_source_types=None,
    document_type_filter=None,
    department_filter=None,
) -> bool:
    if source_type_filter and chunk.get("source_type") != source_type_filter:
        return False
    if corpus_filter and chunk.get("corpus") != corpus_filter:
        return False
    if rag_enabled_filter is not None and chunk.get("rag_enabled") is not rag_enabled_filter:
        return False
    if normalize_text(chunk.get("doc_name", "")) in _normalized_filter_values(exclude_document_names):
        return False
    if normalize_text(chunk.get("source_type", "")) in _normalized_filter_values(exclude_source_types):
        return False
    allowed_types = _normalized_filter_values(document_type_filter)
    if allowed_types and normalize_text(chunk.get("document_type", "")) not in allowed_types:
        return False
    allowed_departments = _normalized_filter_values(department_filter)
    if allowed_departments and normalize_text(chunk.get("department", "")) not in allowed_departments:
        return False
    return True


def _filter_results_by_retrieval_filters(results: list[dict], **filters) -> list[dict]:
    return [doc for doc in results if _matches_retrieval_filters(doc, **filters)]


def _extract_document_number_query(question: str) -> str | None:
    normalized = normalize_text(question)
    searchable = re.sub(r"[_\-.]+", " ", normalized)

    match = re.search(
        r"(?:so|van\s*ban|quyet\s*dinh|quy\s*dinh|quy\s*che|thong\s*bao|qd|qc|tb|vb)\s*[:\-]?\s*(\d{1,6})",
        searchable,
    )
    if not match:
        match = re.search(r"\b(\d{2,6})\s*/\s*(?:qd|vb|tb|qc)\b", searchable)

    return match.group(1) if match else None


def _metadata_search_text(chunk: dict) -> str:
    return " ".join(
        str(chunk.get(field, ""))
        for field in SEARCHABLE_METADATA_FIELDS
        if chunk.get(field) is not None
    )


def _chunk_search_text(chunk: dict) -> str:
    return " ".join([
        str(chunk.get("title", "")),
        _metadata_search_text(chunk),
        str(chunk.get("content", "")),
    ])


def _bm25_document_tokens(chunk: dict) -> list[str]:
    content_tokens = _bm25_tokens(chunk.get("content", ""))
    title_tokens = _bm25_tokens(chunk.get("title", ""))
    metadata_tokens = _bm25_tokens(_metadata_search_text(chunk))

    metadata_repeats = max(1, round(BM25_METADATA_BOOST))
    boosted = content_tokens + title_tokens * 2 + metadata_tokens * metadata_repeats
    for field in ("so_van_ban", "so_van_ban_ngan", "dieu", "muc", "phong_ban"):
        value = chunk.get(field)
        if value is not None:
            boosted += _bm25_tokens(str(value)) * metadata_repeats
    return boosted


def _load_document_index():
    """Load và cache toàn bộ chunk tài liệu cho luồng keyword/IDF search."""
    files = list_documents()
    signature = _document_signature(files)

    if _INDEX_CACHE["signature"] == signature:
        return _INDEX_CACHE["chunks"], _INDEX_CACHE["doc_freq"], _INDEX_CACHE["total_docs"]

    disk_cache = _load_document_index_from_disk(signature)
    if disk_cache is not None:
        chunks, doc_freq, total_docs, skipped_files, bm25_corpus = disk_cache
        _restore_document_index_cache(
            signature,
            chunks,
            doc_freq,
            total_docs,
            skipped_files,
            bm25_corpus,
        )
        _SEARCH_CACHE.clear()
        return chunks, doc_freq, total_docs

    chunks = []
    doc_freq = Counter()
    skipped_files = []
    seen_chunk_hashes = set()

    for file in files:
        if file.get("parse_supported") is False:
            continue

        relative_path = file.get("relative_path") or file["file_name"]
        try:
            file_chunks = build_document_chunks(relative_path)
        except Exception as exc:
            skipped_files.append({
                "relative_path": relative_path,
                "error": str(exc),
            })
            continue

        for chunk in file_chunks:
            chunk_hash = chunk.get("chunk_hash")
            if chunk_hash and chunk_hash in seen_chunk_hashes:
                continue
            if chunk_hash:
                seen_chunk_hashes.add(chunk_hash)
            tokens = get_keywords(_chunk_search_text(chunk))
            chunk["_token_counts"] = Counter(tokens)
            chunk["_token_set"] = set(tokens)
            chunks.append(chunk)
            doc_freq.update(chunk["_token_set"])

    _INDEX_CACHE["signature"] = signature
    _INDEX_CACHE["chunks"] = chunks
    _INDEX_CACHE["doc_freq"] = doc_freq
    _INDEX_CACHE["total_docs"] = len(chunks)
    _INDEX_CACHE["skipped_files"] = skipped_files
    bm25_corpus = [_bm25_document_tokens(chunk) for chunk in chunks]
    _INDEX_CACHE["bm25_corpus"] = bm25_corpus
    _INDEX_CACHE["bm25"] = (
        BM25Okapi(bm25_corpus, k1=BM25_K1, b=BM25_B)
        if any(bm25_corpus)
        else None
    )
    _SEARCH_CACHE.clear()
    _write_document_index_to_disk(
        signature,
        chunks,
        doc_freq,
        len(chunks),
        skipped_files,
        bm25_corpus,
    )

    return chunks, doc_freq, len(chunks)


def _metadata_filter_from_constraints(
    constraints: dict,
    source_type_filter: str | None = None,
    corpus_filter: str | None = None,
    rag_enabled_filter: bool | None = None,
) -> dict:
    metadata_filter = {}

    if source_type_filter:
        metadata_filter["source_type"] = source_type_filter
    if corpus_filter:
        metadata_filter["corpus"] = corpus_filter
    if rag_enabled_filter is not None:
        metadata_filter["rag_enabled"] = rag_enabled_filter

    if constraints.get("so_van_ban"):
        metadata_filter["so_van_ban_ngan"] = str(constraints["so_van_ban"])

    for key in ("dieu", "muc", "chuong"):
        if constraints.get(key) is not None:
            metadata_filter[key] = constraints[key]

    return metadata_filter


def _metadata_value_matches(chunk: dict, key: str, expected) -> bool:
    if expected is None:
        return True

    if key == "ngay":
        expected_date = normalize_date(str(expected))
        searchable_text = normalize_text(
            " ".join([
                str(chunk.get("title", "")),
                str(chunk.get("content", "")),
                str(chunk.get("ngay_ban_hanh", "")),
                str(chunk.get("ngay_hieu_luc", "")),
            ])
        )
        return expected_date in {
            normalize_date(str(chunk.get("ngay_ban_hanh", ""))),
            normalize_date(str(chunk.get("ngay_hieu_luc", ""))),
        } or normalize_text(expected_date) in searchable_text

    if key == "so_van_ban":
        expected_number = str(expected)
        if expected_number in {
            str(chunk.get("so_van_ban_ngan", "")),
            str(chunk.get("so_van_ban", "")).split("/", 1)[0],
        }:
            return True

        searchable_text = normalize_text(
            " ".join([
                str(chunk.get("so_van_ban", "")),
                str(chunk.get("ten_van_ban", "")),
                str(chunk.get("doc_name", "")),
                str(chunk.get("relative_path", "")),
            ])
        )
        searchable_text = re.sub(r"[_\-.]+", " ", searchable_text)
        return any(
            re_pattern.search(searchable_text)
            for re_pattern in (
                re.compile(rf"\bso\s*{re.escape(expected_number)}\b"),
                re.compile(rf"\b{re.escape(expected_number)}\s*/\s*(?:qd|vb|tb|qc)\b"),
                re.compile(rf"\b{re.escape(expected_number)}\b"),
            )
        )

    return str(chunk.get(key, "")).lower() == str(expected).lower()


def _document_number_match_strength(chunk: dict, expected) -> int:
    expected_number = str(expected or "")
    if not expected_number:
        return 0

    if expected_number == str(chunk.get("so_van_ban_ngan", "")):
        return 4

    normalized_identity = normalize_text(
        " ".join([
            str(chunk.get("doc_name", "")),
            str(chunk.get("relative_path", "")),
            str(chunk.get("ten_van_ban", "")),
        ])
    )
    normalized_identity = re.sub(r"[_\-.]+", " ", normalized_identity)
    if re.search(
        rf"\b(?:qd|qc|tb|vb|so|quyet\s*dinh|quy\s*che|thong\s*bao)\s*{re.escape(expected_number)}\b",
        normalized_identity,
    ):
        return 3
    if re.search(
        rf"\b{re.escape(expected_number)}\s*(?:qd|qc|tb|vb|quyet\s*dinh|quy\s*che|thong\s*bao)\b",
        normalized_identity,
    ):
        return 3
    if re.search(rf"\b{re.escape(expected_number)}\b", normalized_identity):
        return 2

    return 1 if _metadata_value_matches(chunk, "so_van_ban", expected_number) else 0


def _metadata_match_count(chunk: dict, constraints: dict) -> int:
    return sum(
        1
        for key, expected in constraints.items()
        if _metadata_value_matches(chunk, key, expected)
    )


def _search_metadata_documents(question: str, limit: int, **retrieval_filters):
    constraints = extract_metadata_constraints(question)
    document_number = _extract_document_number_query(question)
    if document_number and not constraints.get("so_van_ban"):
        constraints["so_van_ban"] = document_number
    if not constraints:
        return [], {}

    chunks, _, _ = _load_document_index()
    results = []

    for chunk in chunks:
        if not _matches_retrieval_filters(chunk, **retrieval_filters):
            continue
        match_count = _metadata_match_count(chunk, constraints)
        if match_count == 0:
            continue

        if constraints.get("so_van_ban") and not _metadata_value_matches(
            chunk, "so_van_ban", constraints["so_van_ban"]
        ):
            continue

        clean_chunk = {
            key: value
            for key, value in chunk.items()
            if not key.startswith("_")
        }
        clean_chunk["score"] = METADATA_EXACT_SCORE + match_count
        clean_chunk["keyword_score"] = score_chunk(
            question,
            f'{chunk.get("title", "")} {_metadata_search_text(chunk)}',
            chunk.get("content", ""),
        )
        clean_chunk["metadata_score"] = match_count
        clean_chunk["document_number_match_strength"] = _document_number_match_strength(
            chunk,
            constraints.get("so_van_ban"),
        )
        clean_chunk["metadata_matched"] = True
        results.append(clean_chunk)

    results.sort(
        key=lambda item: (
            item.get("document_number_match_strength", 0),
            item.get("metadata_score", 0),
            item.get("keyword_score", 0),
            item.get("chunk_index", 0),
        ),
        reverse=True,
    )

    return results[:limit], constraints


def score_chunk(question: str, title: str, content: str, doc_freq=None, total_docs=0, token_counts=None):
    """Tính điểm liên quan keyword/IDF giữa câu hỏi và một chunk tài liệu."""
    query_keywords = get_keywords(question)
    if not query_keywords:
        return 0.0

    title_tokens = set(get_keywords(title))
    content_tokens = set(get_keywords(content))
    token_counts = token_counts or Counter(get_keywords(f"{title} {content}"))
    doc_freq = doc_freq or Counter()
    total_docs = total_docs or 1

    score = 0.0

    for word in query_keywords:
        if word not in token_counts:
            continue

        idf = math.log((1 + total_docs) / (1 + doc_freq.get(word, 0))) + 1
        score += token_counts[word] * idf

        if word in title_tokens:
            score += 4.0 * idf

        if word in content_tokens:
            score += 0.5 * idf

    return round(score, 4)


def _chunk_key(chunk: dict):
    """Tạo khóa định danh chunk để gộp kết quả vector và keyword không bị trùng."""
    content_hash = chunk.get("content_hash")
    chunk_index = chunk.get("chunk_index")

    if content_hash is not None and chunk_index is not None:
        return f"{content_hash}:{chunk_index}"

    return (
        chunk.get("doc_name"),
        chunk.get("title"),
        chunk_index,
        chunk.get("content", "")[:200],
    )


def _search_bm25_documents(
    question: str,
    limit: int,
    source_type_filter: str | None = None,
    **retrieval_filters,
):
    """Search the in-memory document corpus with BM25 and metadata field boosts."""
    chunks, _, _ = _load_document_index()
    bm25 = _INDEX_CACHE.get("bm25")
    query_tokens = _bm25_tokens(question)
    if bm25 is None or not query_tokens:
        return []

    results = []
    for chunk, score in zip(chunks, bm25.get_scores(query_tokens)):
        if not _matches_retrieval_filters(
            chunk,
            source_type_filter=source_type_filter,
            **retrieval_filters,
        ):
            continue
        if float(score) <= BM25_MIN_SCORE:
            continue

        clean_chunk = {
            key: value
            for key, value in chunk.items()
            if not key.startswith("_")
        }
        clean_chunk["score"] = round(float(score), 6)
        clean_chunk["keyword_score"] = round(float(score), 6)
        clean_chunk["bm25_score"] = round(float(score), 6)
        results.append(clean_chunk)

    results.sort(key=lambda item: item["bm25_score"], reverse=True)
    return results[:limit]


def _search_keyword_documents(question: str, limit: int, source_type_filter: str | None = None):
    """Compatibility alias for the old keyword-search helper."""
    return _search_bm25_documents(question, limit, source_type_filter)


def _merge_with_rrf(
    result_sets: list[list[dict]],
    limit: int,
    branch_weights: dict[str, float] | None = None,
):
    """Gộp nhiều danh sách kết quả bằng Reciprocal Rank Fusion dựa trên thứ hạng."""
    fused = {}
    branch_weights = branch_weights or {}

    for result_set in result_sets:
        for rank, chunk in enumerate(result_set, start=1):
            key = _chunk_key(chunk)
            item = fused.setdefault(
                key,
                {
                    "chunk": dict(chunk),
                    "rrf_score": 0.0,
                },
            )
            branches = set(chunk.get("retrieval_branches") or [])
            weight = max(
                (branch_weights.get(branch, 1.0) for branch in branches),
                default=1.0,
            )
            item["rrf_score"] += weight / (RRF_K + rank)

            if "distance" in chunk:
                item["chunk"]["vector_score"] = chunk.get("score")
                item["chunk"]["distance"] = chunk.get("distance")

            if "keyword_score" in chunk:
                item["chunk"]["keyword_score"] = chunk.get("keyword_score")
            if "bm25_score" in chunk:
                item["chunk"]["bm25_score"] = chunk.get("bm25_score")
            merged_branches = set(item["chunk"].get("retrieval_branches") or [])
            merged_branches.update(branches)
            item["chunk"]["retrieval_branches"] = sorted(merged_branches)

    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    results = []

    for item in ranked[:limit]:
        chunk = item["chunk"]
        chunk["rrf_score"] = round(item["rrf_score"], 6)
        chunk["score"] = chunk["rrf_score"]
        results.append(chunk)

    return results


def _compact_debug_sources(results: list[dict], limit: int = 5) -> list[dict]:
    return [
        {
            "title": item.get("title"),
            "doc_name": item.get("doc_name"),
            "relative_path": item.get("relative_path"),
            "phong_ban": item.get("phong_ban"),
            "score": item.get("score"),
            "vector_score": item.get("vector_score"),
            "keyword_score": item.get("keyword_score"),
            "bm25_score": item.get("bm25_score"),
            "rrf_score": item.get("rrf_score"),
            "rerank_score": item.get("rerank_score"),
            "retrieval_branches": item.get("retrieval_branches"),
            "hyde_only": item.get("hyde_only"),
            "distance": item.get("distance"),
            "metadata_matched": item.get("metadata_matched"),
        }
        for item in results[:limit]
    ]


def _hyde_debug_payload(result: dict, preview_chars: int = 500) -> dict:
    payload = {
        key: value
        for key, value in (result or {}).items()
        if key != "text"
    }
    text = " ".join(str((result or {}).get("text") or "").split())
    if text:
        payload["text_preview"] = (
            text[:preview_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
            if len(text) > preview_chars
            else text
        )
    else:
        payload["text_preview"] = None
    return payload


def _probe_title_overlap(question: str, doc: dict) -> int:
    query_terms = set(get_keywords(question))
    if not query_terms:
        return 0
    searchable = normalize_text(
        " ".join([
            str(doc.get("title") or ""),
            _metadata_search_text(doc),
        ])
    )
    return sum(1 for term in query_terms if term in searchable)


def _evaluate_probe_evidence(
    question: str,
    bm25_results: list[dict],
    ann_results: list[dict],
    rrf_results: list[dict],
) -> tuple[bool, dict]:
    top_bm25 = max(
        (float(item.get("bm25_score") or 0) for item in bm25_results),
        default=0.0,
    )
    top_vector = max(
        (
            float(
                item.get("vector_score")
                if item.get("vector_score") is not None
                else item.get("score") or 0
            )
            for item in ann_results
        ),
        default=0.0,
    )
    top_rrf = float(rrf_results[0].get("rrf_score") or 0) if rrf_results else 0.0
    second_rrf = (
        float(rrf_results[1].get("rrf_score") or 0)
        if len(rrf_results) > 1
        else 0.0
    )
    rrf_gap = top_rrf - second_rrf
    title_overlap = max(
        (_probe_title_overlap(question, item) for item in rrf_results[:PROBE_TOP_K]),
        default=0,
    )

    signals = {
        "bm25_passed": top_bm25 >= PROBE_BM25_MIN_SCORE,
        "vector_passed": top_vector >= PROBE_VECTOR_MIN_SCORE,
        "rrf_passed": top_rrf >= PROBE_RRF_MIN_SCORE,
        "rrf_gap_passed": rrf_gap >= PROBE_RRF_SCORE_GAP,
        "title_overlap_passed": title_overlap >= PROBE_MIN_TITLE_OVERLAP,
    }
    signal_count = sum(1 for passed in signals.values() if passed)
    strong_vector_anchor = top_vector >= min(0.95, PROBE_VECTOR_MIN_SCORE + 0.12)
    has_lexical_or_strong_semantic_anchor = (
        signals["bm25_passed"]
        or signals["title_overlap_passed"]
        or strong_vector_anchor
    )
    has_evidence = (
        bool(rrf_results)
        and signal_count >= PROBE_MIN_EVIDENCE_SIGNALS
        and has_lexical_or_strong_semantic_anchor
    )
    debug = {
        "attempted": True,
        "has_confident_evidence": has_evidence,
        "decision": "grounded_hyde" if has_evidence else "clarification_needed",
        "top_bm25_score": round(top_bm25, 6),
        "top_vector_score": round(top_vector, 6),
        "top_rrf_score": round(top_rrf, 6),
        "rrf_score_gap": round(rrf_gap, 6),
        "title_metadata_overlap": title_overlap,
        "signal_count": signal_count,
        "required_signal_count": PROBE_MIN_EVIDENCE_SIGNALS,
        "strong_vector_anchor": strong_vector_anchor,
        "signals": signals,
        "thresholds": {
            "bm25": PROBE_BM25_MIN_SCORE,
            "vector": PROBE_VECTOR_MIN_SCORE,
            "rrf": PROBE_RRF_MIN_SCORE,
            "rrf_gap": PROBE_RRF_SCORE_GAP,
            "title_overlap": PROBE_MIN_TITLE_OVERLAP,
        },
        "evidence_sources": _compact_debug_sources(
            rrf_results,
            limit=PROBE_EVIDENCE_TOP_K,
        ),
    }
    return has_evidence, debug


async def search_documents(
    question: str,
    debug: dict | None = None,
    source_type_filter: str | None = None,
    ambiguity_decision: dict | None = None,
    corpus_filter: str | None = None,
    rag_enabled_filter: bool | None = None,
    exclude_document_names=None,
    exclude_source_types=None,
    document_type_filter=None,
    department_filter=None,
):
    """Run query expansion, BM25, Chroma ANN, RRF and cross-encoder reranking."""
    ambiguity_decision = ambiguity_decision or {
        "action": DIRECT_RETRIEVAL,
        "topic": None,
        "confidence": 1.0,
        "reason": "retrieval_default_direct",
        "clarifying_question": None,
    }
    ambiguity_action = ambiguity_decision.get("action", DIRECT_RETRIEVAL)
    if ambiguity_action == HYDE_RETRIEVAL:
        ambiguity_decision = {
            **ambiguity_decision,
            "action": DIRECT_RETRIEVAL,
            "original_action": HYDE_RETRIEVAL,
            "reason": "hyde_llm_disabled_for_cap_two",
        }
        ambiguity_action = DIRECT_RETRIEVAL
    retrieval_query = _academic_policy_retrieval_query(question)
    effective_final_top_k = _effective_final_top_k(question, source_type_filter)
    active_filters = {
        "source_type_filter": source_type_filter,
        "corpus_filter": corpus_filter,
        "rag_enabled_filter": rag_enabled_filter,
        "exclude_document_names": exclude_document_names,
        "exclude_source_types": exclude_source_types,
        "document_type_filter": document_type_filter,
        "department_filter": department_filter,
    }
    optional_filters = {
        key: value
        for key, value in active_filters.items()
        if key != "source_type_filter" and value is not None and value != set()
    }
    metadata_search_filters = dict(optional_filters)
    if source_type_filter:
        metadata_search_filters["source_type_filter"] = source_type_filter
    filter_signature = (
        source_type_filter or "",
        corpus_filter or "",
        rag_enabled_filter,
        tuple(sorted(_normalized_filter_values(exclude_document_names))),
        tuple(sorted(_normalized_filter_values(exclude_source_types))),
        tuple(sorted(_normalized_filter_values(document_type_filter))),
        tuple(sorted(_normalized_filter_values(department_filter))),
    )
    signature = _current_document_signature()
    cache_key = _search_cache_key(
        question,
        source_type_filter,
        signature,
        ambiguity_action,
        effective_final_top_k,
        retrieval_query,
        filter_signature,
    )
    cached_results = _get_search_cache(cache_key)
    if cached_results is not None:
        if debug is not None:
            probe_cache_hit = ambiguity_action == PROBE_RETRIEVAL
            debug.update({
                "cache_hit": True,
                "ambiguity": ambiguity_decision,
                "source_type_filter": source_type_filter,
                "active_filters": filter_signature,
                "probe_retrieval": {
                    "attempted": False,
                    "has_confident_evidence": bool(cached_results),
                    "decision": (
                        "cached_results_reused"
                        if cached_results
                        else "clarification_needed"
                    ),
                    "cache_hit": True,
                } if probe_cache_hit else {
                    "attempted": False,
                    "decision": "not_requested",
                    "cache_hit": True,
                },
                "grounded_hyde": {
                    "attempted": False,
                    "status": "cached_retrieval_reused",
                    "cache_hit": True,
                },
                "rrf_results": _compact_debug_sources(cached_results),
                "reranking": {"reason": "cached_retrieval_reused"},
                "final_results_count": len(cached_results),
                "final_sources": _compact_debug_sources(cached_results),
                "fallback_reason": (
                    "probe_insufficient_evidence"
                    if probe_cache_hit and not cached_results
                    else None
                ),
                "skipped_files": deepcopy(_INDEX_CACHE.get("skipped_files", [])),
            })
        return cached_results

    candidate_limit = max(RRF_CANDIDATE_TOP_K, effective_final_top_k)
    metadata_results, metadata_constraints = _search_metadata_documents(
        retrieval_query,
        candidate_limit,
        **metadata_search_filters,
    )
    original_metadata_constraints = extract_metadata_constraints(question)

    # Exact document-number requests must never fall through to a different document.
    if original_metadata_constraints.get("so_van_ban") and metadata_constraints.get("so_van_ban") and not metadata_results:
        final_results = []
        _set_search_cache(cache_key, final_results)
        if debug is not None:
            debug.update({
                "cache_hit": False,
                "metadata_constraints": metadata_constraints,
                "source_type_filter": source_type_filter,
                "active_filters": filter_signature,
                "final_search_query": retrieval_query,
                "effective_final_top_k": effective_final_top_k,
                "metadata_results_count": 0,
                "expanded_queries": [question],
                "expansion": {
                    "attempted": False,
                    "reason": "exact_document_number_not_found",
                },
                "bm25_results": [],
                "bm25_errors": [],
                "ann_results": [],
                "rrf_results": [],
                "reranking": {"reason": "no_exact_document"},
                "final_results_count": 0,
                "final_sources": [],
                "skipped_files": deepcopy(_INDEX_CACHE.get("skipped_files", [])),
            })
        return final_results

    if ambiguity_action == CLARIFICATION_NEEDED:
        if debug is not None:
            debug.update({
                "cache_hit": False,
                "ambiguity": ambiguity_decision,
                "final_search_query": retrieval_query,
                "fallback_reason": "clarification_bypassed_for_retrieval",
            })

    metadata_filter = _metadata_filter_from_constraints(
        metadata_constraints if metadata_results else {},
        source_type_filter,
        corpus_filter,
        rag_enabled_filter,
    )
    for doc in metadata_results:
        doc["retrieval_branches"] = ["metadata"]
    result_sets = [metadata_results] if metadata_results else []
    bm25_debug = []
    bm25_errors = []
    ann_debug = []
    vector_errors = []
    probe_debug = {
        "attempted": False,
        "has_confident_evidence": None,
        "decision": "not_requested",
    }
    hyde_result = {
        "text": "",
        "attempted": False,
        "status": "not_requested",
        "text_hash": None,
        "char_count": 0,
        "word_count": 0,
        "error": None,
        "cache_hit": False,
    }
    grounded_hyde_result = {
        "text": "",
        "attempted": False,
        "status": "not_requested",
        "text_hash": None,
        "char_count": 0,
        "word_count": 0,
        "error": None,
        "cache_hit": False,
    }
    if ambiguity_action == HYDE_RETRIEVAL:
        hyde_result = generate_hyde_document(question)
        if hyde_result.get("status") == "need_clarification":
            if debug is not None:
                debug.update({
                    "cache_hit": False,
                    "ambiguity": ambiguity_decision,
                    "hyde": _hyde_debug_payload(hyde_result),
                    "fallback_reason": "hyde_requested_clarification",
                    "final_results_count": 0,
                    "final_sources": [],
                })
            return []

    bm25_results = []
    bm25_error = None
    try:
        bm25_results = _search_bm25_documents(
            retrieval_query,
            min(BM25_TOP_K, candidate_limit),
            source_type_filter,
            **optional_filters,
        )
    except Exception as exc:
        bm25_error = str(exc)
        bm25_errors.append({"query": question, "error": bm25_error})
    for doc in bm25_results:
        doc["retrieval_branches"] = ["bm25_original"]
    if bm25_results:
        result_sets.append(bm25_results)
    bm25_debug.append({
        "query": question,
        "error": bm25_error,
        "results": _compact_debug_sources(bm25_results, limit=BM25_TOP_K),
    })

    local_business_query = (
        corpus_filter == "local_documents"
        and _local_query_profile(question) == "business"
        and not document_type_filter
    )
    if local_business_query:
        business_bm25_error = None
        try:
            business_bm25_results = _search_bm25_documents(
                retrieval_query,
                min(BM25_TOP_K, candidate_limit),
                source_type_filter,
                **{
                    **optional_filters,
                    "document_type_filter": {"business_document"},
                },
            )
        except Exception as exc:
            business_bm25_results = []
            business_bm25_error = str(exc)
            bm25_errors.append({
                "query": question,
                "branch": "bm25_business",
                "error": business_bm25_error,
            })
        for doc in business_bm25_results:
            doc["retrieval_branches"] = ["bm25_business"]
        if business_bm25_results:
            result_sets.append(business_bm25_results)
        bm25_debug.append({
            "query": question,
            "branch": "bm25_business",
            "error": business_bm25_error,
            "results": _compact_debug_sources(
                business_bm25_results,
                limit=BM25_TOP_K,
            ),
        })

    ann_original = []
    ann_original_error = None
    try:
        ann_limit = ANN_TOP_K
        if (
            corpus_filter == "local_documents"
            and _local_query_profile(question) == "business"
        ):
            ann_limit = max(ann_limit, 30)
        ann_original = search_similar_chunks(
            retrieval_query,
            top_k=min(ann_limit, candidate_limit),
            metadata_filter=metadata_filter if metadata_filter else None,
        )
        ann_original = _filter_results_by_retrieval_filters(
            ann_original,
            **active_filters,
        )
    except Exception as exc:
        ann_original_error = str(exc)
        vector_errors.append({"branch": "ann_original", "error": ann_original_error})
    for doc in ann_original:
        doc["retrieval_branches"] = ["ann_original"]
    if ann_original:
        result_sets.append(ann_original)
    ann_debug.append({
        "branch": "ann_original",
        "error": ann_original_error,
        "results": _compact_debug_sources(ann_original, limit=ANN_TOP_K),
    })

    if local_business_query:
        business_ann_error = None
        business_metadata_filter = {
            **(metadata_filter or {}),
            "document_type": "business_document",
        }
        try:
            business_ann_results = search_similar_chunks(
                retrieval_query,
                top_k=min(max(ANN_TOP_K, 30), candidate_limit),
                metadata_filter=business_metadata_filter,
            )
            business_ann_results = _filter_results_by_retrieval_filters(
                business_ann_results,
                **active_filters,
            )
        except Exception as exc:
            business_ann_results = []
            business_ann_error = str(exc)
            vector_errors.append({
                "branch": "ann_business",
                "error": business_ann_error,
            })
        for doc in business_ann_results:
            doc["retrieval_branches"] = ["ann_business"]
        if business_ann_results:
            result_sets.append(business_ann_results)
        ann_debug.append({
            "branch": "ann_business",
            "error": business_ann_error,
            "results": _compact_debug_sources(
                business_ann_results,
                limit=ANN_TOP_K,
            ),
        })

    ann_hyde = []
    ann_grounded_hyde = []
    fallback_reason = None

    if ambiguity_action == PROBE_RETRIEVAL:
        probe_result_sets = [
            result_set
            for result_set in (bm25_results, ann_original)
            if result_set
        ]
        probe_rrf_results = (
            _merge_with_rrf(probe_result_sets, PROBE_TOP_K)
            if probe_result_sets
            else []
        )
        has_probe_evidence, probe_debug = _evaluate_probe_evidence(
            question,
            bm25_results,
            ann_original,
            probe_rrf_results,
        )
        if not has_probe_evidence:
            final_results = []
            _set_search_cache(cache_key, final_results)
            if debug is not None:
                debug.update({
                    "cache_hit": False,
                    "metadata_constraints": metadata_constraints,
                    "source_type_filter": source_type_filter,
                    "metadata_results_count": len(metadata_results),
                    "ambiguity": ambiguity_decision,
                    "probe_retrieval": probe_debug,
                    "grounded_hyde": _hyde_debug_payload(grounded_hyde_result),
                    "bm25_results": bm25_debug,
                    "bm25_errors": bm25_errors,
                    "ann_results": ann_debug,
                    "bm25_original_results": bm25_debug,
                    "ann_original_results": _compact_debug_sources(
                        ann_original,
                        limit=ANN_TOP_K,
                    ),
                    "ann_grounded_hyde_results": [],
                    "vector_errors": vector_errors,
                    "rrf_results": _compact_debug_sources(
                        probe_rrf_results,
                        limit=PROBE_TOP_K,
                    ),
                    "reranking": {"reason": "probe_insufficient_evidence"},
                    "final_results_count": 0,
                    "final_sources": [],
                    "fallback_reason": "probe_insufficient_evidence",
                    "skipped_files": deepcopy(_INDEX_CACHE.get("skipped_files", [])),
                })
            return []

        grounding_docs = probe_rrf_results[:PROBE_EVIDENCE_TOP_K]
        grounded_hyde_result = generate_grounded_hyde_document(
            question,
            grounding_docs,
        )
        grounded_text = grounded_hyde_result.get("text") or ""
        if grounded_text:
            grounded_error = None
            try:
                ann_grounded_hyde = search_similar_chunks(
                    grounded_text,
                    top_k=min(GROUNDED_HYDE_ANN_TOP_K, candidate_limit),
                    metadata_filter=metadata_filter if metadata_filter else None,
                )
                ann_grounded_hyde = _filter_results_by_retrieval_filters(
                    ann_grounded_hyde,
                    **active_filters,
                )
            except Exception as exc:
                grounded_error = str(exc)
                fallback_reason = "grounded_hyde_ann_error_original_retrieval"
                vector_errors.append({
                    "branch": "ann_grounded_hyde",
                    "error": grounded_error,
                })
            for doc in ann_grounded_hyde:
                doc["retrieval_branches"] = ["ann_grounded_hyde"]
            if ann_grounded_hyde:
                result_sets.append(ann_grounded_hyde)
            ann_debug.append({
                "branch": "ann_grounded_hyde",
                "error": grounded_error,
                "results": _compact_debug_sources(
                    ann_grounded_hyde,
                    limit=GROUNDED_HYDE_ANN_TOP_K,
                ),
            })
        elif grounded_hyde_result.get("status") in {
            "error_direct_fallback",
            "need_clarification",
            "disabled",
            "no_grounding_evidence",
            "ungrounded_output_rejected",
        }:
            fallback_reason = "grounded_hyde_error_original_retrieval"

    if ambiguity_action == HYDE_RETRIEVAL:
        hyde_text = hyde_result.get("text") or ""
        if hyde_text:
            hyde_error = None
            try:
                from app.core.config import HYDE_ANN_TOP_K

                ann_hyde = search_similar_chunks(
                    hyde_text,
                    top_k=min(HYDE_ANN_TOP_K, candidate_limit),
                    metadata_filter=metadata_filter if metadata_filter else None,
                )
                ann_hyde = _filter_results_by_retrieval_filters(
                    ann_hyde,
                    **active_filters,
                )
            except Exception as exc:
                hyde_error = str(exc)
                vector_errors.append({"branch": "ann_hyde", "error": hyde_error})
            for doc in ann_hyde:
                doc["retrieval_branches"] = ["ann_hyde"]
            if ann_hyde:
                result_sets.append(ann_hyde)
            ann_debug.append({
                "branch": "ann_hyde",
                "error": hyde_error,
                "results": _compact_debug_sources(ann_hyde, limit=ANN_TOP_K),
            })

    if result_sets:
        rrf_results = _merge_with_rrf(
            result_sets,
            candidate_limit,
            branch_weights=(
                _local_rrf_branch_weights(question)
                if corpus_filter == "local_documents"
                else None
            ),
        )
        if metadata_results:
            metadata_keys = {_chunk_key(chunk) for chunk in metadata_results}
            rrf_results.sort(
                key=lambda item: (
                    _chunk_key(item) in metadata_keys,
                    item.get("document_number_match_strength", 0),
                    item.get("metadata_score", 0),
                    item.get("rrf_score", 0),
                ),
                reverse=True,
            )
        rerank_pool_size = _policy_rerank_pool_size(
            question,
            effective_final_top_k,
        )
        if corpus_filter == "local_documents":
            rerank_pool_size = max(
                rerank_pool_size,
                min(CROSS_ENCODER_TOP_N, candidate_limit),
            )
        final_results, rerank_debug = rerank_documents(
            question,
            rrf_results,
            final_top_k=rerank_pool_size,
        )
        final_results = _prioritize_policy_results(
            question,
            final_results,
        )
        if corpus_filter == "local_documents":
            final_results = _rank_local_documents(
                question,
                final_results,
                effective_final_top_k,
            )
        else:
            final_results = final_results[:effective_final_top_k]
        for doc in final_results:
            branches = set(doc.get("retrieval_branches") or [])
            doc["hyde_only"] = branches in (
                {"ann_hyde"},
                {"ann_grounded_hyde"},
            )
    else:
        rrf_results = []
        final_results = []
        rerank_debug = {"reason": "no_candidates"}

    _set_search_cache(cache_key, final_results)

    if debug is not None:
        debug.update({
            "cache_hit": False,
            "metadata_constraints": metadata_constraints,
            "source_type_filter": source_type_filter,
            "active_filters": {
                "source_type": source_type_filter,
                "corpus": corpus_filter,
                "rag_enabled": rag_enabled_filter,
                "exclude_document_names": sorted(_normalized_filter_values(exclude_document_names)),
                "exclude_source_types": sorted(_normalized_filter_values(exclude_source_types)),
                "document_types": sorted(_normalized_filter_values(document_type_filter)),
                "departments": sorted(_normalized_filter_values(department_filter)),
            },
            "final_search_query": retrieval_query,
            "effective_final_top_k": effective_final_top_k,
            "metadata_results_count": len(metadata_results),
            "ambiguity": ambiguity_decision,
            "probe_retrieval": probe_debug,
            "expanded_queries": [question],
            "expansion": {
                "attempted": False,
                "reason": "replaced_by_ambiguity_hyde",
            },
            "hyde": _hyde_debug_payload(hyde_result),
            "grounded_hyde": _hyde_debug_payload(grounded_hyde_result),
            "bm25_results": bm25_debug,
            "bm25_errors": bm25_errors,
            "ann_results": ann_debug,
            "bm25_original_results": bm25_debug,
            "ann_original_results": _compact_debug_sources(
                ann_original, limit=ANN_TOP_K
            ),
            "ann_hyde_results": _compact_debug_sources(
                ann_hyde, limit=ANN_TOP_K
            ),
            "ann_grounded_hyde_results": _compact_debug_sources(
                ann_grounded_hyde,
                limit=GROUNDED_HYDE_ANN_TOP_K,
            ),
            "vector_errors": vector_errors,
            "rrf_results": _compact_debug_sources(rrf_results, limit=candidate_limit),
            "reranking": rerank_debug,
            "metadata_sources": _compact_debug_sources(metadata_results),
            "final_results_count": len(final_results),
            "final_sources": _compact_debug_sources(final_results),
            "fallback_reason": fallback_reason or (
                "hyde_error_direct_retrieval"
                if hyde_result.get("status") == "error_direct_fallback"
                else None
            ),
            "skipped_files": deepcopy(_INDEX_CACHE.get("skipped_files", [])),
        })

    return final_results
