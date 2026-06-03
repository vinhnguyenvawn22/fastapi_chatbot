from collections import Counter
import math
import re
import unicodedata
from typing import Any

from app.controller.document_controller import build_document_chunks, list_documents
from app.core.config import MIN_SEARCH_SCORE, SEARCH_TOP_K


STOP_WORDS = {
    "là", "gì", "của", "và", "có", "không", "như", "nào",
    "được", "trong", "về", "cho", "các", "những", "một", "này",
    "tôi", "em", "anh", "chị", "hỏi", "muốn", "biết", "thì",
    "ạ", "ơi", "với", "giúp", "xin", "cho", "mình",
}

QUERY_EXPANSION = {
    "support": [
        "đăng ký kỹ thuật hỗ trợ sự kiện",
        "hỗ trợ sự kiện trên support",
        "quy trình đăng ký kỹ thuật",
    ],
    "sự kiện": [
        "đăng ký kỹ thuật hỗ trợ sự kiện",
        "hỗ trợ sự kiện",
        "support",
    ],
    "phần mềm": [
        "quản lý và sử dụng phần mềm",
        "cài đặt phần mềm",
        "ứng dụng phần mềm",
    ],
    "hệ thống mạng": [
        "quản lý sử dụng hệ thống mạng",
        "vận hành hệ thống mạng",
        "kết nối mạng",
        "wifi",
        "internet",
    ],
    "lan wifi": [
        "báo hỏng hệ thống mạng",
        "báo hỏng lan wifi",
        "xử lý sự cố hệ thống mạng",
        "kết nối mạng",
        "wifi",
        "internet",
    ],
    "báo hỏng": [
        "báo hỏng hệ thống mạng",
        "xử lý sự cố",
        "quy trình báo hỏng",
        "support",
    ],
    "camera": [
        "hệ thống camera giám sát",
        "quản lý vận hành khai thác camera",
    ],
    "thiết bị phòng học": [
        "khai thác sử dụng bảo trì thiết bị phòng học",
        "thiết bị trên giảng đường",
        "máy chiếu",
        "loa",
        "micro",
    ],
    "hoãn thi": [
        "vắng mặt dự thi có lý do chính đáng",
        "đơn xin hoãn thi",
        "điểm I",
        "kỳ thi phụ",
    ],
    "phúc khảo": [
        "đơn phúc khảo",
        "xem xét lại điểm thi",
        "thời hạn phúc khảo",
        "nộp đơn phúc khảo",
    ],
    "học phí": [
        "miễn giảm học phí",
        "đóng học phí",
        "hạn nộp học phí",
    ],
    "học bổng": [
        "xét học bổng",
        "điều kiện học bổng",
        "mức học bổng",
    ],
}


INTERNAL_DOCUMENT_KEYWORDS = {
    "quy dinh", "quy che", "quyet dinh", "cong van", "nghi dinh",
    "thong tu", "van ban", "dieu", "muc", "chuong", "khoan",
    "ban hanh", "hieu luc", "so van ban", "don vi ban hanh",
    "hoc phi", "hoc bong", "phuc khao", "hoan thi", "bao luu",
    "tot nghiep", "diem ren luyen", "tin chi", "dang ky hoc phan",
    "mien giam", "ky thi", "thi lai", "diem thi", "diem i",
    "sinh vien", "giang vien", "dao tao", "phong dao tao",
    "thu tuc", "quy trinh", "bieu mau", "don", "xac nhan",
    "support", "su kien", "phan mem", "he thong mang", "wifi",
    "camera", "thiet bi phong hoc", "may chieu", "email", "tai khoan",
    "uneti", "nha truong", "noi quy", "quy tac", "huong dan",
}

DOCUMENT_NUMBER_PREFIXES = (
    "quy dinh",
    "quy che",
    "quyet dinh",
    "cong van",
    "nghi dinh",
    "thong tu",
    "van ban",
    "qd",
    "cv",
)


OUT_OF_SCOPE_KEYWORDS = {
    "bitcoin", "crypto", "tien ao", "thoi tiet", "bong da", "chung khoan",
    "nau an", "phim", "game", "du lich", "mua ve", "gia vang", "ty gia",
    "xem tarot", "tu vi", "xem boi", "nhac", "ca si", "dien thoai nao",
}

GENERAL_QUESTION_KEYWORDS = {
    "xin chao", "hello", "hi", "chao", "cam on", "thanks",
    "ban la ai", "may la ai", "ke chuyen", "viet tho",
}


_INDEX_CACHE: dict[str, Any] = {
    "signature": None,
    "chunks": [],
    "doc_freq": Counter(),
    "total_docs": 0,
}


def normalize_text(text: str = "") -> str:
    """
    Chuẩn hóa text: bỏ dấu tiếng Việt, viết thường, gộp khoảng trắng.
    """
    text = str(text or "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return " ".join(text.lower().split())


NORMALIZED_STOP_WORDS = {normalize_text(word) for word in STOP_WORDS}


def _contains_normalized_keyword(normalized_text: str, keyword: str) -> bool:
    normalized_keyword = normalize_text(keyword)
    if not normalized_keyword:
        return False

    return re.search(
        rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])",
        normalized_text,
    ) is not None


def _contains_any_normalized_keyword(normalized_text: str, keywords: set[str]) -> bool:
    return any(_contains_normalized_keyword(normalized_text, keyword) for keyword in keywords)


def clear_document_index_cache() -> None:
    """
    Xóa cache index local.
    """
    _INDEX_CACHE["signature"] = None
    _INDEX_CACHE["chunks"] = []
    _INDEX_CACHE["doc_freq"] = Counter()
    _INDEX_CACHE["total_docs"] = 0


def apply_uneti_query_expansion(query: str) -> str:
    """
    Mở rộng câu hỏi để tăng khả năng tìm đúng tài liệu.
    """
    normalized_query = normalize_text(query)
    expanded_terms = [query]

    for key, terms in QUERY_EXPANSION.items():
        if normalize_text(key) in normalized_query:
            expanded_terms.extend(terms)

    return " ".join(dict.fromkeys(expanded_terms))


def get_keywords(text: str) -> list[str]:
    """
    Tách từ khóa từ câu/text.
    """
    normalized = normalize_text(text)
    words = re.findall(r"[a-z0-9]+", normalized)

    return [
        word
        for word in words
        if len(word) >= 3 and word not in NORMALIZED_STOP_WORDS
    ]


def _document_signature(files: list[dict[str, Any]]) -> tuple:
    """
    Tạo signature để biết khi nào cần build lại index.
    """
    return tuple(
        (
            file.get("file_name"),
            file.get("file_size_kb"),
            file.get("updated_at"),
        )
        for file in files
    )


def _compact_number(value: Any) -> str:
    """
    Chuẩn hóa số văn bản/điều khoản về dạng chỉ còn chữ số chính.
    Ví dụ: '665/QĐ-ĐHKTKTCN' -> '665'.
    """
    text = normalize_text(str(value or ""))
    match = re.search(r"\b(\d{1,6})\b", text)
    return match.group(1) if match else ""


def _document_number_prefix_pattern() -> str:
    return "|".join(re.escape(prefix) for prefix in DOCUMENT_NUMBER_PREFIXES)


def _contains_document_number(normalized_text: str, number: str) -> bool:
    """
    Match số văn bản bằng ngữ cảnh gần loại văn bản/số hiệu, tránh ăn nhầm năm hoặc số trang.
    """
    number = re.escape(str(number or "").strip())
    if not number:
        return False

    prefixes = _document_number_prefix_pattern()
    separator = r"[\s:_\/\-]*"
    patterns = [
        rf"(?<![a-z0-9])(?:so|{prefixes}){separator}(?:so{separator})?{number}(?![a-z0-9])",
        rf"(?<![a-z0-9]){number}{separator}(?:qd|cv|tt|nd|dh|dhktktcn)(?![a-z0-9])",
    ]

    return any(re.search(pattern, normalized_text) for pattern in patterns)


def _extract_document_number_from_text(text: str) -> str | None:
    """
    Trích số văn bản từ tên file/title/content bằng pattern có ngữ cảnh.
    """
    normalized = normalize_text(text)
    prefixes = _document_number_prefix_pattern()
    separator = r"[\s:_\/\-]*"
    patterns = [
        rf"(?<![a-z0-9])(?:so|{prefixes}){separator}(?:so{separator})?(\d{{1,6}})(?![a-z0-9])",
        rf"(?<![a-z0-9])(\d{{1,6}}){separator}(?:qd|cv|tt|nd|dh|dhktktcn)(?![a-z0-9])",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)

    return None


def _normalize_date(day: str, month: str, year: str) -> str:
    """
    Chuẩn hóa ngày về dd/mm/yyyy.
    """
    if len(year) == 2:
        year = "20" + year

    return f"{day.zfill(2)}/{month.zfill(2)}/{year}"


def _is_valid_date_parts(day: str, month: str, year: str) -> bool:
    try:
        day_value = int(day)
        month_value = int(month)
        year_value = int(("20" + year) if len(year) == 2 else year)
    except Exception:
        return False

    return 1 <= day_value <= 31 and 1 <= month_value <= 12 and 1900 <= year_value <= 2100


def _extract_date_from_text(text: str) -> str | None:
    """
    Trích ngày ở các dạng:
    - 15/06/2024
    - 15-06-2024
    - ngày 15 tháng 6 năm 2024
    """
    raw_text = str(text or "")
    normalized = normalize_text(raw_text)

    date_match = re.search(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})", raw_text)
    if date_match:
        day, month, year = date_match.groups()
        if _is_valid_date_parts(day, month, year):
            return _normalize_date(day, month, year)

    compact_date_match = re.search(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)", raw_text)
    if compact_date_match:
        day, month, year = compact_date_match.groups()
        if _is_valid_date_parts(day, month, year):
            return _normalize_date(day, month, year)

    long_date_match = re.search(
        r"ngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{2,4})",
        normalized,
    )
    if long_date_match:
        day, month, year = long_date_match.groups()
        if _is_valid_date_parts(day, month, year):
            return _normalize_date(day, month, year)

    return None


def _date_variants(date_value: str) -> set[str]:
    """
    Tạo các biến thể ngày để match với content.
    """
    date_value = str(date_value or "").strip()
    variants = {date_value}

    match = re.search(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})", date_value)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year

        dd = day.zfill(2)
        mm = month.zfill(2)

        variants.update(
            {
                f"{dd}/{mm}/{year}",
                f"{int(day)}/{int(month)}/{year}",
                f"{dd}-{mm}-{year}",
                f"{int(day)}-{int(month)}-{year}",
                f"ngay {int(day)} thang {int(month)} nam {year}",
            }
        )

    return {normalize_text(item) for item in variants if item}


def classify_query(question: str) -> str:
    """
    Router/Intent classifier đơn giản cho Data Layer.

    Trả về:
    - internal_document: cần tra tài liệu nội bộ.
    - general_question: câu hỏi chung/giao tiếp, không cần retrieval.
    - out_of_scope: ngoài phạm vi tài liệu nội bộ.
    """
    q = normalize_text(question)

    if not q:
        return "general_question"

    # Nếu câu có dấu hiệu metadata rõ ràng thì luôn coi là câu hỏi tài liệu.
    if extract_query_metadata(question):
        return "internal_document"

    if _contains_any_normalized_keyword(q, OUT_OF_SCOPE_KEYWORDS):
        return "out_of_scope"

    if _contains_any_normalized_keyword(q, GENERAL_QUESTION_KEYWORDS):
        return "general_question"

    if _contains_any_normalized_keyword(q, INTERNAL_DOCUMENT_KEYWORDS):
        return "internal_document"

    # Chatbot này chủ yếu hỏi đáp trên tài liệu nội bộ. Với câu hỏi không rõ intent,
    # vẫn cho retrieval chạy rồi dùng ngưỡng điểm/độ phủ token để loại kết quả yếu.
    return "internal_document"


def extract_query_metadata(question: str) -> dict[str, Any]:
    """
    Trích metadata nghiệp vụ từ câu hỏi để retrieval ưu tiên exact match.

    Ví dụ:
    - "Quy định 665 ban hành ngày nào?" -> {"so_van_ban": "665", "loai_van_ban": "quy định"}
    - "Điều 12 nói gì?" -> {"dieu": "12"}
    - "ngày 15/6/2024" -> {"ngay": "15/06/2024"}
    """
    q = normalize_text(question)
    metadata: dict[str, Any] = {}

    so_van_ban_patterns = [
        r"\b(?:so|quy dinh|quyet dinh|cong van|nghi dinh|thong tu|van ban|qd|cv)\s*(?:so)?\s*[:\-/]?\s*(\d{1,6})\b",
        r"\b(?:so van ban)\s*[:\-/]?\s*(\d{1,6})\b",
    ]

    for pattern in so_van_ban_patterns:
        match = re.search(pattern, q)
        if match:
            metadata["so_van_ban"] = match.group(1)
            break

    # Trường hợp người dùng hỏi: "665 ban hành ngày nào?"
    if "so_van_ban" not in metadata and any(
        keyword in q for keyword in ["ban hanh", "hieu luc", "quy dinh", "quyet dinh", "cong van", "van ban"]
    ):
        fallback_number = re.search(r"\b(\d{3,6})\b", q)
        if fallback_number:
            metadata["so_van_ban"] = fallback_number.group(1)

    dieu_match = re.search(r"\b(?:dieu|đieu)\s+(\d{1,3})\b", q)
    if dieu_match:
        metadata["dieu"] = dieu_match.group(1)

    date_value = _extract_date_from_text(question)
    if date_value:
        metadata["ngay"] = date_value

    for raw_loai, normalized_loai in {
        "quy dinh": "quy định",
        "quy che": "quy chế",
        "quyet dinh": "quyết định",
        "cong van": "công văn",
        "nghi dinh": "nghị định",
        "thong tu": "thông tư",
    }.items():
        if raw_loai in q:
            metadata["loai_van_ban"] = normalized_loai
            break

    return metadata


def _extract_document_metadata(
    chunk: dict[str, Any],
    file: dict[str, Any],
    title: str,
    content: str,
    doc_name: str,
) -> dict[str, Any]:
    """
    Chuẩn hóa metadata nếu chunk/document_controller đã có.
    Nếu chưa có thì cố gắng suy luận nhẹ từ doc_name/title/content.
    Hàm này không thay thế phần parse PDF của document_controller, chỉ giúp Data Layer metadata-aware hơn.
    """
    metadata: dict[str, Any] = {}

    for key in [
        "so_van_ban",
        "ngay_ban_hanh",
        "ngay_hieu_luc",
        "ten_van_ban",
        "don_vi_ban_hanh",
        "loai_van_ban",
    ]:
        value = chunk.get(key) or file.get(key)
        if value not in (None, ""):
            metadata[key] = str(value).strip()

    raw_dieu = chunk.get("dieu")
    if raw_dieu not in (None, ""):
        metadata["dieu"] = _compact_number(raw_dieu) or str(raw_dieu).strip()

    combined_text = f"{doc_name}\n{title}\n{content[:2500]}"
    normalized_combined = normalize_text(combined_text)

    if "dieu" not in metadata:
        dieu_match = re.search(r"\bdieu\s+(\d{1,3})\b", normalized_combined)
        if dieu_match:
            metadata["dieu"] = dieu_match.group(1)

    if "so_van_ban" not in metadata:
        so_van_ban = _extract_document_number_from_text(normalized_combined)
        if so_van_ban:
            metadata["so_van_ban"] = so_van_ban

    if "ngay_ban_hanh" not in metadata:
        ban_hanh_match = re.search(
            r"(?:ban hanh|ky ngay|ngay ban hanh).{0,80}?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
            normalized_combined,
        )
        if ban_hanh_match:
            metadata["ngay_ban_hanh"] = _extract_date_from_text(ban_hanh_match.group(1)) or ban_hanh_match.group(1)

    if "ngay_ban_hanh" not in metadata:
        date_value = _extract_date_from_text(combined_text)
        if date_value:
            metadata["ngay_ban_hanh"] = date_value

    if "ngay_hieu_luc" not in metadata:
        hieu_luc_match = re.search(
            r"(?:hieu luc|co hieu luc).{0,80}?(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})",
            normalized_combined,
        )
        if hieu_luc_match:
            metadata["ngay_hieu_luc"] = _extract_date_from_text(hieu_luc_match.group(1)) or hieu_luc_match.group(1)

    if "loai_van_ban" not in metadata:
        for raw_loai, display_loai in {
            "quy dinh": "quy định",
            "quy che": "quy chế",
            "quyet dinh": "quyết định",
            "cong van": "công văn",
            "nghi dinh": "nghị định",
            "thong tu": "thông tư",
        }.items():
            if raw_loai in normalized_combined:
                metadata["loai_van_ban"] = display_loai
                break

    if "ten_van_ban" not in metadata and title:
        metadata["ten_van_ban"] = title

    return metadata


def _load_document_index() -> tuple[list[dict[str, Any]], Counter, int]:
    """
    Đọc tài liệu trong uploads/ChatAI và build index local.
    Nếu file PDF nào lỗi thì bỏ qua file đó, không làm crash API.
    """
    try:
        files = list_documents()
    except Exception as exc:
        print(f"[ERROR] Không đọc được danh sách tài liệu: {exc}")
        return [], Counter(), 0

    signature = _document_signature(files)

    if _INDEX_CACHE["signature"] == signature:
        return (
            _INDEX_CACHE["chunks"],
            _INDEX_CACHE["doc_freq"],
            _INDEX_CACHE["total_docs"],
        )

    chunks: list[dict[str, Any]] = []
    doc_freq: Counter = Counter()

    for file in files:
        file_name = file.get("file_name")
        if not file_name:
            continue

        try:
            file_chunks = build_document_chunks(file_name)
        except Exception as exc:
            print(f"[WARN] Bỏ qua file lỗi: {file_name} | {exc}")
            continue

        for chunk in file_chunks:
            try:
                title = str(chunk.get("title") or "Không rõ mục")
                content = str(chunk.get("content") or "")

                if not content.strip():
                    continue

                doc_name = str(chunk.get("doc_name") or file_name)
                file_path = str(chunk.get("file_path") or file.get("file_path") or "")

                metadata = _extract_document_metadata(
                    chunk=chunk,
                    file=file,
                    title=title,
                    content=content,
                    doc_name=doc_name,
                )

                tokens = get_keywords(
                    " ".join(
                        [
                            title,
                            content,
                            doc_name,
                            str(metadata.get("so_van_ban", "")),
                            str(metadata.get("loai_van_ban", "")),
                            str(metadata.get("don_vi_ban_hanh", "")),
                        ]
                    )
                )
                token_counts = Counter(tokens)
                token_set = set(tokens)

                clean_chunk = {
                    "doc_name": doc_name,
                    "title": title,
                    "dieu": metadata.get("dieu", chunk.get("dieu")),
                    "content": content,
                    "file_path": file_path,
                    "chunk_index": chunk.get("chunk_index", 0),
                    "so_van_ban": metadata.get("so_van_ban"),
                    "ngay_ban_hanh": metadata.get("ngay_ban_hanh"),
                    "ngay_hieu_luc": metadata.get("ngay_hieu_luc"),
                    "ten_van_ban": metadata.get("ten_van_ban"),
                    "don_vi_ban_hanh": metadata.get("don_vi_ban_hanh"),
                    "loai_van_ban": metadata.get("loai_van_ban"),
                    "_token_counts": token_counts,
                    "_token_set": token_set,
                }

                chunks.append(clean_chunk)
                doc_freq.update(token_set)

            except Exception as exc:
                print(f"[WARN] Bỏ qua chunk lỗi trong file {file_name}: {exc}")
                continue

    _INDEX_CACHE["signature"] = signature
    _INDEX_CACHE["chunks"] = chunks
    _INDEX_CACHE["doc_freq"] = doc_freq
    _INDEX_CACHE["total_docs"] = len(chunks)

    return chunks, doc_freq, len(chunks)


def _score_metadata_match(
    query_metadata: dict[str, Any],
    chunk: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Chấm điểm exact match theo metadata nghiệp vụ.
    Exact match metadata được ưu tiên cao hơn semantic/keyword.
    """
    if not query_metadata:
        return 0.0, []

    score = 0.0
    matched_fields: list[str] = []

    title = str(chunk.get("title") or "")
    content = str(chunk.get("content") or "")
    doc_name = str(chunk.get("doc_name") or "")
    combined = normalize_text(f"{doc_name}\n{title}\n{content[:2500]}")

    query_so_van_ban = _compact_number(query_metadata.get("so_van_ban"))
    doc_so_van_ban = _compact_number(chunk.get("so_van_ban"))

    if query_so_van_ban:
        if doc_so_van_ban == query_so_van_ban or _contains_document_number(combined, query_so_van_ban):
            score += 100.0
            matched_fields.append("so_van_ban")

    query_dieu = _compact_number(query_metadata.get("dieu"))
    doc_dieu = _compact_number(chunk.get("dieu"))

    if query_dieu:
        if doc_dieu == query_dieu or re.search(rf"\bdieu\s+{re.escape(query_dieu)}\b", combined):
            score += 45.0
            matched_fields.append("dieu")

    query_date = str(query_metadata.get("ngay") or "")
    if query_date:
        date_variants = _date_variants(query_date)
        doc_dates = " ".join(
            [
                normalize_text(str(chunk.get("ngay_ban_hanh") or "")),
                normalize_text(str(chunk.get("ngay_hieu_luc") or "")),
                combined,
            ]
        )

        if any(date_variant and date_variant in doc_dates for date_variant in date_variants):
            score += 35.0
            matched_fields.append("ngay")

    query_loai = normalize_text(str(query_metadata.get("loai_van_ban") or ""))
    doc_loai = normalize_text(str(chunk.get("loai_van_ban") or ""))

    if query_loai:
        if query_loai == doc_loai or query_loai in combined:
            score += 12.0
            matched_fields.append("loai_van_ban")

    return score, matched_fields


def _required_strict_metadata_fields(query_metadata: dict[str, Any]) -> set[str]:
    """
    Các field này nếu người dùng hỏi rõ thì kết quả phải khớp đủ, không chỉ khớp một phần.
    """
    return {key for key in ["so_van_ban", "dieu", "ngay"] if query_metadata.get(key)}


def _query_overlap_stats(query: str, chunk: dict[str, Any]) -> tuple[int, int, float]:
    query_tokens = set(get_keywords(query))
    if not query_tokens:
        return 0, 0, 0.0

    chunk_tokens = chunk.get("_token_set")
    if not isinstance(chunk_tokens, set):
        chunk_tokens = set(
            get_keywords(
                " ".join(
                    [
                        str(chunk.get("title") or ""),
                        str(chunk.get("content") or ""),
                        str(chunk.get("doc_name") or ""),
                    ]
                )
            )
        )

    matched_count = len(query_tokens.intersection(chunk_tokens))
    coverage = matched_count / max(len(query_tokens), 1)
    return matched_count, len(query_tokens), coverage


def _chunk_search_text(chunk: dict[str, Any]) -> str:
    return normalize_text(
        " ".join(
            [
                str(chunk.get("doc_name") or ""),
                str(chunk.get("title") or ""),
                str(chunk.get("content") or ""),
            ]
        )
    )


def _passes_required_phrase_filters(query: str, chunk: dict[str, Any]) -> bool:
    """
    Với các intent nghiệp vụ rõ ràng, bắt chunk phải chứa cụm then chốt.
    Tránh tài liệu chung về "mạng/thông tin" vượt lên khi user hỏi quy trình báo hỏng.
    """
    normalized_query = normalize_text(query)
    chunk_text = _chunk_search_text(chunk)

    if _contains_normalized_keyword(normalized_query, "bao hong"):
        if not _contains_normalized_keyword(chunk_text, "bao hong"):
            return False

    query_tokens = set(get_keywords(query))
    chunk_tokens = set(get_keywords(chunk_text))

    if {"lan", "wifi"}.issubset(query_tokens):
        if not {"lan", "wifi"}.issubset(chunk_tokens):
            return False

    return True


def _passes_keyword_quality(
    query: str,
    chunk: dict[str, Any],
    matched_fields: list[str],
) -> tuple[bool, int, float]:
    """
    Chặn các chunk chỉ ăn điểm vì một từ quá chung khi không có metadata match.
    """
    matched_count, query_token_count, coverage = _query_overlap_stats(query, chunk)

    if not _passes_required_phrase_filters(query, chunk):
        return False, matched_count, coverage

    if {"so_van_ban", "dieu", "ngay"}.intersection(set(matched_fields)):
        return True, matched_count, coverage

    if query_token_count == 0:
        return False, matched_count, coverage

    if query_token_count <= 2:
        return matched_count >= 1, matched_count, coverage

    return matched_count >= 2 or coverage >= 0.35, matched_count, coverage


def score_chunk(
    question: str,
    title: str,
    content: str,
    doc_name: str = "",
    doc_freq: Counter | None = None,
    total_docs: int = 0,
    token_counts: Counter | None = None,
) -> float:
    """
    Tính điểm liên quan theo keyword.
    Boost theo title, content, doc_name.
    """
    query_keywords = get_keywords(question)
    if not query_keywords:
        return 0.0

    title_tokens = set(get_keywords(title))
    content_tokens = set(get_keywords(content))
    doc_name_tokens = set(get_keywords(doc_name))

    token_counts = token_counts or Counter(get_keywords(f"{title} {content} {doc_name}"))
    doc_freq = doc_freq or Counter()
    total_docs = total_docs or 1

    score = 0.0

    normalized_question = normalize_text(question)
    normalized_title = normalize_text(title)
    normalized_content = normalize_text(content)
    normalized_doc_name = normalize_text(doc_name)

    if normalized_question and normalized_question in normalized_title:
        score += 30.0

    if normalized_question and normalized_question in normalized_content:
        score += 20.0

    if normalized_question and normalized_question in normalized_doc_name:
        score += 10.0

    for word in query_keywords:
        idf = math.log((1 + total_docs) / (1 + doc_freq.get(word, 0))) + 1

        if word in title_tokens:
            score += 8.0 * idf

        if word in content_tokens:
            score += 5.0 * idf

        if word in doc_name_tokens:
            score += 2.0 * idf

        score += token_counts.get(word, 0) * idf

    return round(score, 4)


def _make_highlight(content: str, max_chars: int = 1200) -> dict[str, list[str]]:
    """
    Tạo highlight giống format Elasticsearch để prompt_builder dùng được.
    """
    clean_content = " ".join(str(content or "").split())
    return {"content": [clean_content[:max_chars]]}


async def search_documents(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Tìm kiếm tài liệu đã upload.

    Nâng cấp chính:
    - Có Query Router/Intent Classifier.
    - Có extract metadata từ câu hỏi.
    - Ưu tiên exact match theo số văn bản/ngày/điều trước keyword.
    - Nếu câu hỏi có metadata rõ nhưng không match metadata nào thì trả [] để tránh bịa.
    """
    try:
        query = (query or "").strip()
        if not query:
            return []

        intent = classify_query(query)
        if intent != "internal_document":
            return []

        query_metadata = extract_query_metadata(query)
        expanded_query = apply_uneti_query_expansion(query)
        chunks, doc_freq, total_docs = _load_document_index()

        results: list[dict[str, Any]] = []

        try:
            min_score = float(MIN_SEARCH_SCORE)
        except Exception:
            min_score = 0.1

        for chunk in chunks:
            try:
                title = str(chunk.get("title") or "")
                content = str(chunk.get("content") or "")
                doc_name = str(chunk.get("doc_name") or "")

                keyword_score = score_chunk(
                    question=expanded_query,
                    title=title,
                    content=content,
                    doc_name=doc_name,
                    doc_freq=doc_freq,
                    total_docs=total_docs,
                    token_counts=chunk.get("_token_counts"),
                )

                metadata_score, matched_fields = _score_metadata_match(query_metadata, chunk)
                passes_quality, matched_token_count, token_coverage = _passes_keyword_quality(
                    query=query,
                    chunk=chunk,
                    matched_fields=matched_fields,
                )
                if not passes_quality:
                    continue

                final_score = keyword_score + metadata_score

                if final_score >= min_score:
                    clean_chunk = {
                        key: value
                        for key, value in chunk.items()
                        if not key.startswith("_") and value not in (None, "")
                    }

                    clean_chunk["_score"] = round(final_score, 4)
                    clean_chunk["score"] = round(final_score, 4)
                    clean_chunk["keyword_score"] = round(keyword_score, 4)
                    clean_chunk["metadata_score"] = round(metadata_score, 4)
                    clean_chunk["metadata_match"] = bool(matched_fields)
                    clean_chunk["metadata_matched_fields"] = matched_fields
                    clean_chunk["matched_token_count"] = matched_token_count
                    clean_chunk["token_coverage"] = round(token_coverage, 4)
                    clean_chunk["query_intent"] = intent
                    clean_chunk["query_metadata"] = query_metadata
                    clean_chunk["highlight"] = _make_highlight(content)

                    results.append(clean_chunk)

            except Exception as exc:
                print(f"[WARN] Bỏ qua chunk khi search: {exc}")
                continue

        required_fields = _required_strict_metadata_fields(query_metadata)
        if required_fields:
            exact_results = [
                item
                for item in results
                if required_fields.issubset(set(item.get("metadata_matched_fields", [])))
            ]

            # Nếu user hỏi rõ số văn bản/điều/ngày mà không có chunk nào match đủ,
            # không trả kết quả semantic mơ hồ để tránh chatbot trả lời sai.
            if not exact_results:
                return []

            results = exact_results

        results.sort(
            key=lambda item: (
                item.get("metadata_score", 0),
                item.get("_score", 0),
            ),
            reverse=True,
        )

        limit = top_k or SEARCH_TOP_K or 5
        return results[:limit]

    except Exception as exc:
        print(f"[ERROR] search_documents failed: {exc}")
        return []
