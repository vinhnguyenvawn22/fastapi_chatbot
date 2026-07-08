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
    "gpa": ["điểm trung bình tích lũy", "điểm hệ 4"],
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
DOCUMENT_INDEX_CACHE_VERSION = 1
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
):
    return (
        normalize_text(question),
        source_type_filter or "",
        retrieval_action,
        signature,
        SEARCH_TOP_K,
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


def _metadata_filter_from_constraints(constraints: dict, source_type_filter: str | None = None) -> dict:
    metadata_filter = {}

    if source_type_filter:
        metadata_filter["source_type"] = source_type_filter

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


def _search_metadata_documents(question: str, limit: int):
    constraints = extract_metadata_constraints(question)
    document_number = _extract_document_number_query(question)
    if document_number and not constraints.get("so_van_ban"):
        constraints["so_van_ban"] = document_number
    if not constraints:
        return [], {}

    chunks, _, _ = _load_document_index()
    results = []

    for chunk in chunks:
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
):
    """Search the in-memory document corpus with BM25 and metadata field boosts."""
    chunks, _, _ = _load_document_index()
    bm25 = _INDEX_CACHE.get("bm25")
    query_tokens = _bm25_tokens(question)
    if bm25 is None or not query_tokens:
        return []

    results = []
    for chunk, score in zip(chunks, bm25.get_scores(query_tokens)):
        if source_type_filter and chunk.get("source_type") != source_type_filter:
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


def _merge_with_rrf(result_sets: list[list[dict]], limit: int):
    """Gộp nhiều danh sách kết quả bằng Reciprocal Rank Fusion dựa trên thứ hạng."""
    fused = {}

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
            item["rrf_score"] += 1 / (RRF_K + rank)

            if "distance" in chunk:
                item["chunk"]["vector_score"] = chunk.get("score")
                item["chunk"]["distance"] = chunk.get("distance")

            if "keyword_score" in chunk:
                item["chunk"]["keyword_score"] = chunk.get("keyword_score")
            if "bm25_score" in chunk:
                item["chunk"]["bm25_score"] = chunk.get("bm25_score")
            branches = set(item["chunk"].get("retrieval_branches") or [])
            branches.update(chunk.get("retrieval_branches") or [])
            item["chunk"]["retrieval_branches"] = sorted(branches)

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
    signature = _current_document_signature()
    cache_key = _search_cache_key(
        question,
        source_type_filter,
        signature,
        ambiguity_action,
    )
    cached_results = _get_search_cache(cache_key)
    if cached_results is not None:
        if debug is not None:
            probe_cache_hit = ambiguity_action == PROBE_RETRIEVAL
            debug.update({
                "cache_hit": True,
                "ambiguity": ambiguity_decision,
                "source_type_filter": source_type_filter,
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

    candidate_limit = max(RRF_CANDIDATE_TOP_K, SEARCH_TOP_K)
    metadata_results, metadata_constraints = _search_metadata_documents(
        question, candidate_limit
    )

    # Exact document-number requests must never fall through to a different document.
    if metadata_constraints.get("so_van_ban") and not metadata_results:
        final_results = []
        _set_search_cache(cache_key, final_results)
        if debug is not None:
            debug.update({
                "cache_hit": False,
                "metadata_constraints": metadata_constraints,
                "source_type_filter": source_type_filter,
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
                "fallback_reason": "clarification_bypassed_for_retrieval",
            })

    metadata_filter = _metadata_filter_from_constraints(
        metadata_constraints if metadata_results else {},
        source_type_filter,
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
            question,
            min(BM25_TOP_K, candidate_limit),
            source_type_filter,
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

    ann_original = []
    ann_original_error = None
    try:
        ann_original = search_similar_chunks(
            question,
            top_k=min(ANN_TOP_K, candidate_limit),
            metadata_filter=metadata_filter if metadata_filter else None,
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
        rrf_results = _merge_with_rrf(result_sets, candidate_limit)
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
        final_results, rerank_debug = rerank_documents(
            question,
            rrf_results,
            final_top_k=min(CROSS_ENCODER_FINAL_TOP_K, SEARCH_TOP_K),
        )
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
