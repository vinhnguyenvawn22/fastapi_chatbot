from collections import Counter, OrderedDict
from copy import deepcopy
import math
import re
import time
import unicodedata

from app.controller.document_controller import build_document_chunks, list_documents
from app.core.config import (
    MIN_SEARCH_SCORE,
    RERANK_AMBIGUOUS_QUERY_KEYWORDS,
    RETRIEVAL_CACHE_MAX_ITEMS,
    RETRIEVAL_CACHE_TTL_SECONDS,
    SEARCH_TOP_K,
    VECTOR_FAST_PATH_CONFIDENCE,
    VECTOR_FAST_PATH_SCORE_GAP,
)
from app.data.query_analyzer import extract_metadata_constraints, normalize_date
from app.data.vector_store import search_similar_chunks


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
}
_SEARCH_CACHE = OrderedDict()
RRF_K = 60
HYBRID_CANDIDATE_MULTIPLIER = 4
METADATA_EXACT_SCORE = 100.0


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


def _document_signature(files):
    """Tạo chữ ký từ danh sách file để biết cache keyword index còn hợp lệ hay không."""
    return tuple(
        (file.get("relative_path") or file["file_name"], file["file_size_kb"], file.get("updated_at"))
        for file in files
    )


def _current_document_signature():
    return _document_signature(list_documents())


def _search_cache_key(question: str, source_type_filter: str | None, signature):
    return (
        normalize_text(question),
        source_type_filter or "",
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


def _load_document_index():
    """Load và cache toàn bộ chunk tài liệu cho luồng keyword/IDF search."""
    files = list_documents()
    signature = _document_signature(files)

    if _INDEX_CACHE["signature"] == signature:
        return _INDEX_CACHE["chunks"], _INDEX_CACHE["doc_freq"], _INDEX_CACHE["total_docs"]

    chunks = []
    doc_freq = Counter()

    for file in files:
        file_chunks = build_document_chunks(file.get("relative_path") or file["file_name"])

        for chunk in file_chunks:
            title = chunk.get("title", "")
            content = chunk.get("content", "")
            tokens = get_keywords(f"{title} {content}")
            chunk["_token_counts"] = Counter(tokens)
            chunk["_token_set"] = set(tokens)
            chunks.append(chunk)
            doc_freq.update(chunk["_token_set"])

    _INDEX_CACHE["signature"] = signature
    _INDEX_CACHE["chunks"] = chunks
    _INDEX_CACHE["doc_freq"] = doc_freq
    _INDEX_CACHE["total_docs"] = len(chunks)

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
                str(chunk.get("title", "")),
                str(chunk.get("content", "")),
            ])
        )
        return any(
            re_pattern.search(searchable_text)
            for re_pattern in (
                re.compile(rf"\bso\s*{re.escape(expected_number)}\b"),
                re.compile(rf"\b{re.escape(expected_number)}\s*/\s*(?:qd|vb|tb|qc)\b"),
                re.compile(rf"\b{re.escape(expected_number)}\b"),
            )
        )

    return str(chunk.get(key, "")).lower() == str(expected).lower()


def _metadata_match_count(chunk: dict, constraints: dict) -> int:
    return sum(
        1
        for key, expected in constraints.items()
        if _metadata_value_matches(chunk, key, expected)
    )


def _search_metadata_documents(question: str, limit: int):
    constraints = extract_metadata_constraints(question)
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
            chunk.get("title", ""),
            chunk.get("content", ""),
        )
        clean_chunk["metadata_score"] = match_count
        clean_chunk["metadata_matched"] = True
        results.append(clean_chunk)

    results.sort(
        key=lambda item: (
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


def _search_keyword_documents(question: str, limit: int, source_type_filter: str | None = None):
    """Tìm các chunk liên quan bằng keyword/IDF, bổ sung tốt cho vector search."""
    # Keyword/IDF search giữ vai trò bắt đúng tên riêng, mã văn bản, điều khoản.
    results = []
    expanded_question = apply_uneti_query_expansion(question)
    chunks, doc_freq, total_docs = _load_document_index()

    for chunk in chunks:
        if source_type_filter and chunk.get("source_type") != source_type_filter:
            continue

        score = score_chunk(
            expanded_question,
            chunk.get("title", ""),
            chunk.get("content", ""),
            doc_freq=doc_freq,
            total_docs=total_docs,
            token_counts=chunk.get("_token_counts"),
        )

        if score >= MIN_SEARCH_SCORE:
            clean_chunk = {
                key: value
                for key, value in chunk.items()
                if not key.startswith("_")
            }
            clean_chunk["score"] = score
            clean_chunk["keyword_score"] = score
            results.append(clean_chunk)

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:limit]


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

    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
    results = []

    for item in ranked[:limit]:
        chunk = item["chunk"]
        chunk["score"] = round(item["rrf_score"], 6)
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
            "distance": item.get("distance"),
            "metadata_matched": item.get("metadata_matched"),
        }
        for item in results[:limit]
    ]


def _result_confidence(result: dict) -> float:
    score = result.get("vector_score")
    if score is None:
        score = result.get("score")

    try:
        return float(score or 0)
    except (TypeError, ValueError):
        return 0.0


def _should_use_hybrid_rerank(question: str, vector_results: list[dict]) -> tuple[bool, str]:
    if not vector_results:
        return True, "no_vector_results"

    if len(get_keywords(question)) <= RERANK_AMBIGUOUS_QUERY_KEYWORDS:
        return True, "ambiguous_or_short_query"

    top_score = _result_confidence(vector_results[0])
    second_score = _result_confidence(vector_results[1]) if len(vector_results) > 1 else 0.0
    score_gap = top_score - second_score

    if top_score >= VECTOR_FAST_PATH_CONFIDENCE and score_gap >= VECTOR_FAST_PATH_SCORE_GAP:
        return False, "top_vector_result_confident"

    return True, "close_or_low_confidence_results"


async def search_documents(
    question: str,
    debug: dict | None = None,
    source_type_filter: str | None = None,
):
    """Truy xuất tài liệu theo hybrid search: vector semantic + keyword/IDF + RRF."""
    # Hybrid search: vector bắt ngữ nghĩa, keyword/IDF bắt chính xác thuật ngữ.
    signature = _current_document_signature()
    cache_key = _search_cache_key(question, source_type_filter, signature)
    cached_results = _get_search_cache(cache_key)
    if cached_results is not None:
        if debug is not None:
            debug.update({
                "cache_hit": True,
                "source_type_filter": source_type_filter,
                "final_results_count": len(cached_results),
                "final_sources": _compact_debug_sources(cached_results),
            })
        return cached_results

    candidate_limit = max(SEARCH_TOP_K * HYBRID_CANDIDATE_MULTIPLIER, SEARCH_TOP_K)
    metadata_results, metadata_constraints = _search_metadata_documents(question, candidate_limit)
    metadata_filter = _metadata_filter_from_constraints(metadata_constraints, source_type_filter)
    if metadata_constraints and not metadata_results:
        _set_search_cache(cache_key, [])
        if debug is not None:
            debug.update({
                "cache_hit": False,
                "metadata_constraints": metadata_constraints,
                "source_type_filter": source_type_filter,
                "metadata_results_count": 0,
                "vector_results_count": 0,
                "keyword_results_count": 0,
                "vector_error": None,
                "hybrid_rerank_used": False,
                "rerank_reason": "metadata_constraints_without_match",
                "final_results_count": 0,
                "final_sources": [],
            })
        return []

    vector_results = []
    vector_error = None

    try:
        vector_results = search_similar_chunks(
            question,
            top_k=candidate_limit,
            metadata_filter=metadata_filter if metadata_filter else None,
        )
    except Exception as exc:
        # Nếu embedding/vector store lỗi, keyword search vẫn giữ hệ thống trả lời được.
        vector_error = str(exc)

    use_hybrid_rerank, rerank_reason = _should_use_hybrid_rerank(question, vector_results)
    if metadata_results:
        use_hybrid_rerank = True
        rerank_reason = "metadata_results_need_keyword_merge"

    if not use_hybrid_rerank:
        final_results = vector_results[:SEARCH_TOP_K]
        _set_search_cache(cache_key, final_results)
        if debug is not None:
            debug.update({
                "cache_hit": False,
                "metadata_constraints": metadata_constraints,
                "source_type_filter": source_type_filter,
                "metadata_results_count": len(metadata_results),
                "vector_results_count": len(vector_results),
                "keyword_results_count": 0,
                "vector_error": vector_error,
                "hybrid_rerank_used": False,
                "rerank_reason": rerank_reason,
                "metadata_sources": _compact_debug_sources(metadata_results),
                "vector_sources": _compact_debug_sources(vector_results),
                "keyword_sources": [],
                "final_results_count": len(final_results),
                "final_sources": _compact_debug_sources(final_results),
            })
        return final_results

    keyword_results = _search_keyword_documents(question, candidate_limit, source_type_filter)
    result_sets = [
        result_set
        for result_set in (metadata_results, vector_results, keyword_results)
        if result_set
    ]

    if not result_sets:
        _set_search_cache(cache_key, [])
        if debug is not None:
            debug.update({
                "cache_hit": False,
                "metadata_constraints": metadata_constraints,
                "source_type_filter": source_type_filter,
                "metadata_results_count": len(metadata_results),
                "vector_results_count": len(vector_results),
                "keyword_results_count": len(keyword_results),
                "vector_error": vector_error,
                "hybrid_rerank_used": use_hybrid_rerank,
                "rerank_reason": rerank_reason,
                "metadata_sources": _compact_debug_sources(metadata_results),
                "vector_sources": _compact_debug_sources(vector_results),
                "keyword_sources": _compact_debug_sources(keyword_results),
                "final_results_count": 0,
                "final_sources": [],
            })
        return []

    results = _merge_with_rrf(result_sets, SEARCH_TOP_K)

    if metadata_results:
        metadata_keys = {_chunk_key(chunk) for chunk in metadata_results}
        results.sort(
            key=lambda item: (
                _chunk_key(item) in metadata_keys,
                item.get("metadata_score", 0),
                item.get("keyword_score", 0) or 0,
                item.get("score", 0) or 0,
            ),
            reverse=True,
        )

    final_results = results[:SEARCH_TOP_K]
    _set_search_cache(cache_key, final_results)

    if debug is not None:
        debug.update({
            "cache_hit": False,
            "metadata_constraints": metadata_constraints,
            "source_type_filter": source_type_filter,
            "metadata_results_count": len(metadata_results),
            "vector_results_count": len(vector_results),
            "keyword_results_count": len(keyword_results),
            "vector_error": vector_error,
            "hybrid_rerank_used": use_hybrid_rerank,
            "rerank_reason": rerank_reason,
            "metadata_sources": _compact_debug_sources(metadata_results),
            "vector_sources": _compact_debug_sources(vector_results),
            "keyword_sources": _compact_debug_sources(keyword_results),
            "final_results_count": len(final_results),
            "final_sources": _compact_debug_sources(final_results),
        })

    return final_results
