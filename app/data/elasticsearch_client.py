from collections import Counter
import math
import unicodedata
from typing import Any

from app.controller.document_controller import build_document_chunks, list_documents
from app.core.config import MIN_SEARCH_SCORE, SEARCH_TOP_K


STOP_WORDS = {
    "là", "gì", "của", "và", "có", "không", "như", "nào",
    "được", "trong", "về", "cho", "các", "những", "một", "này",
    "tôi", "em", "anh", "chị", "hỏi", "muốn", "biết", "thì",
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
    ],
    "camera": [
        "hệ thống camera giám sát",
        "quản lý vận hành khai thác camera",
    ],
    "thiết bị phòng học": [
        "khai thác sử dụng bảo trì thiết bị phòng học",
        "thiết bị trên giảng đường",
    ],
    "hoãn thi": [
        "vắng mặt dự thi có lý do chính đáng",
        "đơn xin hoãn thi",
        "điểm I",
        "kỳ thi phụ",
    ],
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
    words = [
        word.strip(".,;:!?()[]{}\"'")
        for word in normalized.split()
    ]

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

                tokens = get_keywords(f"{title} {content} {doc_name}")
                token_counts = Counter(tokens)
                token_set = set(tokens)

                clean_chunk = {
                    "doc_name": doc_name,
                    "title": title,
                    "dieu": chunk.get("dieu"),
                    "content": content,
                    "file_path": file_path,
                    "chunk_index": chunk.get("chunk_index", 0),
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

    Bản này chạy local trong uploads/ChatAI, không cần Elasticsearch server.
    Hàm có try/except để tránh lỗi 500 khi một file PDF bị lỗi đọc text.
    """
    try:
        query = (query or "").strip()
        if not query:
            return []

        expanded_query = apply_uneti_query_expansion(query)
        chunks, doc_freq, total_docs = _load_document_index()

        results: list[dict[str, Any]] = []

        for chunk in chunks:
            try:
                title = str(chunk.get("title") or "")
                content = str(chunk.get("content") or "")
                doc_name = str(chunk.get("doc_name") or "")

                score = score_chunk(
                    question=expanded_query,
                    title=title,
                    content=content,
                    doc_name=doc_name,
                    doc_freq=doc_freq,
                    total_docs=total_docs,
                    token_counts=chunk.get("_token_counts"),
                )

                try:
                    min_score = float(MIN_SEARCH_SCORE)
                except Exception:
                    min_score = 0.1

                if score >= min_score:
                    clean_chunk = {
                        key: value
                        for key, value in chunk.items()
                        if not key.startswith("_")
                    }

                    clean_chunk["_score"] = score
                    clean_chunk["score"] = score
                    clean_chunk["highlight"] = _make_highlight(content)

                    results.append(clean_chunk)

            except Exception as exc:
                print(f"[WARN] Bỏ qua chunk khi search: {exc}")
                continue

        results.sort(key=lambda item: item.get("_score", 0), reverse=True)

        limit = top_k or SEARCH_TOP_K or 5
        return results[:limit]

    except Exception as exc:
        print(f"[ERROR] search_documents failed: {exc}")
        return []