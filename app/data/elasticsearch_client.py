from collections import Counter
import math
import unicodedata

from app.controller.document_controller import build_document_chunks, list_documents
from app.core.config import MIN_SEARCH_SCORE, SEARCH_TOP_K
from app.data.vector_store import search_similar_chunks


STOP_WORDS = {
    "là", "gì", "của", "và", "có", "không", "như", "nào",
    "được", "trong", "về", "cho", "các", "những", "một", "này",
    "tôi", "em", "anh", "chị", "hỏi", "muốn", "biết", "thì",
}
QUERY_EXPANSION = {
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


def normalize_text(text: str = ""):
    text = str(text or "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return " ".join(text.lower().split())


NORMALIZED_STOP_WORDS = {normalize_text(word) for word in STOP_WORDS}


def clear_document_index_cache():
    _INDEX_CACHE["signature"] = None
    _INDEX_CACHE["chunks"] = []
    _INDEX_CACHE["doc_freq"] = Counter()
    _INDEX_CACHE["total_docs"] = 0


def apply_uneti_query_expansion(query: str) -> str:
    normalized_query = normalize_text(query)
    expanded_terms = [query]

    for key, terms in QUERY_EXPANSION.items():
        if normalize_text(key) in normalized_query:
            expanded_terms.extend(terms)

    return " ".join(dict.fromkeys(expanded_terms))


def get_keywords(text: str):
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
    return tuple(
        (file["file_name"], file["file_size_kb"], file.get("updated_at"))
        for file in files
    )


def _load_document_index():
    files = list_documents()
    signature = _document_signature(files)

    if _INDEX_CACHE["signature"] == signature:
        return _INDEX_CACHE["chunks"], _INDEX_CACHE["doc_freq"], _INDEX_CACHE["total_docs"]

    chunks = []
    doc_freq = Counter()

    for file in files:
        file_chunks = build_document_chunks(file["file_name"])

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


def score_chunk(question: str, title: str, content: str, doc_freq=None, total_docs=0, token_counts=None):
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


async def search_documents(question: str):
    # Ưu tiên vector search để tìm các chunk gần nghĩa với câu hỏi.
    try:
        vector_results = search_similar_chunks(question, top_k=SEARCH_TOP_K)
        if vector_results:
            return vector_results
    except Exception:
        # Nếu embedding/vector store lỗi, fallback keyword search giữ hệ thống vẫn trả lời được.
        pass

    # Fallback keyword + IDF cho trường hợp chưa index vector hoặc vector search lỗi.
    results = []
    expanded_question = apply_uneti_query_expansion(question)
    chunks, doc_freq, total_docs = _load_document_index()

    for chunk in chunks:
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
            results.append(clean_chunk)

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:SEARCH_TOP_K]
