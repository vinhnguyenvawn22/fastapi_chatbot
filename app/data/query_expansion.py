import json
import re

from google import genai

from app.core.config import (
    GEMINI_API_KEY,
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_MAX_VARIANTS,
    QUERY_EXPANSION_MAX_WORDS,
    QUERY_EXPANSION_MODEL,
)
from app.data.query_analyzer import extract_metadata_constraints, normalize_text


_client = genai.Client(api_key=GEMINI_API_KEY)


def should_expand_query(question: str) -> tuple[bool, str]:
    question = str(question or "").strip()
    if not QUERY_EXPANSION_ENABLED:
        return False, "disabled"
    if not question:
        return False, "empty_query"

    metadata = extract_metadata_constraints(question)
    if any(metadata.get(key) is not None for key in ("so_van_ban", "dieu", "muc")):
        return False, "specific_metadata_query"

    if len(question.split()) > QUERY_EXPANSION_MAX_WORDS:
        return False, "query_too_long"

    return True, "eligible"


def _parse_variants(text: str, original: str) -> list[str]:
    raw_text = str(text or "").strip()
    match = re.search(r"\[[\s\S]*\]", raw_text)
    if match:
        raw_text = match.group(0)

    try:
        values = json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        values = [
            line.lstrip("-0123456789. ").strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

    if not isinstance(values, list):
        values = []

    normalized_original = normalize_text(original)
    variants = []
    seen = {normalized_original}
    for value in values:
        variant = " ".join(str(value or "").split()).strip()
        normalized = normalize_text(variant)
        if not variant or not normalized or normalized in seen:
            continue
        variants.append(variant)
        seen.add(normalized)
        if len(variants) >= QUERY_EXPANSION_MAX_VARIANTS:
            break
    return variants


def expand_query_with_gemini(question: str) -> list[str]:
    """Return only expansion variants; the caller always keeps the original query."""
    should_expand, _ = should_expand_query(question)
    if not should_expand:
        return []

    prompt = f"""
Bạn đang tối ưu truy vấn tìm kiếm tài liệu nội bộ tiếng Việt.
    Hãy tạo tối đa {QUERY_EXPANSION_MAX_VARIANTS} biến thể ngắn cho câu hỏi bên dưới.
    Tổng số truy vấn, bao gồm câu hỏi gốc, không được vượt quá 3.

Yêu cầu:
- Giữ nguyên ý định của người dùng.
- Bổ sung từ đồng nghĩa hoặc thuật ngữ hành chính có thể xuất hiện trong văn bản.
- Không thêm số văn bản, Điều, Mục hoặc dữ kiện không có trong câu hỏi.
- Không trả lời câu hỏi.
- Chỉ trả về một JSON array các chuỗi, không thêm giải thích.

Câu hỏi: {question}
""".strip()

    response = _client.models.generate_content(
        model=QUERY_EXPANSION_MODEL,
        contents=prompt,
    )
    return _parse_variants(response.text or "", question)


def build_query_variants(question: str) -> tuple[list[str], dict]:
    """Keep the original query and degrade to it when Gemini expansion fails."""
    question = " ".join(str(question or "").split()).strip()
    should_expand, reason = should_expand_query(question)
    debug = {
        "enabled": QUERY_EXPANSION_ENABLED,
        "attempted": should_expand,
        "reason": reason,
        "error": None,
    }

    variants = []
    if should_expand:
        try:
            variants = expand_query_with_gemini(question)
        except Exception as exc:
            debug["error"] = str(exc)
            debug["reason"] = "gemini_error_fallback_original"

    queries = [question]
    queries.extend(variants[:QUERY_EXPANSION_MAX_VARIANTS])
    debug["expanded_queries"] = queries
    debug["variant_count"] = max(len(queries) - 1, 0)
    return queries, debug
