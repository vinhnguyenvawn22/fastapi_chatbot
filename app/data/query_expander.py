import json
import re
import time
from collections import OrderedDict

from app.core.config import (
    QUERY_EXPANSION_ENABLED,
    QUERY_EXPANSION_CACHE_MAX_ITEMS,
    QUERY_EXPANSION_CACHE_TTL_SECONDS,
    QUERY_EXPANSION_MAX_VARIANTS,
    QUERY_EXPANSION_MAX_WORDS,
)
from app.data.gemini_client import ask_gemini
from app.data.query_analyzer import extract_metadata_constraints, normalize_text


LLM_ERROR_MARKERS = (
    "429",
    "503",
    "resource_exhausted",
    "unavailable",
    "vuot gioi han su dung",
    "vượt giới hạn sử dụng",
    "loi khi goi gemini api",
    "lỗi khi gọi gemini api",
    "he thong ai dang ban",
    "hệ thống ai đang bận",
)
_EXPANSION_CACHE = OrderedDict()


def clear_expansion_cache():
    _EXPANSION_CACHE.clear()


def _get_cached_expansion(key: str):
    cached = _EXPANSION_CACHE.get(key)
    if not cached:
        return None
    created_at, variants = cached
    if time.monotonic() - created_at > QUERY_EXPANSION_CACHE_TTL_SECONDS:
        _EXPANSION_CACHE.pop(key, None)
        return None
    _EXPANSION_CACHE.move_to_end(key)
    return list(variants)


def _set_cached_expansion(key: str, variants: list[str]):
    _EXPANSION_CACHE[key] = (time.monotonic(), tuple(variants))
    _EXPANSION_CACHE.move_to_end(key)
    while len(_EXPANSION_CACHE) > QUERY_EXPANSION_CACHE_MAX_ITEMS:
        _EXPANSION_CACHE.popitem(last=False)


def should_expand_query(question: str) -> tuple[bool, str]:
    question = " ".join(str(question or "").split())
    if not QUERY_EXPANSION_ENABLED:
        return False, "disabled"
    if not question:
        return False, "empty_query"
    if len(question.split()) > QUERY_EXPANSION_MAX_WORDS:
        return False, "query_too_long"
    constraints = extract_metadata_constraints(question)
    if any(constraints.get(key) is not None for key in ("so_van_ban", "dieu", "muc")):
        return False, "specific_metadata_query"
    return True, "eligible"


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(
        r"\s*```(?:json)?\s*(.*?)\s*```\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


def _is_llm_error_response(text: str) -> bool:
    normalized = normalize_text(text)
    return any(marker in normalized for marker in LLM_ERROR_MARKERS)


def _parse_variants(raw_response: str, original: str) -> tuple[list[str], str]:
    text = str(raw_response or "").strip()
    if not text:
        return [original], "invalid_llm_response"
    if _is_llm_error_response(text):
        return [original], "llm_error"

    text = _strip_code_fence(text)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return [original], "invalid_llm_response"

    if not isinstance(parsed, dict) or not isinstance(parsed.get("queries"), list):
        return [original], "invalid_llm_response"

    unique = [original]
    seen = {normalize_text(original)}
    for candidate in parsed["queries"]:
        if not isinstance(candidate, str):
            continue
        candidate = " ".join(candidate.split()).strip()
        normalized = normalize_text(candidate)
        if not candidate or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
        if len(unique) >= QUERY_EXPANSION_MAX_VARIANTS:
            break
    return unique, "expanded" if len(unique) > 1 else "no_valid_variants"


def expand_query(question: str) -> tuple[list[str], dict]:
    original = " ".join(str(question or "").split())
    eligible, reason = should_expand_query(original)
    debug = {
        "enabled": QUERY_EXPANSION_ENABLED,
        "used": False,
        "reason": reason,
        "queries": [original] if original else [],
        "error": None,
    }
    if not eligible:
        return debug["queries"], debug
    cache_key = normalize_text(original)
    cached = _get_cached_expansion(cache_key)
    if cached is not None:
        debug.update({
            "used": len(cached) > 1,
            "reason": "cache_hit",
            "queries": cached,
            "cache_hit": True,
        })
        return cached, debug

    prompt = (
        "Bạn tối ưu truy vấn tìm kiếm tài liệu hành chính tiếng Việt. "
        f"Hãy tạo tối đa {max(QUERY_EXPANSION_MAX_VARIANTS - 1, 0)} cách diễn đạt tương đương. "
        "Giữ nguyên ý định, tên riêng, số liệu và thuật ngữ; không trả lời câu hỏi. "
        'Chỉ trả JSON dạng {"queries": ["biến thể 1", "biến thể 2"]}.\n'
        f"Câu hỏi: {original}"
    )
    try:
        variants, parse_reason = _parse_variants(ask_gemini(prompt), original)
        debug.update({
            "used": len(variants) > 1,
            "reason": parse_reason,
            "queries": variants,
            "error": (
                "Gemini returned an error response"
                if parse_reason == "llm_error"
                else None
            ),
        })
        debug["cache_hit"] = False
        if parse_reason in {"expanded", "no_valid_variants"}:
            _set_cached_expansion(cache_key, variants)
        return variants, debug
    except Exception as exc:
        debug.update({"reason": "llm_error", "error": str(exc)})
        return [original], debug
