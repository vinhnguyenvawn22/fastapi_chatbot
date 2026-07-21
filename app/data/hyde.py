from collections import OrderedDict
from copy import deepcopy
import hashlib
import re
import time

from app.core.config import (
    GROUNDED_HYDE_MAX_EVIDENCE_CHARS,
    HYDE_CACHE_MAX_ITEMS,
    HYDE_CACHE_TTL_SECONDS,
    HYDE_ENABLED,
    HYDE_MAX_WORDS,
    HYDE_MODEL,
)
from app.data.gemini_client import generate_content
from app.data.query_analyzer import normalize_text


NEED_CLARIFICATION = "NEED_CLARIFICATION"
_CACHE = OrderedDict()


def clear_hyde_cache() -> None:
    _CACHE.clear()


def _get_cached(key: str) -> dict | None:
    cached = _CACHE.get(key)
    if not cached:
        return None
    created_at, payload = cached
    if time.monotonic() - created_at > HYDE_CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    result = deepcopy(payload)
    result["cache_hit"] = True
    return result


def _set_cached(key: str, payload: dict) -> None:
    stored = deepcopy(payload)
    stored["cache_hit"] = False
    _CACHE[key] = (time.monotonic(), stored)
    _CACHE.move_to_end(key)
    while len(_CACHE) > HYDE_CACHE_MAX_ITEMS:
        _CACHE.popitem(last=False)


def _debug_payload(status: str, text: str = "", error: str | None = None) -> dict:
    return {
        "attempted": status not in {"disabled", "not_requested"},
        "status": status,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
        "char_count": len(text),
        "word_count": len(text.split()),
        "error": error,
        "cache_hit": False,
    }


def generate_hyde_document(question: str) -> dict:
    key = normalize_text(question)
    if not HYDE_ENABLED:
        return {"text": "", **_debug_payload("disabled")}

    cached = _get_cached(key)
    if cached:
        return cached

    prompt = f"""
Bạn là bộ phận tạo tài liệu giả định phục vụ tìm kiếm trong kho văn bản nội bộ của trường đại học.

NHIỆM VỤ:
Dựa trên câu hỏi của người dùng, hãy viết một đoạn văn giả định mô tả loại thông tin có khả năng xuất hiện trong tài liệu liên quan.

QUY TẮC:
- Không trả lời trực tiếp câu hỏi.
- Không tự tạo số văn bản, Điều, Mục, ngày tháng, tên người hoặc đơn vị.
- Không khẳng định thông tin chưa có căn cứ.
- Giữ đúng chủ đề và ý định của câu hỏi.
- Không mở rộng sang chủ đề khác.
- Sử dụng thuật ngữ hành chính có khả năng xuất hiện trong quy định, quy chế hoặc hướng dẫn.
- Độ dài từ 50 đến {HYDE_MAX_WORDS} từ.
- Chỉ trả về đoạn văn giả định, không giải thích, không Markdown.
- Nếu không xác định được chủ đề, trả chính xác NEED_CLARIFICATION.

CÂU HỎI:
{question}

ĐOẠN VĂN GIẢ ĐỊNH:
""".strip()

    try:
        response = generate_content(model=HYDE_MODEL, contents=prompt)
        text = " ".join(str(response.text or "").split()).strip()
        if text.upper() == NEED_CLARIFICATION:
            result = {
                "text": "",
                **_debug_payload("need_clarification"),
            }
        else:
            words = text.split()
            text = " ".join(words[:HYDE_MAX_WORDS])
            result = {
                "text": text,
                **_debug_payload("success", text=text),
            }
    except Exception as exc:
        result = {
            "text": "",
            **_debug_payload("error_direct_fallback", error=str(exc)),
        }

    _set_cached(key, result)
    return result


def _grounding_text(evidence: list[dict]) -> str:
    blocks = []
    total_chars = 0

    for index, doc in enumerate(evidence, start=1):
        metadata = " | ".join(
            str(doc.get(field) or "").strip()
            for field in (
                "title",
                "doc_name",
                "so_van_ban",
                "dieu",
                "muc",
                "phong_ban",
            )
            if doc.get(field)
        )
        content = " ".join(str(doc.get("content") or "").split())
        block = f"Nguồn {index}: {metadata}\n{content}".strip()
        remaining = GROUNDED_HYDE_MAX_EVIDENCE_CHARS - total_chars
        if remaining <= 0:
            break
        blocks.append(block[:remaining])
        total_chars += len(blocks[-1])

    return "\n\n".join(blocks)


def _legal_references(text: str) -> set[tuple[str, str]]:
    normalized = normalize_text(text)
    return {
        (kind, number)
        for kind, number in re.findall(
            r"\b(so|van ban|qd|quyet dinh|dieu|muc|chuong)\s*[:\-]?\s*(\d{1,6})\b",
            normalized,
        )
    }


def generate_grounded_hyde_document(question: str, evidence: list[dict]) -> dict:
    grounding = _grounding_text(evidence)
    if not HYDE_ENABLED:
        return {"text": "", **_debug_payload("disabled")}
    if not grounding:
        return {"text": "", **_debug_payload("no_grounding_evidence")}

    evidence_hash = hashlib.sha256(grounding.encode("utf-8")).hexdigest()
    key = f"grounded:{normalize_text(question)}:{evidence_hash}"
    cached = _get_cached(key)
    if cached:
        return cached

    prompt = f"""
Bạn tạo một đoạn văn giả định để hỗ trợ tìm kiếm trong kho tài liệu nội bộ.

CÂU HỎI:
{question}

BẰNG CHỨNG THĂM DÒ:
{grounding}

QUY TẮC BẮT BUỘC:
- Chỉ sử dụng chủ đề, thuật ngữ, số văn bản, Điều, Mục, đơn vị và thông tin xuất hiện trong bằng chứng.
- Không tự bổ sung dữ kiện, số hiệu, điều khoản, tên văn bản hoặc chủ đề mới.
- Không trả lời trực tiếp cho người dùng; chỉ mô tả đoạn nội dung có khả năng xuất hiện trong tài liệu liên quan.
- Nếu bằng chứng không liên quan đủ rõ đến câu hỏi, trả chính xác NEED_CLARIFICATION.
- Viết từ 40 đến {HYDE_MAX_WORDS} từ, không Markdown và không giải thích.

ĐOẠN VĂN GIẢ ĐỊNH:
""".strip()

    try:
        response = generate_content(model=HYDE_MODEL, contents=prompt)
        text = " ".join(str(response.text or "").split()).strip()
        if text.upper() == NEED_CLARIFICATION:
            result = {
                "text": "",
                **_debug_payload("need_clarification"),
            }
        else:
            text = " ".join(text.split()[:HYDE_MAX_WORDS])
            introduced_references = _legal_references(text) - _legal_references(grounding)
            if introduced_references:
                result = {
                    "text": "",
                    **_debug_payload(
                        "ungrounded_output_rejected",
                        error="LLM introduced legal references not present in evidence",
                    ),
                }
            else:
                result = {
                    "text": text,
                    **_debug_payload("success", text=text),
                }
    except Exception as exc:
        result = {
            "text": "",
            **_debug_payload("error_direct_fallback", error=str(exc)),
        }

    result["grounding_hash"] = evidence_hash
    result["grounding_source_count"] = len(evidence)
    _set_cached(key, result)
    return result
