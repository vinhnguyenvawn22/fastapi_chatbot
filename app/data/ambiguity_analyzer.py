from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import re
import time

from google import genai

from app.core.config import (
    AMBIGUITY_CACHE_MAX_ITEMS,
    AMBIGUITY_CACHE_TTL_SECONDS,
    AMBIGUITY_CLARIFY_THRESHOLD,
    AMBIGUITY_LLM_ENABLED,
    GEMINI_API_KEY,
    HYDE_MIN_TOPIC_CONFIDENCE,
    HYDE_MODEL,
)
from app.data.query_analyzer import extract_metadata_constraints, normalize_text


DIRECT_RETRIEVAL = "direct_retrieval"
HYDE_RETRIEVAL = "hyde_retrieval"
PROBE_RETRIEVAL = "probe_retrieval"
CLARIFICATION_NEEDED = "clarification_needed"

_client = genai.Client(api_key=GEMINI_API_KEY)
_CACHE = OrderedDict()

TOPIC_TERMS = {
    "camera": {"camera", "giam sat", "hinh anh camera"},
    "email": {"email", "thu dien tu", "gmail", "google workspace"},
    "mang": {"mang", "wifi", "internet", "ket noi"},
    "phong_hoc": {"phong hoc", "may chieu", "thiet bi phong hoc"},
    "tot_nghiep": {"tot nghiep", "xet tot nghiep", "ra truong"},
    "hoc_phan": {"hoc phan", "dang ky mon", "dang ky hoc"},
    "lms": {"lms", "hoc truc tuyen"},
    "tai_khoan": {
        "mat khau", "password", "reset password", "quen pass",
        "dang nhap", "login", "account",
    },
}

DOCUMENT_TERMS = {
    "quy dinh", "quy che", "quyet dinh", "thong bao", "huong dan",
}


@dataclass
class AmbiguityDecision:
    action: str
    topic: str | None
    confidence: float
    reason: str
    clarifying_question: str | None = None
    analyzer: str = "rule"
    cache_hit: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def clear_ambiguity_cache() -> None:
    _CACHE.clear()


def _get_cached(key: str) -> AmbiguityDecision | None:
    cached = _CACHE.get(key)
    if not cached:
        return None
    created_at, payload = cached
    if time.monotonic() - created_at > AMBIGUITY_CACHE_TTL_SECONDS:
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    decision = AmbiguityDecision(**deepcopy(payload))
    decision.cache_hit = True
    return decision


def _set_cached(key: str, decision: AmbiguityDecision) -> None:
    payload = decision.to_dict()
    payload["cache_hit"] = False
    _CACHE[key] = (time.monotonic(), payload)
    _CACHE.move_to_end(key)
    while len(_CACHE) > AMBIGUITY_CACHE_MAX_ITEMS:
        _CACHE.popitem(last=False)


def _detect_topic(normalized: str) -> tuple[str | None, float]:
    best_topic = None
    best_score = 0.0
    for topic, terms in TOPIC_TERMS.items():
        matched = sum(1 for term in terms if term in normalized)
        if not matched:
            continue
        score = min(0.72 + (matched - 1) * 0.1, 0.95)
        if score > best_score:
            best_topic = topic
            best_score = score
    return best_topic, best_score


def _looks_garbled(question: str, normalized: str) -> bool:
    if not normalized:
        return True
    digit_mixed = bool(re.search(r"(?=[a-z]*\d)(?=[a-z\d]*[a-z])[a-z\d]+", normalized))
    repeated_noise = bool(re.search(r"([a-z0-9])\1{3,}", normalized))
    unsupported_chars = bool(re.search(r"[^0-9a-zA-ZÀ-ỹ\s,.;:()/_-]", question))
    unknown_shape = digit_mixed or repeated_noise or unsupported_chars
    return unknown_shape and not any(
        term in normalized
        for terms in TOPIC_TERMS.values()
        for term in terms
    )


def _rule_decision(question: str) -> AmbiguityDecision | None:
    normalized = normalize_text(question)
    metadata = extract_metadata_constraints(question)

    if not normalized:
        return AmbiguityDecision(
            CLARIFICATION_NEEDED,
            None,
            0.0,
            "empty_query",
            "Bạn cần hỏi rõ ràng hơn",
        )

    if metadata or any(term in normalized for term in DOCUMENT_TERMS):
        return AmbiguityDecision(
            DIRECT_RETRIEVAL, None, 1.0, "specific_document_or_metadata"
        )

    topic, topic_confidence = _detect_topic(normalized)
    if topic and topic_confidence >= HYDE_MIN_TOPIC_CONFIDENCE:
        return AmbiguityDecision(
            HYDE_RETRIEVAL,
            topic,
            max(topic_confidence, 0.85),
            "known_topic_hyde",
        )

    if not _looks_garbled(question, normalized):
        return AmbiguityDecision(
            HYDE_RETRIEVAL,
            None,
            0.5,
            "eligible_query_hyde",
        )

    return AmbiguityDecision(
        PROBE_RETRIEVAL,
        None,
        0.0,
        "garbled_query_requires_probe",
    )


def _parse_llm_decision(text: str) -> AmbiguityDecision:
    raw = str(text or "").strip()
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        raw = match.group(0)
    payload = json.loads(raw)
    action = payload.get("action")
    if action not in {
        DIRECT_RETRIEVAL,
        HYDE_RETRIEVAL,
        PROBE_RETRIEVAL,
        CLARIFICATION_NEEDED,
    }:
        raise ValueError("invalid ambiguity action")
    confidence = max(0.0, min(float(payload.get("confidence", 0)), 1.0))
    if confidence < AMBIGUITY_CLARIFY_THRESHOLD:
        action = CLARIFICATION_NEEDED
    return AmbiguityDecision(
        action=action,
        topic=payload.get("topic") or None,
        confidence=confidence,
        reason=str(payload.get("reason") or "llm_analysis"),
        clarifying_question=payload.get("clarifying_question") or (
            "Bạn có thể mô tả rõ hơn nội dung hoặc hệ thống cần hỏi không?"
            if action == CLARIFICATION_NEEDED
            else None
        ),
        analyzer="llm",
    )


def _analyze_with_llm(question: str) -> AmbiguityDecision:
    prompt = f"""
Phân tích độ rõ nghĩa của câu hỏi dùng để tra cứu tài liệu nội bộ trường đại học.
Chỉ trả về JSON:
{{
  "action": "direct_retrieval | probe_retrieval | hyde_retrieval | clarification_needed",
  "topic": "chủ đề ngắn hoặc null",
  "confidence": 0.0,
  "reason": "lý do ngắn",
  "clarifying_question": "câu hỏi làm rõ hoặc null"
}}

Quy tắc:
- direct_retrieval nếu câu hỏi đủ rõ để tìm tài liệu.
- probe_retrieval nếu chưa chắc chủ đề nhưng vẫn có thể thăm dò tài liệu trước.
- hyde_retrieval nếu câu hỏi mơ hồ nhưng vẫn có thể tạo câu trả lời giả định an toàn.
- clarification_needed nếu không xác định được đối tượng, có viết tắt khó hiểu hoặc có nhiều cách hiểu.
- Không tự sửa câu hỏi sang một ý nghĩa mới.

Câu hỏi: {question}
""".strip()
    response = _client.models.generate_content(model=HYDE_MODEL, contents=prompt)
    return _parse_llm_decision(response.text or "")


def analyze_ambiguity(question: str) -> AmbiguityDecision:
    key = normalize_text(question)
    cached = _get_cached(key)
    if cached:
        if cached.action in {CLARIFICATION_NEEDED, HYDE_RETRIEVAL}:
            cached.action = PROBE_RETRIEVAL if cached.action == CLARIFICATION_NEEDED else DIRECT_RETRIEVAL
            cached.clarifying_question = None
            cached.reason = f"{cached.reason}_retrieval_only"
        return cached

    rule_decision = _rule_decision(question)
    if rule_decision is not None:
        if rule_decision.action in {CLARIFICATION_NEEDED, HYDE_RETRIEVAL}:
            rule_decision.action = (
                PROBE_RETRIEVAL
                if rule_decision.action == CLARIFICATION_NEEDED
                else DIRECT_RETRIEVAL
            )
            rule_decision.clarifying_question = None
            rule_decision.reason = f"{rule_decision.reason}_retrieval_only"
        _set_cached(key, rule_decision)
        return rule_decision

    if not AMBIGUITY_LLM_ENABLED:
        decision = AmbiguityDecision(
            DIRECT_RETRIEVAL if key else PROBE_RETRIEVAL,
            None,
            0.5 if key else 0.0,
            "ambiguity_llm_disabled",
            analyzer="rule",
        )
        _set_cached(key, decision)
        return decision

    try:
        decision = _analyze_with_llm(question)
        if decision.action in {CLARIFICATION_NEEDED, HYDE_RETRIEVAL}:
            decision.action = (
                PROBE_RETRIEVAL
                if decision.action == CLARIFICATION_NEEDED
                else DIRECT_RETRIEVAL
            )
            decision.clarifying_question = None
            decision.reason = f"{decision.reason}_retrieval_only"
    except Exception:
        decision = AmbiguityDecision(
            DIRECT_RETRIEVAL,
            None,
            0.5,
            "ambiguity_llm_error_rule_fallback",
            analyzer="fallback",
        )
    _set_cached(key, decision)
    return decision
