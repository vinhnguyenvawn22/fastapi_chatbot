from copy import deepcopy
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import time

from app.core.config import BUSINESS_MAPPING_FILE, BUSINESS_MAPPING_MIN_CONFIDENCE
from app.data.query_analyzer import normalize_text


BUSINESS_STOP_WORDS = {
    "toi", "em", "anh", "chi", "ban", "can", "muon", "hoi", "la", "gi",
    "nao", "nhu", "the", "de", "duoc", "khong", "co", "va", "cua", "cho",
    "trong", "tren", "mot", "cac", "nhung", "hay", "neu", "thi",
}

_MAPPING_CACHE = {
    "path": None,
    "mtime": None,
    "records": [],
}


def _mapping_path() -> Path:
    return Path(BUSINESS_MAPPING_FILE).resolve()


def _tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = [
        token.strip(".,;:!?()[]{}\"'")
        for token in re.split(r"\s+", normalized)
        if token.strip(".,;:!?()[]{}\"'")
    ]
    return [
        token
        for token in tokens
        if len(token) >= 2 and token not in BUSINESS_STOP_WORDS
    ]


def _load_mapping_payload(path: Path) -> dict:
    if not path.exists():
        return {"records": []}

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, list):
        return {"records": payload}

    return payload if isinstance(payload, dict) else {"records": []}


def load_business_mapping() -> list[dict]:
    """Load mapping nghiep vu tu JSON va cache theo thoi gian sua file."""
    path = _mapping_path()
    mtime = path.stat().st_mtime if path.exists() else None

    if _MAPPING_CACHE["path"] == str(path) and _MAPPING_CACHE["mtime"] == mtime:
        return deepcopy(_MAPPING_CACHE["records"])

    payload = _load_mapping_payload(path)
    records = payload.get("records") or []

    clean_records = []
    for index, record in enumerate(records, start=1):
        question = str(record.get("question") or "").strip()
        answer = str(record.get("answer") or "").strip()
        if not question or not answer:
            continue

        keywords = record.get("keywords") or []
        if isinstance(keywords, str):
            keywords = [
                keyword.strip()
                for keyword in keywords.split(",")
                if keyword.strip()
            ]

        clean_record = {
            "id": record.get("id") or f"business_mapping_{index}",
            "unit": record.get("unit") or "PCNTT",
            "file_id": record.get("file_id"),
            "source_file": record.get("source_file"),
            "source_location": record.get("source_location"),
            "question": question,
            "answer": answer,
            "keywords": keywords,
            "stt": record.get("stt"),
        }
        clean_records.append(clean_record)

    _MAPPING_CACHE["path"] = str(path)
    _MAPPING_CACHE["mtime"] = mtime
    _MAPPING_CACHE["records"] = clean_records

    return deepcopy(clean_records)


def clear_business_mapping_cache():
    _MAPPING_CACHE["path"] = None
    _MAPPING_CACHE["mtime"] = None
    _MAPPING_CACHE["records"] = []


def _token_overlap_score(query_tokens: set[str], text: str) -> float:
    if not query_tokens:
        return 0.0

    text_tokens = set(_tokenize(text))
    if not text_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def _keyword_phrase_score(normalized_query: str, keywords: list[str]) -> float:
    if not normalized_query or not keywords:
        return 0.0

    normalized_keywords = [
        normalize_text(keyword)
        for keyword in keywords
        if normalize_text(keyword)
    ]
    if not normalized_keywords:
        return 0.0

    hits = 0
    partial_hits = 0
    query_tokens = set(_tokenize(normalized_query))

    for keyword in normalized_keywords:
        keyword_tokens = set(_tokenize(keyword))
        if keyword and (keyword in normalized_query or normalized_query in keyword):
            hits += 1
        elif keyword_tokens and query_tokens and query_tokens & keyword_tokens:
            partial_hits += 1

    return min(1.0, (hits + partial_hits * 0.45) / len(normalized_keywords))


def _score_record(query: str, record: dict) -> tuple[float, dict]:
    normalized_query = normalize_text(query)
    normalized_question = normalize_text(record.get("question") or "")
    query_tokens = set(_tokenize(query))

    question_similarity = SequenceMatcher(
        None,
        normalized_query,
        normalized_question,
    ).ratio()
    question_token_overlap = _token_overlap_score(
        query_tokens,
        record.get("question") or "",
    )
    keyword_score = _keyword_phrase_score(
        normalized_query,
        record.get("keywords") or [],
    )
    answer_overlap = _token_overlap_score(query_tokens, record.get("answer") or "")
    source_overlap = _token_overlap_score(
        query_tokens,
        " ".join([
            str(record.get("file_id") or ""),
            str(record.get("source_file") or ""),
            str(record.get("source_location") or ""),
        ]),
    )

    exact_bonus = 0.0
    if normalized_query and normalized_question:
        if normalized_query == normalized_question:
            exact_bonus = 0.18
        elif normalized_query in normalized_question or normalized_question in normalized_query:
            exact_bonus = 0.1

    confidence = (
        question_similarity * 0.3
        + question_token_overlap * 0.42
        + keyword_score * 0.18
        + answer_overlap * 0.07
        + source_overlap * 0.03
        + exact_bonus
    )

    reasons = {
        "question_similarity": round(question_similarity, 4),
        "question_token_overlap": round(question_token_overlap, 4),
        "keyword_score": round(keyword_score, 4),
        "answer_overlap": round(answer_overlap, 4),
        "source_overlap": round(source_overlap, 4),
        "exact_bonus": round(exact_bonus, 4),
    }

    return min(1.0, round(confidence, 4)), reasons


def _candidate_from_record(record: dict, confidence: float, reasons: dict, rank: int):
    return {
        "rank": rank,
        "confidence": confidence,
        "matched_question": record.get("question"),
        "answer": record.get("answer"),
        "file_id": record.get("file_id"),
        "source_file": record.get("source_file"),
        "source_location": record.get("source_location"),
        "keywords": record.get("keywords") or [],
        "score_details": reasons,
    }


def search_business_mapping(query: str, top_k: int = 3) -> dict:
    started_at = time.monotonic()
    normalized_query = str(query or "").strip()
    top_k = max(1, min(int(top_k or 3), 10))

    if not normalized_query:
        return {
            "query": query,
            "matched": False,
            "answer": None,
            "confidence": 0.0,
            "matched_question": None,
            "file_id": None,
            "source_file": None,
            "source_location": None,
            "keywords": [],
            "fallback_suggestion": "empty_query",
            "candidates": [],
            "debug": {
                "mapping_file": str(_mapping_path()),
                "candidate_count": 0,
                "elapsed_ms": 0,
            },
        }

    scored = []
    records = load_business_mapping()

    for record in records:
        confidence, reasons = _score_record(normalized_query, record)
        scored.append((confidence, reasons, record))

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].get("question_token_overlap", 0),
            item[1].get("keyword_score", 0),
        ),
        reverse=True,
    )

    candidates = [
        _candidate_from_record(record, confidence, reasons, rank)
        for rank, (confidence, reasons, record) in enumerate(scored[:top_k], start=1)
    ]
    best = candidates[0] if candidates else None
    matched = bool(best and best["confidence"] >= BUSINESS_MAPPING_MIN_CONFIDENCE)

    return {
        "query": normalized_query,
        "matched": matched,
        "answer": best["answer"] if matched else None,
        "confidence": best["confidence"] if best else 0.0,
        "matched_question": best["matched_question"] if matched else None,
        "file_id": best["file_id"] if matched else None,
        "source_file": best["source_file"] if matched else None,
        "source_location": best["source_location"] if matched else None,
        "keywords": best["keywords"] if matched else [],
        "fallback_suggestion": None if matched else "fallback_to_rag_or_chat",
        "candidates": candidates,
        "debug": {
            "mapping_file": str(_mapping_path()),
            "candidate_count": len(records),
            "min_confidence": BUSINESS_MAPPING_MIN_CONFIDENCE,
            "elapsed_ms": round((time.monotonic() - started_at) * 1000, 2),
        },
    }
