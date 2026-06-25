import time
from collections import OrderedDict
from functools import lru_cache

from app.core.config import (
    CROSS_ENCODER_ENABLED,
    CROSS_ENCODER_CACHE_MAX_ITEMS,
    CROSS_ENCODER_CACHE_TTL_SECONDS,
    CROSS_ENCODER_MAX_LENGTH,
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_N,
)
from app.data.query_analyzer import normalize_text


_RERANK_CACHE = OrderedDict()


def clear_rerank_cache():
    _RERANK_CACHE.clear()


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_ENCODER_MODEL, max_length=CROSS_ENCODER_MAX_LENGTH)


def _rerank_text(chunk: dict) -> str:
    return "\n".join(
        str(value)
        for value in (
            chunk.get("title"),
            chunk.get("ten_van_ban"),
            chunk.get("so_van_ban"),
            chunk.get("phong_ban"),
            chunk.get("content"),
        )
        if value
    )


def _chunk_cache_key(chunk: dict):
    return (
        chunk.get("content_hash"),
        chunk.get("relative_path") or chunk.get("doc_name"),
        chunk.get("chunk_index"),
        chunk.get("title"),
    )


def _get_cached_score(key):
    cached = _RERANK_CACHE.get(key)
    if not cached:
        return None
    created_at, score = cached
    if time.monotonic() - created_at > CROSS_ENCODER_CACHE_TTL_SECONDS:
        _RERANK_CACHE.pop(key, None)
        return None
    _RERANK_CACHE.move_to_end(key)
    return score


def _set_cached_score(key, score):
    _RERANK_CACHE[key] = (time.monotonic(), float(score))
    _RERANK_CACHE.move_to_end(key)
    while len(_RERANK_CACHE) > CROSS_ENCODER_CACHE_MAX_ITEMS:
        _RERANK_CACHE.popitem(last=False)


def rerank_chunks(question: str, chunks: list[dict]) -> tuple[list[dict], dict]:
    candidates = [dict(chunk) for chunk in chunks[:CROSS_ENCODER_TOP_N]]
    tail = [dict(chunk) for chunk in chunks[CROSS_ENCODER_TOP_N:]]
    debug = {
        "enabled": CROSS_ENCODER_ENABLED,
        "used": False,
        "model": CROSS_ENCODER_MODEL,
        "candidate_count": len(candidates),
        "error": None,
    }
    if not CROSS_ENCODER_ENABLED or len(candidates) < 2:
        debug["reason"] = "disabled_or_insufficient_candidates"
        return candidates + tail, debug

    try:
        query_key = normalize_text(question)
        missing = []
        for chunk in candidates:
            cache_key = (query_key, _chunk_cache_key(chunk), CROSS_ENCODER_MODEL)
            cached_score = _get_cached_score(cache_key)
            if cached_score is None:
                missing.append((chunk, cache_key))
            else:
                chunk["cross_encoder_score"] = cached_score
        if missing:
            scores = get_cross_encoder().predict(
                [(question, _rerank_text(chunk)) for chunk, _ in missing]
            )
            for (chunk, cache_key), score in zip(missing, scores):
                chunk["cross_encoder_score"] = float(score)
                _set_cached_score(cache_key, score)
        candidates.sort(key=lambda item: item["cross_encoder_score"], reverse=True)
        debug.update({
            "used": True,
            "reason": "reranked",
            "cache_hits": len(candidates) - len(missing),
            "model_scored": len(missing),
        })
        return candidates + tail, debug
    except Exception as exc:
        debug.update({"reason": "model_error", "error": str(exc)})
        return candidates + tail, debug
