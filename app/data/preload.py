import asyncio
import time

from app.core.config import (
    PRELOAD_CROSS_ENCODER,
    PRELOAD_EMBEDDING_MODEL,
    PRELOAD_RAG_COMPONENTS,
)


def _preload_sync() -> dict:
    results = {}

    def run(name, loader):
        started = time.perf_counter()
        try:
            loader()
            results[name] = {
                "status": "ok",
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        except Exception as exc:
            results[name] = {
                "status": "failed",
                "error": str(exc),
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            }

    if PRELOAD_RAG_COMPONENTS:
        from app.data.elasticsearch_client import _load_document_index
        from app.data.vector_store import get_collection

        run("chroma", get_collection)
        run("bm25_index", _load_document_index)
    if PRELOAD_EMBEDDING_MODEL:
        from app.data.embedding_client import get_embedding_model

        run("embedding_model", get_embedding_model)
    if PRELOAD_CROSS_ENCODER:
        from app.data.reranker import get_cross_encoder

        run("cross_encoder", get_cross_encoder)
    return results


async def preload_rag_components() -> dict:
    return await asyncio.to_thread(_preload_sync)
