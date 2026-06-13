from starlette.concurrency import run_in_threadpool

from app.controller.chatbot_controller import _build_sources, _clean_answer_text
from app.data.website_search_client import search_uneti_website


def _add_trace_step(trace: list[dict], name: str, status: str = "ok", **data):
    step = {
        "name": name,
        "status": status,
    }
    step.update(data)
    trace.append(step)


async def search_website_knowledge(request):
    """Tra cuu website UNETI bang pipeline website rieng, co debug nguon."""
    trace = []
    query = str(request.query or "").strip()

    _add_trace_step(
        trace,
        "query_normalization",
        original_length=len(request.query or ""),
        normalized_length=len(query),
        is_empty=not bool(query),
    )

    if not query:
        return {
            "query": request.query,
            "answer": "Vui long nhap cau hoi can tim tren website UNETI.",
            "selected_count": 0,
            "sources": [],
            "trace": trace,
        }

    website_debug = {}
    result = await run_in_threadpool(search_uneti_website, query, website_debug)
    raw_sources = (result.get("sources") or [])[:request.top_k]
    sources = _build_sources(raw_sources, query)

    _add_trace_step(
        trace,
        "website_search",
        debug=website_debug,
        selected_count=len(raw_sources),
        selected_sources=[
            {
                "title": source.get("title"),
                "url": source.get("url"),
                "attachment_url": source.get("attachment_url"),
                "score": source.get("score"),
            }
            for source in raw_sources
        ],
    )

    return {
        "query": query,
        "answer": _clean_answer_text(result.get("answer")),
        "selected_count": len(raw_sources),
        "sources": sources,
        "trace": trace,
    }
