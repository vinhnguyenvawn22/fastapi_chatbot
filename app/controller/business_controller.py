from app.controller.chatbot_controller import _build_sources, _has_confident_evidence
from app.data.elasticsearch_client import search_documents
from app.data.query_analyzer import QueryIntent, classify_query


def _add_trace_step(trace: list[dict], name: str, status: str = "ok", **data):
    step = {
        "name": name,
        "status": status,
    }
    step.update(data)
    trace.append(step)


def _source_identity(doc: dict):
    for field_name in ("url", "link", "source_url", "attachment_url", "doc_name", "ten_van_ban"):
        value = doc.get(field_name)
        if value:
            return str(value).strip().lower()

    return (
        str(doc.get("title") or "").strip().lower(),
        str(doc.get("chunk_index") or ""),
    )


def _select_representative_docs(docs, limit: int):
    selected = []
    seen = set()

    for doc in docs or []:
        identity = _source_identity(doc)
        if identity in seen:
            continue

        selected.append(doc)
        seen.add(identity)

        if len(selected) >= limit:
            break

    return selected


async def search_business_knowledge(request):
    """Tra cuu nguon nghiep vu da xep hang, khong goi LLM."""
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
            "intent": QueryIntent.OUT_OF_SCOPE.value,
            "candidate_count": 0,
            "selected_count": 0,
            "has_confident_evidence": False,
            "evidence_reason": "empty_query",
            "sources": [],
            "trace": trace,
        }

    analysis = classify_query(query)
    _add_trace_step(
        trace,
        "query_classification",
        intent=analysis.intent.value,
        reason=analysis.reason,
        metadata=analysis.metadata,
    )

    if analysis.intent != QueryIntent.INTERNAL_DOCUMENT:
        _add_trace_step(
            trace,
            "routing",
            status="skipped",
            reason="not_internal_document_query",
        )
        return {
            "query": query,
            "intent": analysis.intent.value,
            "candidate_count": 0,
            "selected_count": 0,
            "has_confident_evidence": False,
            "evidence_reason": "not_internal_document_query",
            "sources": [],
            "trace": trace,
        }

    retrieval_debug = {}
    docs = await search_documents(query, debug=retrieval_debug)
    _add_trace_step(
        trace,
        "retrieval",
        candidate_count=len(docs or []),
        debug=retrieval_debug,
        top_scores=[
            {
                "title": doc.get("title"),
                "doc_name": doc.get("doc_name"),
                "score": doc.get("score"),
                "keyword_score": doc.get("keyword_score"),
                "vector_score": doc.get("vector_score"),
            }
            for doc in (docs or [])[:5]
        ],
    )

    has_evidence, evidence_reason = _has_confident_evidence(query, docs or [])
    _add_trace_step(
        trace,
        "evidence_check",
        status="passed" if has_evidence else "failed",
        has_confident_evidence=has_evidence,
        reason=evidence_reason,
    )

    selected_docs = _select_representative_docs(docs, request.top_k)
    sources = _build_sources(selected_docs)
    _add_trace_step(
        trace,
        "source_ranking",
        selected_count=len(selected_docs),
        selected_sources=[
            {
                "title": doc.get("title"),
                "doc_name": doc.get("doc_name"),
                "score": doc.get("score"),
                "keyword_score": doc.get("keyword_score"),
                "vector_score": doc.get("vector_score"),
            }
            for doc in selected_docs
        ],
    )

    return {
        "query": query,
        "intent": analysis.intent.value,
        "candidate_count": len(docs or []),
        "selected_count": len(selected_docs),
        "has_confident_evidence": has_evidence,
        "evidence_reason": evidence_reason,
        "sources": sources,
        "trace": trace,
    }
