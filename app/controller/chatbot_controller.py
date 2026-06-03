from starlette.concurrency import run_in_threadpool

from app.data.elasticsearch_client import search_documents
from app.data.gemini_client import ask_gemini
from app.data.prompt_builder import build_context, build_prompt


SOURCE_PREVIEW_CHARS = 240


def _source_preview(content: str) -> str:
    """Create a short one-line preview for API clients."""
    preview = " ".join(str(content or "").split())

    if len(preview) <= SOURCE_PREVIEW_CHARS:
        return preview

    return f"{preview[:SOURCE_PREVIEW_CHARS].rstrip()}..."


def _build_sources(docs):
    """Convert retrieved chunks into a stable public response shape."""
    sources = []

    for doc in docs:
        scores = {}

        for field_name in ("score", "vector_score", "keyword_score", "distance"):
            score = doc.get(field_name)
            if score is None:
                scores[field_name] = None
                continue

            try:
                scores[field_name] = float(score)
            except (TypeError, ValueError):
                scores[field_name] = None

        sources.append({
            "title": doc.get("title"),
            "doc_name": doc.get("doc_name"),
            "chunk_index": doc.get("chunk_index"),
            "score": scores["score"],
            "vector_score": scores["vector_score"],
            "keyword_score": scores["keyword_score"],
            "distance": scores["distance"],
            "preview": _source_preview(doc.get("content", "")),
        })

    return sources


async def handle_chat(request):
    """Handle one chat turn with retrieval, prompt building, and LLM call."""
    question = request.question.strip()

    if not question:
        return {
            "question": request.question,
            "answer": "Vui lòng nhập câu hỏi.",
            "source": None,
            "sources": [],
        }

    docs = await search_documents(question)

    if not docs:
        return {
            "question": question,
            "answer": "Không tìm thấy nội dung phù hợp trong tài liệu.",
            "source": None,
            "sources": [],
        }

    context = build_context(docs)
    prompt = build_prompt(question, context)
    answer = await run_in_threadpool(ask_gemini, prompt)

    best_doc = docs[0]
    source = f'{best_doc.get("title")} - {best_doc.get("doc_name")}'
    sources = _build_sources(docs)

    return {
        "question": question,
        "answer": answer,
        "source": source,
        "sources": sources,
    }
