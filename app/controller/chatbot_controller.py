from starlette.concurrency import run_in_threadpool

from app.core.config import MIN_SEARCH_SCORE
from app.data.elasticsearch_client import search_documents
from app.data.gemini_client import ask_gemini
from app.data.prompt_builder import build_context, build_prompt
from app.data.query_analyzer import QueryIntent, classify_query


SOURCE_PREVIEW_CHARS = 240
NO_EVIDENCE_ANSWER = "Không tìm thấy căn cứ đủ rõ trong tài liệu đã cung cấp."
OUT_OF_SCOPE_ANSWER = "Câu hỏi này nằm ngoài phạm vi tài liệu nội bộ hiện có."
GENERAL_ADVICE_ANSWER = "Câu hỏi này không cần tra cứu tài liệu nội bộ. Vui lòng hỏi về quy định, quy trình, văn bản hoặc nội dung trong tài liệu đã cung cấp."
MIN_VECTOR_CONFIDENCE = 0.35


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
            "so_van_ban": doc.get("so_van_ban"),
            "ngay_ban_hanh": doc.get("ngay_ban_hanh"),
            "ngay_hieu_luc": doc.get("ngay_hieu_luc"),
            "ten_van_ban": doc.get("ten_van_ban"),
            "don_vi_ban_hanh": doc.get("don_vi_ban_hanh"),
            "loai_van_ban": doc.get("loai_van_ban"),
            "chuong": doc.get("chuong"),
            "muc": doc.get("muc"),
            "dieu": doc.get("dieu"),
            "chunk_index": doc.get("chunk_index"),
            "score": scores["score"],
            "vector_score": scores["vector_score"],
            "keyword_score": scores["keyword_score"],
            "distance": scores["distance"],
            "preview": _source_preview(doc.get("content", "")),
        })

    return sources


def _has_confident_evidence(docs) -> bool:
    for doc in docs:
        if doc.get("metadata_matched"):
            return True

        keyword_score = doc.get("keyword_score")
        if keyword_score is not None and float(keyword_score) >= MIN_SEARCH_SCORE:
            return True

        vector_score = doc.get("vector_score")
        if vector_score is None and doc.get("distance") is not None:
            vector_score = 1 - float(doc["distance"])

        if vector_score is not None and float(vector_score) >= MIN_VECTOR_CONFIDENCE:
            return True

    return False


async def handle_chat(request):
    """Handle one chat turn with retrieval, prompt building, and LLM call."""
    question = request.question.strip()

    if not question:
        return {
            "question": request.question,
            "answer": "Vui lòng nhập câu hỏi.",
            "source": None,
            "sources": [],
            "intent": QueryIntent.OUT_OF_SCOPE.value,
        }

    analysis = classify_query(question)

    if analysis.intent == QueryIntent.OUT_OF_SCOPE:
        return {
            "question": question,
            "answer": OUT_OF_SCOPE_ANSWER,
            "source": None,
            "sources": [],
            "intent": analysis.intent.value,
        }

    if analysis.intent == QueryIntent.GENERAL_ADVICE:
        return {
            "question": question,
            "answer": GENERAL_ADVICE_ANSWER,
            "source": None,
            "sources": [],
            "intent": analysis.intent.value,
        }

    docs = await search_documents(question)

    if not docs or not _has_confident_evidence(docs):
        return {
            "question": question,
            "answer": NO_EVIDENCE_ANSWER,
            "source": None,
            "sources": _build_sources(docs),
            "intent": analysis.intent.value,
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
        "intent": analysis.intent.value,
    }
