import re

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.config import (
    MIN_SEARCH_SCORE,
    MIN_VECTOR_CONFIDENCE,
    SHORT_QUERY_MIN_SEARCH_SCORE,
    SHORT_QUERY_MIN_VECTOR_CONFIDENCE,
)
from app.data.business_knowledge import (
    BUSINESS_FAQ_SOURCE_TYPE,
    build_business_faq_answer,
    search_business_sources,
)
from app.data.elasticsearch_client import get_keywords, normalize_text, search_documents
from app.data.gemini_client import ask_gemini
from app.data.prompt_builder import build_context, build_prompt, build_website_prompt
from app.data.query_analyzer import QueryIntent, classify_query
from app.data.trace_logger import RagTrace, load_trace
from app.data.website_search_client import index_uneti_website


SOURCE_PREVIEW_CHARS = 1100
SOURCE_PREVIEW_SENTENCES = 5
NO_WEBSITE_EVIDENCE_ANSWER = "Không tìm thấy thông tin phù hợp trên website UNETI."
NO_EVIDENCE_ANSWER = "Không tìm thấy căn cứ đủ rõ trong tài liệu đã cung cấp."
OUT_OF_SCOPE_ANSWER = "Câu hỏi này nằm ngoài phạm vi tài liệu nội bộ hiện có."
GENERAL_ADVICE_ANSWER = "Câu hỏi này không cần tra cứu tài liệu nội bộ. Vui lòng hỏi về quy định, quy trình, văn bản hoặc nội dung trong tài liệu đã cung cấp."
SHORT_QUERY_KEYWORD_COUNT = 3


def _clean_answer_text(answer: str | None) -> str:
    return str(answer or "").replace("**", "")


def _confidence_from_source(doc: dict) -> tuple[float | None, str | None]:
    if doc.get("metadata_matched"):
        confidence = 1.0
    else:
        confidence_values = []

        vector_score = doc.get("vector_score")
        if vector_score is None and doc.get("distance") is not None:
            vector_score = 1 - float(doc["distance"])

        if vector_score is not None:
            confidence_values.append(max(0.0, min(float(vector_score), 1.0)))

        keyword_score = doc.get("keyword_score")
        if keyword_score is not None:
            keyword_confidence = float(keyword_score) / max(MIN_SEARCH_SCORE * 4, 1)
            confidence_values.append(max(0.0, min(keyword_confidence, 1.0)))

        if not confidence_values:
            return None, None

        confidence = max(confidence_values)

    if confidence >= 0.75:
        label = "Cao"
    elif confidence >= 0.55:
        label = "Trung bình"
    else:
        label = "Thấp"

    return round(confidence, 4), label


def _clean_preview_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"---\s*Trang\s+\d+\s*---", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_preview_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\b([a-zA-Z])\s+([a-zA-Z])\b", r"\1\2", text)
    parts = re.split(r"(?<=[.!?])\s+|(?<=;)\s+|(?=\b[a-z]\)\s+)", text)
    return [part.strip(" -") for part in parts if len(part.strip(" -")) >= 24]


def _score_preview_sentence(sentence: str, question_keywords: list[str]) -> int:
    normalized_sentence = normalize_text(sentence)
    return sum(1 for keyword in set(question_keywords) if keyword in normalized_sentence)


def _shorten_preview_sentence(sentence: str, limit: int = 260) -> str:
    sentence = sentence.strip(" -")
    if len(sentence) <= limit:
        return sentence

    cut_at = sentence.rfind(". ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = sentence.rfind("; ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = sentence.rfind(", ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = sentence.rfind(" ", 0, limit)
    if cut_at < limit * 0.45:
        cut_at = limit

    return sentence[:cut_at].rstrip(" ,;:-") + "."


def _fallback_preview(text: str) -> str:
    if len(text) <= SOURCE_PREVIEW_CHARS:
        return text

    cut_at = text.rfind(". ", 0, SOURCE_PREVIEW_CHARS)
    if cut_at < SOURCE_PREVIEW_CHARS * 0.55:
        cut_at = text.rfind("; ", 0, SOURCE_PREVIEW_CHARS)
    if cut_at < SOURCE_PREVIEW_CHARS * 0.55:
        cut_at = text.rfind(" ", 0, SOURCE_PREVIEW_CHARS)
    if cut_at < SOURCE_PREVIEW_CHARS * 0.55:
        cut_at = SOURCE_PREVIEW_CHARS

    return text[:cut_at].rstrip(" ,;:-") + "."


def _source_preview(content: str, title: str | None = None, question: str | None = None) -> str:
    title_text = _clean_preview_text(title or "")
    content_text = _clean_preview_text(content)
    question_keywords = get_keywords(question or "")
    sentences = _split_preview_sentences(content_text)

    if sentences:
        ranked_sentences = sorted(
            enumerate(sentences),
            key=lambda item: (_score_preview_sentence(item[1], question_keywords), -item[0]),
            reverse=True,
        )
        selected_indexes = [
            index
            for index, sentence in ranked_sentences
            if _score_preview_sentence(sentence, question_keywords) > 0
        ][:SOURCE_PREVIEW_SENTENCES]

        if not selected_indexes:
            selected_indexes = list(range(min(SOURCE_PREVIEW_SENTENCES, len(sentences))))

        selected_indexes = sorted(selected_indexes)
        summary_items = [
            _shorten_preview_sentence(sentences[index])
            for index in selected_indexes
        ]
        summary = "\n".join(f"- {item}" for item in summary_items)
    else:
        summary = content_text

    if "\n- " not in summary:
        summary = _fallback_preview(summary)

    if title_text and title_text.lower() not in summary.lower():
        preview = f"{title_text}\n{summary}"
    else:
        preview = summary

    return f"Tóm tắt nguồn:\n{preview}"


def _build_sources(docs, question: str | None = None):
    sources = []

    for doc in docs:
        confidence, confidence_label = _confidence_from_source(doc)
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
            "url": doc.get("url"),
            "attachment_url": doc.get("attachment_url"),
            "source_type": doc.get("source_type"),
            "relative_path": doc.get("relative_path"),
            "phong_ban": doc.get("phong_ban"),
            "source_root": doc.get("source_root"),
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
            "file_id": doc.get("file_id"),
            "faq_location": doc.get("faq_location"),
            "audience": doc.get("audience"),
            "mapping_relative_path": doc.get("mapping_relative_path"),
            "score": scores["score"],
            "vector_score": scores["vector_score"],
            "keyword_score": scores["keyword_score"],
            "distance": scores["distance"],
            "confidence": confidence,
            "confidence_percent": round(confidence * 100) if confidence is not None else None,
            "confidence_label": confidence_label,
            "preview": _source_preview(doc.get("content", ""), doc.get("title"), question),
        })

    return sources


def _has_confident_evidence(question: str, docs) -> tuple[bool, str]:
    query_keyword_count = len(get_keywords(question))
    keyword_threshold = (
        SHORT_QUERY_MIN_SEARCH_SCORE
        if query_keyword_count < SHORT_QUERY_KEYWORD_COUNT
        else MIN_SEARCH_SCORE
    )
    vector_threshold = (
        SHORT_QUERY_MIN_VECTOR_CONFIDENCE
        if query_keyword_count < SHORT_QUERY_KEYWORD_COUNT
        else MIN_VECTOR_CONFIDENCE
    )

    for doc in docs:
        if doc.get("metadata_matched"):
            return True, "metadata_matched"

        vector_score = doc.get("vector_score")
        if vector_score is None and doc.get("distance") is not None:
            vector_score = 1 - float(doc["distance"])

        if vector_score is not None and float(vector_score) >= vector_threshold:
            return True, "vector_score_passed"

        keyword_score = doc.get("keyword_score")
        if keyword_score is not None and float(keyword_score) >= keyword_threshold:
            return True, "keyword_score_passed"

    return False, "no_confident_source"


def _finalize(trace: RagTrace, response: dict) -> dict:
    if "answer" in response:
        response["answer"] = _clean_answer_text(response["answer"])

    response["trace_id"] = trace.trace_id
    trace.set_response(response)
    trace.save()
    return response


async def _search_website_and_finalize(trace: RagTrace, question: str, intent: str, reason: str):
    website_debug = {}
    try:
        index_result = await run_in_threadpool(index_uneti_website, question, website_debug)
    except Exception as exc:
        trace.add_step("website_search", {
            "status": "index_error",
            "error": str(exc),
        }, {
            "question": question,
            "reason": reason,
        })
        return _finalize(trace, {
            "question": question,
            "answer": NO_WEBSITE_EVIDENCE_ANSWER,
            "source": None,
            "sources": [],
            "intent": intent,
        })

    trace.add_step("website_search", website_debug, {
        "question": question,
        "reason": reason,
    })

    if not index_result.get("indexed_chunks"):
        trace.add_step("website_index", {
            "indexed_chunks": 0,
            "llm_called": False,
            "reason": "no_text_chunks_from_website",
        })
        return _finalize(trace, {
            "question": question,
            "answer": NO_WEBSITE_EVIDENCE_ANSWER,
            "source": None,
            "sources": [],
            "intent": intent,
        })

    retrieval_debug = {}
    docs = await search_documents(
        question,
        debug=retrieval_debug,
        source_type_filter="website_uneti",
    )
    trace.add_step("retrieval_after_website_index", retrieval_debug, {"question": question})

    website_docs = [
        doc
        for doc in docs
        if doc.get("source_type") == "website_uneti"
    ]

    has_evidence = bool(website_docs)
    evidence_reason = "website_indexed_source" if has_evidence else "no_website_chunks_found"
    trace.add_step("evidence_check_after_website_index", {
        "has_confident_evidence": has_evidence,
        "reason": evidence_reason,
        "query_keyword_count": len(get_keywords(question)),
        "website_source_count": len(website_docs),
        "llm_called": bool(website_docs and has_evidence),
    })

    if not website_docs or not has_evidence:
        return _finalize(trace, {
            "question": question,
            "answer": NO_WEBSITE_EVIDENCE_ANSWER,
            "source": None,
            "sources": _build_sources(website_docs, question),
            "intent": intent,
        })

    context = build_context(website_docs)
    prompt = build_website_prompt(question, context)
    trace.add_step("context_builder", {
        "context_chars": len(context),
        "prompt_chars": len(prompt),
        "source_count": len(website_docs),
        "source_type": "website_uneti",
    })

    answer = await run_in_threadpool(ask_gemini, prompt)
    trace.add_step("llm_call", {
        "answer_chars": len(answer or ""),
        "llm_called": True,
    })

    best_doc = website_docs[0]
    source = best_doc.get("attachment_url") or best_doc.get("url")

    return _finalize(trace, {
        "question": question,
        "answer": answer,
        "source": source,
        "sources": _build_sources(website_docs, question),
        "intent": intent,
    })


async def handle_chat(request):
    question = request.question.strip()
    trace = RagTrace(question)
    trace.add_step("request_received", {"question": question, "is_empty": not bool(question)})

    if not question:
        return _finalize(trace, {
            "question": request.question,
            "answer": "Vui lòng nhập câu hỏi.",
            "source": None,
            "sources": [],
            "intent": QueryIntent.OUT_OF_SCOPE.value,
        })

    analysis = classify_query(question)
    trace.add_step("classify_query", {
        "intent": analysis.intent.value,
        "reason": analysis.reason,
        "metadata": analysis.metadata,
    }, {"question": question})

    if analysis.intent == QueryIntent.OUT_OF_SCOPE:
        trace.add_step("route_decision", {"llm_called": False, "reason": "out_of_scope"})
        return _finalize(trace, {
            "question": question,
            "answer": OUT_OF_SCOPE_ANSWER,
            "source": None,
            "sources": [],
            "intent": analysis.intent.value,
        })

    if analysis.intent == QueryIntent.GENERAL_ADVICE:
        trace.add_step("route_decision", {"llm_called": False, "reason": "general_advice"})
        return _finalize(trace, {
            "question": question,
            "answer": GENERAL_ADVICE_ANSWER,
            "source": None,
            "sources": [],
            "intent": analysis.intent.value,
        })

    if analysis.intent == QueryIntent.WEBSITE_UNETI:
        return await _search_website_and_finalize(
            trace,
            question,
            analysis.intent.value,
            "explicit_website_intent",
        )

    business_debug = {}
    business_docs = search_business_sources(question, debug=business_debug)
    trace.add_step("business_retrieval", business_debug, {"question": question})

    has_business_evidence, business_evidence_reason = _has_confident_evidence(question, business_docs)
    faq_direct_answer = build_business_faq_answer(business_docs)
    trace.add_step("business_evidence_check", {
        "has_confident_evidence": has_business_evidence,
        "reason": business_evidence_reason,
        "query_keyword_count": len(get_keywords(question)),
        "faq_direct_answer": bool(faq_direct_answer),
        "llm_called": bool(business_docs and has_business_evidence and not faq_direct_answer),
    })

    if business_docs and has_business_evidence and faq_direct_answer:
        best_doc = business_docs[0]
        trace.add_step("faq_direct_answer", {
            "answer_chars": len(faq_direct_answer),
            "source_count": len(business_docs),
            "source_type": BUSINESS_FAQ_SOURCE_TYPE,
            "top_source": best_doc.get("title"),
            "file_id": best_doc.get("file_id"),
        })

        source = f'{best_doc.get("faq_location") or best_doc.get("title")} - {best_doc.get("doc_name")}'
        return _finalize(trace, {
            "question": question,
            "answer": faq_direct_answer,
            "source": source,
            "sources": _build_sources(business_docs, question),
            "intent": analysis.intent.value,
        })

    if business_docs and has_business_evidence:
        context = build_context(business_docs)
        prompt = build_prompt(question, context)
        trace.add_step("context_builder", {
            "context_chars": len(context),
            "prompt_chars": len(prompt),
            "source_count": len(business_docs),
            "source_type": "business_document",
        })

        answer = await run_in_threadpool(ask_gemini, prompt)
        trace.add_step("llm_call", {
            "answer_chars": len(answer or ""),
            "llm_called": True,
        })

        best_doc = business_docs[0]
        source = f'{best_doc.get("title")} - {best_doc.get("doc_name")}'

        return _finalize(trace, {
            "question": question,
            "answer": answer,
            "source": source,
            "sources": _build_sources(business_docs, question),
            "intent": analysis.intent.value,
        })

    retrieval_debug = {}
    docs = await search_documents(question, debug=retrieval_debug)
    trace.add_step("retrieval", retrieval_debug, {"question": question})

    has_evidence, evidence_reason = _has_confident_evidence(question, docs)
    trace.add_step("evidence_check", {
        "has_confident_evidence": has_evidence,
        "reason": evidence_reason,
        "query_keyword_count": len(get_keywords(question)),
        "llm_called": bool(docs and has_evidence),
    })

    if not docs or not has_evidence:
        trace.add_step("fallback_decision", {
            "from": "internal_document",
            "to": "website_uneti",
            "reason": evidence_reason,
        })
        return await _search_website_and_finalize(
            trace,
            question,
            "website_uneti_fallback",
            evidence_reason,
        )

    context = build_context(docs)
    prompt = build_prompt(question, context)
    trace.add_step("context_builder", {
        "context_chars": len(context),
        "prompt_chars": len(prompt),
        "source_count": len(docs),
    })

    answer = await run_in_threadpool(ask_gemini, prompt)
    trace.add_step("llm_call", {
        "answer_chars": len(answer or ""),
        "llm_called": True,
    })

    best_doc = docs[0]
    source = f'{best_doc.get("title")} - {best_doc.get("doc_name")}'

    return _finalize(trace, {
        "question": question,
        "answer": answer,
        "source": source,
        "sources": _build_sources(docs, question),
        "intent": analysis.intent.value,
    })


def get_chat_trace(trace_id: str) -> dict:
    try:
        return load_trace(trace_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="trace_id khong hop le") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Khong tim thay trace") from exc
