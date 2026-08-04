import asyncio
from collections.abc import Awaitable, Callable
import re
from typing import Any

from langchain_core.runnables import RunnableLambda

from app.data.business_knowledge import search_business_sources
from app.data.elasticsearch_client import get_keywords, normalize_text, search_documents
from app.data.gemini_client import ask_gemini
from app.data.prompt_builder import build_context, build_prompt, build_website_prompt
from app.data.website_search_client import index_uneti_website


PipelineState = dict[str, Any]
TraceCallback = Callable[[str, dict, dict | None], None]
GEMINI_UNAVAILABLE_ANSWER = (
    "Hệ thống AI đang tạm thời không thể tạo câu trả lời do Gemini hết quota "
    "hoặc gặp lỗi. Vui lòng thử lại sau."
)
GEMINI_ERROR_MARKERS = {
    "gemini_quota_or_rate_limit": "He thong AI tam thoi vuot gioi han",
    "gemini_unavailable": "He thong AI dang ban",
    "gemini_api_error": "Loi khi goi Gemini API",
    "qwen_unavailable": "Khong ket noi duoc Qwen local",
    "qwen_timeout": "Qwen local phan hoi qua thoi gian cho phep",
    "qwen_local_error": "Loi khi goi mo hinh Qwen local",
}


def _trace(state: PipelineState, name: str, output: dict, input_data: dict | None = None) -> None:
    callback: TraceCallback | None = state.get("trace_callback")
    if callback:
        callback(name, output, input_data)


def _short_debug_message(value: Any, limit: int = 500) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if not text:
        return None
    return text[:limit]


def _gemini_error_reason(answer: Any, llm_error: str | None) -> str | None:
    if llm_error:
        return "gemini_exception"

    answer_text = str(answer or "")
    for reason, marker in GEMINI_ERROR_MARKERS.items():
        if marker in answer_text:
            return reason
    return None


def _format_answer_for_display(answer: Any) -> str:
    """Normalize model markdown into the plain, readable style used by chat UI."""
    text = str(answer or "")
    text = re.sub(
        r"(?m)^[ \t]*(?:-|•|\*(?!\*))[ \t]+",
        "• ",
        text,
    )
    return text.replace("**", "").strip()


def _business_procedure_fallback_answer(state: PipelineState) -> str | None:
    question = normalize_text(state.get("question") or "")
    if not any(
        marker in question
        for marker in (
            "cach", "lam sao", "xem o dau", "kiem tra",
            "tra cuu", "dang ky", "gui", "nop",
        )
    ):
        return None

    for doc in (state.get("docs") or [])[:3]:
        content = str(doc.get("content") or "").strip()
        is_business_doc = (
            doc.get("document_type") == "business_document"
            or "HDSD" in str(doc.get("doc_name") or "")
        )
        if not is_business_doc or not re.search(
            r"(?im)^(?:B\d+|Bước\s+\d+)\s*:",
            content,
        ):
            continue

        content = re.sub(
            r"(https?://[^\s]+?\.vn)(?=truy\b)",
            r"\1\nTruy",
            content,
            flags=re.IGNORECASE,
        )
        raw_lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in content.splitlines()
            if line.strip()
        ]
        selected = []
        for line in raw_lines:
            if line.startswith("Chức năng:"):
                selected.append(line)
            elif re.match(r"^(?:B\d+|Bước\s+\d+)\s*:", line, re.IGNORECASE):
                selected.append(re.sub(
                    r"^(?:B\d+|Bước\s+\d+)\s*:\s*",
                    "",
                    line,
                    flags=re.IGNORECASE,
                ))
            elif line.lower().startswith("truy cập trực tiếp"):
                selected.append(line)
            elif re.match(r"^https?://", line, re.IGNORECASE):
                if selected and selected[-1].endswith(":"):
                    selected[-1] = f"{selected[-1]} {line}"
                else:
                    selected.append(line)
            elif line.startswith("Lưu ý:"):
                selected.append(line)

        if len(selected) < 2:
            continue
        title = doc.get("title") or "Nguồn hướng dẫn"
        doc_name = doc.get("doc_name") or "Tài liệu"
        return (
            "\n".join(f"- {line}" for line in selected)
            + f"\n(Nguồn: {title} - {doc_name})"
        )
    return None


def _extractive_fallback_answer(state: PipelineState) -> str:
    docs = state.get("docs") or []
    if not docs:
        return GEMINI_UNAVAILABLE_ANSWER
    business_answer = _business_procedure_fallback_answer(state)
    if business_answer:
        return business_answer
    retrieval_plan = state.get("retrieval_plan") or (state.get("retrieval_debug") or {}).get("retrieval_plan") or {}
    query_text = " ".join([
        str(state.get("question") or ""),
        str(retrieval_plan.get("query") or ""),
        " ".join(str(item) for item in (retrieval_plan.get("must") or [])),
    ])
    query_terms = set(get_keywords(query_text))
    ranked = []
    for doc_index, doc in enumerate(docs[:5]):
        content = re.sub(r"\s+", " ", str(doc.get("content") or "")).strip()
        if not content:
            continue
        sentences = [
            part.strip(" -")
            for part in re.split(r"(?<=[.!?])\s+|(?<=;)\s+", content)
            if len(part.strip(" -")) >= 24
        ]
        if not sentences and content:
            sentences = [content[:700]]
        for sent_index, sentence in enumerate(sentences):
            score = len(query_terms & set(get_keywords(sentence)))
            ranked.append((score, doc_index, sent_index, sentence))
    if not ranked:
        return GEMINI_UNAVAILABLE_ANSWER
    selected = sorted(
        sorted(ranked, key=lambda item: item[0], reverse=True)[:5],
        key=lambda item: (item[1], item[2]),
    )
    lines = [f"- {sentence}" for _, _, _, sentence in selected]
    best_doc = docs[0]
    source = f'{best_doc.get("title") or "Nguon"} - {best_doc.get("doc_name") or "Tai lieu"}'
    return "\n".join(lines) + f"\n(Nguon: {source})"


def _trace_hybrid_retrieval(state: PipelineState, debug: dict) -> None:
    ambiguity = debug.get("ambiguity", {})
    _trace(state, "ambiguity_detection", {
        "ambiguity_action": ambiguity.get("action"),
        "detected_topic": ambiguity.get("topic"),
        "ambiguity_confidence": ambiguity.get("confidence"),
        "ambiguity_reason": ambiguity.get("reason"),
        "clarification_question": ambiguity.get("clarifying_question"),
        "cache_hit": ambiguity.get("cache_hit"),
    })
    _trace(state, "probe_retrieval", debug.get("probe_retrieval", {}))
    _trace(state, "probe_evidence_decision", {
        key: value
        for key, value in debug.get("probe_retrieval", {}).items()
        if key != "evidence_sources"
    })
    _trace(state, "hyde_generation", debug.get("hyde", {}))
    _trace(state, "grounded_hyde_generation", debug.get("grounded_hyde", {}))
    _trace(state, "bm25_retrieval", {
        "queries": debug.get("bm25_original_results", []),
        "errors": debug.get("bm25_errors", []),
    })
    _trace(state, "ann_retrieval", {
        "ann_original_results": debug.get("ann_original_results", []),
        "ann_hyde_results": debug.get("ann_hyde_results", []),
        "ann_grounded_hyde_results": debug.get(
            "ann_grounded_hyde_results",
            [],
        ),
        "errors": debug.get("vector_errors", []),
    })
    _trace(state, "rrf_fusion", {
        "results": debug.get("rrf_results", []),
    })
    _trace(state, "cross_encoder_rerank", debug.get("reranking", {}))


async def _retrieve_internal(state: PipelineState) -> PipelineState:
    debug = {}
    source_type_filter = state.get("source_type_filter") or "local_file"
    docs = await search_documents(
        state["question"],
        debug=debug,
        source_type_filter=source_type_filter,
        corpus_filter="local_documents" if source_type_filter == "local_file" else None,
        rag_enabled_filter=True if source_type_filter == "local_file" else None,
        exclude_document_names=(
            {"PCNTT_MAPPING_FILE.docx"}
            if source_type_filter == "local_file"
            else None
        ),
        exclude_source_types=(
            {"website_uneti", "business_faq_mapping"}
            if source_type_filter == "local_file"
            else None
        ),
        ambiguity_decision=state.get("ambiguity_decision"),
    )
    _trace_hybrid_retrieval(state, debug)
    _trace(state, "lcel_internal_retrieval", debug, {
        "source_route": "internal_document",
        "reason": state.get("reason"),
    })
    return {**state, "docs": docs, "retrieval_debug": debug}


async def _retrieve_local_documents(state: PipelineState) -> PipelineState:
    debug = {}
    docs = await search_documents(
        state["question"],
        debug=debug,
        source_type_filter="local_file",
        corpus_filter="local_documents",
        rag_enabled_filter=True,
        exclude_document_names={"PCNTT_MAPPING_FILE.docx"},
        exclude_source_types={"website_uneti", "business_faq_mapping"},
        document_type_filter=state.get("document_type_filter"),
        department_filter=state.get("department_filter"),
        ambiguity_decision=state.get("ambiguity_decision"),
    )
    _trace_hybrid_retrieval(state, debug)
    _trace(state, "lcel_local_documents_retrieval", debug, {
        "source_route": "local_documents",
        "reason": state.get("reason"),
    })
    return {**state, "docs": docs, "retrieval_debug": debug}


async def _retrieve_business(state: PipelineState) -> PipelineState:
    debug = {}
    docs = await asyncio.to_thread(
        search_business_sources,
        state["question"],
        None,
        debug,
        state.get("query_context"),
    )
    docs = [
        doc for doc in docs
        if doc.get("source_type") in {"business_document", "business_faq_mapping"}
        or normalize_text(doc.get("source_root", "")) == "nghiep_vu"
        or "web support" in normalize_text(doc.get("doc_name", ""))
    ]
    _trace(state, "business_retrieval_plan", {
        "retrieval_plan": debug.get("retrieval_plan"),
        "retrieval_plan_parse_error": debug.get("retrieval_plan_parse_error"),
        "retrieval_plan_llm_called": (debug.get("retrieval_plan") or {}).get("llm_called", False),
        "retrieval_plan_cache_hit": (debug.get("retrieval_plan") or {}).get("cache_hit", False),
        "final_search_query": debug.get("final_search_query"),
        "fallback_reason": debug.get("fallback_reason"),
        "mapping_judge_llm_called": any(
            item.get("llm_used")
            for item in debug.get("mapping_gate_decisions", [])
        ),
    }, {
        "source_route": "business_document",
        "reason": state.get("reason"),
    })
    _trace(state, "lcel_business_retrieval", debug, {
        "source_route": "business_document",
        "reason": state.get("reason"),
    })
    return {**state, "docs": docs, "retrieval_debug": debug}


async def _retrieve_website(state: PipelineState) -> PipelineState:
    index_debug = {}
    index_result = await asyncio.to_thread(index_uneti_website, state["question"], index_debug)
    _trace(state, "lcel_website_index", index_debug, {
        "source_route": "website_uneti",
        "reason": state.get("reason"),
    })

    if not index_result.get("indexed_chunks"):
        return {**state, "docs": [], "retrieval_debug": index_debug}

    debug = {}
    docs = await search_documents(
        state["question"],
        debug=debug,
        source_type_filter="website_uneti",
        ambiguity_decision=state.get("ambiguity_decision"),
    )
    docs = [doc for doc in docs if doc.get("source_type") == "website_uneti"]
    _trace_hybrid_retrieval(state, debug)
    _trace(state, "lcel_website_retrieval", debug, {
        "source_route": "website_uneti",
    })
    return {**state, "docs": docs, "retrieval_debug": debug}


def _build_generation_prompt(state: PipelineState) -> PipelineState:
    docs = state.get("docs") or []
    context = build_context(docs, max_chunks=state.get("max_context_chunks"))
    retrieval_plan = (state.get("retrieval_debug") or {}).get("retrieval_plan")
    if state.get("prompt_type") == "website":
        prompt = build_website_prompt(
            state["question"], context,
            conversation_history=state.get("conversation_history"),
            original_question=state.get("original_question"),
        )
    else:
        prompt = build_prompt(
            state["question"], context, retrieval_plan=retrieval_plan,
            conversation_history=state.get("conversation_history"),
            original_question=state.get("original_question"),
            required_aspects=state.get("required_aspects"),
            generation_guidance=state.get("generation_guidance"),
        )

    _trace(state, "context_selection", {
        "selected_source_count": len(docs),
        "selected_sources": [
            {
                "doc_name": doc.get("doc_name"),
                "title": doc.get("title"),
                "chunk_index": doc.get("chunk_index"),
                "source_type": doc.get("source_type"),
                "aggregate_route": doc.get("aggregate_route"),
                "aggregate_score": doc.get("aggregate_score"),
                "bm25_score": doc.get("bm25_score"),
                "vector_score": doc.get("vector_score"),
                "rrf_score": doc.get("rrf_score"),
                "rerank_score": doc.get("rerank_score"),
            }
            for doc in docs
        ],
        "context_chars": len(context),
    })
    _trace(state, "lcel_prompt_builder", {
        "context_chars": len(context),
        "prompt_chars": len(prompt),
        "source_count": len(docs),
        "prompt_type": state.get("prompt_type", "document"),
        "retrieval_plan": retrieval_plan,
        "interpreted_question": (retrieval_plan or {}).get("query"),
        "has_interpreted_question_block": (
            "CÁCH HỆ THỐNG ĐÃ HIỂU CÂU HỎI:" in prompt
        ),
        "required_aspect_count": len(state.get("required_aspects") or []),
    })
    return {
        **state,
        "context": context,
        "prompt": prompt,
        "retrieval_plan": retrieval_plan,
        "interpreted_question": (retrieval_plan or {}).get("query"),
    }


async def _generate_answer(state: PipelineState) -> PipelineState:
    llm_error = None
    try:
        answer = await asyncio.to_thread(ask_gemini, state["prompt"])
    except Exception as exc:
        llm_error = str(exc)
        answer = ""
    gemini_error_reason = _gemini_error_reason(answer, llm_error)
    fallback_used = gemini_error_reason is not None
    gemini_error_message = _short_debug_message(llm_error or answer) if fallback_used else None
    if fallback_used:
        answer = _extractive_fallback_answer(state) if state.get("docs") else GEMINI_UNAVAILABLE_ANSWER
    answer = _format_answer_for_display(answer)
    _trace(state, "lcel_llm_call", {
        "answer_chars": len(answer or ""),
        "llm_called": True,
        "final_generation_llm_called": True,
        "fallback_used": fallback_used,
        "fallback_reason": gemini_error_reason,
        "error": _short_debug_message(llm_error),
        "gemini_error_message": gemini_error_message,
    })
    return {**state, "answer": answer}


internal_retriever = RunnableLambda(_retrieve_internal).with_config(
    {"run_name": "Retrieval"}
)
local_documents_retriever = RunnableLambda(_retrieve_local_documents).with_config(
    {"run_name": "Local Documents Retrieval"}
)
business_retriever = RunnableLambda(_retrieve_business).with_config(
    {"run_name": "Retrieval"}
)
website_retriever = RunnableLambda(_retrieve_website).with_config(
    {"run_name": "Retrieval"}
)
generation_chain = (
    RunnableLambda(_build_generation_prompt).with_config({"run_name": "Context Builder"})
    | RunnableLambda(_generate_answer).with_config({"run_name": "LLM Generation"})
)


async def retrieve_internal(state: PipelineState) -> PipelineState:
    return await internal_retriever.ainvoke(state)


async def retrieve_local_documents(state: PipelineState) -> PipelineState:
    return await local_documents_retriever.ainvoke(state)


async def retrieve_business(state: PipelineState) -> PipelineState:
    return await business_retriever.ainvoke(state)


async def retrieve_website(state: PipelineState) -> PipelineState:
    return await website_retriever.ainvoke(state)


async def generate_answer(state: PipelineState) -> PipelineState:
    return await generation_chain.ainvoke(state)
