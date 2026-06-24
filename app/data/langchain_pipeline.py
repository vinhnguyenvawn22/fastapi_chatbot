import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from app.data.business_knowledge import search_business_sources
from app.data.elasticsearch_client import search_documents
from app.data.gemini_client import ask_gemini
from app.data.prompt_builder import build_context, build_prompt, build_website_prompt
from app.data.website_search_client import index_uneti_website


PipelineState = dict[str, Any]
TraceCallback = Callable[[str, dict, dict | None], None]


def _trace(state: PipelineState, name: str, output: dict, input_data: dict | None = None) -> None:
    callback: TraceCallback | None = state.get("trace_callback")
    if callback:
        callback(name, output, input_data)


async def _retrieve_internal(state: PipelineState) -> PipelineState:
    debug = {}
    docs = await search_documents(
        state["question"],
        debug=debug,
        source_type_filter=state.get("source_type_filter"),
    )
    _trace(state, "lcel_internal_retrieval", debug, {
        "source_route": "internal_document",
        "reason": state.get("reason"),
    })
    return {**state, "docs": docs, "retrieval_debug": debug}


async def _retrieve_business(state: PipelineState) -> PipelineState:
    debug = {}
    docs = await asyncio.to_thread(search_business_sources, state["question"], None, debug)
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
    )
    docs = [doc for doc in docs if doc.get("source_type") == "website_uneti"]
    _trace(state, "lcel_website_retrieval", debug, {
        "source_route": "website_uneti",
    })
    return {**state, "docs": docs, "retrieval_debug": debug}


def _build_generation_prompt(state: PipelineState) -> PipelineState:
    docs = state.get("docs") or []
    context = build_context(docs)
    if state.get("prompt_type") == "website":
        prompt = build_website_prompt(state["question"], context)
    else:
        prompt = build_prompt(state["question"], context)

    _trace(state, "lcel_prompt_builder", {
        "context_chars": len(context),
        "prompt_chars": len(prompt),
        "source_count": len(docs),
        "prompt_type": state.get("prompt_type", "document"),
    })
    return {**state, "context": context, "prompt": prompt}


async def _generate_answer(state: PipelineState) -> PipelineState:
    answer = await asyncio.to_thread(ask_gemini, state["prompt"])
    _trace(state, "lcel_llm_call", {
        "answer_chars": len(answer or ""),
        "llm_called": True,
    })
    return {**state, "answer": answer}


internal_retriever = RunnableLambda(_retrieve_internal).with_config(
    {"run_name": "internal_document_retriever"}
)
business_retriever = RunnableLambda(_retrieve_business).with_config(
    {"run_name": "business_document_retriever"}
)
website_retriever = RunnableLambda(_retrieve_website).with_config(
    {"run_name": "website_uneti_retriever"}
)
generation_chain = (
    RunnableLambda(_build_generation_prompt).with_config({"run_name": "chat_prompt_template"})
    | RunnableLambda(_generate_answer).with_config({"run_name": "gemini_generation"})
)


async def retrieve_internal(state: PipelineState) -> PipelineState:
    with tracing_context(enabled=False):
        return await internal_retriever.ainvoke(state)


async def retrieve_business(state: PipelineState) -> PipelineState:
    with tracing_context(enabled=False):
        return await business_retriever.ainvoke(state)


async def retrieve_website(state: PipelineState) -> PipelineState:
    with tracing_context(enabled=False):
        return await website_retriever.ainvoke(state)


async def generate_answer(state: PipelineState) -> PipelineState:
    with tracing_context(enabled=False):
        return await generation_chain.ainvoke(state)
