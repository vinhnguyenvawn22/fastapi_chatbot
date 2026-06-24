import asyncio
import os

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.controller import chatbot_controller
from app.controller.document_controller import (
    _extract_document_number_from_filename,
    _is_ignored_document,
)
from app.data.prompt_builder import build_prompt
from app.data.trace_logger import RagTrace
import app.data.elasticsearch_client as document_search
from app.main import app
import app.routers.chat_router as chat_router
from app.schemas.chat_schema import ChatRequest


client = TestClient(app)


@pytest.mark.parametrize(
    ("path", "handler_name"),
    [
        ("/api/chat/", "handle_chat"),
        ("/api/chat/business", "handle_business_chat"),
        ("/api/chat/internal", "handle_internal_chat"),
        ("/api/chat/website", "handle_website_chat"),
    ],
)
def test_four_chat_endpoints_keep_response_schema(monkeypatch, path, handler_name):
    async def fake_handler(request):
        return {
            "question": request.question,
            "answer": "Cau tra loi test.",
            "source": None,
            "sources": [],
            "intent": "internal_document",
            "trace_id": "trace-test",
        }

    monkeypatch.setattr(chat_router, handler_name, fake_handler)
    response = client.post(path, json={"question": "cau hoi test"})

    assert response.status_code == 200
    assert response.json() == {
        "question": "cau hoi test",
        "answer": "Cau tra loi test.",
        "source": None,
        "intent": "internal_document",
        "trace_id": "trace-test",
        "sources": [],
    }


def test_document_prompt_is_preserved_through_chat_prompt_template():
    prompt = build_prompt("Noi dung van ban?", "<NGUON>noi dung</NGUON>")

    assert "Noi dung van ban?" in prompt
    assert "<NGUON>noi dung</NGUON>" in prompt
    assert not prompt.startswith("Human:")


@pytest.mark.parametrize(
    "question",
    [
        "van ban so 880",
        "quy dinh so 799",
        "quy che so 832",
        "QD_1291",
    ],
)
def test_document_number_queries_are_forced_to_internal_retrieval(monkeypatch, question):
    calls = []

    async def fake_internal(trace, routed_question, intent, reason):
        calls.append((routed_question, reason))
        return {
            "question": routed_question,
            "answer": "internal",
            "source": None,
            "sources": [],
            "intent": intent,
            "trace_id": trace.trace_id,
        }

    async def fail_aggregate(*args, **kwargs):
        raise AssertionError("Document-number query must not use aggregate retrieval")

    monkeypatch.setattr(chatbot_controller, "_answer_with_internal_documents", fake_internal)
    monkeypatch.setattr(chatbot_controller, "_answer_with_aggregate_documents", fail_aggregate)

    result = asyncio.run(chatbot_controller.handle_chat(ChatRequest(question=question)))

    assert result["answer"] == "internal"
    assert calls == [(question, "document_number_query")]


def test_aggregate_source_deduplication():
    duplicate = {
        "source_type": "internal_document",
        "relative_path": "quy-dinh/880.pdf",
        "chunk_index": 1,
        "title": "Dieu 1",
    }
    unique = {
        "source_type": "business_document",
        "relative_path": "faq.docx",
        "chunk_index": 2,
        "title": "FAQ",
    }

    result = chatbot_controller._deduplicate_docs(
        [duplicate],
        [dict(duplicate), unique],
    )

    assert result == [duplicate, unique]


def test_document_number_extraction_does_not_use_year_after_qd_prefix():
    file_name = (
        "DHKTKTCN_PDT_QD_2025_12_09_"
        "Quy che dao tao dai hoc chinh quy_832_20092023.docx"
    )

    assert _extract_document_number_from_filename(file_name) == "832"


def test_office_lock_document_is_ignored():
    from pathlib import Path

    assert _is_ignored_document(Path("~$temporary.docx")) is True
    assert _is_ignored_document(Path("official.docx")) is False


def test_trace_write_failure_does_not_break_chat(monkeypatch):
    trace = RagTrace("test question")

    def fail_write(*args, **kwargs):
        raise PermissionError("trace directory is read-only")

    monkeypatch.setattr("pathlib.Path.write_text", fail_write)

    assert trace.save() == trace.trace_id


def test_business_chat_generates_from_original_source_not_mapping_summary(monkeypatch):
    original_doc = {
        "source_type": "business_document",
        "title": "Muc II -> 5 -> 5.3 -> a",
        "content": "Original source instructions.",
        "doc_name": "source.docx",
        "relative_path": "source.docx",
        "chunk_index": 1,
        "keyword_score": 20.0,
        "faq_answer": "Mapping summary must not be returned.",
    }
    generation_calls = []

    async def fake_retrieve_business(state):
        return {**state, "docs": [original_doc]}

    async def fake_generate_answer(state):
        generation_calls.append(state["docs"])
        return {**state, "answer": "Answer generated from original source."}

    monkeypatch.setattr(
        chatbot_controller,
        "retrieve_business",
        fake_retrieve_business,
    )
    monkeypatch.setattr(
        chatbot_controller,
        "generate_answer",
        fake_generate_answer,
    )

    result = asyncio.run(
        chatbot_controller.handle_business_chat(
            ChatRequest(question="How do I reset my password?")
        )
    )

    assert generation_calls == [[original_doc]]
    assert result["answer"] == "Answer generated from original source."
    assert result["answer"] != original_doc["faq_answer"]
    assert result["source"].endswith("source.docx")


def test_document_index_skips_one_broken_file(monkeypatch):
    files = [
        {
            "file_name": "broken.docx",
            "relative_path": "broken.docx",
            "file_size_kb": 1,
            "updated_at": "2026-01-01",
            "parse_supported": True,
        },
        {
            "file_name": "valid.docx",
            "relative_path": "valid.docx",
            "file_size_kb": 1,
            "updated_at": "2026-01-01",
            "parse_supported": True,
        },
    ]

    def fake_build_document_chunks(file_name):
        if file_name == "broken.docx":
            raise ValueError("invalid docx")
        return [{
            "doc_name": "valid.docx",
            "title": "Valid section",
            "content": "Valid document content",
            "chunk_index": 1,
        }]

    monkeypatch.setattr(document_search, "list_documents", lambda: files)
    monkeypatch.setattr(
        document_search,
        "build_document_chunks",
        fake_build_document_chunks,
    )
    document_search.clear_document_index_cache()

    chunks, _, total = document_search._load_document_index()

    assert total == 1
    assert chunks[0]["doc_name"] == "valid.docx"
    assert document_search._INDEX_CACHE["skipped_files"] == [{
        "relative_path": "broken.docx",
        "error": "invalid docx",
    }]
    document_search.clear_document_index_cache()
