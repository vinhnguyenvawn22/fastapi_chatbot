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
import app.data.langchain_pipeline as langchain_pipeline
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


def test_document_prompt_includes_interpreted_question_without_replacing_original():
    prompt = build_prompt(
        "toi muon cham lai bai thi nhu the nao",
        "<NGUON>noi dung phuc khao</NGUON>",
        retrieval_plan={
            "intent": "phuc_khao",
            "domain": "khao_thi",
            "query": "phuc khao ket qua bai thi diem thi hoc phan sinh vien",
            "hyde": "Sinh vien de nghi phuc khao.",
            "must": ["phuc khao", "diem thi"],
            "status": "rule_success",
        },
    )

    assert "CÂU HỎI GỐC:" in prompt
    assert "toi muon cham lai bai thi nhu the nao" in prompt
    assert "CÁCH HỆ THỐNG ĐÃ HIỂU CÂU HỎI:" in prompt
    assert "Intent: phuc_khao" in prompt
    assert "Truy vấn nghiệp vụ: phuc khao ket qua bai thi diem thi hoc phan sinh vien" in prompt
    assert "Sinh vien de nghi phuc khao." not in prompt


def test_document_prompt_includes_fallback_interpreted_block_for_empty_plan():
    prompt = build_prompt(
        "Noi dung van ban?",
        "<NGUON>noi dung</NGUON>",
        retrieval_plan={
            "intent": "unknown",
            "domain": "unknown",
            "query": "",
            "must": [],
            "status": "not_needed",
        },
    )

    assert "CÁCH HỆ THỐNG ĐÃ HIỂU CÂU HỎI:" in prompt
    assert "Intent: unknown" in prompt
    assert "Nhóm nghiệp vụ: unknown" in prompt
    assert "Truy vấn nghiệp vụ: Noi dung van ban?" in prompt


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


def test_aggregate_generation_receives_business_retrieval_plan(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Phuc khao",
        "content": "Sinh vien gui yeu cau phuc khao ket qua bai thi.",
        "doc_name": "support-sv.docx",
        "relative_path": "support-sv.docx",
        "chunk_index": 1,
        "keyword_score": 120.0,
    }
    retrieval_plan = {
        "intent": "phuc_khao",
        "domain": "khao_thi",
        "query": "phuc khao ket qua bai thi diem thi hoc phan sinh vien",
        "must": ["phuc khao", "diem thi"],
        "status": "rule_success",
    }
    generated_states = []

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {"retrieval_plan": retrieval_plan},
        }

    async def fake_internal(state):
        return {**state, "docs": [], "retrieval_debug": {}}

    async def fake_generate(state):
        generated_states.append(state)
        return {**state, "answer": "Sinh vien thuc hien phuc khao theo tai lieu."}

    monkeypatch.setattr(chatbot_controller, "retrieve_business", fake_business)
    monkeypatch.setattr(chatbot_controller, "retrieve_internal", fake_internal)
    monkeypatch.setattr(chatbot_controller, "generate_answer", fake_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "_has_confident_evidence",
        lambda question, docs: (bool(docs), "mock"),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "_filter_usable_sources",
        lambda question, docs: (docs, []),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_aggregate_documents(
            RagTrace("toi muon cham lai bai thi"),
            "toi muon cham lai bai thi",
            "internal_document",
            "document_terms",
        )
    )

    assert result["answer"] == "Sinh vien thuc hien phuc khao theo tai lieu."
    assert generated_states
    assert generated_states[0]["question"] == "toi muon cham lai bai thi"
    assert generated_states[0]["retrieval_debug"]["retrieval_plan"] == retrieval_plan


def test_aggregate_prefers_mapping_guided_business_answer(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Muc II -> 5 -> 5.3 -> a",
        "content": "Huong dan lay lai mat khau Gmail va xu ly su co email UNETI.",
        "doc_name": "ChatbotAI_CBGV_SV_V4.docx",
        "relative_path": "ChatbotAI_CBGV_SV_V4.docx",
        "chunk_index": 1,
        "keyword_score": 123.0,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Dieu 16 email",
        "content": "Ca nhan co trach nhiem doi mat khau ban dau va bao mat thong tin.",
        "doc_name": "quy-dinh-email.pdf",
        "relative_path": "quy-dinh-email.pdf",
        "chunk_index": 1,
        "keyword_score": 25.0,
    }
    generated_docs = []

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "mapping_selected": True,
                "retrieval_method": "location",
                "mapping_question": "Email nha truong quen mat khau",
            },
        }

    async def fake_internal(state):
        return {**state, "docs": [internal_doc], "retrieval_debug": {}}

    async def fake_generate(state):
        generated_docs.append(state["docs"])
        return {**state, "answer": "Dung huong dan nghiep vu de lay lai mat khau Gmail."}

    monkeypatch.setattr(chatbot_controller, "retrieve_business", fake_business)
    monkeypatch.setattr(chatbot_controller, "retrieve_internal", fake_internal)
    monkeypatch.setattr(chatbot_controller, "generate_answer", fake_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "_has_confident_evidence",
        lambda question, docs: (bool(docs), "mock"),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "_filter_usable_sources",
        lambda question, docs: (docs, []),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_aggregate_documents(
            RagTrace("toi muon doi mat khau email"),
            "toi muon doi mat khau email",
            "internal_document",
            "document_terms",
        )
    )

    assert generated_docs == [[business_doc]]
    assert result["source"] == "Muc II -> 5 -> 5.3 -> a - ChatbotAI_CBGV_SV_V4.docx"
    assert result["answer"] == "Dung huong dan nghiep vu de lay lai mat khau Gmail."


def test_aggregate_rejects_keyword_false_positive_and_uses_internal(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Khối lượng giảm trừ",
        "content": "Hướng dẫn xem khối lượng giảm trừ của cán bộ giảng viên trên hệ thống.",
        "doc_name": "support-cbgv.docx",
        "relative_path": "support-cbgv.docx",
        "chunk_index": 1,
        "keyword_score": 81.0,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Đối tượng được miễn, giảm học phí",
        "content": (
            "Quy định các đối tượng sinh viên được miễn, giảm học phí "
            "và hồ sơ đề nghị áp dụng."
        ),
        "doc_name": "mien-giam-hoc-phi.pdf",
        "relative_path": "ctsv/mien-giam-hoc-phi.pdf",
        "chunk_index": 1,
        "keyword_score": 20.0,
    }
    generated_docs = []

    async def fake_business(state):
        return {**state, "docs": [business_doc]}

    async def fake_internal(state):
        return {**state, "docs": [internal_doc]}

    async def fake_generate(state):
        generated_docs.append(state["docs"])
        return {**state, "answer": "Sinh viên thuộc các nhóm quy định được miễn, giảm học phí."}

    monkeypatch.setattr(chatbot_controller, "retrieve_business", fake_business)
    monkeypatch.setattr(chatbot_controller, "retrieve_internal", fake_internal)
    monkeypatch.setattr(chatbot_controller, "generate_answer", fake_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "rerank_chunks",
        lambda question, docs: (docs, {"used": False, "reason": "mock"}),
    )

    result = asyncio.run(
        chatbot_controller.handle_chat(
            ChatRequest(question="đối tượng được miễn giảm học phí")
        )
    )

    assert generated_docs == [[internal_doc | {"lexical_coverage": 1.0}]]
    assert result["source"].endswith("mien-giam-hoc-phi.pdf")
    assert all(source["source_type"] != "business_document" for source in result["sources"])


def test_aggregate_retries_internal_when_combined_generation_has_no_evidence(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Học phí trên cổng hỗ trợ",
        "content": (
            "Hướng dẫn xem mục học phí và thông tin tài chính trên cổng hỗ trợ "
            "dành cho người học và cán bộ phụ trách."
        ),
        "doc_name": "support.docx",
        "relative_path": "support.docx",
        "chunk_index": 1,
        "keyword_score": 20.0,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Miễn giảm học phí",
        "content": (
            "Các đối tượng được miễn giảm học phí thực hiện theo quy định này, "
            "bao gồm điều kiện, hồ sơ và trình tự đề nghị áp dụng chính sách."
        ),
        "doc_name": "policy.pdf",
        "relative_path": "policy.pdf",
        "chunk_index": 1,
        "keyword_score": 20.0,
    }
    call_count = 0

    async def fake_generate(state):
        nonlocal call_count
        call_count += 1
        answer = (
            chatbot_controller.NO_EVIDENCE_ANSWER
            if call_count == 1
            else "Câu trả lời từ tài liệu nội bộ."
        )
        return {**state, "answer": answer}

    monkeypatch.setattr(
        chatbot_controller,
        "retrieve_business",
        lambda state: asyncio.sleep(0, result={**state, "docs": [business_doc]}),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "retrieve_internal",
        lambda state: asyncio.sleep(0, result={**state, "docs": [internal_doc]}),
    )
    monkeypatch.setattr(chatbot_controller, "generate_answer", fake_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "rerank_chunks",
        lambda question, docs: (docs, {"used": False, "reason": "mock"}),
    )

    result = asyncio.run(
        chatbot_controller.handle_chat(
            ChatRequest(question="miễn giảm học phí")
        )
    )

    assert call_count == 2
    assert result["answer"] == "Câu trả lời từ tài liệu nội bộ."
    assert result["source"].endswith("policy.pdf")


def test_aggregate_no_evidence_does_not_return_misleading_sources(monkeypatch):
    irrelevant_doc = {
        "source_type": "business_document",
        "title": "Khối lượng giảm trừ",
        "content": "Hướng dẫn khối lượng công tác dành cho cán bộ giảng viên.",
        "doc_name": "support.docx",
        "relative_path": "support.docx",
        "chunk_index": 1,
        "keyword_score": 81.0,
    }

    monkeypatch.setattr(
        chatbot_controller,
        "retrieve_business",
        lambda state: asyncio.sleep(0, result={**state, "docs": [irrelevant_doc]}),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "retrieve_internal",
        lambda state: asyncio.sleep(0, result={**state, "docs": []}),
    )

    async def fake_website(trace, question, intent, reason):
        return chatbot_controller._finalize(trace, {
            "question": question,
            "answer": chatbot_controller.NO_EVIDENCE_ANSWER,
            "source": None,
            "sources": [],
            "intent": intent,
        })

    monkeypatch.setattr(chatbot_controller, "_search_website_and_finalize", fake_website)

    result = asyncio.run(
        chatbot_controller.handle_chat(
            ChatRequest(question="đối tượng được miễn giảm học phí")
        )
    )

    assert result["source"] is None
    assert result["sources"] == []


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


def test_ambiguous_chat_returns_clarification_without_retrieval(monkeypatch):
    class Decision:
        def to_dict(self):
            return {
                "action": "clarification_needed",
                "topic": None,
                "confidence": 0.2,
                "reason": "garbled_query",
                "clarifying_question": (
                    "Bạn muốn kiểm tra đầu ra của chức năng hoặc hệ thống nào?"
                ),
                "analyzer": "rule",
                "cache_hit": False,
            }

    async def fail_retrieval(*args, **kwargs):
        raise AssertionError("Retrieval must not run for clarification")

    monkeypatch.setattr(chatbot_controller, "analyze_ambiguity", lambda question: Decision())
    monkeypatch.setattr(
        chatbot_controller,
        "_answer_with_aggregate_documents",
        fail_retrieval,
    )

    result = asyncio.run(
        chatbot_controller.handle_chat(
            ChatRequest(question="xtet đầu ra ta4 kiểu gì")
        )
    )

    assert result["intent"] == "clarification_needed"
    assert result["sources"] == []
    assert result["answer"] == "Bạn muốn kiểm tra đầu ra của chức năng hoặc hệ thống nào?"


def test_probe_failure_returns_clarification_after_retrieval(monkeypatch):
    async def fake_retrieve_internal(state):
        return {
            **state,
            "docs": [],
            "retrieval_debug": {
                "fallback_reason": "probe_insufficient_evidence",
                "ambiguity": {
                    "action": "probe_retrieval",
                    "reason": "unknown_topic_requires_probe",
                },
            },
        }

    monkeypatch.setattr(
        chatbot_controller,
        "retrieve_internal",
        fake_retrieve_internal,
    )

    trace = RagTrace("chủ đề lạ")
    result = asyncio.run(
        chatbot_controller._answer_with_internal_documents(
            trace,
            "chủ đề lạ",
            "internal_document",
            "test",
            {"action": "probe_retrieval"},
        )
    )

    assert result["answer"] == "Bạn cần hỏi rõ ràng hơn"
    assert result["intent"] == "clarification_needed"


def test_gemini_error_returns_unavailable_message_without_source_summary(monkeypatch):
    monkeypatch.setattr(
        langchain_pipeline,
        "ask_gemini",
        lambda prompt: "He thong AI dang ban, vui long thu lai sau it phut.",
    )
    traces = []
    state = {
        "prompt": "prompt",
        "docs": [{
            "title": "Điều kiện tốt nghiệp",
            "content": "Sinh viên phải tích lũy đủ học phần và hoàn thành nghĩa vụ.",
        }],
        "trace_callback": lambda name, output, input_data=None: traces.append(
            (name, output)
        ),
    }

    result = asyncio.run(langchain_pipeline._generate_answer(state))

    assert result["answer"] == langchain_pipeline.GEMINI_UNAVAILABLE_ANSWER
    assert "Thông tin tóm tắt" not in result["answer"]
    assert "Điều kiện tốt nghiệp" not in result["answer"]
    assert traces[-1][1]["fallback_used"] is True
    assert traces[-1][1]["fallback_reason"] == "gemini_unavailable"
    assert "He thong AI dang ban" in traces[-1][1]["gemini_error_message"]


def test_gemini_exception_returns_unavailable_message_without_source_summary(monkeypatch):
    monkeypatch.setattr(
        langchain_pipeline,
        "ask_gemini",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("Gemini offline")),
    )
    state = {
        "prompt": "prompt",
        "docs": [{
            "title": "Quy định camera",
            "content": "Hệ thống camera được quản lý và vận hành theo quy định.",
        }],
    }

    result = asyncio.run(langchain_pipeline._generate_answer(state))

    assert result["answer"] == langchain_pipeline.GEMINI_UNAVAILABLE_ANSWER
    assert "Quy định camera" not in result["answer"]


def test_hyde_only_source_requires_rerank_score(monkeypatch):
    monkeypatch.setattr(chatbot_controller, "HYDE_MIN_RERANK_SCORE", 0.5)
    low = {
        "hyde_only": True,
        "rerank_score": 0.2,
        "vector_score": 0.95,
    }
    high = {
        "hyde_only": True,
        "rerank_score": 0.8,
        "vector_score": 0.95,
    }

    assert chatbot_controller._has_confident_evidence("camera", [low])[0] is False
    assert chatbot_controller._has_confident_evidence("camera", [high])[0] is True


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
