import asyncio
import os
import uuid

import pytest
from fastapi.testclient import TestClient


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.controller import chatbot_controller
from app.controller.document_controller import (
    _extract_document_number_from_filename,
    _is_ignored_document,
)
from app.data.prompt_builder import build_context, build_prompt
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
        ("/api/chat/local-documents", "handle_local_documents_chat"),
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
    response = client.post(
        path, json={"question": "cau hoi test", "request_id": str(uuid.uuid4())}
    )

    assert response.status_code == 200
    data = response.json()
    assert {
        key: data[key]
        for key in ("question", "answer", "source", "intent", "trace_id", "sources")
    } == {
        "question": "cau hoi test",
        "answer": "Cau tra loi test.",
        "source": None,
        "intent": "internal_document",
        "trace_id": "trace-test",
        "sources": [],
    }
    assert data["thread_id"]
    assert data["user_message_id"]
    assert data["assistant_message_id"]


def test_document_prompt_is_preserved_through_chat_prompt_template():
    prompt = build_prompt("Noi dung van ban?", "<NGUON>noi dung</NGUON>")

    assert "Noi dung van ban?" in prompt
    assert "<NGUON>noi dung</NGUON>" in prompt
    assert not prompt.startswith("Human:")


def test_local_documents_retriever_uses_hard_filters(monkeypatch):
    captured = {}

    async def fake_search(question, **kwargs):
        captured["question"] = question
        captured.update(kwargs)
        return []

    monkeypatch.setattr(langchain_pipeline, "search_documents", fake_search)
    result = asyncio.run(
        langchain_pipeline.retrieve_local_documents({
            "question": "Quy dinh canh bao hoc tap?",
            "reason": "test",
        })
    )

    assert result["docs"] == []
    assert captured["source_type_filter"] == "local_file"
    assert captured["corpus_filter"] == "local_documents"
    assert captured["rag_enabled_filter"] is True
    assert captured["exclude_document_names"] == {"PCNTT_MAPPING_FILE.docx"}
    assert captured["exclude_source_types"] == {
        "website_uneti",
        "business_faq_mapping",
    }


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

    result = asyncio.run(chatbot_controller.handle_chat(
        ChatRequest(question=question, request_id="controller-document-number")
    ))

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


def test_multihop_subquestions_disable_llm_retrieval_helpers(monkeypatch):
    business_states = []
    internal_states = []

    async def fake_business(state):
        business_states.append(state)
        return {**state, "docs": [], "retrieval_debug": {}}

    async def fake_internal(state):
        internal_states.append(state)
        return {**state, "docs": [], "retrieval_debug": {}}

    monkeypatch.setattr(chatbot_controller, "retrieve_business", fake_business)
    monkeypatch.setattr(chatbot_controller, "retrieve_internal", fake_internal)

    asyncio.run(
        chatbot_controller._retrieve_multihop_evidence(
            RagTrace("nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi"),
            {
                "question": "nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi",
                "query_context": {},
            },
            "nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi",
        )
    )

    assert business_states
    assert internal_states
    assert all(
        state["query_context"].get("skip_retrieval_plan_llm") is True
        for state in business_states
    )
    assert all(
        state["ambiguity_decision"]["action"] == "direct_retrieval"
        for state in internal_states
    )
    assert all(
        state["ambiguity_decision"]["reason"] == "multi_hop_subquestion_no_llm"
        for state in internal_states
    )


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

    assert len(generated_docs) == 1
    assert [doc["doc_name"] for doc in generated_docs[0]] == [business_doc["doc_name"]]
    assert generated_docs[0][0]["aggregate_route"] == "business"
    assert result["source"] == "Muc II -> 5 -> 5.3 -> a - ChatbotAI_CBGV_SV_V4.docx"
    assert result["answer"] == "Dung huong dan nghiep vu de lay lai mat khau Gmail."


def test_exam_retake_procedure_uses_support_source_not_hoc_lai(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Mot cua -> Khao thi",
        "content": (
            "Dang ky thi lai la thu tuc Mot cua - Khao thi. "
            "Buoc 1: Dang nhap he thong support. "
            "Buoc 2: Chon Mot cua - Khao thi -> Dang ky thi lai. "
            "Buoc 3: Gui yeu cau."
        ),
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "chunk_index": 10,
        "keyword_score": 300.0,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Dieu 11. Dang ky hoc lai, hoc cai thien diem",
        "content": "Sinh vien dang ky hoc lai hoac hoc cai thien diem theo quy che dao tao.",
        "doc_name": "quy-che-dao-tao.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "chunk_index": 11,
        "keyword_score": 200.0,
    }

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "mapping_selected": True,
                "retrieval_method": "keyword",
                "information_need": "procedure_ui",
            },
        }

    async def fake_internal(state):
        return {**state, "docs": [internal_doc], "retrieval_debug": {}}

    async def fake_generate(state):
        raise AssertionError("Exam retake procedure should not be generated from hoc-lai internal docs")

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
            RagTrace("huong dan dang ki thi lai"),
            "huong dan dang ki thi lai",
            "internal_document",
            "document_terms",
        )
    )

    assert result["source"] == "Mot cua -> Khao thi - 2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx"
    assert result["sources"][0]["doc_name"] == "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx"
    assert "dang-ky-thi-lai" in result["answer"]
    assert "hoc lai" not in chatbot_controller.normalize_text(result["answer"])


def test_cancel_exam_retake_procedure_uses_cancel_support_source(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Mot cua -> Khao thi -> Huy dang ky thi lai",
        "content": (
            "Huy dang ky thi lai cho phep sinh vien gui yeu cau. "
            "Buoc 1: Dang nhap he thong support. "
            "Buoc 2: Chon Mot cua - Khao thi -> Huy dang ky thi lai. "
            "Buoc 3: Gui yeu cau."
        ),
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "chunk_index": 16,
        "keyword_score": 300.0,
    }

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "mapping_selected": True,
                "retrieval_method": "keyword",
                "information_need": "procedure_ui",
            },
        }

    async def fake_internal(state):
        return {**state, "docs": [], "retrieval_debug": {}}

    async def fake_generate(state):
        raise AssertionError("Cancel exam retake should use the support source directly")

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
            RagTrace("toi muon huy dang ki thi lai lam the nao"),
            "toi muon huy dang ki thi lai lam the nao",
            "internal_document",
            "document_terms",
        )
    )

    assert "huy-dang-ky-thi-lai" in result["answer"]
    assert result["sources"][0]["chunk_index"] == 16


def test_aggregate_soft_merge_keeps_internal_and_business_sources(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Web Support - Hoan thi",
        "content": "Huong dan sinh vien vao Mot cua - Khao thi de gui yeu cau hoan thi tren support.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "chunk_index": 15,
        "keyword_score": 120.0,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Dieu 15. Thi ket thuc hoc phan",
        "content": "Sinh vien vang mat du thi co ly do chinh dang duoc phep hoan thi theo quy che dao tao.",
        "doc_name": "quy-che-dao-tao.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "chunk_index": 30,
        "keyword_score": 115.0,
    }
    generated_docs = []

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "mapping_selected": True,
                "retrieval_method": "keyword",
                "information_need": "procedure_ui",
            },
        }

    async def fake_internal(state):
        return {**state, "docs": [internal_doc], "retrieval_debug": {}}

    async def fake_generate(state):
        generated_docs.append(state["docs"])
        return {**state, "answer": "Sinh vien can xem quy dinh hoan thi va gui yeu cau tren support."}

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
    monkeypatch.setattr(
        chatbot_controller,
        "rerank_chunks",
        lambda question, docs: (docs, {"used": False, "reason": "mock"}),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_aggregate_documents(
            RagTrace("toi muon biet dieu kien hoan thi va cach thuc hien tren support"),
            "toi muon biet dieu kien hoan thi va cach thuc hien tren support",
            "internal_document",
            "document_terms",
        )
    )

    assert len(generated_docs) == 1
    assert {doc["source_type"] for doc in generated_docs[0]} == {
        "business_document",
        "official_document",
    }
    assert {source["source_type"] for source in result["sources"]} == {
        "business_document",
        "official_document",
    }


def test_diverse_aggregate_selection_limits_single_document_dominance():
    same_doc_chunks = [
        {
            "source_type": "official_document",
            "title": f"Dieu {index}",
            "content": "Noi dung quy dinh lien quan",
            "doc_name": "quy-che-a.docx",
            "relative_path": "quy-che-a.docx",
            "chunk_index": index,
            "keyword_score": 100 - index,
        }
        for index in range(1, 6)
    ]
    other_doc = {
        "source_type": "official_document",
        "title": "Dieu khac",
        "content": "Noi dung quy dinh lien quan tu tai lieu khac",
        "doc_name": "quy-che-b.docx",
        "relative_path": "quy-che-b.docx",
        "chunk_index": 1,
        "keyword_score": 90.0,
    }

    selected, debug = chatbot_controller._select_diverse_aggregate_sources(
        "quy dinh dieu kien",
        [],
        same_doc_chunks + [other_doc],
        query_context={"information_need": "policy_document"},
        limit=5,
    )

    assert debug["doc_name_count"] >= 2
    assert sum(1 for doc in selected if doc["doc_name"] == "quy-che-a.docx") <= 3
    assert any(doc["doc_name"] == "quy-che-b.docx" for doc in selected)


def test_internal_retrieval_defaults_to_local_document_filter(monkeypatch):
    captured = {}

    async def fake_search_documents(question, debug=None, **kwargs):
        captured.update(kwargs)
        if debug is not None:
            debug["final_sources"] = []
        return []

    monkeypatch.setattr(langchain_pipeline, "search_documents", fake_search_documents)

    result = asyncio.run(
        langchain_pipeline.retrieve_internal({
            "question": "toi muon hoan thi thi phai lam sao",
            "reason": "test",
            "trace_callback": None,
        })
    )

    assert captured["source_type_filter"] == "local_file"
    assert captured["corpus_filter"] == "local_documents"
    assert captured["rag_enabled_filter"] is True
    assert result["docs"] == []


def test_build_context_accepts_chunk_override():
    docs = [
        {
            "doc_name": f"doc-{index}.pdf",
            "title": f"Dieu {index}",
            "chunk_index": index,
            "content": f"Noi dung chunk {index}",
        }
        for index in range(1, 6)
    ]

    context = build_context(docs, max_chunks=5)

    assert context.count("<NGUON") == 5
    assert 'ten_tai_lieu="doc-5.pdf"' in context


def test_academic_policy_question_blocks_business_direct_answer(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Dang ky muon thiet bi",
        "content": "Huong dan dang ky muon thiet bi phong hoc tren cong ho tro.",
        "doc_name": "support-cbgv.docx",
        "relative_path": "support-cbgv.docx",
        "chunk_index": 1,
        "keyword_score": 90.0,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Hoan thi ket thuc hoc phan",
        "content": (
            "Sinh vien xin hoan thi ket thuc hoc phan phai thuc hien theo quy che "
            "dao tao va quy dinh hoc vu cua nha truong."
        ),
        "doc_name": "quy-che-dao-tao.pdf",
        "relative_path": "quy-che-dao-tao.pdf",
        "chunk_index": 2,
        "keyword_score": 80.0,
    }
    generated_docs = []

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "mapping_selected": True,
                "retrieval_method": "location",
                "mapping_gate_score": 100,
            },
        }

    async def fake_internal(state):
        assert state["source_type_filter"] == "local_file"
        return {**state, "docs": [internal_doc], "retrieval_debug": {}}

    async def fake_generate(state):
        generated_docs.append(state["docs"])
        return {**state, "answer": "Sinh vien thuc hien hoan thi theo quy che dao tao."}

    monkeypatch.setattr(chatbot_controller, "retrieve_business", fake_business)
    monkeypatch.setattr(chatbot_controller, "retrieve_internal", fake_internal)
    monkeypatch.setattr(chatbot_controller, "generate_answer", fake_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "_business_direct_answer",
        lambda question, docs: "Direct business answer",
    )
    monkeypatch.setattr(
        chatbot_controller,
        "_filter_usable_sources",
        lambda question, docs: (docs, []),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "_has_confident_evidence",
        lambda question, docs: (bool(docs), "mock"),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "rerank_chunks",
        lambda question, docs: (docs, {"used": False, "reason": "mock"}),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_aggregate_documents(
            RagTrace("toi muon hoan thi thi phai lam sao"),
            "toi muon hoan thi thi phai lam sao",
            "internal_document",
            "document_terms",
        )
    )

    assert generated_docs
    assert any(doc["doc_name"] == internal_doc["doc_name"] for doc in generated_docs[0])
    assert result["answer"] == "Sinh vien thuc hien hoan thi theo quy che dao tao."
    assert any(source["source_type"] == "official_document" for source in result["sources"])


def test_course_registration_change_policy_source_filter_prefers_article_10():
    question = "cach huy hoc phan da dang ky"
    direct_doc = {
        "source_type": "official_document",
        "title": "Dieu 10. Rut bot hoc phan da dang ky",
        "content": "Sinh vien duoc rut bot hoc phan trong thoi gian dang ky hoc phan.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "dieu": 10,
    }
    noisy_doc = {
        "source_type": "official_document",
        "title": "Quy doi chung chi tieng Anh",
        "content": "Hoc phan da dang ky chung chi tieng Anh TOEIC IELTS.",
        "doc_name": "TTNNTH_Quy doi chung chi tieng Anh.docx",
        "relative_path": "tieng-anh.docx",
    }

    assert chatbot_controller._matches_academic_policy_source(question, direct_doc)
    assert not chatbot_controller._matches_academic_policy_source(question, noisy_doc)
    assert (
        chatbot_controller._score_aggregate_evidence(question, direct_doc, {})
        > chatbot_controller._score_aggregate_evidence(question, noisy_doc, {})
    )


def test_credit_load_warning_policy_source_filter_prefers_training_regulation():
    question = "em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi"
    direct_doc = {
        "source_type": "official_document",
        "title": "Dieu 9. Dang ky khoi luong hoc tap",
        "content": "Sinh vien bi canh bao hoc tap khong duoc dang ky qua 16 tin chi.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "dieu": 9,
    }
    noisy_doc = {
        "source_type": "business_document",
        "title": "Muc I -> 2 -> 2.2",
        "content": "Thoi khoa bieu lich hoc lich thi tren Web Support SV.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "web-support-sv.docx",
    }

    assert chatbot_controller._matches_academic_policy_source(question, direct_doc)
    assert not chatbot_controller._matches_academic_policy_source(question, noisy_doc)
    assert (
        chatbot_controller._score_aggregate_evidence(question, direct_doc, {})
        > chatbot_controller._score_aggregate_evidence(question, noisy_doc, {})
    )


def test_credit_load_warning_aggregate_selection_drops_web_support_noise():
    question = "em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi"
    business_doc = {
        "source_type": "business_document",
        "title": "Muc I -> 2 -> 2.2",
        "content": "Thoi khoa bieu lich hoc lich thi tren Web Support SV.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "web-support-sv.docx",
        "chunk_index": 7,
        "keyword_score": 90,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Dieu 9. Dang ky khoi luong hoc tap",
        "content": "Sinh vien bi canh bao hoc tap khong duoc dang ky qua 16 tin chi.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "chunk_index": 19,
        "dieu": 9,
        "keyword_score": 50,
    }

    selected, debug = chatbot_controller._select_diverse_aggregate_sources(
        question,
        [business_doc],
        [internal_doc],
        {"information_need": "policy_document"},
        limit=3,
    )

    assert debug["selected_sources"][0]["source_type"] == "official_document"
    assert selected[0]["title"] == "Dieu 9. Dang ky khoi luong hoc tap"


def test_credit_load_warning_aggregate_selection_prefers_16_credit_clause():
    question = "em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi"
    generic_clause = {
        "source_type": "official_document",
        "title": "Dieu 9. Dang ky khoi luong hoc tap (1)",
        "content": (
            "Khoi luong hoc tap toi da la 3/2 so tin chi trung binh mot hoc ky. "
            "Sinh vien vua bi canh bao hoc tap o hoc ky truoc do."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "chunk_index": 18,
        "dieu": 9,
        "keyword_score": 100,
    }
    specific_clause = {
        "source_type": "official_document",
        "title": "Dieu 9. Dang ky khoi luong hoc tap (2)",
        "content": (
            "Sinh vien dang trong thoi gian bi canh bao ket qua hoc tap chi duoc "
            "dang ky khoi luong hoc tap khong qua 16 tin chi cho moi hoc ky. "
            "Quy dinh nay ap dung rieng cho sinh vien diem trung binh hoc ky yeu kem "
            "hoac dang bi canh bao ket qua hoc tap trong hoc ky moi."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "chunk_index": 19,
        "dieu": 9,
        "keyword_score": 80,
    }

    selected, debug = chatbot_controller._select_diverse_aggregate_sources(
        question,
        [],
        [generic_clause, specific_clause],
        {"information_need": "policy_document"},
        limit=2,
    )

    assert debug["selected_sources"][0]["title"] == "Dieu 9. Dang ky khoi luong hoc tap (2)"
    assert selected[0]["chunk_index"] == 19


def test_credit_load_warning_aggregate_uses_deterministic_16_credit_answer(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Muc I -> 2 -> 2.2",
        "content": "Thoi khoa bieu lich hoc lich thi tren Web Support SV.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "web-support-sv.docx",
        "chunk_index": 7,
        "keyword_score": 90,
    }
    internal_doc = {
        "source_type": "official_document",
        "title": "Dieu 9. Dang ky khoi luong hoc tap (2)",
        "content": (
            "Sinh vien dang trong thoi gian bi canh bao ket qua hoc tap chi duoc "
            "dang ky khoi luong hoc tap khong qua 16 tin chi cho moi hoc ky. "
            "Quy dinh nay ap dung rieng cho sinh vien diem trung binh hoc ky yeu kem "
            "hoac dang bi canh bao ket qua hoc tap trong hoc ky moi."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "chunk_index": 19,
        "dieu": 9,
        "keyword_score": 80,
    }

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "information_need": "policy_document",
                "retrieval_method": "generic_hybrid",
            },
        }

    async def fake_internal(state):
        return {**state, "docs": [internal_doc], "retrieval_debug": {}}

    async def fail_generate(state):
        raise AssertionError("Gemini generation should be skipped for clear 16-credit clause")

    monkeypatch.setattr(chatbot_controller, "retrieve_business", fake_business)
    monkeypatch.setattr(chatbot_controller, "retrieve_internal", fake_internal)
    monkeypatch.setattr(chatbot_controller, "generate_answer", fail_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "_retrieve_multihop_evidence",
        lambda *args, **kwargs: asyncio.sleep(0, result=([], [], {})),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_aggregate_documents(
            RagTrace("em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi"),
            "em dang bi canh bao hoc tap thi toi da duoc dang ky bao nhieu tin chi",
            "internal_document",
            "document_terms",
        )
    )

    assert "không quá 16 tín chỉ" in result["answer"]
    assert "3/2" in result["answer"]
    assert result["sources"][0]["chunk_index"] == 19


def test_transfer_school_policy_filter_rejects_master_regulation():
    question = "toi muon chuyen truong khong phai chuyen chuong trinh dao tao"
    undergraduate_doc = {
        "source_type": "official_document",
        "title": "Dieu 28. Chuyen truong",
        "content": "Sinh vien duoc chuyen truong khi co dong y cua Hieu truong va cung nganh dao tao.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "dieu": 28,
        "keyword_score": 40,
    }
    master_doc = {
        "source_type": "official_document",
        "title": "Chuyen truong thac si",
        "content": "Hoc vien thac si duoc chuyen truong theo quy che dao tao trinh do thac si.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che tuyen sinh va dao tao trinh do thac si_834_20092023.docx",
        "relative_path": "thac-si.docx",
        "keyword_score": 100,
    }

    assert chatbot_controller._matches_academic_policy_source(question, undergraduate_doc)
    assert not chatbot_controller._matches_academic_policy_source(question, master_doc)
    assert (
        chatbot_controller._score_aggregate_evidence(question, undergraduate_doc, {})
        > chatbot_controller._score_aggregate_evidence(question, master_doc, {})
    )


def test_elective_failed_course_policy_filter_prefers_article_11():
    question = "neu em bi F mon tu chon thi co the chon mon khac thay the khong"
    direct_doc = {
        "source_type": "official_document",
        "title": "Dieu 11. Hoc lai, hoc cai thien diem",
        "content": "Hoc phan tu chon bi diem F F+ thi sinh vien co the hoc doi sang hoc phan khac tuong duong.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "relative_path": "quy-che-dao-tao.docx",
        "dieu": 11,
    }
    business_doc = {
        "source_type": "business_document",
        "title": "Lop hoc phan",
        "content": "Huong dan xem lop hoc phan tren Web Support SV.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
    }

    assert chatbot_controller._matches_academic_policy_source(question, direct_doc)
    assert not chatbot_controller._matches_academic_policy_source(question, business_doc)


def test_f_grade_comparison_policy_filter_accepts_articles_16_and_11():
    question = "diem F+ va F khac nhau nhu the nao co phai hoc lai ca hai khong"
    grade_doc = {
        "source_type": "official_document",
        "title": "Dieu 16. Thang diem danh gia",
        "content": "Diem hoc phan duoc quy doi sang diem chu F+ va F theo thang diem.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "dieu": 16,
    }
    retake_doc = {
        "source_type": "official_document",
        "title": "Dieu 11. Hoc lai, hoc doi",
        "content": "Hoc phan bat buoc khong dat phai hoc lai; hoc phan tu chon co the hoc doi hoc phan tuong duong.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "dieu": 11,
    }
    noisy_doc = {
        "source_type": "official_document",
        "title": "Diem hoc vien thac si",
        "content": "Hoc vien thac si co diem F.",
        "doc_name": "quy-che-thac-si-834.docx",
    }

    assert chatbot_controller._matches_academic_policy_source(question, grade_doc)
    assert chatbot_controller._matches_academic_policy_source(question, retake_doc)
    assert not chatbot_controller._matches_academic_policy_source(question, noisy_doc)


def test_credit_definition_policy_filter_prefers_article_2():
    question = "mot tin chi tuong duong voi bao nhieu tiet hoc ly thuyet va thuc hanh"
    direct_doc = {
        "source_type": "official_document",
        "title": "Dieu 3. Phuong thuc to chuc dao tao (3)",
        "content": "Mot tin chi bang 15 tiet ly thuyet, 30 tiet thuc hanh, 45 60 gio lam tieu luan.",
        "doc_name": "DHKTKTCN_PDT_QD_2025_12_09_Quy che dao tao dai hoc chinh quy_832_20092023.docx",
        "dieu": 3,
    }
    noisy_doc = {
        "source_type": "official_document",
        "title": "Dieu kien tot nghiep",
        "content": "Sinh vien tot nghiep can du chung chi va diem trung binh tich luy.",
        "doc_name": "quy-che-dao-tao.docx",
        "dieu": 24,
    }

    assert chatbot_controller._matches_academic_policy_source(question, direct_doc)
    assert not chatbot_controller._matches_academic_policy_source(question, noisy_doc)


def test_graduation_answer_combines_requirements_and_good_classification():
    question = "dieu kien tot nghiep la gi va dieu kien tot nghiep loai gioi"
    requirements_doc = {
        "source_type": "local_file",
        "title": "Dieu 24. Dieu kien xet tot nghiep va cong nhan tot nghiep",
        "content": (
            "Dieu kien xet tot nghiep. Sinh vien duoc Truong xet va cong nhan tot nghiep "
            "khi tich luy du hoc phan, diem trung binh tich luy tu 2,00, co chung chi "
            "ngoai ngu va tin hoc."
        ),
        "doc_name": "Quy che dao tao dai hoc chinh quy.docx",
        "dieu": 24,
    }
    classification_doc = {
        "source_type": "local_file",
        "title": "Dieu 25. Cap bang tot nghiep",
        "content": (
            "Hang tot nghiep: loai gioi co diem trung binh tich luy tu 3,20 den 3,59. "
            "Hang tot nghiep bi giam mot muc neu hoc lai vuot qua 5% tong so tin chi "
            "hoac bi ky luat tu muc canh cao."
        ),
        "doc_name": "Quy che dao tao dai hoc chinh quy.docx",
        "dieu": 25,
    }

    answer, sources = chatbot_controller._graduation_classification_answer(
        question,
        [classification_doc, requirements_doc],
    )

    assert "Điều kiện được xét và công nhận tốt nghiệp" in answer
    assert "Điều kiện xếp loại tốt nghiệp giỏi" in answer
    assert "3,20 đến 3,59" in answer
    assert "vượt quá 5%" in answer
    assert [doc["dieu"] for doc in sources] == [24, 25]


def test_exam_defer_answer_combines_policy_and_procedure_sources():
    question = (
        "Dieu kien hoan thi la gi va "
        "thu tuc xin hoan thi thuc hien nhu the nao?"
    )
    policy_doc = {
        "source_type": "local_file",
        "title": "Dieu 16. Cach tinh diem hoc phan",
        "content": (
            "Diem I duoc ap dung khi sinh vien bi om hoac tai nan khong the du thi "
            "va duoc Nha truong cho phep; ly do khach quan duoc Truong Khoa chap thuan."
        ),
        "doc_name": "Quy che dao tao dai hoc chinh quy.docx",
        "dieu": 16,
    }
    procedure_doc = {
        "source_type": "local_file",
        "title": "1.5. Hoan thi",
        "content": (
            "Huong dan hoan thi tai Mot cua - Khao thi. Buoc 1 dang nhap. "
            "Buoc 2 truy cap support.uneti.edu.vn/mot-cua/khao-thi/hoan-thi. "
            "Buoc 3 dien du lieu. Buoc 4 chon hoc phan va Gui yeu cau."
        ),
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
    }

    answer, sources = chatbot_controller._exam_defer_answer(
        question,
        [policy_doc, procedure_doc],
    )

    assert "1. Điều kiện hoãn thi" in answer
    assert "2. Thủ tục xin hoãn thi" in answer
    assert "support.uneti.edu.vn/mot-cua/khao-thi/hoan-thi" in answer
    assert "MC-KT-05" in answer
    assert [doc["title"] for doc in sources] == [
        "Dieu 16. Cach tinh diem hoc phan",
        "1.5. Hoan thi",
    ]


def test_local_endpoint_retrieves_each_detected_aspect_separately(monkeypatch):
    question = (
        "Dieu kien chuyen truong la gi va "
        "ho so chuyen truong can nhung gi?"
    )
    base_doc = {
        "source_type": "local_file",
        "title": "Thong tin chung",
        "content": "Thong tin chung ve sinh vien.",
        "doc_name": "base.docx",
        "relative_path": "base.docx",
        "chunk_index": 1,
        "keyword_score": 50.0,
    }
    condition_doc = {
        "source_type": "local_file",
        "title": "Dieu kien chuyen truong",
        "content": "Cac dieu kien sinh vien duoc xem xet chuyen truong.",
        "doc_name": "condition.docx",
        "relative_path": "condition.docx",
        "chunk_index": 1,
        "keyword_score": 90.0,
    }
    dossier_doc = {
        "source_type": "local_file",
        "title": "Ho so chuyen truong",
        "content": (
            "Ho so chuyen truong gom don xin chuyen truong, bang diem "
            "va cac giay to minh chung."
        ),
        "doc_name": "dossier.docx",
        "relative_path": "dossier.docx",
        "chunk_index": 1,
        "keyword_score": 90.0,
    }
    retrieval_questions = []
    generated_states = []

    async def fake_retrieve(state):
        retrieval_questions.append(state["question"])
        if state["question"] == question:
            docs = [base_doc]
        elif state["question"] == "Dieu kien chuyen truong la gi":
            docs = [condition_doc]
        else:
            docs = [dossier_doc]
        return {**state, "docs": docs, "retrieval_debug": {}}

    async def fake_generate(state):
        generated_states.append(state)
        return {
            **state,
            "answer": (
                "[Y_1]\nTra loi dieu kien.\n[/Y_1]\n"
                "[Y_2]\nTra loi ho so.\n[/Y_2]"
            ),
        }

    monkeypatch.setattr(chatbot_controller, "retrieve_local_documents", fake_retrieve)
    monkeypatch.setattr(chatbot_controller, "generate_answer", fake_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "_has_confident_evidence",
        lambda question, docs: (bool(docs), "mock"),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_local_documents(
            RagTrace(question),
            question,
            "internal_document",
            "test_multi_aspect",
        )
    )

    assert retrieval_questions[0] == question
    assert "Dieu kien chuyen truong la gi" in retrieval_questions
    assert "ho so chuyen truong can nhung gi" in retrieval_questions
    assert len(generated_states) == 1
    assert len(generated_states[0]["required_aspects"]) == 2
    assert [doc["doc_name"] for doc in generated_states[0]["docs"][:2]] == [
        "condition.docx",
        "dossier.docx",
    ]
    assert result["answer"] == "Tra loi dieu kien.\nTra loi ho so."


def test_attendance_exam_policy_drops_business_and_prefers_training_regulation(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Diem danh tren web support",
        "content": "Huong dan sinh vien xem diem danh va so tiet vang tren web support.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "chunk_index": 1,
        "keyword_score": 100.0,
    }
    direct_internal = {
        "source_type": "official_document",
        "title": "Dieu 13. Danh gia hoc phan",
        "content": (
            "Sinh vien nghi hoc tren 50% so tiet trong chuong trinh se bi cam thi "
            "ca ky thi chinh va ky thi phu, diem thi duoc tinh la 0 diem."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "Phong Dao tao/DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "chunk_index": 26,
        "keyword_score": 100.0,
    }
    generic_internal = {
        "source_type": "official_document",
        "title": "Dieu 26. To chuc thuc hien",
        "content": "Nghi hoc dai ngay khong ly do co the bi xu ly ky luat.",
        "doc_name": "DHKTKTCN_PCTCTSV_Quy che Cong tac sinh vien.pdf",
        "relative_path": "Phong CTCTSV/DHKTKTCN_PCTCTSV_Quy che Cong tac sinh vien.pdf",
        "chunk_index": 59,
        "keyword_score": 80.0,
    }
    generated_docs = []

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "mapping_selected": True,
                "retrieval_method": "keyword",
                "information_need": "policy_document",
            },
        }

    async def fake_internal(state):
        return {
            **state,
            "docs": [generic_internal, direct_internal],
            "retrieval_debug": {},
        }

    async def fake_generate(state):
        generated_docs.append(state["docs"])
        return {**state, "answer": "Sinh vien nghi hoc tren 50% so tiet se bi cam thi."}

    monkeypatch.setattr(chatbot_controller, "retrieve_business", fake_business)
    monkeypatch.setattr(chatbot_controller, "retrieve_internal", fake_internal)
    monkeypatch.setattr(chatbot_controller, "generate_answer", fake_generate)
    monkeypatch.setattr(
        chatbot_controller,
        "_filter_usable_sources",
        lambda question, docs: (docs, []),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "_has_confident_evidence",
        lambda question, docs: (bool(docs), "mock"),
    )
    monkeypatch.setattr(
        chatbot_controller,
        "rerank_chunks",
        lambda question, docs: (docs, {"used": False, "reason": "mock"}),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_aggregate_documents(
            RagTrace("nghi hoc khong phep co bi cam thi khong"),
            "nghi hoc khong phep co bi cam thi khong",
            "internal_document",
            "document_terms",
        )
    )

    assert len(generated_docs) == 1
    assert [doc["title"] for doc in generated_docs[0]] == [direct_internal["title"]]
    assert generated_docs[0][0]["aggregate_route"] == "internal"
    assert result["sources"][0]["doc_name"].startswith("DHKTKTCN_PDT")
    assert all(source["source_type"] != "business_document" for source in result["sources"])


def test_absence_permission_comparison_does_not_select_final_exam_rule(monkeypatch):
    business_doc = {
        "source_type": "business_document",
        "title": "Diem danh tren web support",
        "content": "Huong dan sinh vien xem diem danh va so tiet vang tren web support.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "chunk_index": 1,
        "keyword_score": 100.0,
    }
    attendance_internal = {
        "source_type": "official_document",
        "title": "Dieu 13. Danh gia hoc phan",
        "content": (
            "Diem chuyen can duoc danh gia theo thoi gian tham gia hoc tap tren lop. "
            "Sinh vien nghi hoc tren 50% so tiet trong chuong trinh se bi cam thi."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "Phong Dao tao/DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "chunk_index": 26,
        "keyword_score": 95.0,
    }
    final_exam_internal = {
        "source_type": "official_document",
        "title": "Dieu 15. Thi ket thuc hoc phan",
        "content": (
            "Sinh vien vang mat trong ky thi ket thuc hoc phan neu co ly do chinh dang "
            "duoc du thi ky thi phu."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "Phong Dao tao/DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "chunk_index": 30,
        "keyword_score": 120.0,
    }
    generated_docs = []

    async def fake_business(state):
        return {
            **state,
            "docs": [business_doc],
            "retrieval_debug": {
                "mapping_selected": True,
                "retrieval_method": "keyword",
                "information_need": "policy_document",
            },
        }

    async def fake_internal(state):
        return {
            **state,
            "docs": [final_exam_internal, attendance_internal],
            "retrieval_debug": {},
        }

    async def fake_generate(state):
        generated_docs.append(state["docs"])
        return {**state, "answer": "Tai lieu neu cach tinh diem chuyen can theo so tiet vang."}

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
        "rerank_chunks",
        lambda question, docs: (docs, {"used": False, "reason": "mock"}),
    )

    result = asyncio.run(
        chatbot_controller._answer_with_aggregate_documents(
            RagTrace("nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi"),
            "nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi",
            "internal_document",
            "document_terms",
        )
    )

    assert len(generated_docs) == 1
    assert [doc["title"] for doc in generated_docs[0]] == ["Dieu 13. Danh gia hoc phan"]
    assert result["sources"][0]["title"] == "Dieu 13. Danh gia hoc phan"
    assert all(source["title"] != "Dieu 15. Thi ket thuc hoc phan" for source in result["sources"])
    assert all(source["source_type"] != "business_document" for source in result["sources"])


def test_absence_permission_comparison_partial_fallback_uses_attendance_evidence():
    doc = {
        "source_type": "official_document",
        "title": "Dieu 13. Danh gia hoc phan",
        "content": (
            "Diem chuyen can duoc danh gia theo thoi gian tham gia hoc tap tren lop. "
            "Co nghi hoc; nghi hoc duoi 10% so tiet trong chuong trinh duoc tinh 8 diem. "
            "Nghi hoc tu 50% tro len: 0 diem. Sinh vien nghi hoc tren 50% so tiet se bi cam thi."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "Phong Dao tao/DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "chunk_index": 26,
        "keyword_score": 95.0,
    }

    answer, docs = chatbot_controller._absence_permission_comparison_answer(
        "nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi",
        [doc],
    )

    assert answer is not None
    assert "chưa nêu rõ" in answer
    assert "Nghỉ từ 50% trở lên: 0 điểm" in answer
    assert docs == [doc]


def test_absence_permission_comparison_fallback_keeps_support_and_policy_sources():
    support_doc = {
        "source_type": "business_document",
        "title": "Theo doi diem danh",
        "content": "He thong hien thi so tiet vang, nghi co phep va nghi khong phep cua sinh vien.",
        "doc_name": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "relative_path": "2026.03.25.AI_HDSD TREN WEB SUPPORT SV.docx",
        "chunk_index": 9,
        "keyword_score": 100.0,
    }
    policy_doc = {
        "source_type": "official_document",
        "title": "Dieu 13. Danh gia hoc phan",
        "content": (
            "Diem chuyen can duoc danh gia theo thoi gian tham gia hoc tap tren lop. "
            "Co nghi hoc; nghi hoc duoi 10% so tiet trong chuong trinh duoc tinh 8 diem. "
            "Nghi hoc tu 50% tro len: 0 diem. Sinh vien nghi hoc tren 50% so tiet se bi cam thi."
        ),
        "doc_name": "DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "relative_path": "Phong Dao tao/DHKTKTCN_PDT_QD_Quy che dao tao dai hoc chinh quy.docx",
        "chunk_index": 26,
        "keyword_score": 95.0,
    }

    answer, docs = chatbot_controller._absence_permission_comparison_answer(
        "nghi hoc khong phep va nghi hoc co phep khac nhau nhung gi",
        [support_doc, policy_doc],
    )

    assert answer is not None
    assert "ghi nhận riêng" in answer
    assert "không nêu" in answer
    assert [doc["source_type"] for doc in docs] == ["business_document", "official_document"]


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
            ChatRequest(
                question="đối tượng được miễn giảm học phí",
                request_id="controller-false-positive",
            )
        )
    )

    assert len(generated_docs) == 1
    assert [doc["doc_name"] for doc in generated_docs[0]] == [internal_doc["doc_name"]]
    assert generated_docs[0][0]["lexical_coverage"] == 1.0
    assert generated_docs[0][0]["aggregate_route"] == "internal"
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
            ChatRequest(
                question="miễn giảm học phí",
                request_id="controller-retry-internal",
            )
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
            ChatRequest(
                question="đối tượng được miễn giảm học phí",
                request_id="controller-no-evidence",
            )
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


def test_ambiguous_chat_continues_to_retrieval(monkeypatch):
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

    async def fake_retrieval(trace, question, intent, reason, ambiguity_decision=None):
        return {
            "question": question,
            "answer": "Khong tim thay can cu du ro trong tai lieu da cung cap.",
            "source": None,
            "sources": [],
            "intent": intent,
        }

    monkeypatch.setattr(chatbot_controller, "analyze_ambiguity", lambda question: Decision())
    monkeypatch.setattr(
        chatbot_controller,
        "_answer_with_aggregate_documents",
        fake_retrieval,
    )

    result = asyncio.run(
        chatbot_controller.handle_chat(
            ChatRequest(
                question="xtet đầu ra ta4 kiểu gì",
                request_id="controller-ambiguous",
            )
        )
    )

    assert result["intent"] != "clarification_needed"
    assert result["sources"] == []
    assert result["answer"] == "Khong tim thay can cu du ro trong tai lieu da cung cap."


def test_probe_failure_returns_no_evidence_after_retrieval(monkeypatch):
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

    assert result["answer"] == chatbot_controller.NO_EVIDENCE_ANSWER
    assert result["intent"] != "clarification_needed"


def test_gemini_error_uses_extractive_fallback_from_sources(monkeypatch):
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

    assert "Sinh viên phải tích lũy" in result["answer"]
    assert "Điều kiện tốt nghiệp" in result["answer"]
    assert traces[-1][1]["fallback_used"] is True
    assert traces[-1][1]["fallback_reason"] == "gemini_unavailable"
    assert "He thong AI dang ban" in traces[-1][1]["gemini_error_message"]


def test_gemini_exception_uses_extractive_fallback_from_sources(monkeypatch):
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

    assert "camera" in result["answer"]
    assert "Quy định camera" in result["answer"]


def test_gemini_error_uses_single_business_procedure_source(monkeypatch):
    monkeypatch.setattr(
        langchain_pipeline,
        "ask_gemini",
        lambda prompt: "Loi khi goi Gemini API. Vui long thu lai sau.",
    )
    state = {
        "question": "Tôi kiểm tra số giờ coi thi và chấm thi ở đâu?",
        "prompt": "prompt",
        "docs": [
            {
                "title": "Màn Khối lượng coi - chấm thi",
                "doc_name": "AI_HDSD TREN WEB SUPPORT CBGV.docx",
                "document_type": "business_document",
                "content": (
                    "Tài liệu hướng dẫn\n"
                    "Chức năng: Xem khối lượng coi thi và chấm thi.\n"
                    "B1: Đăng nhập tại https://support.uneti.edu.vntruy cập trực tiếp đường dẫn:\n"
                    "https://support.uneti.edu.vn/cong-tac-giang-vien/tra-cuu/khoi-luong-coi-cham-thi\n"
                    "B3: Chọn năm học và học kỳ.\n"
                    "B4: Xem số tiết và trạng thái.\n"
                    "Lưu ý: Kiểm tra kỹ số tiết."
                ),
            },
            {
                "title": "Quy định thanh tra",
                "doc_name": "quy-dinh.docx",
                "content": "Thanh tra công tác chấm thi.",
            },
        ],
    }

    result = asyncio.run(langchain_pipeline._generate_answer(state))

    assert "Xem khối lượng coi thi và chấm thi" in result["answer"]
    assert "support.uneti.edu.vn/cong-tac-giang-vien" in result["answer"]
    assert "Chọn năm học và học kỳ" in result["answer"]
    assert "Quy định thanh tra" not in result["answer"]


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
            ChatRequest(
                question="How do I reset my password?",
                request_id="controller-business-reset",
            )
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


@pytest.mark.parametrize(
    ("question", "expected_parts"),
    [
        (
            "Man Nhan su dung de lam gi?",
            ["thông tin nhân sự cá nhân", "khối lượng giảm trừ"],
        ),
        (
            "Muon xem lop hoc phan giang vien thi vao duong dan nao?",
            ["https://support.uneti.edu.vn/cong-tac-giang-vien/tra-cuu/lop-hoc-phan-giang-vien"],
        ),
        (
            "Toi muon dang ky muon thiet bi phong hoc thi lam the nao?",
            ["Đăng nhập", "Đăng ký sử dụng thiết bị", "Chọn lịch dạy", "Gửi yêu cầu"],
        ),
        (
            "Quy trinh xu ly ho so thu tuc hanh chinh gom may buoc?",
            ["5 bước", "Nộp hồ sơ", "Tiếp nhận hồ sơ", "Xử lý hồ sơ", "Phê duyệt hồ sơ", "Trả kết quả"],
        ),
        (
            "Quy trình xử lý hồ sơ thủ tục hành chính giảng viên",
            ["5 bước", "Nộp hồ sơ", "Tiếp nhận hồ sơ", "Xử lý hồ sơ", "Phê duyệt hồ sơ", "Trả kết quả"],
        ),
        (
            "Trang thai minh chung kiem dinh gom nhung gi?",
            ["Chờ duyệt", "Đã duyệt", "Cần bổ sung"],
        ),
    ],
)
def test_business_chat_answers_cbgv_support_questions_directly(question, expected_parts):
    result = asyncio.run(
        chatbot_controller.handle_business_chat(
            ChatRequest(question=question, request_id="controller-business-direct")
        )
    )

    assert "Thông tin tóm tắt từ các nguồn đã truy xuất" not in result["answer"]
    assert result["source"].endswith("2026.03.25.AI_HDSD TREN WEB SUPPORT CBGV.docx")
    for expected in expected_parts:
        assert expected in result["answer"]
