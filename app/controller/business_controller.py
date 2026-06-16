import re

from app.controller.chatbot_controller import _build_sources, _has_confident_evidence
from app.data.business_knowledge import build_business_faq_answer, search_business_sources
from app.data.elasticsearch_client import get_keywords, normalize_text
from app.data.query_analyzer import QueryIntent, classify_query


def _add_trace_step(trace: list[dict], name: str, status: str = "ok", **data):
    step = {
        "name": name,
        "status": status,
    }
    step.update(data)
    trace.append(step)


def _source_identity(doc: dict):
    for field_name in ("url", "link", "source_url", "attachment_url"):
        value = doc.get(field_name)
        if value:
            return str(value).strip().lower()

    if doc.get("doc_name") and doc.get("chunk_index") is not None:
        return (
            str(doc.get("doc_name")).strip().lower(),
            str(doc.get("chunk_index")),
        )

    for field_name in ("doc_name", "ten_van_ban"):
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


def _extract_business_answer(docs: list[dict], query: str) -> str:
    """Tao cau tra loi ngan tu nguon nghiep vu, khong goi LLM."""
    if not docs:
        return "Khong tim thay thong tin phu hop trong bo tai lieu nghiep vu."

    faq_answer = build_business_faq_answer(docs)
    if faq_answer:
        return faq_answer

    normalized_query = normalize_text(query)
    source_name = docs[0].get("title") or docs[0].get("doc_name")

    if "dieu kien" in normalized_query and "tot nghiep" in normalized_query:
        combined_content = " ".join(str(doc.get("content") or "") for doc in docs[:3])
        anchor = re.search(
            r"Sinh\s+viên\s+được\s+Trường\s+xét\s+và\s+công\s+nhận\s+tốt\s+nghiệp\s+khi\s+có\s+đủ\s+các\s+điều\s+kiện\s+sau\s*:",
            combined_content,
            flags=re.IGNORECASE,
        )
        if anchor:
            combined_content = combined_content[anchor.end():]
            next_section = re.search(r"\s+2\.\s+Sau\s+mỗi\s+học\s+kỳ", combined_content, flags=re.IGNORECASE)
            if next_section:
                combined_content = combined_content[:next_section.start()]

        condition_items = []
        seen_labels = set()

        for match in re.finditer(
            r"(?P<label>a|b|c|d|đ|e)\)\s+(?P<text>.*?)(?=\s+(?:a|b|c|d|đ|e)\)\s+|\s+2\.\s+|$)",
            combined_content,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            label = match.group("label").lower()
            text = " ".join(match.group("text").split()).strip(" ;.")
            if label in seen_labels or not text:
                continue
            if "tot nghiep" not in normalize_text(text) and label == "a":
                pass
            condition_items.append(f"- {label}) {text}.")
            seen_labels.add(label)

        ordered_items = []
        for label in ("a", "b", "c", "d", "đ", "e"):
            for item in condition_items:
                if item.startswith(f"- {label})"):
                    ordered_items.append(item)
                    break

        if len(ordered_items) >= 4:
            source_text = f"Nguon: {source_name}\n" if source_name else ""
            return (
                source_text
                + "Sinh vien duoc xet va cong nhan tot nghiep khi co du cac dieu kien:\n"
                + "\n".join(ordered_items)
            )

    sources = _build_sources(docs[:2], query)
    keywords = set(get_keywords(query))
    candidates = []

    for index, source in enumerate(sources, start=1):
        preview = source.get("preview") or ""
        preview = preview.replace("Tom tat nguon:", "").replace("Tóm tắt nguồn:", "").strip()
        if not preview:
            continue

        for line in preview.splitlines():
            line = line.strip(" -")
            if len(line) < 24:
                continue

            normalized_line = normalize_text(line)
            score = sum(1 for keyword in keywords if keyword in normalized_line)
            if "ai " in normalize_text(query) and any(
                phrase in normalized_line
                for phrase in (" do ", "trach nhiem", "quan ly", "truong don vi")
            ):
                score += 4
            if score <= 0:
                continue
            candidates.append((score, index, line))

    if not candidates:
        return "Da tim thay nguon nghiep vu phu hop, vui long xem danh sach sources de doi chieu."

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected_lines = []
    seen = set()

    for _, source_index, line in candidates:
        normalized_line = normalize_text(line)
        if normalized_line in seen:
            continue
        selected_lines.append(f"- {line}")
        seen.add(normalized_line)
        if len(selected_lines) >= 4:
            break

    source_names = [
        source.get("title") or source.get("doc_name")
        for source in sources
        if source.get("title") or source.get("doc_name")
    ]
    source_text = f"Nguon: {source_names[0]}\n" if source_names else ""

    return source_text + "\n".join(selected_lines)


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

    retrieval_debug = {}
    docs = search_business_sources(query, debug=retrieval_debug)
    _add_trace_step(
        trace,
        "business_retrieval",
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
    sources = _build_sources(selected_docs, query)
    answer = _extract_business_answer(selected_docs, query)
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
        "answer": answer,
        "sources": sources,
        "trace": trace,
    }
