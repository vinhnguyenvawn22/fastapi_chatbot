from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import math
import re
import zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader

from app.controller.document_controller import build_document_chunks, chunk_text, list_documents
from app.core.config import BUSINESS_DOCUMENTS_DIR, BUSINESS_SEARCH_TOP_K, MIN_SEARCH_SCORE
from app.data.elasticsearch_client import get_keywords, normalize_text


_BUSINESS_INDEX_CACHE = {
    "signature": None,
    "chunks": [],
    "doc_freq": Counter(),
    "total_docs": 0,
}
_BUSINESS_SEARCH_CACHE = {}

_XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def clear_business_knowledge_cache():
    _BUSINESS_INDEX_CACHE["signature"] = None
    _BUSINESS_INDEX_CACHE["chunks"] = []
    _BUSINESS_INDEX_CACHE["doc_freq"] = Counter()
    _BUSINESS_INDEX_CACHE["total_docs"] = 0
    _BUSINESS_SEARCH_CACHE.clear()


def _business_path() -> Path:
    return Path(BUSINESS_DOCUMENTS_DIR).resolve()


def _supported_files() -> list[Path]:
    root = _business_path()
    if not root.exists():
        return []

    files = []
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("~$"):
            continue
        if file_path.suffix.lower() in {".pdf", ".docx"}:
            files.append(file_path)

    return sorted(files)


def _signature(files: list[Path], official_files: list[dict]):
    business_signature = tuple(
        (
            str(file_path.relative_to(_business_path())),
            file_path.stat().st_size,
            file_path.stat().st_mtime,
        )
        for file_path in files
    )
    official_signature = tuple(
        (
            file.get("relative_path") or file.get("file_name"),
            file.get("file_size_kb"),
            file.get("updated_at"),
        )
        for file in official_files
    )
    return business_signature, official_signature


def _clean_text(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    parts = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"Trang {page_index}: {text}")
    return "\n".join(parts)


def _extract_docx_text(file_path: Path) -> str:
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    parts = []

    with zipfile.ZipFile(file_path) as archive:
        if "word/document.xml" not in archive.namelist():
            return ""

        root = ET.fromstring(archive.read("word/document.xml"))
        for paragraph in root.findall(".//w:p", namespace):
            texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
            text = _clean_text("".join(texts))
            if text:
                parts.append(text)

    return "\n".join(parts)


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings = []
    for item in root.findall("main:si", _XLSX_NS):
        texts = [node.text or "" for node in item.findall(".//main:t", _XLSX_NS)]
        strings.append(_clean_text("".join(texts)))
    return strings


def _xlsx_sheet_paths(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib.get("Id"): rel.attrib.get("Target", "")
        for rel in rels.findall("rel:Relationship", _XLSX_NS)
    }

    sheets = []
    for sheet in workbook.findall("main:sheets/main:sheet", _XLSX_NS):
        name = sheet.attrib.get("name", "Sheet")
        rel_id = sheet.attrib.get(f"{{{_XLSX_NS['office_rel']}}}id")
        target = rel_map.get(rel_id)
        if target:
            sheets.append((name, f"xl/{target.lstrip('/')}"))
    return sheets


def _xlsx_cell_text(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find("main:v", _XLSX_NS)
    if value is None or value.text is None:
        inline = cell.find(".//main:t", _XLSX_NS)
        return _clean_text(inline.text if inline is not None else "")

    raw_value = value.text
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(raw_value)]
        except (ValueError, IndexError):
            return ""
    return _clean_text(raw_value)


def _extract_xlsx_text(file_path: Path) -> str:
    rows = []
    with zipfile.ZipFile(file_path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        for sheet_name, sheet_path in _xlsx_sheet_paths(archive):
            if sheet_path not in archive.namelist():
                continue
            sheet_root = ET.fromstring(archive.read(sheet_path))
            for row in sheet_root.findall(".//main:row", _XLSX_NS):
                values = [
                    _xlsx_cell_text(cell, shared_strings)
                    for cell in row.findall("main:c", _XLSX_NS)
                ]
                values = [value for value in values if value]
                if values:
                    rows.append(f"{sheet_name}: " + " | ".join(values))
    return "\n".join(rows)


def _extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(file_path)
    if suffix == ".docx":
        return _extract_docx_text(file_path)
    if suffix == ".xlsx":
        return _extract_xlsx_text(file_path)
    return ""


def _score_chunk(query: str, chunk: dict, doc_freq: Counter, total_docs: int) -> float:
    query_tokens = get_keywords(query)
    if not query_tokens:
        return 0.0

    token_counts = chunk.get("_token_counts") or Counter()
    title_tokens = set(get_keywords(chunk.get("title", "")))
    score = 0.0

    for token in query_tokens:
        if token not in token_counts:
            continue

        idf = math.log((1 + total_docs) / (1 + doc_freq.get(token, 0))) + 1
        score += token_counts[token] * idf
        if token in title_tokens:
            score += 5.0 * idf

    normalized_query = normalize_text(query)
    normalized_title = normalize_text(chunk.get("title", ""))
    normalized_content = normalize_text(chunk.get("content", ""))
    doc_name = normalize_text(chunk.get("doc_name", ""))

    if "tot nghiep" in normalized_query and "dieu kien" in normalized_query:
        if "dieu 24" in normalized_content or "dieu 24" in normalized_title:
            score += 80.0
        if "sinh vien duoc truong xet va cong nhan tot nghiep" in normalized_content:
            score += 120.0
        if "cac dieu kien sau" in normalized_content:
            score += 60.0
        if re.search(r"\ba\)\s+cho den thoi diem xet tot nghiep", normalized_content):
            score += 40.0
        if re.search(r"\bd\)\s+co cac chung chi", normalized_content):
            score += 35.0
        if "khoa luan" in normalized_content or "do an tot nghiep" in normalized_content:
            score -= 45.0
        if "hang tot nghiep" in normalized_content or "cap bang tot nghiep" in normalized_content:
            score -= 45.0

    if doc_name.endswith("mapping.docx") or doc_name.endswith("danh gia.xlsx"):
        score *= 0.45

    return round(max(score, 0.0), 4)


def _load_business_index():
    files = _supported_files()
    official_files = list_documents()
    signature = _signature(files, official_files)

    if _BUSINESS_INDEX_CACHE["signature"] == signature:
        return (
            _BUSINESS_INDEX_CACHE["chunks"],
            _BUSINESS_INDEX_CACHE["doc_freq"],
            _BUSINESS_INDEX_CACHE["total_docs"],
        )

    chunks = []
    doc_freq = Counter()
    root = _business_path()

    for file_path in files:
        try:
            text = _extract_text(file_path)
        except Exception:
            continue

        relative_path = file_path.relative_to(root).as_posix()
        stat = file_path.stat()
        split_chunks = chunk_text(text, chunk_size=1200, overlap=180)

        for index, content in enumerate(split_chunks, start=1):
            title = f"{file_path.stem} ({index})" if len(split_chunks) > 1 else file_path.stem
            tokens = get_keywords(f"{title} {content}")
            chunk = {
                "doc_name": file_path.name,
                "relative_path": relative_path,
                "source_root": root.name,
                "title": title,
                "content": content,
                "chunk_index": index,
                "source_type": "business_document",
                "file_path": str(file_path),
                "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                "_token_counts": Counter(tokens),
                "_token_set": set(tokens),
            }
            chunks.append(chunk)
            doc_freq.update(chunk["_token_set"])

    for file in official_files:
        try:
            file_chunks = build_document_chunks(file.get("relative_path") or file["file_name"])
        except Exception:
            continue

        for chunk in file_chunks:
            tokens = get_keywords(f"{chunk.get('title', '')} {chunk.get('content', '')}")
            chunk["_token_counts"] = Counter(tokens)
            chunk["_token_set"] = set(tokens)
            chunks.append(chunk)
            doc_freq.update(chunk["_token_set"])

    _BUSINESS_INDEX_CACHE["signature"] = signature
    _BUSINESS_INDEX_CACHE["chunks"] = chunks
    _BUSINESS_INDEX_CACHE["doc_freq"] = doc_freq
    _BUSINESS_INDEX_CACHE["total_docs"] = len(chunks)
    _BUSINESS_SEARCH_CACHE.clear()

    return chunks, doc_freq, len(chunks)


def search_business_sources(query: str, limit: int | None = None, debug: dict | None = None) -> list[dict]:
    query = str(query or "").strip()
    limit = limit or BUSINESS_SEARCH_TOP_K
    cache_key = (normalize_text(query), limit)

    if cache_key in _BUSINESS_SEARCH_CACHE:
        cached = deepcopy(_BUSINESS_SEARCH_CACHE[cache_key])
        if debug is not None:
            debug.update(cached.get("debug", {}))
            debug["cache_hit"] = True
        return cached.get("results", [])

    chunks, doc_freq, total_docs = _load_business_index()
    results = []

    for chunk in chunks:
        score = _score_chunk(query, chunk, doc_freq, total_docs)
        if score < MIN_SEARCH_SCORE:
            continue

        clean_chunk = {
            key: value
            for key, value in chunk.items()
            if not key.startswith("_")
        }
        clean_chunk["score"] = score
        clean_chunk["keyword_score"] = score
        results.append(clean_chunk)

    results.sort(key=lambda item: item["score"], reverse=True)
    final_results = results[:limit]
    debug_data = {
        "cache_hit": False,
        "business_documents_dir": str(_business_path()),
        "indexed_chunk_count": total_docs,
        "candidate_count": len(results),
        "final_results_count": len(final_results),
        "final_sources": [
            {
                "title": item.get("title"),
                "doc_name": item.get("doc_name"),
                "relative_path": item.get("relative_path"),
                "score": item.get("score"),
            }
            for item in final_results
        ],
    }

    _BUSINESS_SEARCH_CACHE[cache_key] = {
        "results": deepcopy(final_results),
        "debug": deepcopy(debug_data),
    }

    if debug is not None:
        debug.update(debug_data)

    return final_results
