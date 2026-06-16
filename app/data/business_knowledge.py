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
FAQ_MAPPING_DOC_NAME = "PCNTT_MAPPING_FILE.docx"
BUSINESS_FAQ_SOURCE_TYPE = "business_faq_mapping"
BUSINESS_FAQ_MIN_SCORE = max(MIN_SEARCH_SCORE, 14.0)

_XLSX_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "office_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
_DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_BUSINESS_FAQ_QUERY_EXPANSION = {
    "web support": ["support uneti", "support.uneti.edu.vn"],
    "support": ["web support", "support uneti", "support.uneti.edu.vn"],
    "xem diem": ["ket qua hoc tap", "diem hoc ky", "diem thanh phan"],
    "diem": ["ket qua hoc tap", "diem hoc ky", "diem thanh phan"],
    "thoi khoa bieu": ["lich hoc", "lich thi"],
    "lich hoc": ["hoc tap", "lich hoc lich thi"],
    "lich thi": ["hoc tap", "lich hoc lich thi"],
    "bao hong": ["bao hong thiet bi", "su co thiet bi", "phong hoc", "giang duong"],
    "hong": ["bao hong", "su co", "thiet bi"],
    "may chieu": ["thiet bi", "bao hong thiet bi", "phong hoc", "giang duong"],
    "may tinh": ["thiet bi", "bao hong thiet bi", "phong hoc", "giang duong"],
    "quen mat khau": ["email google workspace", "lms", "tu khac phuc"],
    "email truong": ["email google workspace", "lms"],
    "khoi luong": ["cong tac giang vien", "khoi luong giang day", "coi thi", "cham thi"],
    "coi thi": ["khoi luong coi cham thi", "cong tac giang vien"],
    "cham thi": ["khoi luong coi cham thi", "cong tac giang vien"],
    "khao sat": ["khao sat noi bo", "khao sat bat buoc", "phieu khao sat"],
    "bi chan": ["khao sat noi bo bat buoc", "hoan thanh khao sat"],
    "khong su dung duoc": ["khao sat noi bo bat buoc", "hoan thanh khao sat"],
    "mot cua": ["thu tuc hanh chinh", "danh gia thu tuc", "thong ke mot cua"],
    "danh gia thu tuc": ["muc do hai long", "thu tuc hanh chinh mot cua"],
    "tin tuc thong bao": ["module tin tuc thong bao", "thong bao tren support"],
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


def _docx_cell_text(cell: ET.Element) -> str:
    texts = [node.text or "" for node in cell.findall(".//w:t", _DOCX_NS)]
    return _clean_text("".join(texts))


def _extract_docx_tables(file_path: Path) -> list[list[list[str]]]:
    tables = []

    with zipfile.ZipFile(file_path) as archive:
        if "word/document.xml" not in archive.namelist():
            return []

        root = ET.fromstring(archive.read("word/document.xml"))
        for table in root.findall(".//w:tbl", _DOCX_NS):
            rows = []
            for row in table.findall("./w:tr", _DOCX_NS):
                cells = [
                    _docx_cell_text(cell)
                    for cell in row.findall("./w:tc", _DOCX_NS)
                ]
                if any(cells):
                    rows.append(cells)
            if rows:
                tables.append(rows)

    return tables


def _doc_name_with_extension(source_file_name: str) -> str:
    source_file_name = _clean_text(source_file_name)
    if not source_file_name:
        return ""
    if Path(source_file_name).suffix.lower() in {".docx", ".pdf", ".xlsx"}:
        return source_file_name
    return f"{source_file_name}.docx"


def _resolve_relative_path(root: Path, source_file_name: str, fallback: str) -> str:
    doc_name = _doc_name_with_extension(source_file_name)
    if doc_name:
        candidate = root / doc_name
        if candidate.exists():
            return candidate.relative_to(root).as_posix()
    return fallback


def _parse_file_mapping(table: list[list[str]]) -> dict[str, dict]:
    file_map = {}
    rows = table[1:] if table else []

    for row in rows:
        if len(row) < 2:
            continue

        file_id = _clean_text(row[0])
        source_file_name = _clean_text(row[1])
        if not file_id or not source_file_name:
            continue

        file_map[file_id] = {
            "file_id": file_id,
            "source_file_name": source_file_name,
            "structure": _clean_text(row[3] if len(row) > 3 else ""),
            "audience": _clean_text(row[4] if len(row) > 4 else ""),
        }

    if "PCNTT_FILE_02" not in file_map:
        for row in rows:
            source_file_name = _clean_text(row[1] if len(row) > 1 else "")
            audience = _clean_text(row[4] if len(row) > 4 else "")
            normalized_row = normalize_text(f"{source_file_name} {audience}")
            if "support sv" in normalized_row:
                file_map["PCNTT_FILE_02"] = {
                    "file_id": "PCNTT_FILE_02",
                    "source_file_name": source_file_name,
                    "structure": _clean_text(row[3] if len(row) > 3 else ""),
                    "audience": audience,
                }
                break

    if "PCNTT_FILE_02" not in file_map:
        for row in rows:
            source_file_name = _clean_text(row[1] if len(row) > 1 else "")
            audience = _clean_text(row[4] if len(row) > 4 else "")
            normalized_row = normalize_text(f"{source_file_name} {audience}")
            if "support" in normalized_row and "sinh vien" in normalized_row:
                file_map["PCNTT_FILE_02"] = {
                    "file_id": "PCNTT_FILE_02",
                    "source_file_name": source_file_name,
                    "structure": _clean_text(row[3] if len(row) > 3 else ""),
                    "audience": audience,
                }
                break

    return file_map


def _expanded_business_faq_query(query: str) -> str:
    normalized_query = normalize_text(query)
    expanded_terms = [query]

    for key, terms in _BUSINESS_FAQ_QUERY_EXPANSION.items():
        if key in normalized_query:
            expanded_terms.extend(terms)

    return " ".join(dict.fromkeys(expanded_terms))


def _faq_keyword_phrases(keywords: str) -> list[str]:
    return [
        normalize_text(part)
        for part in re.split(r"[,;|]", keywords or "")
        if normalize_text(part)
    ]


def _build_business_faq_rows(file_path: Path, root: Path) -> list[dict]:
    try:
        tables = _extract_docx_tables(file_path)
    except Exception:
        return []

    if len(tables) < 2:
        return []

    file_map = _parse_file_mapping(tables[0])
    fallback_relative_path = file_path.relative_to(root).as_posix()
    rows = []

    for table_index, table in enumerate(tables[1:], start=2):
        for row_index, row in enumerate(table[1:], start=1):
            if len(row) < 6:
                continue

            stt = _clean_text(row[0])
            file_id = _clean_text(row[1])
            question = _clean_text(row[2])
            answer = _clean_text(row[3])
            location = _clean_text(row[4])
            keywords = _clean_text(row[5])

            if not file_id or not question or not answer:
                continue

            source_info = file_map.get(file_id, {})
            source_file_name = source_info.get("source_file_name") or file_id
            doc_name = _doc_name_with_extension(source_file_name) or file_path.name
            relative_path = _resolve_relative_path(root, source_file_name, fallback_relative_path)
            audience = source_info.get("audience") or ""
            title = question
            content = "\n".join([
                f"Cau hoi thuong gap: {question}",
                f"Cau tra loi chuan: {answer}",
                f"Vi tri chinh xac trong file goc: {location}",
                f"Tu khoa tim kiem: {keywords}",
                f"Doi tuong: {audience}",
            ]).strip()

            rows.append({
                "doc_name": doc_name,
                "relative_path": relative_path,
                "mapping_relative_path": fallback_relative_path,
                "source_root": root.name,
                "title": title,
                "content": content,
                "chunk_index": int(stt) if stt.isdigit() else row_index,
                "source_type": BUSINESS_FAQ_SOURCE_TYPE,
                "file_path": str(root / relative_path),
                "updated_at": datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat(),
                "file_id": file_id,
                "faq_question": question,
                "faq_answer": answer,
                "faq_location": location,
                "faq_keywords": keywords,
                "audience": audience,
                "mapping_table_index": table_index,
                "ten_van_ban": source_file_name,
            })

    return rows


def _extract_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    parts = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"Trang {page_index}: {text}")
    return "\n".join(parts)


def _extract_docx_text(file_path: Path) -> str:
    parts = []

    with zipfile.ZipFile(file_path) as archive:
        if "word/document.xml" not in archive.namelist():
            return ""

        root = ET.fromstring(archive.read("word/document.xml"))
        for paragraph in root.findall(".//w:p", _DOCX_NS):
            texts = [node.text or "" for node in paragraph.findall(".//w:t", _DOCX_NS)]
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


def _score_business_faq(query: str, chunk: dict) -> float:
    expanded_query = _expanded_business_faq_query(query)
    query_tokens = set(get_keywords(expanded_query))
    original_tokens = set(get_keywords(query))
    if not query_tokens:
        return 0.0

    question = chunk.get("faq_question") or chunk.get("title") or ""
    answer = chunk.get("faq_answer") or ""
    keywords = chunk.get("faq_keywords") or ""
    audience = chunk.get("audience") or ""
    location = chunk.get("faq_location") or ""

    question_tokens = set(get_keywords(question))
    answer_tokens = set(get_keywords(answer))
    keyword_tokens = set(get_keywords(keywords))
    audience_tokens = set(get_keywords(audience))
    location_tokens = set(get_keywords(location))

    score = 0.0
    score += len(query_tokens & question_tokens) * 9.0
    score += len(query_tokens & keyword_tokens) * 11.0
    score += len(query_tokens & answer_tokens) * 3.0
    score += len(query_tokens & audience_tokens) * 4.0
    score += len(query_tokens & location_tokens) * 1.0

    original_overlap = len(original_tokens & (question_tokens | keyword_tokens | answer_tokens))
    if original_overlap >= 2:
        score += original_overlap * 4.0

    normalized_query = normalize_text(query)
    normalized_question = normalize_text(question)
    normalized_keywords = normalize_text(keywords)
    normalized_answer = normalize_text(answer)

    if normalized_query and normalized_query in normalized_question:
        score += 100.0
    elif normalized_question and normalized_question in normalized_query:
        score += 80.0

    for phrase in _faq_keyword_phrases(keywords):
        if phrase and phrase in normalized_query:
            score += 30.0
        elif phrase and all(token in query_tokens for token in get_keywords(phrase)):
            score += 12.0

    if any(term in normalized_query for term in ("lam the nao", "lam sao", "cach", "nhu the nao", "o dau")):
        if any(term in normalized_question for term in ("lam the nao", "cach", "o dau", "truy cap")):
            score += 20.0
        if any(term in normalized_answer for term in ("buoc 1", "dang nhap", "chon", "truy cap")):
            score += 12.0
        if any(term in normalized_question for term in ("muc dich", "dung de lam gi", "giup")):
            score -= 45.0

    if "o dau" in normalized_query and "o dau" in normalized_question:
        score += 70.0

    if "nhu the nao" in normalized_query and "nhu the nao" in normalized_question:
        score += 70.0

    if "lam the nao" in normalized_query and "lam the nao" in normalized_question:
        score += 70.0

    combined_normalized = normalize_text(f"{question} {answer} {keywords}")

    if "diem" in original_tokens:
        if "diem" in combined_normalized:
            score += 35.0
        else:
            score -= 25.0

    if any(term in normalized_query for term in ("bao hong", "hong", "may chieu", "may tinh")):
        if any(term in combined_normalized for term in ("bao hong", "thiet bi", "su co")):
            score += 25.0
        if any(term in normalized_query for term in ("phong hoc", "giang duong")) and any(
            term in combined_normalized for term in ("phong", "toa nha", "giang duong")
        ):
            score += 15.0

    if any(term in normalized_query for term in ("coi thi", "cham thi")):
        if "coi thi" in combined_normalized:
            score += 35.0
        if "cham thi" in combined_normalized:
            score += 35.0

    if "khoi luong" in normalized_query:
        if chunk.get("file_id") == "PCNTT_FILE_03":
            score += 15.0
        if "khoi luong" not in combined_normalized:
            score -= 15.0

    if any(term in normalized_query for term in ("sinh vien", "sv")):
        if "sinh vien" in normalize_text(audience) or "sinh vien" in normalized_question:
            score += 18.0
        if "giang vien" in normalize_text(audience) and "sinh vien" not in normalized_question:
            score -= 18.0

    if any(term in normalized_query for term in ("giang vien", "can bo", "cbgv")):
        if any(term in normalize_text(audience) for term in ("giang vien", "can bo")):
            score += 18.0
        if "sinh vien" in normalize_text(audience) and not any(term in normalized_question for term in ("sinh vien", "sv")):
            score -= 18.0

    if any(term in normalized_query for term in ("support", "web support", "support uneti")):
        score += 8.0

    return round(max(score, 0.0), 4)


def _score_chunk(query: str, chunk: dict, doc_freq: Counter, total_docs: int) -> float:
    if chunk.get("source_type") == BUSINESS_FAQ_SOURCE_TYPE:
        return _score_business_faq(query, chunk)

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


def build_business_faq_answer(docs: list[dict], max_items: int = 1) -> str | None:
    faq_docs = [
        doc for doc in docs or []
        if doc.get("source_type") == BUSINESS_FAQ_SOURCE_TYPE and doc.get("faq_answer")
    ]
    if not faq_docs:
        return None

    try:
        top_score = float(faq_docs[0].get("score") or 0)
    except (TypeError, ValueError):
        top_score = 0.0

    selected = []
    seen_answers = set()
    for doc in faq_docs:
        try:
            score = float(doc.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0

        if selected and top_score and score < max(BUSINESS_FAQ_MIN_SCORE, top_score * 0.72):
            continue

        normalized_answer = normalize_text(doc.get("faq_answer", ""))
        if normalized_answer in seen_answers:
            continue

        selected.append(doc)
        seen_answers.add(normalized_answer)

        if len(selected) >= max_items:
            break

    if not selected:
        return None

    if len(selected) == 1:
        doc = selected[0]
        source = doc.get("faq_location") or doc.get("title") or "Khong ro vi tri"
        doc_name = doc.get("doc_name") or doc.get("ten_van_ban") or "Khong ro tai lieu"
        return f'{doc.get("faq_answer")}\n(Nguồn: {source} - {doc_name})'

    answer_lines = [
        f'- {doc.get("faq_answer")}'
        for doc in selected
    ]
    source_parts = []
    for doc in selected:
        source = doc.get("faq_location") or doc.get("title") or "Khong ro vi tri"
        doc_name = doc.get("doc_name") or doc.get("ten_van_ban") or "Khong ro tai lieu"
        source_text = f"{source} - {doc_name}"
        if source_text not in source_parts:
            source_parts.append(source_text)

    return "\n".join(answer_lines) + f"\n(Nguồn: {'; '.join(source_parts)})"


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
            if file_path.name == FAQ_MAPPING_DOC_NAME:
                faq_rows = _build_business_faq_rows(file_path, root)
                if faq_rows:
                    for faq_row in faq_rows:
                        tokens = get_keywords(
                            " ".join([
                                faq_row.get("title", ""),
                                faq_row.get("content", ""),
                                faq_row.get("faq_keywords", ""),
                                faq_row.get("audience", ""),
                            ])
                        )
                        faq_row["_token_counts"] = Counter(tokens)
                        faq_row["_token_set"] = set(tokens)
                        chunks.append(faq_row)
                        doc_freq.update(faq_row["_token_set"])
                    continue

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
        min_score = (
            BUSINESS_FAQ_MIN_SCORE
            if chunk.get("source_type") == BUSINESS_FAQ_SOURCE_TYPE
            else MIN_SEARCH_SCORE
        )
        if score < min_score:
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
        "faq_candidate_count": sum(
            1 for item in results if item.get("source_type") == BUSINESS_FAQ_SOURCE_TYPE
        ),
        "faq_final_count": sum(
            1 for item in final_results if item.get("source_type") == BUSINESS_FAQ_SOURCE_TYPE
        ),
        "final_sources": [
            {
                "title": item.get("title"),
                "doc_name": item.get("doc_name"),
                "relative_path": item.get("relative_path"),
                "source_type": item.get("source_type"),
                "file_id": item.get("file_id"),
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
