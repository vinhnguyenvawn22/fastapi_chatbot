from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import math
import re
import zipfile
import xml.etree.ElementTree as ET

from pypdf import PdfReader

from app.controller.document_controller import build_document_chunks, chunk_text, list_documents
from app.core.config import (
    BUSINESS_DOCUMENTS_DIR,
    BUSINESS_INDEX_CACHE_ENABLED,
    BUSINESS_INDEX_CACHE_FILE,
    BUSINESS_GENERIC_FINAL_TOP_K,
    BUSINESS_GENERIC_KEYWORD_TOP_K,
    BUSINESS_GENERIC_VECTOR_ENABLED,
    BUSINESS_GENERIC_VECTOR_MAX_RUNTIME_EMBED_CHUNKS,
    BUSINESS_GENERIC_VECTOR_MIN_SCORE,
    BUSINESS_GENERIC_VECTOR_TOP_K,
    BUSINESS_MAPPING_LLM_JUDGE_ENABLED,
    BUSINESS_SEARCH_TOP_K,
    HYDE_ENABLED,
    HYDE_MAX_WORDS,
    MIN_SEARCH_SCORE,
)
from app.data.elasticsearch_client import get_keywords, normalize_text
from app.data.gemini_client import ask_gemini


_BUSINESS_INDEX_CACHE = {
    "signature": None,
    "chunks": [],
    "doc_freq": Counter(),
    "total_docs": 0,
}
_BUSINESS_SEARCH_CACHE = {}
_BUSINESS_VECTOR_CACHE = {}
FAQ_MAPPING_DOC_NAME = "PCNTT_MAPPING_FILE.docx"
BUSINESS_FAQ_SOURCE_TYPE = "business_faq_mapping"
BUSINESS_FAQ_MIN_SCORE = max(MIN_SEARCH_SCORE, 14.0)
BUSINESS_SOURCE_TYPE = "business_document"
BUSINESS_GUIDED_VECTOR_MIN_SCORE = 0.35
BUSINESS_INDEX_CACHE_VERSION = 3
BUSINESS_MAPPING_MIN_TOPIC_OVERLAP = 2
SURVEY_FALLBACK_DOC_NAME = "2026.03.03.ChatbotAI_CBGV_SV_V4.docx"
PROCEDURE_EVALUATION_LOCATION = "Mục III -> 6"

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
    "diem danh": ["tra cuu diem danh", "chuyen can", "so buoi vang", "ty le vang"],
    "chuyen can": ["diem danh", "so tiet vang", "nghi co phep", "nghi khong phep"],
    "thoi khoa bieu": ["lich hoc", "lich thi"],
    "lich hoc": ["hoc tap", "lich hoc lich thi"],
    "lich thi": ["hoc tap", "lich hoc lich thi"],
    "cham lai": ["phuc khao", "phuc khao bai thi", "diem thi", "ket qua thi", "hoc phan"],
    "bai thi": ["phuc khao", "diem thi", "ket qua thi", "hoc phan"],
    "phuc khao": ["cham lai bai thi", "diem thi", "ket qua thi", "hoc phan"],
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
_BUSINESS_HYDE_ERROR_MARKERS = (
    "he thong ai tam thoi vuot gioi han",
    "he thong ai dang ban",
    "loi khi goi gemini api",
    "khong tim thay can cu du ro",
)
_BUSINESS_GENERIC_INTENT_TERMS = {
    "lam",
    "the",
    "nao",
    "sao",
    "nhu",
    "dau",
    "muon",
    "can",
    "hoi",
    "cach",
    "bai",
    "lai",
    "toi",
    "em",
    "anh",
    "chi",
    "duoc",
}
_BUSINESS_DISTINCTIVE_TERMS = {
    "lms",
    "email",
    "support",
    "password",
    "mat",
    "khau",
    "may",
    "chieu",
    "danh",
    "chuyen",
    "can",
    "vang",
}
_BUSINESS_DOMAIN_TERMS = {
    "exam_academic": {
        "cham",
        "thi",
        "diem",
        "phuc",
        "khao",
        "ket",
        "qua",
        "hoc",
        "phan",
        "sinh",
        "vien",
    },
    "leave_academic": {
        "nghi",
        "tam",
        "ngung",
        "bao",
        "luu",
        "thoi",
        "hoc",
        "sinh",
        "vien",
    },
    "support_equipment": {
        "bao",
        "hong",
        "thiet",
        "bi",
        "may",
        "chieu",
        "phong",
        "hoc",
        "giang",
        "duong",
    },
    "schedule_lookup": {
        "lich",
        "thi",
        "hoc",
        "thoi",
        "khoa",
        "bieu",
        "tra",
        "cuu",
    },
    "attendance_lookup": {
        "diem",
        "danh",
        "chuyen",
        "can",
        "vang",
        "nghi",
        "phep",
        "tiet",
        "buoi",
        "ty",
        "le",
        "hoc",
        "ky",
        "tra",
        "cuu",
    },
}
_BUSINESS_DOMAIN_CORE_TERMS = {
    "exam_academic": {
        "cham",
        "thi",
        "diem",
        "phuc",
        "khao",
    },
    "leave_academic": {
        "nghi",
        "tam",
        "ngung",
        "bao",
        "luu",
        "thoi",
        "hoc",
    },
    "support_equipment": {
        "bao",
        "hong",
        "thiet",
        "bi",
        "may",
        "chieu",
    },
    "schedule_lookup": {
        "lich",
        "thi",
        "hoc",
        "thoi",
        "khoa",
        "bieu",
    },
    "attendance_lookup": {
        "diem",
        "danh",
        "chuyen",
        "can",
        "vang",
        "nghi",
    },
}
_BUSINESS_DOMAIN_PHRASES = {
    "exam_academic": {
        "cham lai",
        "bai thi",
        "diem thi",
        "phuc khao",
        "ket qua thi",
        "khieu nai diem",
    },
    "leave_academic": {
        "nghi hoc",
        "tam ngung hoc",
        "bao luu",
        "thoi hoc",
    },
    "support_equipment": {
        "bao hong",
        "may chieu",
        "thiet bi",
    },
    "schedule_lookup": {
        "lich thi",
        "lich hoc",
        "thoi khoa bieu",
    },
    "attendance_lookup": {
        "diem danh",
        "tra cuu diem danh",
        "chuyen can",
        "diem chuyen can",
        "so buoi vang",
        "so tiet vang",
        "ty le vang",
    },
}
_MAPPING_JUDGE_CACHE = {}


def _empty_retrieval_plan(status: str = "not_needed", error: str | None = None) -> dict:
    return {
        "intent": "unknown",
        "domain": "unknown",
        "query": "",
        "hyde": "",
        "must": [],
        "avoid": [],
        "clarification_needed": False,
        "clarification_question": None,
        "status": status,
        "error": error,
        "parse_error": None,
        "llm_called": False,
        "cache_hit": False,
    }


def _is_exam_regrade_query(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        phrase in normalized
        for phrase in (
            "phuc khao",
            "cham lai",
            "cham lai bai thi",
            "xem xet lai diem",
            "khieu nai diem",
        )
    ) or (
        "bai thi" in normalized
        and any(term in normalized for term in ("diem", "cham", "ket qua"))
    )


def clear_business_knowledge_cache():
    _BUSINESS_INDEX_CACHE["signature"] = None
    _BUSINESS_INDEX_CACHE["chunks"] = []
    _BUSINESS_INDEX_CACHE["doc_freq"] = Counter()
    _BUSINESS_INDEX_CACHE["total_docs"] = 0
    _BUSINESS_SEARCH_CACHE.clear()
    _MAPPING_JUDGE_CACHE.clear()
    _BUSINESS_VECTOR_CACHE.clear()


def _business_path() -> Path:
    return Path(BUSINESS_DOCUMENTS_DIR).resolve()


def _business_index_cache_path() -> Path:
    return Path(BUSINESS_INDEX_CACHE_FILE).resolve()


def _json_safe_signature(signature):
    return json.loads(json.dumps(signature))


def _serialize_chunk(chunk: dict) -> dict:
    serialized = dict(chunk)
    token_counts = serialized.get("_token_counts")
    token_set = serialized.get("_token_set")

    if isinstance(token_counts, Counter):
        serialized["_token_counts"] = dict(token_counts)
    if isinstance(token_set, set):
        serialized["_token_set"] = sorted(token_set)

    return serialized


def _deserialize_chunk(chunk: dict) -> dict:
    deserialized = dict(chunk)
    deserialized["_token_counts"] = Counter(deserialized.get("_token_counts") or {})
    deserialized["_token_set"] = set(deserialized.get("_token_set") or [])
    return deserialized


def _load_business_index_from_disk(signature):
    if not BUSINESS_INDEX_CACHE_ENABLED:
        return None

    cache_path = _business_index_cache_path()
    if not cache_path.is_file():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return None

    if payload.get("version") != BUSINESS_INDEX_CACHE_VERSION:
        return None
    if payload.get("signature") != _json_safe_signature(signature):
        return None

    chunks = [
        _deserialize_chunk(chunk)
        for chunk in payload.get("chunks", [])
        if isinstance(chunk, dict)
    ]
    doc_freq = Counter(payload.get("doc_freq") or {})
    total_docs = int(payload.get("total_docs") or len(chunks))

    return chunks, doc_freq, total_docs


def _write_business_index_to_disk(signature, chunks, doc_freq, total_docs):
    if not BUSINESS_INDEX_CACHE_ENABLED:
        return

    cache_path = _business_index_cache_path()
    payload = {
        "version": BUSINESS_INDEX_CACHE_VERSION,
        "signature": _json_safe_signature(signature),
        "chunks": [_serialize_chunk(chunk) for chunk in chunks],
        "doc_freq": dict(doc_freq),
        "total_docs": total_docs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)
        temp_path.replace(cache_path)
    except Exception:
        return


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


def _resolve_relative_path(root: Path, source_file_name: str) -> str | None:
    doc_name = _doc_name_with_extension(source_file_name)
    if doc_name:
        candidate = root / doc_name
        if candidate.exists():
            return candidate.relative_to(root).as_posix()
    return None


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


def _business_hyde_debug(status: str, text: str = "", error: str | None = None) -> dict:
    return {
        "attempted": status not in {"disabled", "not_needed"},
        "status": status,
        "char_count": len(text),
        "word_count": len(text.split()),
        "text_preview": text[:500] if text else None,
        "error": error,
    }


def _is_business_hyde_error(text: str) -> bool:
    normalized = normalize_text(text)
    return any(marker in normalized for marker in _BUSINESS_HYDE_ERROR_MARKERS)


def _generate_business_hyde_query(question: str) -> dict:
    question = " ".join(str(question or "").split())
    if not HYDE_ENABLED:
        return {"text": "", **_business_hyde_debug("disabled")}
    if not question:
        return {"text": "", **_business_hyde_debug("empty_question")}

    prompt = f"""
Bạn là bộ tạo truy vấn giả định HyDE cho hệ thống tra cứu tài liệu nghiệp vụ nội bộ của nhà trường.

Đoạn bạn tạo ra chỉ dùng để tìm tài liệu khi file mapping không có kết quả phù hợp hoặc kết quả mapping bị nghi ngờ sai chủ đề. Đây không phải câu trả lời cuối cùng cho người dùng.

Nhiệm vụ:
Đọc câu hỏi gốc của người dùng, xác định đúng đối tượng nghiệp vụ chính, rồi tạo một đoạn mô tả giả định ngắn giúp hệ thống tìm được tài liệu liên quan.

Yêu cầu:
- Giữ đúng chủ đề và các danh từ chính trong câu hỏi gốc.
- Không trả lời trực tiếp cho người dùng.
- Không bịa tên phòng ban, thời hạn, đường dẫn, biểu mẫu, số bước, điều kiện hoặc quy định cụ thể nếu câu hỏi không nêu.
- Chuyển cách hỏi đời thường sang thuật ngữ nghiệp vụ có khả năng xuất hiện trong tài liệu.
- Ưu tiên thuật ngữ có khả năng xuất hiện trong hướng dẫn nghiệp vụ, quy trình, quy chế, thông báo hoặc biểu mẫu.
- Tập trung vào quy trình nghiệp vụ, thao tác hệ thống, vai trò xử lý, biểu mẫu, hồ sơ, trạng thái, phê duyệt, tiếp nhận, kiểm tra, lưu hồ sơ, trả kết quả.
- Nếu câu hỏi có các từ chung như “làm thế nào”, “làm sao”, “như nào”, “ở đâu”, hãy bỏ qua các từ đó khi xác định chủ đề.
- Không tự chuyển câu hỏi sang chủ đề tìm kiếm bài viết, giao diện, website, chatbot, thanh tìm kiếm, nút bấm, danh sách, bộ lọc nếu người dùng không hỏi rõ về các nội dung đó.
- Nếu câu hỏi liên quan sinh viên/học vụ/thi cử, dùng thuật ngữ như: sinh viên, học phần, điểm thi, kết quả thi, phúc khảo, bảo lưu, tạm ngừng học, thôi học, học lại, đăng ký học phần, học phí, hồ sơ, đơn đề nghị.
- Nếu câu hỏi liên quan xử lý yêu cầu trên hệ thống, dùng thuật ngữ như: tạo yêu cầu, tiếp nhận, phân công, xử lý, phê duyệt, cập nhật trạng thái, tra cứu, phản hồi, hoàn tất.
- Không quá 80 từ.
- Không markdown, không JSON.

Ví dụ:
Câu hỏi: tôi muốn chấm lại bài thi thì làm thế nào
Đoạn mô tả giả định: Sinh viên đề nghị phúc khảo bài thi, xem xét lại điểm thi hoặc khiếu nại kết quả thi cần thực hiện thủ tục theo quy định, hồ sơ hoặc đơn đề nghị liên quan đến kết quả thi và học phần.

Câu hỏi: tôi muốn nghỉ học thì làm như nào
Đoạn mô tả giả định: Sinh viên xin nghỉ học tạm thời, tạm ngừng học, bảo lưu kết quả học tập hoặc xin thôi học cần thực hiện thủ tục, hồ sơ hoặc đơn đề nghị theo quy định đào tạo của nhà trường.

Câu hỏi: tôi muốn báo hỏng máy chiếu
Đoạn mô tả giả định: Quy trình tiếp nhận và xử lý yêu cầu báo hỏng thiết bị phòng học, máy chiếu hoặc thiết bị giảng dạy; người dùng tạo yêu cầu, đơn vị phụ trách kiểm tra, xử lý và cập nhật trạng thái.

Câu hỏi: làm sao để xem lịch thi
Đoạn mô tả giả định: Hướng dẫn tra cứu lịch thi trên hệ thống nghiệp vụ; người dùng đăng nhập, chọn chức năng lịch thi hoặc kế hoạch thi, xem thông tin học phần, thời gian, phòng thi và trạng thái hiển thị.

Câu hỏi gốc:
{question}

Đoạn mô tả giả định:
""".strip()

    try:
        text = " ".join(ask_gemini(prompt).split()).strip()
    except Exception as exc:
        return {"text": "", **_business_hyde_debug("error", error=str(exc))}

    if not text:
        return {"text": "", **_business_hyde_debug("empty_response")}
    if _is_business_hyde_error(text):
        return {"text": "", **_business_hyde_debug("llm_error", text=text)}

    text = " ".join(text.split()[: min(HYDE_MAX_WORDS, 80)])
    return {"text": text, **_business_hyde_debug("success", text=text)}


def _trim_phrase_list(values, max_items: int = 5) -> list[str]:
    phrases = []
    for value in values or []:
        phrase = " ".join(str(value or "").split()).strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= max_items:
            break
    return phrases


def _is_cbgv_admin_process_steps_query(question: str) -> bool:
    normalized = normalize_text(question)
    has_admin_process = (
        "thu tuc hanh chinh" in normalized
        and "ho so" in normalized
        and (
            "quy trinh xu ly" in normalized
            or "xu ly ho so" in normalized
            or "quy trinh" in normalized
        )
    )
    asks_for_steps = any(
        term in normalized
        for term in (
            "gom may buoc",
            "bao nhieu buoc",
            "cac buoc",
            "may buoc",
            "nhung buoc",
        )
    )
    asks_process = "giang vien" in normalized or "cbgv" in normalized or "can bo" in normalized
    asks_student = any(term in normalized for term in ("sinh vien", "sv", "nguoi hoc"))
    return has_admin_process and (asks_for_steps or asks_process) and not asks_student


def _rule_based_retrieval_plan(question: str) -> dict | None:
    normalized = normalize_text(question)
    if _is_cbgv_admin_process_steps_query(question):
        return {
            **_empty_retrieval_plan("rule_success"),
            "intent": "quy_trinh_xu_ly_ho_so_thu_tuc_hanh_chinh",
            "domain": "cbgv_thu_tuc_hanh_chinh",
            "query": "quy trinh xu ly ho so thu tuc hanh chinh 5 buoc cbgv",
            "hyde": (
                "Can bo giang vien tra cuu quy trinh xu ly ho so thu tuc hanh chinh "
                "gom cac buoc nop ho so tiep nhan xu ly phe duyet tra ket qua."
            ),
            "must": [
                "quy trinh xu ly ho so thu tuc hanh chinh",
                "nop ho so",
                "tiep nhan ho so",
                "xu ly ho so",
                "tra ket qua",
            ],
            "avoid": ["web support sv", "sinh vien"],
            "clarification_needed": False,
            "clarification_question": None,
        }
    if "xem lai diem" in normalized and "thi" not in normalized:
        return {
            **_empty_retrieval_plan("rule_success"),
            "intent": "xem_diem",
            "domain": "hoc_tap_sinh_vien",
            "query": "xem diem ket qua hoc tap sinh vien phuc khao diem thi",
            "must": ["diem"],
            "avoid": ["web support cbgv", "giang vien", "can bo"],
            "clarification_needed": False,
            "clarification_question": (
                "Bạn muốn xem điểm đã công bố, hay muốn gửi yêu cầu phúc khảo điểm thi?"
            ),
        }
    if _is_exam_regrade_query(question):
        return {
            **_empty_retrieval_plan("rule_success"),
            "intent": "phuc_khao",
            "domain": "khao_thi",
            "query": "phuc khao ket qua bai thi diem thi hoc phan sinh vien",
            "hyde": (
                "Sinh vien de nghi phuc khao khi cho rang diem thi hoac ket qua bai thi "
                "da cong bo chua chinh xac."
            ),
            "must": ["phuc khao", "diem thi", "bai thi", "hoc phan"],
            "avoid": ["bai bao", "tap chi", "thanh tra cham thi", "hoi dong thi"],
        }
    return None


def _retrieval_plan_prompt(question: str) -> str:
    return f"""
Bạn là bộ phân tích truy vấn cho hệ thống RAG tra cứu tài liệu nghiệp vụ nội bộ trường đại học.

Nhiệm vụ của bạn KHÔNG phải trả lời người dùng. Nhiệm vụ là hiểu câu hỏi, viết lại truy vấn tìm tài liệu, và xác định cách truy hồi phù hợp.

Hãy đọc câu hỏi người dùng và trả về JSON hợp lệ theo schema:

{{
  "intent": "ten_intent_ngan_gon_hoac_unclear",
  "domain": "nhom_nghiep_vu",
  "query": "truy vấn tìm kiếm ngắn bằng thuật ngữ nghiệp vụ",
  "hyde": "mô tả giả định ngắn để tìm tài liệu, không bịa quy trình cụ thể",
  "must": ["từ hoặc cụm từ nên có trong tài liệu đúng"],
  "avoid": ["từ hoặc cụm từ thường xuất hiện trong tài liệu sai miền"],
  "clarification_needed": true hoặc false,
  "clarification_question": "câu hỏi làm rõ nếu cần, ngược lại null"
}}

Quy tắc:
- Không trả lời trực tiếp câu hỏi của người dùng.
- Không bịa tên phòng ban, đường dẫn, biểu mẫu, thời hạn, số bước, điều kiện hoặc quy định nếu câu hỏi không nêu.
- Chuyển ngôn ngữ đời thường sang thuật ngữ nghiệp vụ có khả năng xuất hiện trong tài liệu.
- Giữ lại ý chính của người dùng.
- Nếu câu hỏi đủ rõ, đặt clarification_needed = false.
- Nếu câu hỏi có nhiều cách hiểu hợp lý và có nguy cơ tìm sai tài liệu, đặt clarification_needed = true.
- Trường "query" phải ngắn, ưu tiên danh từ và thuật ngữ nghiệp vụ.
- Trường "hyde" tối đa 45 từ.
- Trường "must" tối đa 5 cụm từ.
- Trường "avoid" tối đa 5 cụm từ.
- Chỉ trả về JSON, không markdown, không giải thích.

Một số quy đổi thuật ngữ:
- "chấm lại bài thi", "xem lại điểm thi", "khiếu nại điểm", "điểm thi sai" => "phúc khảo", domain "khao_thi"
- "nghỉ học một thời gian", "dừng học tạm thời" => "tạm ngừng học", "bảo lưu", domain "hoc_vu"
- "xem điểm", "điểm học tập" => "kết quả học tập", domain "hoc_tap"
- "báo hỏng máy chiếu", "hỏng thiết bị phòng học" => "báo hỏng thiết bị", domain "thiet_bi"
- "quên mật khẩu", "không đăng nhập được email/LMS" => "khôi phục mật khẩu", "đăng nhập", domain "tai_khoan"
- "lịch học", "thời khóa biểu", "lịch thi" => domain "lich_hoc_lich_thi"

Ví dụ 1:
Câu hỏi: tôi muốn chấm lại bài thi như thế nào
JSON:
{{
  "intent": "phuc_khao",
  "domain": "khao_thi",
  "query": "phúc khảo kết quả bài thi điểm thi học phần sinh viên",
  "hyde": "Sinh viên đề nghị phúc khảo khi cho rằng điểm thi hoặc kết quả bài thi đã công bố chưa chính xác.",
  "must": ["phúc khảo", "điểm thi", "bài thi", "học phần"],
  "avoid": ["bài báo", "tạp chí", "thanh tra chấm thi", "hội đồng thi"],
  "clarification_needed": false,
  "clarification_question": null
}}

Ví dụ 2:
Câu hỏi: em muốn xem lại điểm
JSON:
{{
  "intent": "unclear",
  "domain": "hoc_tap_or_khao_thi",
  "query": "xem điểm học tập hoặc phúc khảo điểm thi",
  "hyde": "",
  "must": ["điểm"],
  "avoid": [],
  "clarification_needed": true,
  "clarification_question": "Bạn muốn xem điểm đã công bố, hay muốn gửi yêu cầu phúc khảo điểm thi?"
}}

Câu hỏi người dùng:
{question}

JSON:
""".strip()


def _parse_retrieval_plan(raw_response: str, question: str) -> dict:
    raw_text = str(raw_response or "").strip()
    try:
        parsed = json.loads(_strip_json_code_fence(raw_text))
    except Exception as exc:
        fallback = _empty_retrieval_plan("parse_error", error=str(exc))
        fallback["query"] = question
        fallback["raw_response"] = raw_text[:1000]
        fallback["parse_error"] = str(exc)
        return fallback

    plan = _empty_retrieval_plan("success")
    plan.update({
        "intent": str(parsed.get("intent") or "unknown").strip() or "unknown",
        "domain": str(parsed.get("domain") or "unknown").strip() or "unknown",
        "query": " ".join(str(parsed.get("query") or "").split()),
        "hyde": " ".join(str(parsed.get("hyde") or "").split()[:45]),
        "must": _trim_phrase_list(parsed.get("must"), 5),
        "avoid": _trim_phrase_list(parsed.get("avoid"), 5),
        "clarification_needed": False,
        "clarification_question": None,
        "raw_response": raw_text[:1000],
    })
    if plan["clarification_needed"] and not plan["clarification_question"]:
        plan["clarification_question"] = "Bạn cần hỏi rõ ràng hơn"
    return plan


def _generate_business_retrieval_plan(question: str) -> dict:
    question = " ".join(str(question or "").split())
    if not question:
        return _empty_retrieval_plan("empty_question")

    rule_plan = _rule_based_retrieval_plan(question)
    if rule_plan is not None:
        return rule_plan

    if not HYDE_ENABLED:
        disabled = _empty_retrieval_plan("disabled")
        disabled["query"] = question
        return disabled

    try:
        raw_response = ask_gemini(_retrieval_plan_prompt(question))
    except Exception as exc:
        fallback = _empty_retrieval_plan("error", error=str(exc))
        fallback["query"] = question
        fallback["llm_called"] = True
        return fallback

    plan = _parse_retrieval_plan(raw_response, question)
    plan["llm_called"] = True
    if plan.get("parse_error"):
        plan["query"] = question
        plan["hyde"] = ""
        plan["must"] = []
        plan["avoid"] = []
    return plan


def _plan_search_query(question: str, retrieval_plan: dict) -> str:
    parts = [
        question,
        retrieval_plan.get("query") or "",
        retrieval_plan.get("hyde") or "",
        " ".join(retrieval_plan.get("must") or []),
    ]
    return " ".join(dict.fromkeys(" ".join(parts).split()))


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
            source_relative_path = _resolve_relative_path(root, source_file_name)
            relative_path = source_relative_path or source_file_name
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
                "source_relative_path": source_relative_path,
                "source_file_found": bool(source_relative_path),
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

    rows.extend(_supplemental_business_faq_rows(file_path, root, file_map, rows))
    return rows


def _supplemental_business_faq_rows(
    file_path: Path,
    root: Path,
    file_map: dict[str, dict],
    existing_rows: list[dict],
) -> list[dict]:
    existing_questions = {
        normalize_text(row.get("faq_question") or row.get("title") or "")
        for row in existing_rows
    }
    supplemental = [
        {
            "file_id": "PCNTT_FILE_02",
            "question": "Làm thế nào để xem điểm danh theo học kỳ?",
            "answer": (
                "Sinh viên đăng nhập https://support.uneti.edu.vn, chọn Tra cứu -> Điểm danh, "
                "chọn học kỳ cần xem, theo dõi bảng thông tin điểm danh và nhấn Xem chi tiết "
                "để xem từng buổi học."
            ),
            "location": "Tra cứu -> Điểm danh",
            "keywords": (
                "điểm danh, tra cứu điểm danh, chuyên cần, học kỳ, số buổi vắng, "
                "số tiết vắng, tỷ lệ vắng, nghỉ có phép, nghỉ không phép, xem chi tiết"
            ),
        },
        {
            "file_id": "PCNTT_FILE_02",
            "question": "Tôi muốn chấm lại bài thi thì làm thế nào?",
            "answer": (
                "Sinh viên đăng nhập https://support.uneti.edu.vn, chọn Thủ tục hành chính -> "
                "Một cửa -> Khảo thí -> Phúc khảo, điền thông tin yêu cầu phúc khảo và gửi hồ sơ."
            ),
            "location": "Thủ tục hành chính -> Một cửa -> Khảo thí -> Phúc khảo",
            "keywords": (
                "phúc khảo, chấm lại bài thi, điểm thi, bài thi, gửi yêu cầu phúc khảo, "
                "đơn phúc khảo, kết quả bài thi, khảo thí"
            ),
        },
        {
            "file_id": "PCNTT_FILE_02",
            "question": "Làm thế nào để gửi yêu cầu phúc khảo/chấm lại bài thi?",
            "answer": (
                "Sinh viên đăng nhập https://support.uneti.edu.vn, chọn Thủ tục hành chính -> "
                "Một cửa -> Khảo thí -> Phúc khảo, điền thông tin yêu cầu phúc khảo và gửi hồ sơ."
            ),
            "location": "Thủ tục hành chính -> Một cửa -> Khảo thí -> Phúc khảo",
            "keywords": (
                "phúc khảo, chấm lại bài thi, điểm thi, bài thi, gửi yêu cầu phúc khảo, "
                "đơn phúc khảo, kết quả bài thi, khảo thí"
            ),
        },
        {
            "file_id": "PCNTT_FILE_03",
            "question": "Giảng viên xem khối lượng coi thi/chấm thi ở đâu?",
            "answer": (
                "Giảng viên đăng nhập https://support.uneti.edu.vn, chọn Công tác giảng viên -> "
                "Tra cứu -> Khối lượng công tác giảng viên, sau đó chọn tab Khối lượng coi, chấm thi."
            ),
            "location": "Công tác giảng viên -> Tra cứu -> Khối lượng công tác giảng viên",
            "keywords": (
                "khối lượng coi thi, khối lượng chấm thi, công tác giảng viên, "
                "tra cứu khối lượng, học kỳ, giờ cấu trúc"
            ),
        },
        {
            "file_id": "PCNTT_FILE_03",
            "question": "Giảng viên xem lớp học phần giảng viên ở đâu?",
            "answer": (
                "Giảng viên đăng nhập https://support.uneti.edu.vn, chọn Công tác giảng viên -> "
                "Tra cứu -> Khối lượng công tác giảng viên -> Lớp học phần giảng viên."
            ),
            "location": "Công tác giảng viên -> Tra cứu -> Lớp học phần giảng viên",
            "keywords": (
                "lớp học phần giảng viên, công tác giảng viên, tra cứu lớp học phần, "
                "lịch dạy, học kỳ"
            ),
        },
    ]

    rows = []
    next_index = max(
        [int(row.get("chunk_index") or 0) for row in existing_rows if str(row.get("chunk_index") or "").isdigit()]
        or [0]
    ) + 1
    for item in supplemental:
        if normalize_text(item["question"]) in existing_questions:
            continue

        source_info = file_map.get(item["file_id"], {})
        source_file_name = source_info.get("source_file_name") or item["file_id"]
        doc_name = _doc_name_with_extension(source_file_name) or file_path.name
        source_relative_path = _resolve_relative_path(root, source_file_name)
        relative_path = source_relative_path or source_file_name
        audience = source_info.get("audience") or ""
        content = "\n".join([
            f"Cau hoi thuong gap: {item['question']}",
            f"Cau tra loi chuan: {item['answer']}",
            f"Vi tri chinh xac trong file goc: {item['location']}",
            f"Tu khoa tim kiem: {item['keywords']}",
            f"Doi tuong: {audience}",
        ]).strip()

        rows.append({
            "doc_name": doc_name,
            "relative_path": relative_path,
            "source_relative_path": source_relative_path,
            "source_file_found": bool(source_relative_path),
            "mapping_relative_path": file_path.relative_to(root).as_posix(),
            "source_root": root.name,
            "title": item["question"],
            "content": content,
            "chunk_index": next_index,
            "source_type": BUSINESS_FAQ_SOURCE_TYPE,
            "file_path": str(root / relative_path),
            "updated_at": datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).isoformat(),
            "file_id": item["file_id"],
            "faq_question": item["question"],
            "faq_answer": item["answer"],
            "faq_location": item["location"],
            "faq_keywords": item["keywords"],
            "audience": audience,
            "mapping_table_index": "supplemental",
            "ten_van_ban": source_file_name,
        })
        next_index += 1

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

    if "lms" in original_tokens:
        if "lms" in combined_normalized:
            score += 65.0
        else:
            score -= 90.0
        if any(term in normalized_query for term in ("khong dang nhap", "loi dang nhap", "quen mat khau")):
            if any(term in combined_normalized for term in ("su co", "tu khac phuc", "email", "mat khau")):
                score += 35.0

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
    combined_normalized = f"{normalized_title} {normalized_content}"

    cbgv_topic_terms = (
        "nhan su",
        "lop hoc phan",
        "giang vien",
        "can bo",
        "cbgv",
        "dang ky muon thiet bi",
        "muon thiet bi",
        "thiet bi phong hoc",
        "ho so thu tuc hanh chinh",
        "quy trinh xu ly ho so",
        "minh chung kiem dinh",
        "kiem dinh",
    )
    student_topic_terms = ("sinh vien", "sv", "nguoi hoc")
    asks_student_topic = any(term in normalized_query for term in student_topic_terms)
    asks_cbgv_topic = any(term in normalized_query for term in cbgv_topic_terms)

    is_cbgv_doc = "web support cbgv" in doc_name
    is_sv_doc = "web support sv" in doc_name

    if asks_cbgv_topic and is_cbgv_doc:
        score += 65.0
    if asks_cbgv_topic and is_sv_doc and not asks_student_topic:
        score -= 90.0
    if asks_student_topic and is_sv_doc:
        score += 35.0

    phrase_boosts = [
        (
            ("dang ky muon thiet bi", "muon thiet bi", "thiet bi phong hoc"),
            ("man dang ky muon thiet bi", "dang ky su dung thiet bi", "chon lich day va thiet bi"),
            220.0,
        ),
        (
            ("lop hoc phan giang vien", "duong dan", "vao duong dan"),
            ("lop hoc phan giang vien", "tra cuu/lop-hoc-phan-giang-vien"),
            210.0,
        ),
        (
            ("man nhan su", "nhan su dung de lam gi"),
            ("man nhan su", "thong tin nhan su ca nhan", "khoi luong giam tru"),
            190.0,
        ),
        (
            ("quy trinh xu ly ho so", "ho so thu tuc hanh chinh", "gom may buoc"),
            ("quy trinh xu ly ho so thu tuc hanh chinh", "buoc 1: nop ho so", "buoc 5. tra ket qua"),
            240.0,
        ),
        (
            ("trang thai minh chung", "minh chung kiem dinh"),
            ("cho duyet", "da duyet", "can bo sung", "minh chung hien thi"),
            220.0,
        ),
    ]
    for query_phrases, content_phrases, boost in phrase_boosts:
        if any(phrase in normalized_query for phrase in query_phrases) and any(
            phrase in combined_normalized for phrase in content_phrases
        ):
            score += boost

    if _is_exam_regrade_query(query):
        searchable = f"{normalized_title} {normalized_content} {doc_name}"
        if "phuc khao" in searchable:
            score += 240.0
        if any(term in searchable for term in ("1.2. phuc khao", "man phuc khao cho phep", "giay tiep nhan yeu cau phuc khao")):
            score += 120.0
        if any(term in searchable for term in ("ket qua bai thi", "ket qua hoc tap", "diem thi", "hoc phan")):
            score += 90.0
        if "support sv" in doc_name or "support sinh vien" in searchable:
            score += 70.0
        if "khao thi" in searchable and "dam bao chat luong" in searchable:
            score += 45.0
        if any(term in searchable for term in ("bai bao", "tap chi", "ban bien tap")):
            score -= 220.0
        if any(term in searchable for term in ("thanh tra", "kiem tra cong tac cham thi", "hoi dong thi")) and "phuc khao" not in searchable:
            score -= 180.0
        if any(term in searchable for term in ("huy dang ky thi lai", "hoan thi", "dang ky thi lai")) and "man phuc khao cho phep" not in searchable:
            score -= 80.0

    if "thi lai" in normalized_query and any(term in normalized_query for term in ("dang ky", "dang ki", "huy", "huong dan", "lam the nao", "lam sao")):
        searchable = f"{normalized_title} {normalized_content} {doc_name}"
        if "web support sv" in doc_name or "support sv" in searchable:
            score += 120.0
        if "mot cua" in searchable and "khao thi" in searchable:
            score += 90.0
        if "dang ky thi lai" in searchable:
            score += 180.0
        if "huy dang ky thi lai" in searchable and "huy" in normalized_query:
            score += 220.0
        if any(term in searchable for term in ("khcn", "nghien cuu", "de tai", "nhiem vu kh&cn")):
            score -= 260.0
        if "dang ky hoc lai" in searchable or "hoc cai thien" in searchable:
            score -= 160.0

    if "hoan thi" in normalized_query and any(term in normalized_query for term in ("huong dan", "lam the nao", "lam sao", "phai lam sao", "cach", "thu tuc")):
        searchable = f"{normalized_title} {normalized_content} {doc_name}"
        if "web support sv" in doc_name or "support sv" in searchable:
            score += 140.0
        if "mot cua" in searchable and "khao thi" in searchable:
            score += 100.0
        if "hoan thi" in searchable:
            score += 180.0
        if any(term in searchable for term in ("khcn", "nghien cuu", "de tai", "nhiem vu kh&cn")):
            score -= 280.0

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
    # Compatibility shim: mapping summaries must never become final answers.
    return None

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

    disk_cache = _load_business_index_from_disk(signature)
    if disk_cache is not None:
        chunks, doc_freq, total_docs = disk_cache
        _BUSINESS_INDEX_CACHE["signature"] = signature
        _BUSINESS_INDEX_CACHE["chunks"] = chunks
        _BUSINESS_INDEX_CACHE["doc_freq"] = doc_freq
        _BUSINESS_INDEX_CACHE["total_docs"] = total_docs
        _BUSINESS_SEARCH_CACHE.clear()
        return chunks, doc_freq, total_docs

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
    _write_business_index_to_disk(signature, chunks, doc_freq, len(chunks))

    return chunks, doc_freq, len(chunks)


def _clean_index_chunk(chunk: dict) -> dict:
    return {
        key: value
        for key, value in chunk.items()
        if not key.startswith("_") and key != "faq_answer"
    }


def _mapping_candidates(query: str, chunks: list[dict]) -> list[dict]:
    candidates = []
    for chunk in chunks:
        if chunk.get("source_type") != BUSINESS_FAQ_SOURCE_TYPE:
            continue

        score = _score_business_faq(query, chunk)
        if score < BUSINESS_FAQ_MIN_SCORE:
            continue

        candidate = _clean_index_chunk(chunk)
        candidate["mapping_score"] = score
        candidates.append(candidate)

    candidates.sort(key=lambda item: item["mapping_score"], reverse=True)
    return candidates


def _should_search_cbgv_source_directly(query: str) -> bool:
    normalized = normalize_text(query)
    direct_terms = (
        "man nhan su",
        "nhan su dung de lam gi",
        "lop hoc phan giang vien",
        "dang ky muon thiet bi",
        "muon thiet bi",
        "su dung thiet bi",
        "quy trinh xu ly ho so",
        "ho so thu tuc hanh chinh",
        "trang thai minh chung",
        "minh chung kiem dinh",
    )
    return any(term in normalized for term in direct_terms) and "sinh vien" not in normalized


def _should_keep_cbgv_mapping_candidates(query: str) -> bool:
    normalized = normalize_text(query)
    mapping_friendly_terms = (
        "lich day",
        "lich coi thi",
        "lop hoc phan giang vien",
        "khoi luong cong tac",
        "khoi luong giang day",
        "khoi luong coi thi",
        "khoi luong cham thi",
        "tra cuu khoi luong",
        "cong tac giang vien",
    )
    return any(term in normalized for term in mapping_friendly_terms) and "sinh vien" not in normalized


def _meaningful_business_terms(text: str) -> set[str]:
    return {
        term
        for term in get_keywords(text)
        if term not in _BUSINESS_GENERIC_INTENT_TERMS
    }


def _mapping_topic_overlap(query: str, mapping: dict) -> int:
    query_terms = _meaningful_business_terms(query)
    mapping_terms = _meaningful_business_terms(
        " ".join([
            str(mapping.get("faq_question") or ""),
            str(mapping.get("faq_keywords") or ""),
            str(mapping.get("faq_answer") or ""),
            str(mapping.get("faq_location") or ""),
        ])
    )
    return len(query_terms & mapping_terms)


def _strip_json_code_fence(text: str) -> str:
    match = re.fullmatch(
        r"\s*```(?:json)?\s*(.*?)\s*```\s*",
        str(text or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else str(text or "").strip()


def _mapping_judge_prompt(query: str, mapping: dict, topic_overlap: int) -> str:
    return f"""
Bạn là bộ kiểm tra độ phù hợp của mapping trong hệ thống tra cứu tài liệu nghiệp vụ nội bộ nhà trường.

Nhiệm vụ:
So sánh câu hỏi gốc của người dùng với một mapping ứng viên, rồi quyết định mapping đó có đúng chủ đề để dùng truy xuất tài liệu hay không.

Nguyên tắc quan trọng:
Mapping chỉ được chấp nhận nếu nó nói về đúng đối tượng nghiệp vụ chính trong câu hỏi gốc. Nếu mapping chỉ trùng các từ chung hoặc trùng đối tượng audience như “sinh viên”, “cán bộ”, “giảng viên” thì chưa đủ để chấp nhận.

Quy tắc đánh giá:
- Bỏ qua các từ chung khi so sánh, gồm: làm thế nào, làm sao, như nào, ở đâu, tôi muốn, em muốn, cần, hỏi, bài, lại, được, có thể.
- Không coi audience như “sinh viên”, “cán bộ”, “giảng viên” là bằng chứng đủ để mapping đúng chủ đề.
- Nếu câu hỏi hỏi về thi cử, điểm thi, phúc khảo, chấm lại bài thi, kết quả thi, học phần thì mapping cũng phải liên quan đến thi cử, điểm, phúc khảo, học phần hoặc kết quả thi.
- Nếu câu hỏi hỏi về nghỉ học, tạm ngừng học, bảo lưu, thôi học thì mapping cũng phải liên quan đến nghỉ học, tạm ngừng học, bảo lưu, thôi học hoặc thủ tục học vụ.
- Nếu câu hỏi hỏi về báo hỏng thiết bị thì mapping cũng phải liên quan đến báo hỏng, thiết bị, phòng học, máy chiếu hoặc xử lý sự cố.
- Nếu câu hỏi hỏi về lịch học, lịch thi, thời khóa biểu thì mapping cũng phải liên quan đến tra cứu lịch học, lịch thi hoặc thời khóa biểu.
- Nếu câu hỏi hỏi về tài khoản, email, LMS, mật khẩu thì mapping cũng phải liên quan đến tài khoản, email, LMS, đăng nhập, mật khẩu hoặc xác thực.
- Nếu mapping chỉ nói về giao diện, tìm kiếm bài viết, thanh tìm kiếm, danh sách, bộ lọc, nút bấm, website hoặc chatbot mà câu hỏi gốc không hỏi rõ về các nội dung đó, phải reject.
- Nếu không có thuật ngữ nghiệp vụ quan trọng nào trùng hoặc tương đương giữa câu hỏi và mapping, phải reject.
- Nếu mapping_topic_overlap bằng 0 hoặc gần như không có liên quan nghiệp vụ, phải reject.
- Nếu còn nghi ngờ, ưu tiên reject để chuyển sang HyDE/manual search.

Không được:
- Không trả lời câu hỏi của người dùng.
- Không bịa thêm thông tin ngoài câu hỏi và mapping.
- Không giải thích ngoài JSON.
- Không dùng markdown.

Chỉ trả về JSON hợp lệ theo đúng schema sau:

{{
  "decision": "accept" hoặc "reject",
  "confidence": 0.0 đến 1.0,
  "reason": "lý do ngắn gọn",
  "matched_topic": "chủ đề trùng nếu accept, hoặc null",
  "missing_topic": "chủ đề câu hỏi gốc mà mapping không đáp ứng nếu reject, hoặc null"
}}

Câu hỏi gốc:
{query}

Mapping ứng viên:
- Câu hỏi mapping: {mapping.get("faq_question") or mapping.get("title") or ""}
- Câu trả lời/tóm tắt mapping: {mapping.get("faq_answer") or ""}
- Từ khóa mapping: {mapping.get("faq_keywords") or ""}
- Vị trí mapping: {mapping.get("faq_location") or ""}
- Đối tượng/audience: {mapping.get("audience") or ""}
- Topic overlap đã tính bằng rule: {topic_overlap}

JSON:
""".strip()


def _judge_mapping_with_llm(query: str, mapping: dict, topic_overlap: int) -> dict:
    cache_key = (
        normalize_text(query),
        normalize_text(mapping.get("faq_question") or mapping.get("title") or ""),
        normalize_text(mapping.get("faq_keywords") or ""),
        topic_overlap,
    )
    if cache_key in _MAPPING_JUDGE_CACHE:
        cached = deepcopy(_MAPPING_JUDGE_CACHE[cache_key])
        cached["cache_hit"] = True
        return cached

    prompt = _mapping_judge_prompt(query, mapping, topic_overlap)
    try:
        raw_response = ask_gemini(prompt)
        parsed = json.loads(_strip_json_code_fence(raw_response))
    except Exception as exc:
        result = {
            "decision": "reject",
            "confidence": 0.0,
            "reason": "mapping_judge_error",
            "matched_topic": None,
            "missing_topic": "unknown",
            "error": str(exc),
            "cache_hit": False,
        }
    else:
        decision = str(parsed.get("decision") or "").strip().lower()
        if decision not in {"accept", "reject"}:
            decision = "reject"
        result = {
            "decision": decision,
            "confidence": parsed.get("confidence"),
            "reason": parsed.get("reason") or "mapping_judge_decision",
            "matched_topic": parsed.get("matched_topic"),
            "missing_topic": parsed.get("missing_topic"),
            "error": None,
            "cache_hit": False,
        }

    _MAPPING_JUDGE_CACHE[cache_key] = deepcopy(result)
    return result


_AUDIENCE_GENERIC_TERMS = {
    "sinh vien", "sv", "giang vien", "can bo", "cbgv", "thu tuc",
    "quy trinh", "ho so", "yeu cau", "he thong", "support", "xem",
    "dang ky", "lam the nao",
}
_EXPLICIT_SV_TERMS = {"sinh vien", "sv", "nguoi hoc", "em muon", "em can"}
_EXPLICIT_CBGV_TERMS = {"giang vien", "can bo", "cbgv", "thay co", "thay", "co"}
_SV_BUSINESS_TERMS = {
    "diem", "hoc phan", "phuc khao", "lich hoc", "hoc phi", "dang ky hoc phan",
    "cham lai bai thi", "xem lai diem thi", "khieu nai diem",
    "diem thi sai", "gui yeu cau phuc khao", "don phuc khao",
    "ket qua bai thi",
    "diem danh", "tra cuu diem danh", "chuyen can", "diem chuyen can",
    "so buoi vang", "so tiet vang", "ty le vang", "ren luyen",
    "thoi khoa bieu", "chuong trinh dao tao",
}
_CBGV_BUSINESS_TERMS = {
    "lich day", "coi thi", "cham thi", "muon thiet bi phong hoc",
    "ho tro thiet bi", "cong tac giang vien", "dang ky muon thiet bi",
    "may chieu", "khoi luong", "khoi luong cong tac", "khoi luong giang day",
    "khoi luong coi thi", "khoi luong cham thi", "lop hoc phan giang vien",
    "lich coi thi", "nhan su giang vien", "minh chung kiem dinh",
    "ho so thu tuc hanh chinh", "muon thiet bi", "bao hong thiet bi",
}
_DOCUMENT_INTENT_TERMS = {
    "quyet dinh", "quy che", "thong bao", "van ban", "quy dinh",
    "dieu", "muc", "chuong",
}


def _phrase_hits(text: str, phrases: set[str] | list[str]) -> list[str]:
    normalized = normalize_text(text)
    return [phrase for phrase in phrases if phrase and phrase in normalized]


def _infer_audience_from_text(text: str) -> str:
    normalized = normalize_text(text)
    has_sv = any(term in normalized for term in ("support sv", "sinh vien", "student", " nguoi hoc"))
    has_cbgv = any(
        term in normalized
        for term in ("support cbgv", "cbgv", "can bo", "giang vien", "thay co")
    )
    if has_sv and not has_cbgv:
        return "sv"
    if has_cbgv and not has_sv:
        return "cbgv"
    return "unknown"


def _mapping_audience(mapping: dict) -> str:
    text = " ".join([
        str(mapping.get("audience") or ""),
        str(mapping.get("doc_name") or ""),
        str(mapping.get("source_file_name") or ""),
        str(mapping.get("title") or ""),
        str(mapping.get("relative_path") or ""),
        str(mapping.get("source_relative_path") or ""),
    ])
    return _infer_audience_from_text(text)


def _query_audience(query: str) -> tuple[str, dict]:
    normalized = normalize_text(query)
    explicit_sv = _phrase_hits(normalized, _EXPLICIT_SV_TERMS)
    explicit_cbgv = _phrase_hits(normalized, _EXPLICIT_CBGV_TERMS)
    sv_business = _phrase_hits(normalized, _SV_BUSINESS_TERMS)
    cbgv_business = _phrase_hits(normalized, _CBGV_BUSINESS_TERMS)
    if explicit_sv and explicit_cbgv:
        audience = "mixed"
    elif explicit_sv:
        audience = "mixed" if cbgv_business else "sv"
    elif explicit_cbgv:
        audience = "mixed" if sv_business else "cbgv"
    elif sv_business and not cbgv_business:
        audience = "sv"
    elif cbgv_business and not sv_business:
        audience = "cbgv"
    elif sv_business and cbgv_business:
        audience = "mixed"
    else:
        audience = "unknown"
    return audience, {
        "explicit_sv": explicit_sv,
        "explicit_cbgv": explicit_cbgv,
        "sv_business": sv_business,
        "cbgv_business": cbgv_business,
    }


def _mapping_text(mapping: dict) -> str:
    return " ".join([
        str(mapping.get("faq_question") or mapping.get("title") or ""),
        str(mapping.get("faq_keywords") or ""),
        str(mapping.get("faq_answer") or ""),
        str(mapping.get("faq_location") or ""),
        str(mapping.get("doc_name") or ""),
        str(mapping.get("relative_path") or ""),
    ])


def _context_cache_key(query_context: dict | None) -> tuple:
    context = query_context or {}
    return (
        context.get("audience_hint") or "unknown",
        context.get("audience_source") or "unknown",
        context.get("information_need") or "unknown",
        bool(context.get("skip_retrieval_plan_llm")),
    )


def _business_query_with_spelling_variants(query: str) -> str:
    normalized = normalize_text(query)
    extras: list[str] = []
    if "thi lai" in normalized and "dang ki" in normalized:
        extras.append("dang ky thi lai")
    if "thi lai" in normalized:
        extras.append("dang ky thi lai")
    if "thi lai" in normalized and "huy" in normalized:
        extras.append("huy dang ky thi lai")
    if "hoan thi" in normalized:
        extras.append("hoan thi mot cua khao thi gui yeu cau")
    if not extras:
        return query
    return " ".join(dict.fromkeys([query, *extras]))


def _query_context_audience(query: str, query_context: dict | None) -> tuple[str, dict]:
    query_audience, audience_signals = _query_audience(query)
    context = query_context or {}
    context_audience = context.get("audience_hint")
    if context_audience in {"sv", "cbgv", "mixed"}:
        return context_audience, {
            **audience_signals,
            "context_audience": context_audience,
            "context_source": context.get("audience_source"),
        }
    return query_audience, audience_signals


def _mapping_gate_decision(
    query: str,
    mapping: dict,
    query_context: dict | None = None,
) -> dict:
    topic_overlap = _mapping_topic_overlap(query, mapping)
    query_audience, audience_signals = _query_context_audience(query, query_context)
    mapping_audience = _mapping_audience(mapping)
    information_need = (query_context or {}).get("information_need") or "unknown"
    reasons = []
    penalties = []
    counted_signals = []
    normalized_query = normalize_text(query)
    normalized_mapping = normalize_text(_mapping_text(mapping))

    if (
        "khoi luong" in normalized_query
        and any(term in normalized_mapping for term in ("coi thi", "cham thi"))
        and not any(term in normalized_query for term in ("coi thi", "cham thi"))
    ):
        return {
            "decision": "reject",
            "reason": "specific_exam_workload_without_query_signal",
            "score": 0,
            "confidence": 0.0,
            "topic_overlap": topic_overlap,
            "llm_used": False,
            "hard_reject_reason": "specific_exam_workload_without_query_signal",
            "reasons": reasons,
            "penalties": penalties,
            "counted_signals": counted_signals,
            "query_audience": query_audience,
            "mapping_audience": mapping_audience,
            "information_need": information_need,
            "explicit_role_signals": audience_signals,
            "business_role_signals": audience_signals,
        }

    if (
        (query_audience == "sv" and mapping_audience == "cbgv")
        or (query_audience == "cbgv" and mapping_audience == "sv")
    ):
        return {
            "decision": "reject",
            "reason": "audience_mismatch",
            "score": 0,
            "confidence": 0.0,
            "topic_overlap": topic_overlap,
            "llm_used": False,
            "hard_reject_reason": "audience_mismatch",
            "reasons": reasons,
            "penalties": penalties,
            "counted_signals": counted_signals,
            "query_audience": query_audience,
            "mapping_audience": mapping_audience,
            "information_need": information_need,
            "explicit_role_signals": audience_signals,
            "business_role_signals": audience_signals,
        }

    if not _business_domain_matches(query, mapping):
        return {
            "decision": "reject",
            "reason": "domain_mismatch",
            "score": 0,
            "confidence": 0.0,
            "topic_overlap": topic_overlap,
            "llm_used": False,
            "hard_reject_reason": "domain_mismatch",
            "reasons": reasons,
            "penalties": penalties,
            "counted_signals": counted_signals,
            "query_audience": query_audience,
            "mapping_audience": mapping_audience,
            "information_need": information_need,
        }

    query_terms = _meaningful_business_terms(query)
    mapping_text = _mapping_text(mapping)
    mapping_terms = _meaningful_business_terms(mapping_text)
    overlap_terms = query_terms & mapping_terms
    non_generic_overlap = {
        term for term in overlap_terms if term not in _AUDIENCE_GENERIC_TERMS
    }

    score = 0
    normalized_mapping_text = normalize_text(mapping_text)
    phrase_candidates = [
        phrase
        for phrase in _faq_keyword_phrases(query)
        if len(get_keywords(phrase)) >= 2 and phrase not in counted_signals
    ]
    phrase_candidates.extend(
        phrase
        for phrase in _faq_keyword_phrases(mapping.get("faq_keywords", ""))
        if phrase in normalize_text(query)
    )
    for phrase in dict.fromkeys(phrase_candidates):
        if phrase and phrase in normalized_mapping_text and phrase not in _AUDIENCE_GENERIC_TERMS:
            score += 35
            reasons.append(f"phrase_match:{phrase}")
            counted_signals.append(phrase)
            break

    location_text = normalize_text(
        " ".join([
            str(mapping.get("faq_location") or ""),
            str(mapping.get("title") or ""),
            str(mapping.get("doc_name") or ""),
            str(mapping.get("relative_path") or ""),
        ])
    )
    for phrase in dict.fromkeys(phrase_candidates):
        if phrase and phrase in location_text and phrase not in counted_signals:
            score += 25
            reasons.append(f"location_match:{phrase}")
            counted_signals.append(phrase)
            break

    distinctive_hits = sorted((non_generic_overlap & _BUSINESS_DISTINCTIVE_TERMS) - set(counted_signals))
    if distinctive_hits:
        hit_score = min(len(distinctive_hits) * 45, 70)
        score += hit_score
        reasons.append(f"distinctive_overlap:{','.join(distinctive_hits[:3])}")
        counted_signals.extend(distinctive_hits[:3])

    remaining_overlap = sorted(non_generic_overlap - set(counted_signals))
    if remaining_overlap:
        hit_score = min(len(remaining_overlap) * 10, 30)
        score += hit_score
        reasons.append(f"keyword_overlap:{','.join(remaining_overlap[:3])}")
        counted_signals.extend(remaining_overlap[:3])

    if query_audience in {"sv", "cbgv"} and query_audience == mapping_audience:
        score += 20
        reasons.append("audience_match")
    elif query_audience == "mixed" and mapping_audience in {"sv", "cbgv"}:
        score -= 15
        penalties.append("mixed_audience_mapping_penalty")

    if topic_overlap > 0:
        score += 10
        reasons.append("domain_or_topic_overlap")

    if (
        information_need == "procedure_ui"
        and query_audience == "sv"
        and mapping_audience == "sv"
        and topic_overlap > 0
    ):
        score += 35
        reasons.append("query_context_procedure_audience_match")

    if overlap_terms and not non_generic_overlap:
        score -= 20
        penalties.append("generic_only_overlap")

    if len(query_terms) >= 2 and topic_overlap <= 0:
        return {
            "decision": "reject",
            "reason": "zero_topic_overlap",
            "score": score,
            "confidence": 0.0,
            "topic_overlap": topic_overlap,
            "llm_used": False,
            "hard_reject_reason": "zero_topic_overlap",
            "reasons": reasons,
            "penalties": penalties,
            "counted_signals": counted_signals,
            "query_audience": query_audience,
            "mapping_audience": mapping_audience,
            "information_need": information_need,
        }

    if (
        len(query_terms) >= 2
        and len(overlap_terms) < BUSINESS_MAPPING_MIN_TOPIC_OVERLAP
        and BUSINESS_MAPPING_LLM_JUDGE_ENABLED
        and not (query_context or {}).get("skip_retrieval_plan_llm")
    ):
        judge = _judge_mapping_with_llm(query, mapping, topic_overlap)
        return {
            **judge,
            "topic_overlap": topic_overlap,
            "llm_used": True,
        }

    if score < 35:
        decision = "reject"
        reason = "score_below_borderline"
    elif score < 55:
        decision = "reject"
        reason = "borderline_score_generic_fallback"
    else:
        decision = "accept"
        reason = "rule_score_passed"

    return {
        "decision": decision,
        "reason": reason,
        "score": score,
        "confidence": round(min(max(score / 100, 0), 1), 4),
        "topic_overlap": topic_overlap,
        "threshold_accept": 55,
        "threshold_borderline": 35,
        "hard_reject_reason": None if decision == "accept" else reason,
        "reasons": reasons,
        "penalties": penalties,
        "counted_signals": counted_signals,
        "query_audience": query_audience,
        "mapping_audience": mapping_audience,
        "information_need": information_need,
        "explicit_role_signals": audience_signals,
        "business_role_signals": audience_signals,
        "llm_used": False,
    }


def _text_matches_query_domain(query: str, text: str) -> bool:
    normalized_query = normalize_text(query)
    normalized_text = normalize_text(text)
    text_terms = _meaningful_business_terms(normalized_text)

    if any(
        phrase in normalized_query
        for phrase in ("phuc khao", "cham lai", "bai thi", "diem thi", "ket qua thi")
    ):
        return any(
            phrase in normalized_text
            for phrase in (
                "phuc khao",
                "cham lai",
                "bai thi",
                "diem thi",
                "ket qua thi",
                "khao thi",
                "hoc phan",
            )
        )

    for domain_name, phrases in _BUSINESS_DOMAIN_PHRASES.items():
        if not any(phrase in normalized_query for phrase in phrases):
            continue
        domain_terms = _BUSINESS_DOMAIN_CORE_TERMS.get(
            domain_name,
            _BUSINESS_DOMAIN_TERMS.get(domain_name, set()),
        )
        if any(phrase in normalized_text for phrase in phrases):
            return True
        if len(text_terms & domain_terms) >= 2:
            return True
        return False

    return True


def _business_domain_matches(query: str, mapping: dict) -> bool:
    normalized_query = normalize_text(query)
    normalized_mapping = normalize_text(
        " ".join([
            str(mapping.get("faq_question") or ""),
            str(mapping.get("faq_keywords") or ""),
            str(mapping.get("faq_answer") or ""),
            str(mapping.get("faq_location") or ""),
        ])
    )
    for domain_name, phrases in _BUSINESS_DOMAIN_PHRASES.items():
        if not any(phrase in normalized_query for phrase in phrases):
            continue
        domain_terms = _BUSINESS_DOMAIN_CORE_TERMS.get(
            domain_name,
            _BUSINESS_DOMAIN_TERMS.get(domain_name, set()),
        )
        mapping_domain_hits = _meaningful_business_terms(normalized_mapping) & domain_terms
        if not any(phrase in normalized_mapping for phrase in phrases) and not (
            len(mapping_domain_hits) >= 2
        ):
            return False

    query_terms = _meaningful_business_terms(query)
    if not query_terms:
        return True

    mapping_terms = _meaningful_business_terms(
        " ".join([
            str(mapping.get("faq_question") or ""),
            str(mapping.get("faq_keywords") or ""),
            str(mapping.get("faq_answer") or ""),
            str(mapping.get("faq_location") or ""),
        ])
    )
    for domain_terms in _BUSINESS_DOMAIN_TERMS.values():
        query_hits = query_terms & domain_terms
        if len(query_hits) < 2:
            continue
        if not mapping_terms & domain_terms:
            return False
    return True


def _mapping_is_suspected_wrong_topic(
    query: str,
    mapping: dict | None,
    query_context: dict | None = None,
) -> bool:
    if not mapping:
        return False
    return _mapping_gate_decision(query, mapping, query_context)["decision"] == "reject"


def _source_chunks_for_mapping(mapping: dict, chunks: list[dict]) -> list[dict]:
    source_relative_path = mapping.get("source_relative_path")
    doc_name = mapping.get("doc_name")
    if not mapping.get("source_file_found") or not source_relative_path:
        return []

    return [
        chunk
        for chunk in chunks
        if chunk.get("source_type") != BUSINESS_FAQ_SOURCE_TYPE
        and chunk.get("source_root") == mapping.get("source_root")
        and (
            chunk.get("relative_path") == source_relative_path
            or chunk.get("doc_name") == doc_name
        )
    ]


def _is_survey_mapping_or_query(query: str, mapping: dict | None = None) -> bool:
    searchable = normalize_text(
        " ".join([
            str(query or ""),
            str((mapping or {}).get("faq_question") or ""),
            str((mapping or {}).get("faq_answer") or ""),
            str((mapping or {}).get("faq_keywords") or ""),
            str((mapping or {}).get("faq_location") or ""),
        ])
    )
    return "khao sat" in searchable or "phieu khao sat" in searchable


def _is_procedure_evaluation_mapping_or_query(
    query: str,
    mapping: dict | None = None,
) -> bool:
    searchable = normalize_text(
        " ".join([
            str(query or ""),
            str((mapping or {}).get("faq_question") or ""),
            str((mapping or {}).get("faq_answer") or ""),
            str((mapping or {}).get("faq_keywords") or ""),
            str((mapping or {}).get("faq_location") or ""),
        ])
    )
    return (
        "danh gia thu tuc" in searchable
        or (
            "danh gia" in searchable
            and "thu tuc hanh chinh" in searchable
        )
        or "muc do hai long" in searchable
        or "phan hoi ve muc do hai long" in searchable
    )


def _retarget_procedure_evaluation_location(mapping: dict) -> tuple[dict, bool]:
    if not _is_procedure_evaluation_mapping_or_query("", mapping):
        return mapping, False

    current_location = mapping.get("faq_location")
    if current_location == PROCEDURE_EVALUATION_LOCATION:
        return mapping, False

    retargeted = dict(mapping)
    retargeted.update({
        "faq_location": PROCEDURE_EVALUATION_LOCATION,
        "procedure_evaluation_location_override": True,
        "procedure_evaluation_original_location": current_location,
    })
    return retargeted, True


def _retarget_survey_mapping_source(
    mapping: dict,
    chunks: list[dict],
) -> tuple[dict, list[dict], bool]:
    """Survey FAQ rows are mapped to SV support, but detailed survey docs live in the catalog file."""
    if not _is_survey_mapping_or_query("", mapping):
        return mapping, [], False

    target_doc = normalize_text(SURVEY_FALLBACK_DOC_NAME)
    target_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("source_type") != BUSINESS_FAQ_SOURCE_TYPE
        and normalize_text(chunk.get("doc_name", "")) == target_doc
    ]
    if not target_chunks:
        return mapping, [], False

    retargeted = dict(mapping)
    first_chunk = target_chunks[0]
    retargeted.update({
        "doc_name": first_chunk.get("doc_name") or SURVEY_FALLBACK_DOC_NAME,
        "relative_path": first_chunk.get("relative_path") or SURVEY_FALLBACK_DOC_NAME,
        "source_relative_path": first_chunk.get("relative_path") or SURVEY_FALLBACK_DOC_NAME,
        "source_file_found": True,
        "file_path": first_chunk.get("file_path") or str(_business_path() / SURVEY_FALLBACK_DOC_NAME),
        "source_root": first_chunk.get("source_root") or mapping.get("source_root"),
        "survey_source_override": True,
        "survey_original_doc_name": mapping.get("doc_name"),
        "survey_original_relative_path": mapping.get("source_relative_path"),
    })
    return retargeted, target_chunks, True


def _location_anchors(location: str) -> list[str]:
    normalized = normalize_text(location)
    normalized = normalized.replace("→", "->")
    parts = [
        part.strip(" .:-")
        for part in re.split(r"\s*(?:->|>)\s*", normalized)
        if part.strip(" .:-")
    ]

    anchors = []
    for part in reversed(parts):
        compact = re.sub(r"^(?:muc|chuong|dieu)\s+", "", part).strip()
        if re.fullmatch(r"\d+(?:\.\d+)+", compact):
            anchors.append(compact)
        elif re.fullmatch(r"buoc\s+\d+", compact):
            anchors.append(compact)
        elif re.fullmatch(r"\d+", compact) and len(parts) <= 2:
            anchors.append(compact)

    return list(dict.fromkeys(anchors))


def _location_score(chunk: dict, location: str) -> float:
    content = normalize_text(chunk.get("content", ""))
    anchors = _location_anchors(location)
    score = 0.0

    for index, anchor in enumerate(anchors):
        pattern = rf"(?<![\d.]){re.escape(anchor)}(?:\.|\s|$)"
        if re.search(pattern, content):
            score += 120.0 if index == 0 else 45.0

    return score


def _text_ngrams(text: str, size: int) -> set[str]:
    tokens = get_keywords(text)
    if len(tokens) < size:
        return set()
    return {
        " ".join(tokens[index:index + size])
        for index in range(len(tokens) - size + 1)
    }


def _semantic_location_score(mapping: dict, chunk: dict) -> float:
    """Validate logical FAQ locations that are not printed in the source file."""
    location = normalize_text(mapping.get("faq_location", ""))
    if not location.startswith("muc ") or not _location_anchors(location):
        return 0.0

    mapping_text = " ".join([
        str(mapping.get("faq_question") or ""),
        str(mapping.get("faq_answer") or ""),
    ])
    source_text = " ".join([
        str(chunk.get("title") or ""),
        str(chunk.get("content") or ""),
    ])
    mapping_tokens = set(get_keywords(mapping_text))
    source_tokens = set(get_keywords(source_text))
    if not mapping_tokens or not source_tokens:
        return 0.0

    token_coverage = len(mapping_tokens & source_tokens) / len(mapping_tokens)
    trigram_matches = len(_text_ngrams(mapping_text, 3) & _text_ngrams(source_text, 3))
    fourgram_matches = len(_text_ngrams(mapping_text, 4) & _text_ngrams(source_text, 4))
    if token_coverage < 0.55 or (trigram_matches < 2 and fourgram_matches < 1):
        return 0.0

    return round(
        token_coverage * 100.0
        + trigram_matches * 8.0
        + fourgram_matches * 12.0,
        4,
    )


def _location_windows(mapping: dict, source_chunks: list[dict]) -> list[dict]:
    anchors = _location_anchors(mapping.get("faq_location", ""))
    file_path = Path(str(mapping.get("file_path") or ""))
    if not anchors or not file_path.is_file() or not source_chunks:
        return []

    try:
        text = _extract_text(file_path)
    except Exception:
        return []

    anchor = anchors[0]
    pattern = re.compile(
        rf"(?mi)^\s*{re.escape(anchor)}\s*[.)\-:]?\s+"
    )
    matches = list(pattern.finditer(text))
    windows = []
    base_chunk = source_chunks[0]

    for index, match in enumerate(matches, start=1):
        start = max(0, match.start() - 180)
        end = min(len(text), match.start() + 3200)
        content = text[start:end].strip()
        if not content:
            continue

        window = _clean_index_chunk(base_chunk)
        window["content"] = content
        window["chunk_index"] = index
        windows.append(window)

    return windows


def _guided_keyword_score(query: str, mapping: dict, chunk: dict) -> float:
    searchable = normalize_text(
        f'{chunk.get("title", "")} {chunk.get("content", "")}'
    )
    normalized_query = normalize_text(query)
    searchable_tokens = set(get_keywords(searchable))
    query_tokens = set(get_keywords(query))
    keyword_tokens = set(get_keywords(mapping.get("faq_keywords", "")))
    summary_tokens = set(get_keywords(mapping.get("faq_answer", "")))

    score = 0.0
    score += len(query_tokens & searchable_tokens) * 9.0
    score += len(keyword_tokens & searchable_tokens) * 6.0
    score += len(summary_tokens & searchable_tokens) * 1.5

    for phrase in _faq_keyword_phrases(mapping.get("faq_keywords", "")):
        if len(get_keywords(phrase)) >= 2 and phrase in searchable:
            score += 18.0

    if "thi lai" in normalized_query:
        if "huy" in normalized_query:
            if "huy dang ky thi lai" in searchable:
                score += 220.0
            elif "dang ky thi lai" in searchable:
                score += 35.0
        elif "dang ky thi lai" in normalized_query and "dang ky thi lai" in searchable:
            score += 140.0
        if any(term in searchable for term in ("thi thu", "on tap", "ket qua hoc tap", "xac nhan ct&ctsv")):
            score -= 80.0

    if "hoan thi" in normalized_query:
        if "hoan thi" in searchable:
            score += 180.0
        if "mot cua" in searchable and "khao thi" in searchable:
            score += 90.0
        if any(term in searchable for term in ("khcn", "nghien cuu", "de tai", "nhiem vu kh&cn")):
            score -= 180.0

    if "khao sat" in normalized_query:
        if "khao sat noi bo" in searchable:
            score += 80.0
        if "khao sat ben ngoai" in searchable:
            score += 80.0
        if any(term in normalized_query for term in ("loai", "khac biet", "may loai")):
            if any(term in searchable for term in ("hai hinh thuc", "2 loai", "khao sat noi bo – yeu cau dang nhap")):
                score += 160.0
            if all(term in searchable for term in ("khao sat noi bo", "khao sat ben ngoai")):
                score += 120.0
            if any(term in searchable for term in ("buoc 3", "buoc 4", "nop khao sat")):
                score -= 60.0
        if any(term in normalized_query for term in ("quy trinh", "tham gia", "cach lam", "cac buoc", "nop phieu")):
            if any(term in searchable for term in ("buoc 1", "buoc 2", "buoc 3", "nop khao sat")):
                score += 140.0

    if _is_procedure_evaluation_mapping_or_query(normalized_query, mapping):
        if "danh gia thu tuc hanh chinh da xu ly" in searchable:
            score += 240.0
        if "mot-cua/danh-gia-thu-tuc-hanh-chinh" in searchable:
            score += 180.0
        if "thu tuc hanh chinh" in searchable and "danh gia" in searchable:
            score += 120.0
        if "muc do hai long" in searchable:
            score += 100.0
        if "5 muc" in searchable or "05 muc" in searchable:
            score += 90.0
        if "chi nhung thu tuc co trang thai da xu ly" in searchable:
            score += 80.0
        if any(term in searchable for term in ("khcn", "nghien cuu khoa hoc", "nhiem vu kh&cn")):
            score -= 180.0

    return round(score, 4)


def _mapping_keyword_coverage(mapping: dict, chunk: dict) -> float:
    keyword_tokens = set(get_keywords(mapping.get("faq_keywords", "")))
    if not keyword_tokens:
        return 1.0

    searchable_tokens = set(get_keywords(
        f'{chunk.get("title", "")} {chunk.get("content", "")}'
    ))
    return len(keyword_tokens & searchable_tokens) / len(keyword_tokens)


def _has_specific_mapping_phrase(mapping: dict, chunk: dict) -> bool:
    searchable = normalize_text(
        f'{chunk.get("title", "")} {chunk.get("content", "")}'
    )
    generic_phrases = {
        "xem chi tiet",
        "cach truy cap",
        "duong dan",
        "thong tin ho tro",
    }
    return any(
        phrase not in generic_phrases
        and len(get_keywords(phrase)) >= 2
        and phrase in searchable
        for phrase in _faq_keyword_phrases(mapping.get("faq_keywords", ""))
    )


def _keyword_windows(
    query: str,
    mapping: dict,
    source_chunks: list[dict],
) -> list[dict]:
    file_path = Path(str(mapping.get("file_path") or ""))
    if not file_path.is_file() or not source_chunks:
        return []

    try:
        text = _extract_text(file_path)
    except Exception:
        return []

    lines = text.splitlines()
    normalized_lines = [normalize_text(line) for line in lines]
    phrases = _faq_keyword_phrases(mapping.get("faq_keywords", ""))
    phrases.extend(
        phrase
        for phrase in _faq_keyword_phrases(query.replace(" va ", ","))
        if len(get_keywords(phrase)) >= 2
    )

    windows = []
    seen_starts = set()
    base_chunk = source_chunks[0]
    for phrase in phrases:
        if len(get_keywords(phrase)) < 2:
            continue

        for line_index, normalized_line in enumerate(normalized_lines):
            if phrase not in normalized_line:
                continue
            window_start = max(0, line_index - 3)
            bucket = window_start // 4
            if bucket in seen_starts:
                continue
            seen_starts.add(bucket)

            window = _clean_index_chunk(base_chunk)
            window["content"] = "\n".join(
                lines[window_start:min(len(lines), line_index + 22)]
            ).strip()
            window["chunk_index"] = len(windows) + 1
            windows.append(window)

    return windows


def _decorate_guided_chunk(
    chunk: dict,
    mapping: dict,
    retrieval_method: str,
    score: float,
    keyword_score: float | None = None,
    vector_score: float | None = None,
) -> dict:
    result = _clean_index_chunk(chunk)
    result.update({
        "source_type": BUSINESS_SOURCE_TYPE,
        "file_id": mapping.get("file_id"),
        "faq_location": (
            mapping.get("faq_location")
            if retrieval_method == "location"
            else None
        ),
        "audience": mapping.get("audience"),
        "mapping_relative_path": mapping.get("mapping_relative_path"),
        "mapping_score": mapping.get("mapping_score"),
        "retrieval_method": retrieval_method,
        "matched_location": (
            mapping.get("faq_location")
            if retrieval_method == "location"
            else None
        ),
        "score": round(float(score), 4),
    })
    if keyword_score is not None:
        result["keyword_score"] = round(float(keyword_score), 4)
    if vector_score is not None:
        result["vector_score"] = round(float(vector_score), 4)
    if retrieval_method == "location" and mapping.get("faq_location"):
        result["title"] = mapping["faq_location"]
    return result


def _search_location_in_source(
    query: str,
    mapping: dict,
    source_chunks: list[dict],
    limit: int,
) -> list[dict]:
    ranked = []
    location_chunks = _location_windows(mapping, source_chunks)
    candidates = location_chunks + source_chunks
    seen_candidates = set()

    for chunk in candidates:
        candidate_key = (
            chunk.get("relative_path"),
            chunk.get("chunk_index"),
            hashlib.sha256(str(chunk.get("content") or "").encode("utf-8")).hexdigest(),
        )
        if candidate_key in seen_candidates:
            continue
        seen_candidates.add(candidate_key)

        anchor_score = _location_score(chunk, mapping.get("faq_location", ""))
        semantic_score = _semantic_location_score(mapping, chunk)
        # Numeric anchors such as "1.2" can occur elsewhere in the same file.
        # Require the surrounding source text to agree with the selected FAQ row.
        if semantic_score <= 0:
            continue
        keyword_score = _guided_keyword_score(query, mapping, chunk)
        if keyword_score < MIN_SEARCH_SCORE:
            continue
        if _mapping_keyword_coverage(mapping, chunk) < 0.4:
            continue
        ranked.append((semantic_score, anchor_score, keyword_score, chunk))

    if ranked:
        semantic_threshold = max(item[0] for item in ranked) * 0.82
        ranked = [item for item in ranked if item[0] >= semantic_threshold]
    ranked.sort(
        key=lambda item: item[0] + min(item[1], 10.0) + item[2],
        reverse=True,
    )
    return [
        _decorate_guided_chunk(
            chunk,
            mapping,
            "location",
            semantic_score + min(anchor_score, 10.0) + keyword_score,
            keyword_score=keyword_score,
        )
        for semantic_score, anchor_score, keyword_score, chunk in ranked[:limit]
    ]


def _search_keywords_in_source(
    query: str,
    mapping: dict,
    source_chunks: list[dict],
    limit: int,
) -> list[dict]:
    ranked = []
    candidates = _keyword_windows(query, mapping, source_chunks) + source_chunks
    for chunk in candidates:
        score = _guided_keyword_score(query, mapping, chunk)
        if score >= MIN_SEARCH_SCORE:
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        _decorate_guided_chunk(
            chunk,
            mapping,
            "keyword",
            score,
            keyword_score=score,
        )
        for score, chunk in ranked[:limit]
    ]


def _search_vectors_in_source(
    query: str,
    mapping: dict,
    source_chunks: list[dict],
    limit: int,
) -> tuple[list[dict], str | None]:
    try:
        from app.data.embedding_client import embed_documents, embed_query

        guidance = " ".join([
            query,
            mapping.get("faq_keywords", ""),
            mapping.get("faq_answer", ""),
        ])
        query_vector = embed_query(guidance)
        document_vectors = embed_documents([
            chunk.get("content", "")
            for chunk in source_chunks
        ])
    except Exception as exc:
        return [], str(exc)

    ranked = []
    for chunk, vector in zip(source_chunks, document_vectors):
        score = sum(left * right for left, right in zip(query_vector, vector))
        if score >= BUSINESS_GUIDED_VECTOR_MIN_SCORE:
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        _decorate_guided_chunk(
            chunk,
            mapping,
            "vector",
            score,
            vector_score=score,
        )
        for score, chunk in ranked[:limit]
    ], None


def _compact_guided_sources(results: list[dict]) -> list[dict]:
    return [
        {
            "doc_name": item.get("doc_name"),
            "relative_path": item.get("relative_path"),
            "chunk_index": item.get("chunk_index"),
            "retrieval_method": item.get("retrieval_method"),
            "matched_location": item.get("matched_location"),
            "score": item.get("score"),
            "keyword_score": item.get("keyword_score"),
            "vector_score": item.get("vector_score"),
        }
        for item in results
    ]


def _search_generic_business_chunks(
    query: str,
    chunks: list[dict],
    doc_freq: Counter,
    total_docs: int,
    limit: int,
    retrieval_method: str,
    original_query: str | None = None,
    retrieval_plan: dict | None = None,
) -> list[dict]:
    generic_results = []
    original_query = original_query or query
    plan_must = [normalize_text(item) for item in (retrieval_plan or {}).get("must", [])]
    plan_avoid = [normalize_text(item) for item in (retrieval_plan or {}).get("avoid", [])]
    for chunk in chunks:
        if chunk.get("source_type") == BUSINESS_FAQ_SOURCE_TYPE:
            continue
        searchable_text = " ".join([
            str(chunk.get("title") or ""),
            str(chunk.get("content") or ""),
            str(chunk.get("doc_name") or ""),
        ])
        if not _text_matches_query_domain(original_query, searchable_text):
            continue
        score = _score_chunk(query, chunk, doc_freq, total_docs)
        normalized_searchable = normalize_text(searchable_text)
        must_hits = [phrase for phrase in plan_must if phrase and phrase in normalized_searchable]
        avoid_hits = [phrase for phrase in plan_avoid if phrase and phrase in normalized_searchable]
        score += len(must_hits) * 45.0
        score -= len(avoid_hits) * 80.0
        if score < MIN_SEARCH_SCORE:
            continue
        original_score = _score_chunk(original_query, chunk, doc_freq, total_docs)
        result = _clean_index_chunk(chunk)
        result["score"] = score
        result["keyword_score"] = score
        result["original_keyword_score"] = original_score
        result["retrieval_plan_must_hits"] = must_hits
        result["retrieval_plan_avoid_hits"] = avoid_hits
        result["retrieval_method"] = retrieval_method
        generic_results.append(result)

    generic_results.sort(
        key=lambda item: (
            item.get("score") or 0,
            item.get("original_keyword_score") or 0,
        ),
        reverse=True,
    )
    return generic_results[:limit]


def _chunk_vector_cache_key(chunk: dict) -> tuple:
    content = str(chunk.get("content") or "")
    digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()
    return (
        chunk.get("source_root"),
        chunk.get("relative_path") or chunk.get("doc_name"),
        chunk.get("chunk_index"),
        digest,
    )


def _search_generic_business_vectors(
    query_text: str,
    chunks: list[dict],
    keyword_candidates: list[dict],
    retrieval_plan: dict,
    limit: int,
    debug: dict,
) -> list[dict]:
    debug.update({
        "vector_cache_hit_count": 0,
        "vector_cache_miss_count": 0,
        "runtime_embedded_chunk_count": 0,
        "vector_disabled_reason": None,
        "vector_error": None,
    })
    if not BUSINESS_GENERIC_VECTOR_ENABLED:
        debug["vector_disabled_reason"] = "disabled"
        return []
    query_text = " ".join(str(query_text or "").split())
    if not query_text:
        debug["vector_disabled_reason"] = "empty_vector_query"
        return []

    candidate_keys = {
        (
            item.get("source_root"),
            item.get("relative_path") or item.get("doc_name"),
            item.get("chunk_index"),
        )
        for item in keyword_candidates
    }
    candidate_chunks = [
        chunk for chunk in chunks
        if chunk.get("source_type") != BUSINESS_FAQ_SOURCE_TYPE
        and (
            not candidate_keys
            or (
                chunk.get("source_root"),
                chunk.get("relative_path") or chunk.get("doc_name"),
                chunk.get("chunk_index"),
            ) in candidate_keys
        )
    ]
    if not candidate_chunks:
        candidate_chunks = [
            chunk for chunk in chunks
            if chunk.get("source_type") != BUSINESS_FAQ_SOURCE_TYPE
        ][:BUSINESS_GENERIC_VECTOR_MAX_RUNTIME_EMBED_CHUNKS]

    candidate_chunks = candidate_chunks[:BUSINESS_GENERIC_VECTOR_MAX_RUNTIME_EMBED_CHUNKS]
    try:
        from app.data.embedding_client import embed_documents, embed_query

        query_vector = embed_query(query_text)
        missing_chunks = []
        vectors = []
        for chunk in candidate_chunks:
            key = _chunk_vector_cache_key(chunk)
            cached = _BUSINESS_VECTOR_CACHE.get(key)
            if cached is not None:
                debug["vector_cache_hit_count"] += 1
                vectors.append(cached)
            else:
                debug["vector_cache_miss_count"] += 1
                missing_chunks.append((key, chunk))
                vectors.append(None)

        if missing_chunks:
            embedded = embed_documents([chunk.get("content", "") for _, chunk in missing_chunks])
            debug["runtime_embedded_chunk_count"] = len(embedded)
            embedded_iter = iter(embedded)
            vectors = []
            for chunk in candidate_chunks:
                key = _chunk_vector_cache_key(chunk)
                cached = _BUSINESS_VECTOR_CACHE.get(key)
                if cached is None:
                    cached = next(embedded_iter)
                    _BUSINESS_VECTOR_CACHE[key] = cached
                vectors.append(cached)
    except Exception as exc:
        debug["vector_error"] = str(exc)
        return []

    plan_avoid = [normalize_text(item) for item in (retrieval_plan or {}).get("avoid", [])]
    results = []
    for chunk, vector in zip(candidate_chunks, vectors):
        if not vector:
            continue
        score = sum(left * right for left, right in zip(query_vector, vector))
        if score < BUSINESS_GENERIC_VECTOR_MIN_SCORE:
            continue
        normalized_searchable = normalize_text(
            f'{chunk.get("title", "")} {chunk.get("content", "")} {chunk.get("doc_name", "")}'
        )
        avoid_hits = [phrase for phrase in plan_avoid if phrase and phrase in normalized_searchable]
        result = _clean_index_chunk(chunk)
        result["vector_score"] = round(float(score), 4)
        result["retrieval_plan_avoid_hits"] = avoid_hits
        result["retrieval_method"] = "generic_vector"
        results.append(result)

    results.sort(key=lambda item: item.get("vector_score") or 0, reverse=True)
    return results[:limit]


def _generic_result_key(item: dict) -> tuple:
    return (
        item.get("source_type"),
        item.get("relative_path") or item.get("doc_name"),
        item.get("chunk_index"),
        item.get("title"),
    )


def _merge_generic_business_results(
    keyword_results: list[dict],
    vector_results: list[dict],
    query: str,
    retrieval_plan: dict,
) -> tuple[list[dict], dict]:
    merged = {}
    top_keyword_score = max(
        [float(item.get("keyword_score") or item.get("score") or 0) for item in keyword_results] or [0.0]
    )
    keyword_denominator = max(top_keyword_score, MIN_SEARCH_SCORE * 4, 1)
    weak_keyword_branch = top_keyword_score < MIN_SEARCH_SCORE
    top_vector_score = max([float(item.get("vector_score") or 0) for item in vector_results] or [0.0])
    for item in keyword_results + vector_results:
        key = _generic_result_key(item)
        existing = merged.get(key)
        if existing is None:
            existing = dict(item)
            merged[key] = existing
        else:
            existing.update({k: v for k, v in item.items() if v is not None})

    plan_avoid = [normalize_text(item) for item in (retrieval_plan or {}).get("avoid", [])]
    query_audience, _ = _query_audience(query)
    for item in merged.values():
        keyword_score = float(item.get("keyword_score") or item.get("score") or 0)
        keyword_norm = min(max(keyword_score / keyword_denominator, 0.0), 1.0)
        if weak_keyword_branch:
            keyword_norm = min(keyword_norm, 0.35)
        vector_score = float(item.get("vector_score") or 0)
        vector_norm = min(max(vector_score, 0.0), 1.0)
        combined = 0.55 * keyword_norm + 0.45 * vector_norm
        penalties = []
        reasons = []
        normalized_searchable = normalize_text(
            f'{item.get("title", "")} {item.get("content", "")} {item.get("doc_name", "")} {item.get("relative_path", "")}'
        )
        avoid_hits = [phrase for phrase in plan_avoid if phrase and phrase in normalized_searchable]
        if avoid_hits:
            combined -= 0.25
            penalties.append(f"avoid:{','.join(avoid_hits[:3])}")
        item_audience = _infer_audience_from_text(normalized_searchable)
        if (
            (query_audience == "sv" and item_audience == "cbgv")
            or (query_audience == "cbgv" and item_audience == "sv")
        ):
            combined -= 0.20
            penalties.append("audience_mismatch")
        combined = min(max(combined, 0.0), 1.0)
        item["keyword_norm"] = round(keyword_norm, 4)
        item["vector_norm"] = round(vector_norm, 4)
        item["combined_score"] = round(combined * 100, 4)
        item["score"] = item["combined_score"]
        item["retrieval_method"] = "generic_hybrid"
        item["generic_hybrid_reasons"] = reasons
        item["generic_hybrid_penalties"] = penalties
        item["counted_signals"] = []

    ranked = sorted(merged.values(), key=lambda item: item.get("combined_score") or 0, reverse=True)
    debug = {
        "score_scale": "generic_hybrid_0_100",
        "score_scope": "request_local_not_cross_request_comparable",
        "top_keyword_score": round(top_keyword_score, 4),
        "keyword_normalization_denominator": round(keyword_denominator, 4),
        "weak_keyword_branch": weak_keyword_branch,
        "top_vector_score": round(top_vector_score, 4),
        "vector_normalization_method": "clamp_0_1",
    }
    return ranked[:BUSINESS_GENERIC_FINAL_TOP_K], debug


def _search_generic_business_hybrid(
    query: str,
    chunks: list[dict],
    doc_freq: Counter,
    total_docs: int,
    original_query: str,
    retrieval_plan: dict,
    debug: dict,
) -> list[dict]:
    keyword_results = _search_generic_business_chunks(
        query,
        chunks,
        doc_freq,
        total_docs,
        BUSINESS_GENERIC_KEYWORD_TOP_K,
        "generic_keyword",
        original_query=original_query,
        retrieval_plan=retrieval_plan,
    )
    vector_search_text = (
        retrieval_plan.get("hyde")
        or retrieval_plan.get("query")
        or original_query
        or query
    )
    vector_debug = {}
    vector_results = _search_generic_business_vectors(
        vector_search_text,
        chunks,
        keyword_results,
        retrieval_plan,
        BUSINESS_GENERIC_VECTOR_TOP_K,
        vector_debug,
    )
    merged_results, score_debug = _merge_generic_business_results(
        keyword_results,
        vector_results,
        original_query,
        retrieval_plan,
    )
    debug.update(vector_debug)
    debug.update(score_debug)
    debug.update({
        "generic_keyword_count": len(keyword_results),
        "generic_vector_count": len(vector_results),
        "generic_merged_count": len({_generic_result_key(item) for item in keyword_results + vector_results}),
        "generic_final_count": len(merged_results),
        "keyword_top_sources": _compact_guided_sources(keyword_results[:5]),
        "vector_top_sources": _compact_guided_sources(vector_results[:5]),
        "merged_top_sources": _compact_guided_sources(merged_results[:5]),
        "vector_search_text": vector_search_text,
        "retrieval_method": "generic_hybrid" if merged_results else None,
    })
    return merged_results


def search_business_sources(
    query: str,
    limit: int | None = None,
    debug: dict | None = None,
    query_context: dict | None = None,
) -> list[dict]:
    query = str(query or "").strip()
    retrieval_query = _business_query_with_spelling_variants(query)
    limit = limit or BUSINESS_SEARCH_TOP_K
    context_key = _context_cache_key(query_context)
    cache_key = (
        "retrieval_plan_v2_generic_hybrid",
        normalize_text(query),
        limit,
        context_key,
    )

    if cache_key in _BUSINESS_SEARCH_CACHE:
        cached = deepcopy(_BUSINESS_SEARCH_CACHE[cache_key])
        if debug is not None:
            debug.update(cached.get("debug", {}))
            debug["cache_hit"] = True
        return cached.get("results", [])

    context_audience = (query_context or {}).get("audience_hint")
    context_information_need = (query_context or {}).get("information_need")
    normalized_query = normalize_text(query)
    force_student_support_source = (
        context_audience == "sv"
        and context_information_need == "procedure_ui"
        and any(term in normalized_query for term in ("thi lai", "hoan thi"))
    )
    force_direct_source = _should_search_cbgv_source_directly(query) or (
        context_audience == "cbgv"
        and context_information_need == "procedure_ui"
        and not _should_keep_cbgv_mapping_candidates(query)
    )
    chunks, doc_freq, total_docs = _load_business_index()
    if force_direct_source or force_student_support_source:
        source_marker = "web support sv" if force_student_support_source else "web support cbgv"
        chunks = [
            chunk for chunk in chunks
            if source_marker in normalize_text(
                " ".join([
                    str(chunk.get("doc_name") or ""),
                    str(chunk.get("relative_path") or ""),
                ])
            )
        ]
        doc_freq = Counter()
        for chunk in chunks:
            doc_freq.update(set(_meaningful_business_terms(chunk.get("content", ""))))
        total_docs = len(chunks)
    mappings = [] if (force_direct_source or force_student_support_source) else _mapping_candidates(query, chunks)
    selected_mapping = None
    mapping_rejected_reason = None
    rejected_mapping_count = 0
    mapping_gate_decisions = []
    for mapping in mappings:
        gate_decision = _mapping_gate_decision(query, mapping, query_context)
        mapping_gate_decisions.append({
            "question": mapping.get("faq_question"),
            "score": mapping.get("mapping_score"),
            "gate_score": gate_decision.get("score"),
            "decision": gate_decision.get("decision"),
            "reason": gate_decision.get("reason"),
            "hard_reject_reason": gate_decision.get("hard_reject_reason"),
            "topic_overlap": gate_decision.get("topic_overlap"),
            "llm_used": gate_decision.get("llm_used"),
            "confidence": gate_decision.get("confidence"),
            "matched_topic": gate_decision.get("matched_topic"),
            "missing_topic": gate_decision.get("missing_topic"),
            "query_audience": gate_decision.get("query_audience"),
            "mapping_audience": gate_decision.get("mapping_audience"),
            "information_need": gate_decision.get("information_need"),
            "reasons": gate_decision.get("reasons"),
            "penalties": gate_decision.get("penalties"),
            "score_components": gate_decision.get("reasons") or [],
        })
        if gate_decision.get("decision") == "reject":
            rejected_mapping_count += 1
            if mapping_rejected_reason is None:
                mapping_rejected_reason = gate_decision.get("reason") or "suspected_wrong_topic"
            continue
        selected_mapping = mapping
        selected_mapping["mapping_gate_score"] = gate_decision.get("score")
        break

    final_results = []
    retrieval_method = None
    vector_error = None
    generic_hybrid_debug = {}
    retrieval_plan = _empty_retrieval_plan("not_needed")
    final_search_query = retrieval_query
    source_chunks = []
    mapping_ambiguous = False
    survey_source_override = False
    procedure_evaluation_location_override = False
    procedure_evaluation_original_location = None

    if (
        query_context
        and not selected_mapping
        and (query_context.get("information_need") or "unknown") == "unknown"
        and len(_meaningful_business_terms(query)) <= 1
    ):
        mapping_ambiguous = True
        mapping_rejected_reason = "overly_generic_query"

    if selected_mapping:
        if _is_procedure_evaluation_mapping_or_query(query, selected_mapping):
            selected_mapping, procedure_evaluation_location_override = (
                _retarget_procedure_evaluation_location(selected_mapping)
            )
            procedure_evaluation_original_location = selected_mapping.get(
                "procedure_evaluation_original_location"
            )
        if _is_survey_mapping_or_query(query, selected_mapping):
            selected_mapping, source_chunks, survey_source_override = _retarget_survey_mapping_source(
                selected_mapping,
                chunks,
            )
        if not source_chunks:
            source_chunks = _source_chunks_for_mapping(selected_mapping, chunks)
        if source_chunks:
            final_results = _search_location_in_source(
                retrieval_query,
                selected_mapping,
                source_chunks,
                limit,
            )
            retrieval_method = "location" if final_results else None

            if not final_results:
                final_results = _search_keywords_in_source(
                    retrieval_query,
                    selected_mapping,
                    source_chunks,
                    limit,
                )
                retrieval_method = "keyword" if final_results else None

            if not final_results:
                final_results, vector_error = _search_vectors_in_source(
                    retrieval_query,
                    selected_mapping,
                    source_chunks,
                    limit,
                )
                retrieval_method = "vector" if final_results else None
    if not final_results and not mapping_ambiguous:
        if (query_context or {}).get("skip_retrieval_plan_llm"):
            retrieval_plan = _empty_retrieval_plan("skipped_for_multihop")
            retrieval_plan["query"] = query
        else:
            retrieval_plan = _generate_business_retrieval_plan(query)
        final_search_query = _plan_search_query(query, retrieval_plan)
        if final_search_query:
            final_results = _search_generic_business_hybrid(
                final_search_query,
                chunks,
                doc_freq,
                total_docs,
                original_query=query,
                retrieval_plan=retrieval_plan,
                debug=generic_hybrid_debug,
            )
            retrieval_method = "generic_hybrid" if final_results else None

    if not final_results and not mapping_ambiguous:
        final_results = _search_generic_business_chunks(
            retrieval_query,
            chunks,
            doc_freq,
            total_docs,
            limit,
            "generic_keyword",
            original_query=query,
            retrieval_plan=retrieval_plan,
        )
        retrieval_method = "generic_keyword" if final_results else retrieval_method

    debug_data = {
        "cache_hit": False,
        "audience_hint": (query_context or {}).get("audience_hint"),
        "audience_source": (query_context or {}).get("audience_source"),
        "audience_confidence": (query_context or {}).get("audience_confidence"),
        "information_need": (query_context or {}).get("information_need"),
        "query_context": deepcopy(query_context or {}),
        "business_documents_dir": str(_business_path()),
        "indexed_chunk_count": total_docs,
        "candidate_count": len(mappings),
        "force_direct_source": force_direct_source,
        "final_results_count": len(final_results),
        "mapping_selected": bool(selected_mapping),
        "mapping_ambiguous": mapping_ambiguous,
        "mapping_rejected_reason": mapping_rejected_reason,
        "rejected_mapping_count": rejected_mapping_count,
        "mapping_score": selected_mapping.get("mapping_score") if selected_mapping else None,
        "mapping_gate_score": selected_mapping.get("mapping_gate_score") if selected_mapping else None,
        "mapping_question": selected_mapping.get("faq_question") if selected_mapping else None,
        "top_mapping_score": mappings[0].get("mapping_score") if mappings else None,
        "top_mapping_question": mappings[0].get("faq_question") if mappings else None,
        "top_mapping_topic_overlap": (
            _mapping_topic_overlap(query, mappings[0])
            if mappings
            else None
        ),
        "mapping_gate_decisions": mapping_gate_decisions[:5],
        "file_id": selected_mapping.get("file_id") if selected_mapping else None,
        "source_file": selected_mapping.get("doc_name") if selected_mapping else None,
        "source_file_found": bool(source_chunks) if selected_mapping else None,
        "survey_source_override": survey_source_override,
        "survey_original_source_file": (
            selected_mapping.get("survey_original_doc_name")
            if selected_mapping
            else None
        ),
        "procedure_evaluation_location_override": procedure_evaluation_location_override,
        "procedure_evaluation_original_location": procedure_evaluation_original_location,
        "requested_location": selected_mapping.get("faq_location") if selected_mapping else None,
        "matched_location": (
            selected_mapping.get("faq_location")
            if retrieval_method == "location"
            else None
        ),
        "retrieval_method": retrieval_method,
        "retrieval_plan": retrieval_plan,
        "retrieval_plan_parse_error": retrieval_plan.get("parse_error"),
        "final_search_query": final_search_query,
        "fallback_reason": None,
        "business_hyde": {
            "text": retrieval_plan.get("hyde") or retrieval_plan.get("query") or "",
            "status": retrieval_plan.get("status"),
            "retrieval_plan_migrated": True,
        },
        "vector_error": vector_error,
        "final_sources": _compact_guided_sources(final_results),
    }
    debug_data.update(generic_hybrid_debug)

    _BUSINESS_SEARCH_CACHE[cache_key] = {
        "results": deepcopy(final_results),
        "debug": deepcopy(debug_data),
    }

    if debug is not None:
        debug.update(debug_data)

    return final_results
