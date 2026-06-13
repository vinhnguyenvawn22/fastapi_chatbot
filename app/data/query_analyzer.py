import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum


class QueryIntent(str, Enum):
    INTERNAL_DOCUMENT = "internal_document"
    WEBSITE_UNETI = "website_uneti"
    GENERAL_ADVICE = "general_advice"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class QueryAnalysis:
    intent: QueryIntent
    metadata: dict[str, str | int] = field(default_factory=dict)
    reason: str = ""


WEBSITE_TERMS = {
    "website", "trang web", "web uneti", "uneti.edu.vn", "api.uneti.edu.vn",
    "bai viet", "tin tuc", "tin moi", "thong bao tren web", "tren website",
    "link", "duong dan", "cong thong tin",
}

WEBSITE_NEWS_TERMS = {
    "lich nghi", "ngay nghi", "nghi le", "gio to", "hung vuong",
    "30/4", "1/5", "quoc te lao dong", "tuyen sinh", "diem chuan",
    "thong tin tuyen sinh", "xet tuyen", "nhap hoc",
}

BUSINESS_SUPPORT_TERMS = {
    "web support", "support uneti", "support.uneti.edu.vn",
    "module tin tuc thong bao", "tin tuc thong bao",
    "ket qua hoc tap", "xem diem", "diem thanh phan",
    "lich hoc lich thi", "thoi khoa bieu",
    "bao hong", "bao hong thiet bi", "su co thiet bi",
    "khoi luong cong tac", "cong tac giang vien", "khoi luong giang day",
    "coi thi", "cham thi", "khao sat noi bo", "khao sat bat buoc",
    "danh gia thu tuc", "thong ke mot cua", "email google workspace",
}

DOCUMENT_TERMS = {
    "quy dinh", "quy che", "quyet dinh", "thong bao", "van ban",
    "tai lieu", "noi bo", "dieu", "muc", "chuong", "ban hanh",
    "hieu luc", "hoc vu", "dang ky hoc phan", "tot nghiep", "tin chi",
    "hoc phi", "thi", "phong hoc", "email", "lms",
}

GENERAL_TERMS = {
    "tu van", "goi y", "nen", "cach hoc", "kinh nghiem", "loi khuyen",
    "giai thich", "tom tat", "viet giup", "soan giup",
}

OUT_OF_SCOPE_TERMS = {
    "thoi tiet", "bong da", "gia vang", "chung khoan", "bitcoin",
    "nau an", "du lich", "mua gi", "phim", "am nhac",
}


def normalize_text(text: str = "") -> str:
    text = str(text or "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    text = text.replace("Ä‘", "d").replace("Ä", "D")
    return " ".join(text.lower().split())


def normalize_date(value: str) -> str:
    value = str(value or "").strip()

    match = re.search(
        r"(\d{1,2})\s*(?:/|-|\.|tháng)\s*(\d{1,2})\s*(?:/|-|\.|năm)?\s*(\d{4})",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        return value

    day, month, year = match.groups()
    return f"{int(day):02d}/{int(month):02d}/{year}"


def extract_metadata_constraints(text: str) -> dict[str, str | int]:
    normalized = normalize_text(text)
    metadata: dict[str, str | int] = {}

    so_van_ban = re.search(
        r"(?:so|van\s*ban|quyet\s*dinh|quy\s*dinh|qd)\s*[:\-]?\s*(\d{1,6})",
        normalized,
    )
    if not so_van_ban:
        so_van_ban = re.search(
            r"\b(\d{2,6})\s*/\s*(?:qd|qđ|vb|tb|qc|qd-)",
            normalized,
        )
    if so_van_ban:
        metadata["so_van_ban"] = so_van_ban.group(1)

    dieu = re.search(r"\bdieu\s+(\d{1,4})\b", normalized)
    if dieu:
        metadata["dieu"] = int(dieu.group(1))

    muc = re.search(r"\bmuc\s+(\d{1,4})\b", normalized)
    if muc:
        metadata["muc"] = int(muc.group(1))

    chuong = re.search(r"\bchuong\s+([ivxlcdm]+|\d{1,4})\b", normalized)
    if chuong:
        metadata["chuong"] = chuong.group(1).upper()

    date_match = re.search(
        r"(\d{1,2}\s*(?:/|-|\.|tháng)\s*\d{1,2}\s*(?:/|-|\.|năm)?\s*\d{4})",
        text,
        flags=re.IGNORECASE,
    )
    if date_match:
        metadata["ngay"] = normalize_date(date_match.group(1))

    return metadata


def classify_query(question: str) -> QueryAnalysis:
    normalized = normalize_text(question)
    metadata = extract_metadata_constraints(question)

    if not normalized:
        return QueryAnalysis(QueryIntent.OUT_OF_SCOPE, reason="empty_question")

    if any(term in normalized for term in BUSINESS_SUPPORT_TERMS):
        return QueryAnalysis(QueryIntent.INTERNAL_DOCUMENT, metadata, "business_support_terms")

    if any(term in normalized for term in WEBSITE_TERMS):
        return QueryAnalysis(QueryIntent.WEBSITE_UNETI, metadata, "website_terms")

    if any(term in normalized for term in WEBSITE_NEWS_TERMS):
        return QueryAnalysis(QueryIntent.WEBSITE_UNETI, metadata, "website_news_terms")

    if metadata:
        return QueryAnalysis(QueryIntent.INTERNAL_DOCUMENT, metadata, "metadata_query")

    if any(term in normalized for term in OUT_OF_SCOPE_TERMS):
        return QueryAnalysis(QueryIntent.OUT_OF_SCOPE, reason="out_of_scope_terms")

    if any(term in normalized for term in DOCUMENT_TERMS):
        return QueryAnalysis(QueryIntent.INTERNAL_DOCUMENT, metadata, "document_terms")

    if any(term in normalized for term in GENERAL_TERMS):
        return QueryAnalysis(QueryIntent.GENERAL_ADVICE, metadata, "general_terms")

    return QueryAnalysis(QueryIntent.INTERNAL_DOCUMENT, metadata, "default_to_document")
