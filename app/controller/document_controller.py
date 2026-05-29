from pathlib import Path
from pypdf import PdfReader

from app.core.config import DOCUMENTS_DIR


def list_documents():
    documents_path = Path(DOCUMENTS_DIR)

    if not documents_path.exists():
        return []

    files = []

    for file_path in documents_path.glob("*.pdf"):
        files.append({
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size_kb": round(file_path.stat().st_size / 1024, 2)
        })

    return files


def extract_pdf_text(file_name: str):
    documents_path = Path(DOCUMENTS_DIR)
    file_path = documents_path / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_name}")

    reader = PdfReader(str(file_path))

    text_parts = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_parts.append(f"\n--- Trang {page_index + 1} ---\n{text}")

    return "\n".join(text_parts).strip()