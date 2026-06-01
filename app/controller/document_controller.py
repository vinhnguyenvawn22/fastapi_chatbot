from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re

from fastapi import HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    MAX_UPLOAD_SIZE_MB,
)


PDF_MIME_TYPES = {"application/pdf", "application/octet-stream"}
UNSAFE_FILENAME_PATTERN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
SECTION_PATTERN = re.compile(
    r"(?im)^(Điều\s+\d+[\.\:\s]+.*|Mục\s+\d+[\.\:\s]+.*|Chương\s+\d+[\.\:\s]+.*)$"
)


def _documents_path() -> Path:
    return Path(DOCUMENTS_DIR).resolve()


def _safe_pdf_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = UNSAFE_FILENAME_PATTERN.sub("_", name)
    name = re.sub(r"\s+", " ", name)

    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF")

    return name


def _safe_relative_pdf_path(file_name: str) -> Path:
    raw_name = str(file_name or "").strip().replace("\\", "/")

    if not raw_name:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    relative_path = Path(raw_name)

    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    for part in relative_path.parts:
        if not part or part in {".", ".."}:
            raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

        if UNSAFE_FILENAME_PATTERN.search(part):
            raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    if relative_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF")

    return relative_path


def _resolve_document_path(file_name: str) -> Path:
    documents_path = _documents_path()
    relative_path = _safe_relative_pdf_path(file_name)
    file_path = (documents_path / relative_path).resolve()

    if not file_path.is_relative_to(documents_path):
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    return file_path


def _file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def _unique_file_path(file_path: Path) -> Path:
    if not file_path.exists():
        return file_path

    stem = file_path.stem
    suffix = file_path.suffix
    parent = file_path.parent

    for index in range(1, 1000):
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate

    raise HTTPException(status_code=409, detail="Không thể tạo tên file không trùng")


def list_documents():
    documents_path = _documents_path()

    if not documents_path.exists():
        return []

    files = []

    for file_path in sorted(documents_path.glob("*.pdf")):
        stat = file_path.stat()
        files.append({
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size_kb": round(stat.st_size / 1024, 2),
            "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })

    return files


def extract_pdf_text(file_name: str):
    file_path = _resolve_document_path(file_name)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Không tìm thấy file: {file_name}")

    try:
        reader = PdfReader(str(file_path))
    except PdfReadError as exc:
        raise HTTPException(status_code=400, detail="File PDF không đọc được") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Không thể mở file PDF") from exc

    text_parts = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_parts.append(f"\n--- Trang {page_index + 1} ---\n{text}")

    return "\n".join(text_parts).strip()


async def upload_document(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    if file.content_type and file.content_type not in PDF_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF")

    safe_name = _safe_pdf_filename(file.filename)
    documents_path = _documents_path()
    documents_path.mkdir(parents=True, exist_ok=True)

    file_path = _unique_file_path((documents_path / safe_name).resolve())
    max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
    bytes_written = 0

    try:
        with file_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File vượt quá giới hạn {MAX_UPLOAD_SIZE_MB}MB",
                    )

                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Không thể lưu file upload") from exc
    finally:
        await file.close()

    try:
        PdfReader(str(file_path))
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File upload không phải PDF hợp lệ") from exc

    # Invalidate in-memory retrieval cache after a successful upload.
    try:
        from app.data.elasticsearch_client import clear_document_index_cache

        clear_document_index_cache()
    except Exception:
        pass

    return {
        "message": "Upload tài liệu thành công",
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size_kb": round(file_path.stat().st_size / 1024, 2),
        "content_hash": _file_sha256(file_path),
    }


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0")

    if overlap < 0:
        raise ValueError("overlap không được âm")

    if overlap >= chunk_size:
        overlap = max(chunk_size // 5, 0)

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks


def build_document_chunks(file_name: str):
    file_path = _resolve_document_path(file_name)
    text = extract_pdf_text(file_name)
    chunks = split_text_by_metadata(text)
    stat = file_path.stat()
    content_hash = _file_sha256(file_path)

    documents = []

    for index, chunk in enumerate(chunks, start=1):
        documents.append({
            "doc_name": file_path.name,
            "title": chunk["title"],
            "dieu": chunk["dieu"],
            "chunk_index": index,
            "content": chunk["content"],
            "file_path": str(file_path),
            "source_type": "official_document",
            "is_active": True,
            "content_hash": content_hash,
            "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })

    return documents


def split_text_by_metadata(text: str):
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        fallback_chunks = chunk_text(text)
        return [
            {
                "title": f"Đoạn {index}",
                "dieu": None,
                "content": chunk,
            }
            for index, chunk in enumerate(fallback_chunks, start=1)
        ]

    chunks = []

    if matches[0].start() > 0:
        intro = text[:matches[0].start()].strip()
        if intro:
            for index, chunk in enumerate(chunk_text(intro), start=1):
                title = "Phần mở đầu" if index == 1 else f"Phần mở đầu ({index})"
                chunks.append({
                    "title": title,
                    "dieu": None,
                    "content": chunk,
                })

    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        dieu_match = re.search(r"Điều\s+(\d+)", title, flags=re.IGNORECASE)
        dieu = int(dieu_match.group(1)) if dieu_match else None

        split_chunks = chunk_text(content)
        for split_index, split_chunk in enumerate(split_chunks, start=1):
            chunk_title = title if len(split_chunks) == 1 else f"{title} ({split_index})"
            chunks.append({
                "title": chunk_title,
                "dieu": dieu,
                "content": split_chunk,
            })

    return chunks
