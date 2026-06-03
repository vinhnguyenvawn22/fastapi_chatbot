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
    """Trả về đường dẫn tuyệt đối tới thư mục lưu tài liệu PDF."""
    return Path(DOCUMENTS_DIR).resolve()


def _safe_pdf_filename(filename: str) -> str:
    """Chuẩn hóa tên file upload và chặn tên không hợp lệ hoặc không phải PDF."""
    name = Path(filename).name.strip()
    name = UNSAFE_FILENAME_PATTERN.sub("_", name)
    name = re.sub(r"\s+", " ", name)

    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file PDF")

    return name


def _resolve_document_path(file_name: str) -> Path:
    """Tạo đường dẫn PDF an toàn bên trong DOCUMENTS_DIR, tránh path traversal."""
    documents_path = _documents_path()
    safe_name = _safe_pdf_filename(file_name)
    file_path = (documents_path / safe_name).resolve()

    if documents_path not in file_path.parents and file_path != documents_path:
        raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

    return file_path


def _file_sha256(file_path: Path) -> str:
    """Tính mã SHA-256 của file để định danh nội dung tài liệu."""
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def _unique_file_path(file_path: Path) -> Path:
    """Tạo đường dẫn không trùng bằng cách thêm hậu tố _1, _2, ... nếu file đã tồn tại."""
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
    """Liệt kê các file PDF hiện có kèm kích thước và thời điểm cập nhật."""
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
    """Đọc PDF và trích xuất text theo từng trang để phục vụ xem nội dung và chunking."""
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
    """Nhận PDF upload, lưu vào thư mục tài liệu, kiểm tra hợp lệ và index vào vector store."""
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

    vector_index_status = "skipped"
    vector_indexed_chunks = 0

    # Index các chunk mới sau upload để lần chat sau ưu tiên vector search.
    try:
        from app.data.vector_store import index_chunks

        document_chunks = build_document_chunks(file_path.name)
        vector_indexed_chunks = index_chunks(document_chunks)
        vector_index_status = "indexed"
    except Exception as exc:
        vector_index_status = f"failed: {exc}"

    return {
        "message": "Upload tài liệu thành công",
        "file_name": file_path.name,
        "file_path": str(file_path),
        "file_size_kb": round(file_path.stat().st_size / 1024, 2),
        "content_hash": _file_sha256(file_path),
        "vector_index_status": vector_index_status,
        "vector_indexed_chunks": vector_indexed_chunks,
    }


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Chia text đệ quy theo danh sách separator để giữ đoạn/câu tự nhiên nhất có thể."""
    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [
            text[start:start + chunk_size].strip()
            for start in range(0, len(text), chunk_size)
            if text[start:start + chunk_size].strip()
        ]

    separator = separators[0]
    raw_parts = text.split(separator)

    if len(raw_parts) == 1:
        return _split_recursive(text, chunk_size, separators[1:])

    parts = [
        f"{part}{separator}" if index < len(raw_parts) - 1 else part
        for index, part in enumerate(raw_parts)
    ]
    chunks = []
    current = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        candidate = part if not current else f"{current} {part}"

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.extend(_split_recursive(current, chunk_size, separators[1:]))

        current = part

    if current:
        chunks.extend(_split_recursive(current, chunk_size, separators[1:]))

    return chunks


def _with_overlap(chunks: list[str], overlap: int, chunk_size: int) -> list[str]:
    """Thêm phần overlap giữa các chunk liền kề để giảm mất ngữ cảnh tại điểm cắt."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]

    for previous, current in zip(chunks, chunks[1:]):
        prefix = previous[-overlap:].strip()
        if len(previous) > overlap and " " in prefix:
            prefix = prefix.split(" ", 1)[1].strip()

        combined = f"{prefix} {current}".strip()

        if len(combined) > chunk_size + overlap:
            combined = combined[-(chunk_size + overlap):].strip()

        overlapped.append(combined)

    return overlapped


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Chia văn bản dài thành các chunk có overlap, ưu tiên cắt theo đoạn/câu trước."""
    if chunk_size <= 0:
        raise ValueError("chunk_size phải lớn hơn 0")

    if overlap < 0:
        raise ValueError("overlap không được âm")

    if overlap >= chunk_size:
        overlap = max(chunk_size // 5, 0)

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    chunks = _split_recursive(text, chunk_size, separators)

    return _with_overlap(chunks, overlap, chunk_size)


def split_text_by_metadata(text: str):
    """Tách text theo tiêu đề Điều/Mục/Chương; nếu không có tiêu đề thì fallback chunk thường."""
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


def build_document_chunks(file_name: str):
    """Chuyển một PDF thành danh sách chunk có metadata để index và truy xuất RAG."""
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


