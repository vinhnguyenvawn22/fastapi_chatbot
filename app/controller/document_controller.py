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
from app.data.query_analyzer import normalize_date, normalize_text


PDF_MIME_TYPES = {"application/pdf", "application/octet-stream"}
UNSAFE_FILENAME_PATTERN = re.compile(r'[<>:"\\|?*\x00-\x1f]+')
SECTION_PATTERN = re.compile(
    r"(?im)^\s*((?:Điều|Dieu)\s+\d+[\.\:\s]+.*|(?:Mục|Muc)\s+\d+[\.\:\s]+.*|(?:Chương|Chuong)\s+(?:[IVXLCDM]+|\d+)[\.\:\s]+.*)$"
)
DOCUMENT_TYPE_PATTERN = re.compile(
    r"\b(Quyết định|Quyet dinh|Quy định|Quy dinh|Quy chế|Quy che|Thông báo|Thong bao|Hướng dẫn|Huong dan|Kế hoạch|Ke hoach)\b",
    flags=re.IGNORECASE,
)


def _documents_path() -> Path:
    return Path(DOCUMENTS_DIR).resolve()


def _relative_document_path(file_path: Path) -> str:
    return file_path.resolve().relative_to(_documents_path()).as_posix()


def _extract_source_metadata(file_path: Path) -> dict:
    documents_path = _documents_path()
    relative_path = _relative_document_path(file_path)
    parent_parts = Path(relative_path).parts[:-1]
    phong_ban = None

    for part in reversed(parent_parts):
        normalized = normalize_text(part)
        if normalized.startswith("phong ") or normalized.startswith("trung tam "):
            phong_ban = part
            break

    if not phong_ban and parent_parts:
        phong_ban = parent_parts[0]

    return {
        "phong_ban": phong_ban,
        "relative_path": relative_path,
        "source_root": documents_path.name,
    }


def _safe_pdf_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    name = UNSAFE_FILENAME_PATTERN.sub("_", name)
    name = re.sub(r"\s+", " ", name)

    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Ten file khong hop le")

    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chi ho tro file PDF")

    return name


def _resolve_document_path(file_name: str | Path) -> Path:
    documents_path = _documents_path()
    raw_name = str(file_name or "").replace("\\", "/").strip()

    if not raw_name:
        raise HTTPException(status_code=400, detail="Ten file khong hop le")

    if Path(raw_name).is_absolute():
        file_path = Path(raw_name).resolve()
    else:
        parts = raw_name.split("/")
        if any(part in {"", ".", ".."} for part in parts):
            raise HTTPException(status_code=400, detail="Ten file khong hop le")
        if UNSAFE_FILENAME_PATTERN.search("".join(parts)):
            raise HTTPException(status_code=400, detail="Ten file khong hop le")
        if not raw_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Chi ho tro file PDF")
        file_path = (documents_path / raw_name).resolve()

    if documents_path not in file_path.parents:
        raise HTTPException(status_code=400, detail="Ten file khong hop le")

    if file_path.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Chi ho tro file PDF")

    return file_path


def _file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()


def _first_match(pattern: str, text: str, flags=re.IGNORECASE):
    match = re.search(pattern, text, flags=flags)
    return match.group(1).strip() if match else None


def _extract_document_metadata(text: str, file_name: str) -> dict:
    lines = [
        " ".join(line.split())
        for line in text.splitlines()
        if line and " ".join(line.split())
    ]
    header_text = "\n".join(lines[:80])
    normalized_header = normalize_text(header_text)

    so_van_ban = _first_match(
        r"(?:Số|So)\s*[:\-]?\s*([0-9]{1,6}(?:/[A-Za-z0-9.\-]+)?)",
        header_text,
    )
    if not so_van_ban:
        match = re.search(
            r"(?:so|van\s*ban|quyet\s*dinh|quy\s*dinh|qd)\s*[:\-]?\s*([0-9]{1,6}(?:\s*/\s*[a-z0-9.\-]+)?)",
            normalized_header,
        )
        if not match:
            match = re.search(r"\b([0-9]{2,6})\s*/\s*(?:qd|vb|tb|qc)", normalized_header)
        if not match:
            match = re.search(r"\b([0-9]{2,6})\b", normalize_text(file_name))
        if match:
            so_van_ban = re.sub(r"\s+", "", match.group(1)).upper()

    so_van_ban_ngan = None
    if so_van_ban:
        short_match = re.search(r"\d{1,6}", so_van_ban)
        so_van_ban_ngan = short_match.group(0) if short_match else so_van_ban

    ngay_ban_hanh = _first_match(
        r"ngày\s+(\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
        header_text,
    )
    ngay_hieu_luc = _first_match(
        r"hiệu lực(?:\s+thi hành)?(?:\s+kể)?\s+từ\s+ngày\s+(\d{1,2}\s*(?:/|-|\.|tháng)\s*\d{1,2}\s*(?:/|-|\.|năm)?\s*\d{4})",
        header_text,
    )

    type_match = DOCUMENT_TYPE_PATTERN.search(header_text)
    loai_van_ban = type_match.group(1) if type_match else None

    don_vi_ban_hanh = None
    for line in lines[:20]:
        normalized_line = normalize_text(line)
        if any(term in normalized_line for term in ("bo ", "truong ", "phong ", "khoa ", "uy ban")):
            don_vi_ban_hanh = line
            break

    ten_van_ban = None
    for line in lines[:80]:
        normalized_line = normalize_text(line)
        if len(line) >= 12 and any(
            term in normalized_line
            for term in ("quy dinh", "quy che", "quyet dinh", "thong bao", "huong dan")
        ):
            ten_van_ban = line
            break

    return {
        "so_van_ban": so_van_ban,
        "so_van_ban_ngan": so_van_ban_ngan,
        "ngay_ban_hanh": normalize_date(ngay_ban_hanh) if ngay_ban_hanh else None,
        "ngay_hieu_luc": normalize_date(ngay_hieu_luc) if ngay_hieu_luc else None,
        "ten_van_ban": ten_van_ban or Path(file_name).stem,
        "don_vi_ban_hanh": don_vi_ban_hanh,
        "loai_van_ban": loai_van_ban,
    }


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

    raise HTTPException(status_code=409, detail="Khong the tao ten file khong trung")


def list_documents():
    documents_path = _documents_path()

    if not documents_path.exists():
        return []

    files = []

    for file_path in sorted(documents_path.rglob("*.pdf")):
        stat = file_path.stat()
        source_metadata = _extract_source_metadata(file_path)
        files.append({
            "file_name": file_path.name,
            "relative_path": source_metadata["relative_path"],
            "file_path": str(file_path),
            "file_size_kb": round(stat.st_size / 1024, 2),
            "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "phong_ban": source_metadata["phong_ban"],
            "source_root": source_metadata["source_root"],
        })

    return files


def extract_pdf_text(file_name: str):
    file_path = _resolve_document_path(file_name)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Khong tim thay file: {file_name}")

    try:
        reader = PdfReader(str(file_path))
    except PdfReadError as exc:
        raise HTTPException(status_code=400, detail="File PDF khong doc duoc") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Khong the mo file PDF") from exc

    text_parts = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_parts.append(f"\n--- Trang {page_index + 1} ---\n{text}")

    return "\n".join(text_parts).strip()


async def upload_document(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Ten file khong hop le")

    if file.content_type and file.content_type not in PDF_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Chi ho tro file PDF")

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
                        detail=f"File vuot qua gioi han {MAX_UPLOAD_SIZE_MB}MB",
                    )

                buffer.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Khong the luu file upload") from exc
    finally:
        await file.close()

    try:
        PdfReader(str(file_path))
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File upload khong phai PDF hop le") from exc

    try:
        from app.data.elasticsearch_client import clear_document_index_cache

        clear_document_index_cache()
    except Exception:
        pass

    vector_index_status = "skipped"
    vector_indexed_chunks = 0

    try:
        from app.data.vector_store import index_chunks

        document_chunks = build_document_chunks(_relative_document_path(file_path))
        vector_indexed_chunks = index_chunks(document_chunks)
        vector_index_status = "indexed"
    except Exception as exc:
        vector_index_status = f"failed: {exc}"

    return {
        "message": "Upload tai lieu thanh cong",
        "file_name": file_path.name,
        "relative_path": _relative_document_path(file_path),
        "file_path": str(file_path),
        "file_size_kb": round(file_path.stat().st_size / 1024, 2),
        "content_hash": _file_sha256(file_path),
        "vector_index_status": vector_index_status,
        "vector_indexed_chunks": vector_indexed_chunks,
    }


def _split_recursive(text: str, chunk_size: int, separators: list[str]) -> list[str]:
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
    if chunk_size <= 0:
        raise ValueError("chunk_size phai lon hon 0")

    if overlap < 0:
        raise ValueError("overlap khong duoc am")

    if overlap >= chunk_size:
        overlap = max(chunk_size // 5, 0)

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    chunks = _split_recursive(text, chunk_size, separators)

    return _with_overlap(chunks, overlap, chunk_size)


def split_text_by_metadata(text: str):
    matches = list(SECTION_PATTERN.finditer(text))

    if not matches:
        fallback_chunks = chunk_text(text)
        return [
            {
                "title": f"Đoạn {index}",
                "dieu": None,
                "muc": None,
                "chuong": None,
                "content": chunk,
            }
            for index, chunk in enumerate(fallback_chunks, start=1)
        ]

    chunks = []
    current_chuong = None
    current_muc = None

    if matches[0].start() > 0:
        intro = text[:matches[0].start()].strip()
        if intro:
            for index, chunk in enumerate(chunk_text(intro), start=1):
                title = "Phần mở đầu" if index == 1 else f"Phần mở đầu ({index})"
                chunks.append({
                    "title": title,
                    "dieu": None,
                    "muc": None,
                    "chuong": None,
                    "content": chunk,
                })

    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        chuong_match = re.search(r"(?:Chương|Chuong)\s+([IVXLCDM]+|\d+)", title, flags=re.IGNORECASE)
        if chuong_match:
            current_chuong = chuong_match.group(1).upper()
            current_muc = None

        muc_match = re.search(r"(?:Mục|Muc)\s+(\d+)", title, flags=re.IGNORECASE)
        if muc_match:
            current_muc = int(muc_match.group(1))

        dieu_match = re.search(r"(?:Điều|Dieu)\s+(\d+)", title, flags=re.IGNORECASE)
        dieu = int(dieu_match.group(1)) if dieu_match else None

        split_chunks = chunk_text(content)
        for split_index, split_chunk in enumerate(split_chunks, start=1):
            chunk_title = title if len(split_chunks) == 1 else f"{title} ({split_index})"
            chunks.append({
                "title": chunk_title,
                "dieu": dieu,
                "muc": current_muc,
                "chuong": current_chuong,
                "content": split_chunk,
            })

    return chunks


def build_document_chunks(file_name: str):
    file_path = _resolve_document_path(file_name)
    text = extract_pdf_text(file_name)
    chunks = split_text_by_metadata(text)
    document_metadata = _extract_document_metadata(text, file_path.name)
    source_metadata = _extract_source_metadata(file_path)
    stat = file_path.stat()
    content_hash = _file_sha256(file_path)

    documents = []

    for index, chunk in enumerate(chunks, start=1):
        documents.append({
            "doc_name": file_path.name,
            "relative_path": source_metadata["relative_path"],
            "phong_ban": source_metadata["phong_ban"],
            "source_root": source_metadata["source_root"],
            "title": chunk["title"],
            "dieu": chunk["dieu"],
            "muc": chunk.get("muc"),
            "chuong": chunk.get("chuong"),
            "chunk_index": index,
            "content": chunk["content"],
            "file_path": str(file_path),
            "source_type": "official_document",
            "is_active": True,
            "content_hash": content_hash,
            "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            **document_metadata,
        })

    return documents
