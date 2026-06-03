from fastapi import APIRouter, UploadFile, File
from app.controller.document_controller import (
    upload_document,
    list_documents,
    extract_pdf_text,
)

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """API upload một file PDF và index tài liệu vào hệ thống RAG."""
    return await upload_document(file)


@router.get("/")
async def get_documents():
    """API trả về danh sách PDF hiện có trong thư mục tài liệu."""
    return {
        "files": list_documents()
    }


@router.get("/{file_name}/text")
async def get_document_text(file_name: str):
    """API đọc text đã trích xuất từ một file PDF cụ thể."""
    text = extract_pdf_text(file_name)

    return {
        "file_name": file_name,
        "text": text,
    }
