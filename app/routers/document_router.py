from fastapi import APIRouter, UploadFile, File
from app.controller.document_controller import (
    upload_document,
    list_documents,
    extract_pdf_text,
)

router = APIRouter()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    return await upload_document(file)


@router.get("/")
async def get_documents():
    return {
        "files": list_documents()
    }


@router.get("/{file_name}/text")
async def get_document_text(file_name: str):
    text = extract_pdf_text(file_name)

    return {
        "file_name": file_name,
        "text": text,
    }