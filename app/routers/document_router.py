from fastapi import APIRouter, File, HTTPException, UploadFile


router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"],
)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint upload tài liệu PDF.

    Router chỉ kiểm tra request cơ bản.
    Phần lưu file, đọc PDF, chunking, ingest dữ liệu sẽ do phần controller/data-layer xử lý sau.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Chỉ hỗ trợ upload file PDF")

        return {
            "message": "Router upload đã nhận file",
            "file_name": file.filename,
            "content_type": file.content_type,
            "note": "Phần xử lý tài liệu sẽ được nối với controller sau.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))