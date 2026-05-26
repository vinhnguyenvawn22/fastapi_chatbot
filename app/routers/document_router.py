from fastapi import APIRouter

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document():
    return {"message": "Document upload feature will be implemented later"}