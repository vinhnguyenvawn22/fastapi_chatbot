from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/")
async def health_check():
    """API kiểm tra nhanh server còn hoạt động."""
    return {"status": "ok"}
