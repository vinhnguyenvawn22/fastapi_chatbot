from fastapi import APIRouter


router = APIRouter(
    prefix="/api",
    tags=["Health"],
)


@router.get("/health")
def health_check():
    """
    Kiểm tra trạng thái server.
    """
    return {
        "status": "ok",
        "message": "UNETI P.CNTT Chatbot API is running",
    }