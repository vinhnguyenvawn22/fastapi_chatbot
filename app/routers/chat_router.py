from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=3, ge=1, le=12)


router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"],
)


@router.post("")
def chat(request: ChatRequest):
    """
    Endpoint hỏi đáp chatbot.

    Router chỉ nhận request và trả response.
    Phần xử lý chatbot/RAG/controller sẽ do thành viên khác phụ trách.
    """
    try:
        return {
            "message": "Router chat đã nhận request",
            "question": request.question,
            "top_k": request.top_k,
            "note": "Phần xử lý chatbot sẽ được nối với controller sau.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ui")
def chat_ui(request: ChatRequest):
    """
    Endpoint trả dữ liệu phục vụ giao diện chatbot.

    Router chỉ nhận request. Logic tạo dữ liệu UI sẽ được nối với controller sau.
    """
    try:
        return {
            "message": "Router chat UI đã nhận request",
            "question": request.question,
            "top_k": request.top_k,
            "ui_mock": {
                "title": "UNETI P.CNTT Chatbot",
                "description": "API phục vụ giao diện chatbot P.CNTT",
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))