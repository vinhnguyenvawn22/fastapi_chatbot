from fastapi import APIRouter
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.controller.chatbot_controller import handle_chat

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """API nhận câu hỏi người dùng và trả về câu trả lời chatbot theo schema chuẩn."""
    result = await handle_chat(request)
    return ChatResponse(**result)
