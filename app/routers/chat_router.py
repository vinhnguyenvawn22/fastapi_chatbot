from fastapi import APIRouter
from app.schemas.chat_schema import ChatRequest, ChatResponse, TraceResponse
from app.controller.chatbot_controller import get_chat_trace, handle_chat

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """API nhận câu hỏi người dùng và trả về câu trả lời chatbot theo schema chuẩn."""
    result = await handle_chat(request)
    return ChatResponse(**result)


@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def trace_detail(trace_id: str):
    """Tra cuu debug trace cua mot cau hoi theo trace_id."""
    return TraceResponse(**get_chat_trace(trace_id))
