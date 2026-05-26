from fastapi import APIRouter

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.controller.chatbot_controller import handle_chat

router = APIRouter(prefix="/chat", tags=["Chatbot"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    return await handle_chat(request)