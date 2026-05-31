from fastapi import APIRouter
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.controller.chatbot_controller import handle_chat

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = await handle_chat(request)
    return ChatResponse(**result)