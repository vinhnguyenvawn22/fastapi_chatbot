from fastapi import APIRouter

from app.controller.chatbot_controller import (
    get_chat_trace,
    handle_business_chat,
    handle_chat,
    handle_internal_chat,
    handle_website_chat,
)
from app.schemas.chat_schema import ChatRequest, ChatResponse, TraceResponse


router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Endpoint tong hop: tu phan loai va chon nguon phu hop."""
    result = await handle_chat(request)
    return ChatResponse(**result)


@router.post("/business", response_model=ChatResponse)
async def chat_business(request: ChatRequest):
    """Endpoint chi tra cuu tai lieu nghiep vu."""
    result = await handle_business_chat(request)
    return ChatResponse(**result)


@router.post("/internal", response_model=ChatResponse)
async def chat_internal(request: ChatRequest):
    """Endpoint chi tra cuu tai lieu noi bo."""
    result = await handle_internal_chat(request)
    return ChatResponse(**result)


@router.post("/website", response_model=ChatResponse)
async def chat_website(request: ChatRequest):
    """Endpoint chi tra cuu website UNETI."""
    result = await handle_website_chat(request)
    return ChatResponse(**result)


@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def trace_detail(trace_id: str):
    """Tra cuu debug trace cua mot cau hoi theo trace_id."""
    return TraceResponse(**get_chat_trace(trace_id))
