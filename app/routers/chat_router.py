from fastapi import APIRouter, HTTPException, Request, Response, status

from app.controller.chatbot_controller import (
    get_chat_trace,
    handle_business_chat,
    handle_chat,
    handle_internal_chat,
    handle_local_documents_chat,
    handle_website_chat,
)
from app.schemas.chat_schema import (
    ChatRequest, ChatResponse, MessageResponse, ThreadCreateRequest,
    ThreadDetailResponse, ThreadResponse, TraceResponse,
)
from app.services.conversation_service import ConversationService


router = APIRouter()


async def _run_conversation(http_request: Request, request: ChatRequest, handler):
    service = ConversationService(http_request.app.state.conversation_repository)
    return await service.chat(http_request.state.chat_owner_id, request, handler)


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """Endpoint tong hop: tu phan loai va chon nguon phu hop."""
    result = await _run_conversation(http_request, request, handle_chat)
    return ChatResponse(**result)


@router.post("/business", response_model=ChatResponse)
async def chat_business(request: ChatRequest, http_request: Request):
    """Endpoint chi tra cuu tai lieu nghiep vu."""
    result = await _run_conversation(http_request, request, handle_business_chat)
    return ChatResponse(**result)


@router.post("/internal", response_model=ChatResponse)
async def chat_internal(request: ChatRequest, http_request: Request):
    """Endpoint chi tra cuu tai lieu noi bo."""
    result = await _run_conversation(http_request, request, handle_internal_chat)
    return ChatResponse(**result)


@router.post("/local-documents", response_model=ChatResponse)
async def chat_local_documents(request: ChatRequest, http_request: Request):
    """Endpoint chi tra cuu corpus tai lieu local."""
    result = await _run_conversation(http_request, request, handle_local_documents_chat)
    return ChatResponse(**result)


@router.post("/website", response_model=ChatResponse)
async def chat_website(request: ChatRequest, http_request: Request):
    """Endpoint chi tra cuu website UNETI."""
    result = await _run_conversation(http_request, request, handle_website_chat)
    return ChatResponse(**result)


@router.get("/traces/{trace_id}", response_model=TraceResponse)
async def trace_detail(trace_id: str):
    """Tra cuu debug trace cua mot cau hoi theo trace_id."""
    return TraceResponse(**get_chat_trace(trace_id))


@router.post("/threads", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(payload: ThreadCreateRequest, request: Request):
    service = ConversationService(request.app.state.conversation_repository)
    return service.create_thread(request.state.chat_owner_id, payload.title)


@router.get("/threads", response_model=list[ThreadResponse])
async def list_threads(request: Request):
    return request.app.state.conversation_repository.list_threads(request.state.chat_owner_id)


@router.get("/threads/{thread_id}", response_model=ThreadDetailResponse)
async def thread_detail(thread_id: str, request: Request):
    service = ConversationService(request.app.state.conversation_repository)
    thread = service.require_thread(request.state.chat_owner_id, thread_id)
    return request.app.state.conversation_repository.get_thread_detail(
        request.state.chat_owner_id, thread["thread_id"]
    )


@router.get("/threads/{thread_id}/messages", response_model=list[MessageResponse])
async def list_messages(thread_id: str, request: Request):
    service = ConversationService(request.app.state.conversation_repository)
    thread = service.require_thread(request.state.chat_owner_id, thread_id)
    return request.app.state.conversation_repository.list_messages(
        request.state.chat_owner_id, thread["thread_id"]
    ) or []


@router.delete("/threads/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_thread(thread_id: str, request: Request):
    service = ConversationService(request.app.state.conversation_repository)
    thread = service.require_thread(request.state.chat_owner_id, thread_id)
    if not request.app.state.conversation_repository.delete_thread(
        request.state.chat_owner_id, thread["thread_id"]
    ):
        raise HTTPException(status_code=404, detail="Khong tim thay cuoc tro chuyen")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
