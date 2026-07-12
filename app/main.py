from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.routers.page_router import router as page_router
from app.routers.business_router import router as business_router
from app.routers.chat_router import router as chat_router
from app.routers.document_router import router as document_router
from app.routers.health_router import router as health_router
from app.routers.website_router import router as website_router
from app.data.preload import preload_rag_components
from app.data.gemini_client import reset_gemini_call_count
from app.core.config import (
    CHAT_COOKIE_SAMESITE, CHAT_COOKIE_SECURE, CHAT_DATABASE_FILE,
    CHAT_SESSION_COOKIE_NAME, CHAT_SESSION_MAX_AGE_SECONDS,
)
from app.data.conversation_repository import ConversationRepository
from app.services.conversation_service import new_session_token, session_hash


@asynccontextmanager
async def lifespan(app: FastAPI):
    repository = ConversationRepository(CHAT_DATABASE_FILE)
    repository.initialize()
    app.state.conversation_repository = repository
    app.state.rag_preload = await preload_rag_components()
    yield

app = FastAPI(
    title="FastAPI Chatbot",
    description="Chatbot RAG using FastAPI and Gemini",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def reset_gemini_counter_middleware(request, call_next):
    reset_gemini_call_count()
    return await call_next(request)


@app.middleware("http")
async def anonymous_chat_session_middleware(request: Request, call_next):
    token = request.cookies.get(CHAT_SESSION_COOKIE_NAME)
    is_new = not bool(token)
    if is_new:
        token = new_session_token()
    repository = getattr(request.app.state, "conversation_repository", None)
    if repository is None:
        repository = ConversationRepository(CHAT_DATABASE_FILE)
        repository.initialize()
        request.app.state.conversation_repository = repository
    request.state.chat_owner_id = repository.get_or_create_owner(session_hash(token))
    response = await call_next(request)
    if is_new:
        response.set_cookie(
            CHAT_SESSION_COOKIE_NAME,
            token,
            max_age=CHAT_SESSION_MAX_AGE_SECONDS,
            httponly=True,
            secure=CHAT_COOKIE_SECURE,
            samesite=CHAT_COOKIE_SAMESITE,
            path="/",
        )
    return response

app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(business_router, prefix="/api/nghiep-vu", tags=["Nghiep vu"])
app.include_router(website_router, prefix="/api/website", tags=["Website"])
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])
app.include_router(page_router)
