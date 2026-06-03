from fastapi import FastAPI
from app.routers.page_router import router as page_router

from app.routers.chat_router import router as chat_router
from app.routers.document_router import router as document_router
from app.routers.health_router import router as health_router

app = FastAPI(
    title="FastAPI Chatbot",
    description="Chatbot RAG using FastAPI and Gemini",
    version="1.0.0"
)

app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(document_router, prefix="/api/documents", tags=["Documents"])
app.include_router(page_router)