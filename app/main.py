from fastapi import FastAPI

from app.routers.chat_router import router as chat_router
from app.routers.document_router import router as document_router
from app.routers.health_router import router as health_router


app = FastAPI(
    title="FastAPI Chatbot",
    description="Chatbot RAG using FastAPI and Gemini",
    version="1.0.0",
)


app.include_router(health_router)
app.include_router(chat_router)
app.include_router(document_router)


@app.get("/")
def root():
    return {
        "message": "FastAPI Chatbot API",
        "docs": "/docs",
        "health": "/api/health",
    }