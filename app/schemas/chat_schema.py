from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """Schema dữ liệu đầu vào cho API chat."""
    question: str


class ChatResponse(BaseModel):
    """Schema dữ liệu đầu ra của API chat."""
    question: str
    answer: str
    source: Optional[str] = None
