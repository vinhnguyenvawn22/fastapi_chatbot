from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Schema du lieu dau vao cho API chat."""
    question: str


class ChatSource(BaseModel):
    """Mot nguon tai lieu duoc dung de tao cau tra loi."""
    title: Optional[str] = None
    doc_name: Optional[str] = None
    chunk_index: Optional[int] = None
    score: Optional[float] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    distance: Optional[float] = None
    preview: Optional[str] = None


class ChatResponse(BaseModel):
    """Schema du lieu dau ra cua API chat."""
    question: str
    answer: str
    source: Optional[str] = None
    sources: list[ChatSource] = Field(default_factory=list)
