from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """Schema du lieu dau vao cho API chat."""
    question: str


class ChatSource(BaseModel):
    """Mot nguon tai lieu duoc dung de tao cau tra loi."""
    title: Optional[str] = None
    doc_name: Optional[str] = None
    so_van_ban: Optional[str] = None
    ngay_ban_hanh: Optional[str] = None
    ngay_hieu_luc: Optional[str] = None
    ten_van_ban: Optional[str] = None
    don_vi_ban_hanh: Optional[str] = None
    loai_van_ban: Optional[str] = None
    chuong: Optional[str] = None
    muc: Optional[int] = None
    dieu: Optional[int] = None
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
    intent: Optional[str] = None
    sources: list[ChatSource] = Field(default_factory=list)
