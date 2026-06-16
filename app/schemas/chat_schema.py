from pydantic import BaseModel, Field
from typing import Any, Optional


class ChatRequest(BaseModel):
    """Schema du lieu dau vao cho API chat."""
    question: str


class ChatSource(BaseModel):
    """Mot nguon tai lieu duoc dung de tao cau tra loi."""
    title: Optional[str] = None
    doc_name: Optional[str] = None
    url: Optional[str] = None
    attachment_url: Optional[str] = None
    source_type: Optional[str] = None
    relative_path: Optional[str] = None
    phong_ban: Optional[str] = None
    source_root: Optional[str] = None
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
    file_id: Optional[str] = None
    faq_location: Optional[str] = None
    audience: Optional[str] = None
    mapping_relative_path: Optional[str] = None
    score: Optional[float] = None
    vector_score: Optional[float] = None
    keyword_score: Optional[float] = None
    distance: Optional[float] = None
    confidence: Optional[float] = None
    confidence_percent: Optional[int] = None
    confidence_label: Optional[str] = None
    preview: Optional[str] = None


class ChatResponse(BaseModel):
    """Schema du lieu dau ra cua API chat."""
    question: str
    answer: str
    source: Optional[str] = None
    intent: Optional[str] = None
    trace_id: Optional[str] = None
    sources: list[ChatSource] = Field(default_factory=list)


class TraceStep(BaseModel):
    name: str
    status: str
    started_at: str
    finished_at: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class TraceResponse(BaseModel):
    trace_id: str
    question: str
    created_at: str
    updated_at: str
    steps: list[TraceStep] = Field(default_factory=list)
    response: dict[str, Any] | None = None
