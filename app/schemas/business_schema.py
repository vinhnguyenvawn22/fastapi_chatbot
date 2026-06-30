from typing import Any

from pydantic import BaseModel, Field

from app.schemas.chat_schema import ChatSource


class BusinessSearchRequest(BaseModel):
    """Schema dau vao cho API tra cuu nghiep vu."""

    query: str = Field(..., description="Noi dung can tra cuu trong tai lieu nghiep vu.")
    top_k: int = Field(
        default=2,
        ge=1,
        le=5,
        description="So nguon tieu bieu muon tra ve sau khi xep hang.",
    )


class BusinessSearchResponse(BaseModel):
    """Schema dau ra cua API tra cuu nghiep vu."""

    query: str
    intent: str | None = None
    candidate_count: int = 0
    selected_count: int = 0
    has_confident_evidence: bool = False
    evidence_reason: str | None = None
    answer: str | None = None
    sources: list[ChatSource] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class BusinessAskRequest(BaseModel):
    """Schema dau vao cho API hoi dap nghiep vu theo mapping."""

    query: str = Field(..., description="Cau hoi nguoi dung can map vao bo FAQ nghiep vu.")
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="So ung vien mapping gan nhat tra ve de debug.",
    )


class BusinessMappingCandidate(BaseModel):
    """Mot ung vien mapping duoc xep hang theo do phu hop."""

    rank: int
    confidence: float
    matched_question: str | None = None
    answer: str | None = None
    file_id: str | None = None
    source_file: str | None = None
    source_location: str | None = None
    keywords: list[str] = Field(default_factory=list)
    score_details: dict[str, Any] = Field(default_factory=dict)


class BusinessAskResponse(BaseModel):
    """Schema dau ra cua API hoi dap nghiep vu theo mapping."""

    query: str
    matched: bool = False
    answer: str | None = None
    confidence: float = 0.0
    matched_question: str | None = None
    file_id: str | None = None
    source_file: str | None = None
    source_location: str | None = None
    keywords: list[str] = Field(default_factory=list)
    fallback_suggestion: str | None = None
    candidates: list[BusinessMappingCandidate] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
