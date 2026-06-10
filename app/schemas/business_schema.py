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
    sources: list[ChatSource] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
