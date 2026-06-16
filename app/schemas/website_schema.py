from typing import Any

from pydantic import BaseModel, Field

from app.schemas.chat_schema import ChatSource


class WebsiteSearchRequest(BaseModel):
    """Schema dau vao cho API tra cuu website UNETI."""

    query: str = Field(..., description="Noi dung can tim tren website UNETI.")
    top_k: int = Field(
        default=2,
        ge=1,
        le=5,
        description="So nguon website tieu bieu muon tra ve.",
    )


class WebsiteSearchResponse(BaseModel):
    """Schema dau ra cua API tra cuu website UNETI."""

    query: str
    answer: str
    selected_count: int = 0
    sources: list[ChatSource] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
