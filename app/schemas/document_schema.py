from pydantic import BaseModel


class DocumentResponse(BaseModel):
    """Schema phản hồi đơn giản cho các thao tác tài liệu."""
    message: str
