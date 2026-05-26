from pydantic import BaseModel


class DocumentResponse(BaseModel):
    message: str