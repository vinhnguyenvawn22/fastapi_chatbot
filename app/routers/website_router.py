from fastapi import APIRouter

from app.controller.website_controller import search_website_knowledge
from app.schemas.website_schema import WebsiteSearchRequest, WebsiteSearchResponse


router = APIRouter()


@router.post("/search", response_model=WebsiteSearchResponse)
async def website_search(request: WebsiteSearchRequest):
    """API tra cuu truc tiep website UNETI."""
    result = await search_website_knowledge(request)
    return WebsiteSearchResponse(**result)
