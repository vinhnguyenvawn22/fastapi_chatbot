from fastapi import APIRouter

from app.controller.business_controller import search_business_knowledge
from app.schemas.business_schema import BusinessSearchRequest, BusinessSearchResponse


router = APIRouter()


@router.post("/search", response_model=BusinessSearchResponse)
async def business_search(request: BusinessSearchRequest):
    """API tra cuu nguon nghiep vu da xep hang, khong goi LLM."""
    result = await search_business_knowledge(request)
    return BusinessSearchResponse(**result)
