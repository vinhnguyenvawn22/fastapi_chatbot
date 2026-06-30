from fastapi import APIRouter

from app.controller.business_controller import ask_business_mapping, search_business_knowledge
from app.schemas.business_schema import (
    BusinessAskRequest,
    BusinessAskResponse,
    BusinessSearchRequest,
    BusinessSearchResponse,
)


router = APIRouter()


@router.post("/ask", response_model=BusinessAskResponse)
async def business_ask(request: BusinessAskRequest):
    """API hoi dap nghiep vu theo FAQ mapping, khong goi LLM khi da match."""
    result = await ask_business_mapping(request)
    return BusinessAskResponse(**result)


@router.post("/search", response_model=BusinessSearchResponse)
async def business_search(request: BusinessSearchRequest):
    """API tra cuu nguon nghiep vu da xep hang, khong goi LLM."""
    result = await search_business_knowledge(request)
    return BusinessSearchResponse(**result)
