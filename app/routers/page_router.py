from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
LANDING_FILE = BASE_DIR / "templates" / "landing.html"
CHAT_FILE = BASE_DIR / "templates" / "chat_ui.html"


@router.get("/")
async def landing_page():
    return FileResponse(
        path=str(LANDING_FILE),
        media_type="text/html"
    )


@router.get("/chat-ui")
async def chat_page():
    return FileResponse(
        path=str(CHAT_FILE),
        media_type="text/html"
    )
