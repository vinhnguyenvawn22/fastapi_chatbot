from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
HTML_FILE = BASE_DIR / "templates" / "chat_ui.html"


@router.get("/")
async def chat_page():
    return FileResponse(
        path=str(HTML_FILE),
        media_type="text/html"
    )