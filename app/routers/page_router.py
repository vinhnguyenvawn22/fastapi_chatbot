from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["Pages"])

BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = BASE_DIR / "templates" / "chat_ui.html"


@router.get("/chat-ui", response_class=HTMLResponse)
def chat_ui_page():
    """
    Đọc file HTML tĩnh và trả về giao diện chatbot P.CNTT.
    """
    html_content = TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html_content)
