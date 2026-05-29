import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "uploads/ChatAI")

if not GEMINI_API_KEY:
    raise ValueError("Thiếu GEMINI_API_KEY trong file .env")