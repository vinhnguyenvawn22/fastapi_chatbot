import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "uploads/ChatAI")

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
MIN_SEARCH_SCORE = float(os.getenv("MIN_SEARCH_SCORE", "2"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))

# Cấu hình embedding/vector store để có thể đổi model hoặc nơi lưu DB qua .env.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "storage/chroma_db")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "document_chunks")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", str(SEARCH_TOP_K)))
VECTOR_MAX_DISTANCE = float(os.getenv("VECTOR_MAX_DISTANCE", "0.75"))

if not GEMINI_API_KEY:
    raise ValueError("Thiếu GEMINI_API_KEY trong file .env")
