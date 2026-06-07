import os
from dotenv import load_dotenv


load_dotenv(encoding="utf-8-sig")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "uploads/Tổng hợp văn bản AI")

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
MIN_SEARCH_SCORE = float(os.getenv("MIN_SEARCH_SCORE", "4"))
SHORT_QUERY_MIN_SEARCH_SCORE = float(os.getenv("SHORT_QUERY_MIN_SEARCH_SCORE", "10"))
MIN_VECTOR_CONFIDENCE = float(os.getenv("MIN_VECTOR_CONFIDENCE", "0.45"))
SHORT_QUERY_MIN_VECTOR_CONFIDENCE = float(os.getenv("SHORT_QUERY_MIN_VECTOR_CONFIDENCE", "0.55"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "20000"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "storage/chroma_db")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "document_chunks")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", str(SEARCH_TOP_K)))
VECTOR_MAX_DISTANCE = float(os.getenv("VECTOR_MAX_DISTANCE", "0.75"))

UNETI_WEBSITE_DOMAIN = os.getenv("UNETI_WEBSITE_DOMAIN", "uneti.edu.vn")
DISCOVERY_PROJECT_NUMBER = os.getenv("DISCOVERY_PROJECT_NUMBER")
DISCOVERY_LOCATION = os.getenv("DISCOVERY_LOCATION", "global")
DISCOVERY_COLLECTION_ID = os.getenv("DISCOVERY_COLLECTION_ID", "default_collection")
DISCOVERY_ENGINE_ID = os.getenv("DISCOVERY_ENGINE_ID")
DISCOVERY_SERVING_CONFIG_ID = os.getenv("DISCOVERY_SERVING_CONFIG_ID", "default_search")
WEBSITE_SEARCH_TOP_K = int(os.getenv("WEBSITE_SEARCH_TOP_K", "10"))
WEBSITE_RERANK_TOP_K = int(os.getenv("WEBSITE_RERANK_TOP_K", "2"))
WEBSITE_MIN_SOURCE_SCORE = float(os.getenv("WEBSITE_MIN_SOURCE_SCORE", "50"))
WEBSITE_FETCH_TIMEOUT = float(os.getenv("WEBSITE_FETCH_TIMEOUT", "15"))
WEBSITE_FETCH_MAX_BYTES = int(os.getenv("WEBSITE_FETCH_MAX_BYTES", str(20 * 1024 * 1024)))
WEBSITE_EXTRACT_MAX_CHARS = int(os.getenv("WEBSITE_EXTRACT_MAX_CHARS", "12000"))

if not GEMINI_API_KEY:
    raise ValueError("Thieu GEMINI_API_KEY trong file .env")
