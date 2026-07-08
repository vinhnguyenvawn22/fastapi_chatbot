import os
from dotenv import load_dotenv


load_dotenv(encoding="utf-8-sig")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "uploads/Tổng hợp văn bản AI")
DOCUMENT_INDEX_CACHE_ENABLED = os.getenv(
    "DOCUMENT_INDEX_CACHE_ENABLED",
    "true",
).lower() in {"1", "true", "yes", "on"}
DOCUMENT_INDEX_CACHE_FILE = os.getenv(
    "DOCUMENT_INDEX_CACHE_FILE",
    "storage/document_index/index.json",
)
BUSINESS_DOCUMENTS_DIR = os.getenv("BUSINESS_DOCUMENTS_DIR", "documents/nghiep_vu")
BUSINESS_SEARCH_TOP_K = int(os.getenv("BUSINESS_SEARCH_TOP_K", "5"))
BUSINESS_INDEX_CACHE_ENABLED = os.getenv(
    "BUSINESS_INDEX_CACHE_ENABLED",
    "true",
).lower() in {"1", "true", "yes", "on"}
BUSINESS_INDEX_CACHE_FILE = os.getenv(
    "BUSINESS_INDEX_CACHE_FILE",
    "storage/business_knowledge_index/index.json",
)

MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "20"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))
SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "3"))
RRF_CANDIDATE_TOP_K = int(
    os.getenv(
        "RRF_CANDIDATE_TOP_K",
        os.getenv("HYBRID_CANDIDATE_TOP_K", "30"),
    )
)
# Backward-compatible alias for deployments that still import the old name.
HYBRID_CANDIDATE_TOP_K = RRF_CANDIDATE_TOP_K
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "20"))
BM25_MIN_SCORE = float(os.getenv("BM25_MIN_SCORE", "0"))
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))
BM25_METADATA_BOOST = float(os.getenv("BM25_METADATA_BOOST", "2.0"))
ANN_TOP_K = int(os.getenv("ANN_TOP_K", "20"))
RRF_K = int(os.getenv("RRF_K", "60"))
MIN_SEARCH_SCORE = float(os.getenv("MIN_SEARCH_SCORE", "4"))
SHORT_QUERY_MIN_SEARCH_SCORE = float(os.getenv("SHORT_QUERY_MIN_SEARCH_SCORE", "10"))
MIN_VECTOR_CONFIDENCE = float(os.getenv("MIN_VECTOR_CONFIDENCE", "0.45"))
SHORT_QUERY_MIN_VECTOR_CONFIDENCE = float(os.getenv("SHORT_QUERY_MIN_VECTOR_CONFIDENCE", "0.55"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "3"))
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "900"))
RETRIEVAL_CACHE_MAX_ITEMS = int(os.getenv("RETRIEVAL_CACHE_MAX_ITEMS", "128"))
EMBEDDING_CACHE_MAX_ITEMS = int(os.getenv("EMBEDDING_CACHE_MAX_ITEMS", "512"))
VECTOR_FAST_PATH_CONFIDENCE = float(os.getenv("VECTOR_FAST_PATH_CONFIDENCE", "0.82"))
VECTOR_FAST_PATH_SCORE_GAP = float(os.getenv("VECTOR_FAST_PATH_SCORE_GAP", "0.12"))
RERANK_AMBIGUOUS_QUERY_KEYWORDS = int(os.getenv("RERANK_AMBIGUOUS_QUERY_KEYWORDS", "4"))
QUERY_EXPANSION_ENABLED = os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", GEMINI_MODEL)
QUERY_EXPANSION_MAX_VARIANTS = int(os.getenv("QUERY_EXPANSION_MAX_VARIANTS", "2"))
QUERY_EXPANSION_MAX_WORDS = int(os.getenv("QUERY_EXPANSION_MAX_WORDS", "6"))
QUERY_EXPANSION_CACHE_TTL_SECONDS = int(os.getenv("QUERY_EXPANSION_CACHE_TTL_SECONDS", "1800"))
QUERY_EXPANSION_CACHE_MAX_ITEMS = int(os.getenv("QUERY_EXPANSION_CACHE_MAX_ITEMS", "256"))
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
HYDE_MODEL = os.getenv("HYDE_MODEL", GEMINI_MODEL)
HYDE_MAX_WORDS = int(os.getenv("HYDE_MAX_WORDS", "100"))
HYDE_ANN_TOP_K = int(os.getenv("HYDE_ANN_TOP_K", "20"))
HYDE_MIN_TOPIC_CONFIDENCE = float(os.getenv("HYDE_MIN_TOPIC_CONFIDENCE", "0.65"))
HYDE_MIN_RERANK_SCORE = float(os.getenv("HYDE_MIN_RERANK_SCORE", "0.0"))
PROBE_TOP_K = int(os.getenv("PROBE_TOP_K", "6"))
PROBE_EVIDENCE_TOP_K = int(os.getenv("PROBE_EVIDENCE_TOP_K", "3"))
PROBE_BM25_MIN_SCORE = float(os.getenv("PROBE_BM25_MIN_SCORE", "1.5"))
PROBE_VECTOR_MIN_SCORE = float(os.getenv("PROBE_VECTOR_MIN_SCORE", "0.42"))
PROBE_RRF_MIN_SCORE = float(os.getenv("PROBE_RRF_MIN_SCORE", "0.015"))
PROBE_RRF_SCORE_GAP = float(os.getenv("PROBE_RRF_SCORE_GAP", "0.001"))
PROBE_MIN_TITLE_OVERLAP = int(os.getenv("PROBE_MIN_TITLE_OVERLAP", "1"))
PROBE_MIN_EVIDENCE_SIGNALS = int(os.getenv("PROBE_MIN_EVIDENCE_SIGNALS", "2"))
GROUNDED_HYDE_ANN_TOP_K = int(os.getenv("GROUNDED_HYDE_ANN_TOP_K", "20"))
GROUNDED_HYDE_MAX_EVIDENCE_CHARS = int(
    os.getenv("GROUNDED_HYDE_MAX_EVIDENCE_CHARS", "3600")
)
AMBIGUITY_CLARIFY_THRESHOLD = float(
    os.getenv("AMBIGUITY_CLARIFY_THRESHOLD", "0.40")
)
AMBIGUITY_CACHE_TTL_SECONDS = int(
    os.getenv("AMBIGUITY_CACHE_TTL_SECONDS", "900")
)
AMBIGUITY_CACHE_MAX_ITEMS = int(os.getenv("AMBIGUITY_CACHE_MAX_ITEMS", "128"))
AMBIGUITY_LLM_ENABLED = os.getenv("AMBIGUITY_LLM_ENABLED", "false").lower() in {
    "1", "true", "yes", "on",
}
HYDE_CACHE_TTL_SECONDS = int(os.getenv("HYDE_CACHE_TTL_SECONDS", "900"))
HYDE_CACHE_MAX_ITEMS = int(os.getenv("HYDE_CACHE_MAX_ITEMS", "128"))
CROSS_ENCODER_ENABLED = os.getenv("CROSS_ENCODER_ENABLED", "true").lower() in {
    "1", "true", "yes", "on",
}
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
)
CROSS_ENCODER_TOP_N = int(os.getenv("CROSS_ENCODER_TOP_N", "20"))
CROSS_ENCODER_MAX_LENGTH = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
CROSS_ENCODER_FINAL_TOP_K = int(
    os.getenv("CROSS_ENCODER_FINAL_TOP_K", str(SEARCH_TOP_K))
)
CROSS_ENCODER_MIN_SCORE = float(os.getenv("CROSS_ENCODER_MIN_SCORE", "-100"))
CROSS_ENCODER_CACHE_TTL_SECONDS = int(os.getenv("CROSS_ENCODER_CACHE_TTL_SECONDS", "1800"))
CROSS_ENCODER_CACHE_MAX_ITEMS = int(os.getenv("CROSS_ENCODER_CACHE_MAX_ITEMS", "2048"))
CROSS_ENCODER_LEXICAL_FAST_PATH = float(os.getenv("CROSS_ENCODER_LEXICAL_FAST_PATH", "0.8"))
PRELOAD_RAG_COMPONENTS = os.getenv("PRELOAD_RAG_COMPONENTS", "false").lower() in {"1", "true", "yes", "on"}
PRELOAD_EMBEDDING_MODEL = os.getenv("PRELOAD_EMBEDDING_MODEL", "false").lower() in {"1", "true", "yes", "on"}
PRELOAD_CROSS_ENCODER = os.getenv("PRELOAD_CROSS_ENCODER", "false").lower() in {"1", "true", "yes", "on"}
MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", str(MAX_CONTEXT_CHUNKS)))
RETRIEVAL_CACHE_SIZE = int(os.getenv("RETRIEVAL_CACHE_SIZE", str(RETRIEVAL_CACHE_MAX_ITEMS)))
KEYWORD_CONFIDENT_SCORE = float(os.getenv("KEYWORD_CONFIDENT_SCORE", "40"))
RERANK_SCORE_GAP_RATIO = float(os.getenv("RERANK_SCORE_GAP_RATIO", str(VECTOR_FAST_PATH_SCORE_GAP)))
AMBIGUOUS_QUERY_KEYWORD_COUNT = int(os.getenv("AMBIGUOUS_QUERY_KEYWORD_COUNT", str(RERANK_AMBIGUOUS_QUERY_KEYWORDS)))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "storage/chroma_db")
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "document_chunks")
VECTOR_SEARCH_TOP_K = int(os.getenv("VECTOR_SEARCH_TOP_K", str(SEARCH_TOP_K)))
VECTOR_MAX_DISTANCE = float(os.getenv("VECTOR_MAX_DISTANCE", "0.75"))

BUSINESS_MAPPING_FILE = os.getenv(
    "BUSINESS_MAPPING_FILE",
    "storage/business_mapping/pcntt_mapping.json",
)
BUSINESS_MAPPING_MIN_CONFIDENCE = float(
    os.getenv("BUSINESS_MAPPING_MIN_CONFIDENCE", "0.58")
)
BUSINESS_MAPPING_LLM_JUDGE_ENABLED = os.getenv(
    "BUSINESS_MAPPING_LLM_JUDGE_ENABLED",
    "false",
).lower() in {"1", "true", "yes", "on"}
BUSINESS_GENERIC_VECTOR_ENABLED = os.getenv(
    "BUSINESS_GENERIC_VECTOR_ENABLED",
    "true",
).lower() in {"1", "true", "yes", "on"}
BUSINESS_GENERIC_VECTOR_MIN_SCORE = float(os.getenv("BUSINESS_GENERIC_VECTOR_MIN_SCORE", "0.35"))
BUSINESS_GENERIC_VECTOR_TOP_K = int(os.getenv("BUSINESS_GENERIC_VECTOR_TOP_K", "20"))
BUSINESS_GENERIC_KEYWORD_TOP_K = int(os.getenv("BUSINESS_GENERIC_KEYWORD_TOP_K", "20"))
BUSINESS_GENERIC_FINAL_TOP_K = int(os.getenv("BUSINESS_GENERIC_FINAL_TOP_K", "6"))
BUSINESS_GENERIC_VECTOR_MAX_RUNTIME_EMBED_CHUNKS = int(
    os.getenv("BUSINESS_GENERIC_VECTOR_MAX_RUNTIME_EMBED_CHUNKS", "50")
)

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
