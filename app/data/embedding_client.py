from functools import lru_cache

from app.core.config import EMBEDDING_CACHE_MAX_ITEMS, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    """Tải và cache model SentenceTransformer để tái sử dụng cho mọi lần embedding."""
    # Tải model embedding khi quá trình upload/index cần tạo vector lần đầu.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=256)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    model = get_embedding_model()
    vector = model.encode(text or "", normalize_embeddings=True)
    return tuple(float(value) for value in vector.tolist())


def embed_query(text: str) -> list[float]:
    """Tạo vector embedding đã normalize cho một câu hỏi người dùng."""
    # Tạo embedding cho câu hỏi để dùng trong vector search.
    return list(_embed_query_cached(" ".join(str(text or "").split())))


@lru_cache(maxsize=EMBEDDING_CACHE_MAX_ITEMS)
def _embed_query_cached(text: str) -> tuple[float, ...]:
    model = get_embedding_model()
    vector = model.encode(text or "", normalize_embeddings=True)
    return tuple(float(value) for value in vector.tolist())


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Tạo vector embedding hàng loạt cho các chunk tài liệu khi index."""
    # Tạo embedding hàng loạt cho các chunk để index vào vector store.
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]
