from functools import lru_cache

from app.core.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embedding_model():
    # Tải model embedding khi quá trình upload/index cần tạo vector lần đầu.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_query(text: str) -> list[float]:
    # Tạo embedding cho câu hỏi để dùng trong vector search.
    model = get_embedding_model()
    vector = model.encode(text or "", normalize_embeddings=True)
    return vector.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    # Tạo embedding hàng loạt cho các chunk để index vào vector store.
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]
