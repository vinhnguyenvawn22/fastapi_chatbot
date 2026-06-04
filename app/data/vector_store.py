from pathlib import Path
from typing import Any

import chromadb

from app.core.config import (
    VECTOR_COLLECTION_NAME,
    VECTOR_MAX_DISTANCE,
    VECTOR_SEARCH_TOP_K,
    VECTOR_STORE_DIR,
)
from app.data.embedding_client import embed_documents, embed_query


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Lọc metadata về các kiểu dữ liệu đơn giản mà ChromaDB hỗ trợ."""
    # Chroma chỉ nhận metadata kiểu đơn giản, nên cần bỏ None/list/dict.
    cleaned = {}

    for key, value in metadata.items():
        if value is None:
            continue

        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)

    return cleaned


def _chunk_id(chunk: dict) -> str:
    """Tạo ID ổn định cho chunk để upsert không sinh bản ghi trùng khi reindex."""
    # ID ổn định giúp upsert không tạo trùng chunk khi index lại cùng tài liệu.
    content_hash = chunk.get("content_hash") or chunk.get("doc_name") or "unknown"
    chunk_index = chunk.get("chunk_index", 0)
    return f"{content_hash}:{chunk_index}"


def _build_chroma_where(metadata_filter: dict | None):
    if not metadata_filter:
        return None

    filters = [
        {key: value}
        for key, value in metadata_filter.items()
        if value is not None
    ]

    if not filters:
        return None

    if len(filters) == 1:
        return filters[0]

    return {"$and": filters}


def get_collection():
    """Mở hoặc tạo Chroma collection dùng để lưu vector tài liệu bền vững trên disk."""
    # Tạo Chroma persistent collection để vector vẫn còn sau khi restart server.
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    return client.get_or_create_collection(
        name=VECTOR_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict]) -> int:
    """Embedding và upsert danh sách chunk vào ChromaDB, trả về số chunk đã xử lý."""
    # Index chunk vào Chroma sau khi upload để chat không cần parse PDF lại.
    if not chunks:
        return 0

    collection = get_collection()
    documents = [chunk.get("content", "") for chunk in chunks]
    embeddings = embed_documents(documents)

    ids = [_chunk_id(chunk) for chunk in chunks]
    metadatas = [
        _clean_metadata({key: value for key, value in chunk.items() if key != "content"})
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def search_similar_chunks(
    question: str,
    top_k: int = VECTOR_SEARCH_TOP_K,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Tìm top chunk gần nghĩa nhất với câu hỏi bằng embedding và cosine distance."""
    # Tìm các chunk gần nghĩa nhất với câu hỏi bằng cosine distance trong Chroma.
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_vector = embed_query(question)
    query_args = {
        "query_embeddings": [query_vector],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    where = _build_chroma_where(metadata_filter)
    if where:
        query_args["where"] = where

    result = collection.query(**query_args)

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    chunks = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        if distance is not None and distance > VECTOR_MAX_DISTANCE:
            continue

        chunk = dict(metadata or {})
        chunk["content"] = document or ""
        chunk["score"] = round(1 - float(distance or 0), 4)
        chunk["distance"] = round(float(distance or 0), 4)
        chunks.append(chunk)

    return chunks


def delete_document(doc_name: str) -> None:
    """Xóa toàn bộ vector chunk thuộc một tài liệu theo tên file."""
    # Xóa vector theo tên tài liệu khi cần re-index hoặc gỡ tài liệu.
    collection = get_collection()
    collection.delete(where={"doc_name": doc_name})


def clear_collection() -> None:
    """Xóa toàn bộ vector trong collection, thường dùng trước khi reindex toàn bộ."""
    collection = get_collection()
    existing = collection.get(include=[])
    ids = existing.get("ids", [])

    if ids:
        collection.delete(ids=ids)
