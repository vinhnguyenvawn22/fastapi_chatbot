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
    # ID ổn định giúp upsert không tạo trùng chunk khi index lại cùng tài liệu.
    content_hash = chunk.get("content_hash") or chunk.get("doc_name") or "unknown"
    chunk_index = chunk.get("chunk_index", 0)
    return f"{content_hash}:{chunk_index}"


def get_collection():
    # Tạo Chroma persistent collection để vector vẫn còn sau khi restart server.
    Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    return client.get_or_create_collection(
        name=VECTOR_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(chunks: list[dict]) -> int:
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


def search_similar_chunks(question: str, top_k: int = VECTOR_SEARCH_TOP_K) -> list[dict]:
    # Tìm các chunk gần nghĩa nhất với câu hỏi bằng cosine distance trong Chroma.
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_vector = embed_query(question)
    result = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

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
    # Xóa vector theo tên tài liệu khi cần re-index hoặc gỡ tài liệu.
    collection = get_collection()
    collection.delete(where={"doc_name": doc_name})
