from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    # Cho phép chạy script trực tiếp bằng: python scripts/reindex_documents.py
    sys.path.insert(0, str(ROOT_DIR))

from app.controller.document_controller import build_document_chunks, list_documents
from app.data.vector_store import get_collection, index_chunks


def reindex_documents():
    # Đọc toàn bộ PDF hiện có để tạo lại vector index mà không cần upload lại.
    files = list_documents()
    indexed_files = []
    failed_files = []
    total_chunks = 0

    for file_info in files:
        file_name = file_info["file_name"]

        try:
            # Tách PDF thành chunks kèm metadata trước khi đưa vào vector store.
            chunks = build_document_chunks(file_name)

            # Upsert chunks vào Chroma; ID ổn định giúp không tạo trùng khi chạy lại.
            indexed_count = index_chunks(chunks)
            total_chunks += indexed_count
            indexed_files.append({
                "file_name": file_name,
                "chunks": indexed_count,
            })
            print(f"[OK] {file_name}: {indexed_count} chunks")
        except Exception as exc:
            failed_files.append({
                "file_name": file_name,
                "error": str(exc),
            })
            print(f"[ERROR] {file_name}: {exc}")

    # Đếm số vector hiện có sau khi reindex để xác nhận Chroma đã lưu dữ liệu.
    vector_count = get_collection().count()

    return {
        "indexed_file_count": len(indexed_files),
        "total_chunks_indexed": total_chunks,
        "vector_count": vector_count,
        "failed_files": failed_files,
    }


def main():
    result = reindex_documents()

    print("\n=== Reindex summary ===")
    print(f"Indexed files: {result['indexed_file_count']}")
    print(f"Indexed chunks: {result['total_chunks_indexed']}")
    print(f"Vector count: {result['vector_count']}")

    if result["failed_files"]:
        print("\nFailed files:")
        for item in result["failed_files"]:
            print(f"- {item['file_name']}: {item['error']}")
    else:
        print("Failed files: 0")


if __name__ == "__main__":
    main()
