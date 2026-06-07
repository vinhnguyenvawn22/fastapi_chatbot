from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    # Cho phép chạy script trực tiếp bằng: python scripts/reindex_documents.py
    sys.path.insert(0, str(ROOT_DIR))

from app.controller.document_controller import build_document_chunks, list_documents
from app.data.vector_store import clear_collection, get_collection, index_chunks


def reindex_documents():
    """Xóa vector cũ, đọc lại toàn bộ PDF và index lại tất cả chunk vào Chroma."""
    # Đọc toàn bộ PDF hiện có để tạo lại vector index mà không cần upload lại.
    files = list_documents()
    indexed_files = []
    failed_files = []
    total_chunks = 0

    clear_collection()

    for file_info in files:
        file_name = file_info.get("relative_path") or file_info["file_name"]

        try:
            # Tách PDF thành chunks kèm metadata trước khi đưa vào vector store.
            chunks = build_document_chunks(file_name)

            # Upsert chunks vào Chroma; ID ổn định giúp không tạo trùng khi chạy lại.
            indexed_count = index_chunks(chunks)
            total_chunks += indexed_count
            indexed_files.append({
                "file_name": file_info["file_name"],
                "relative_path": file_name,
                "phong_ban": file_info.get("phong_ban"),
                "chunks": indexed_count,
            })
            print(f"[OK] {file_name}: {indexed_count} chunks")
        except Exception as exc:
            failed_files.append({
                "file_name": file_info["file_name"],
                "relative_path": file_name,
                "phong_ban": file_info.get("phong_ban"),
                "error": str(exc),
            })
            print(f"[ERROR] {file_name}: {exc}")

    # Đếm số vector hiện có sau khi reindex để xác nhận Chroma đã lưu dữ liệu.
    vector_count = get_collection().count()

    return {
        "indexed_file_count": len(indexed_files),
        "discovered_pdf_count": len(files),
        "total_chunks_indexed": total_chunks,
        "vector_count": vector_count,
        "failed_files": failed_files,
    }


def main():
    """Chạy reindex từ command line và in báo cáo tổng kết."""
    result = reindex_documents()

    print("\n=== Reindex summary ===")
    print(f"Discovered PDFs: {result['discovered_pdf_count']}")
    print(f"Indexed files: {result['indexed_file_count']}")
    print(f"Indexed chunks: {result['total_chunks_indexed']}")
    print(f"Vector count: {result['vector_count']}")

    if result["failed_files"]:
        print("\nFailed files:")
        for item in result["failed_files"]:
            print(f"- {item['relative_path']}: {item['error']}")
    else:
        print("Failed files: 0")


if __name__ == "__main__":
    main()
