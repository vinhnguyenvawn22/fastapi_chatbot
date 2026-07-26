from pathlib import Path
import argparse
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.controller.document_controller import build_document_chunks, list_documents
from app.core.config import (
    LOCAL_DOCUMENTS_INDEX_VERSION,
    LOCAL_DOCUMENTS_MANIFEST_FILE,
)
from app.data.vector_store import clear_collection, get_collection, index_chunks


def _write_manifest(records: list[dict], report: dict) -> None:
    manifest_path = Path(LOCAL_DOCUMENTS_MANIFEST_FILE).resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_version": LOCAL_DOCUMENTS_INDEX_VERSION,
        "records": records,
        "report": report,
    }
    temp_path = manifest_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(manifest_path)


def reindex_documents(dry_run: bool = False):
    files = list_documents()
    indexed_files = []
    failed_files = []
    unsupported_files = []
    excluded_files = []
    duplicate_files = []
    manifest_records = []
    canonical_content_hashes = {}
    indexed_chunk_hashes = set()
    total_chunks = 0

    if not dry_run:
        clear_collection()

    for file_info in files:
        file_name = file_info.get("relative_path") or file_info["file_name"]

        if file_info.get("parse_supported") is False:
            unsupported_files.append({
                "file_name": file_info["file_name"],
                "relative_path": file_name,
                "file_type": file_info.get("file_type"),
                "phong_ban": file_info.get("phong_ban"),
            })
            print(f"[SKIP] {file_name}: unsupported file type")
            continue

        try:
            chunks = build_document_chunks(file_name)
            first_chunk = chunks[0] if chunks else {}
            record = {
                "relative_path": file_name,
                "file_hash": first_chunk.get("file_hash"),
                "content_hash": first_chunk.get("content_hash"),
                "modified_time": file_info.get("updated_at"),
                "parser_version": first_chunk.get("parser_version"),
                "chunker_version": first_chunk.get("chunker_version"),
                "index_status": "pending",
                "chunk_count": 0,
                "parse_status": "success",
                "parse_error": None,
                "duplicate_of": None,
            }

            if first_chunk.get("rag_enabled") is False:
                record["index_status"] = "excluded"
                record["exclude_reason"] = first_chunk.get("exclude_reason")
                manifest_records.append(record)
                excluded_files.append({
                    "file_name": file_info["file_name"],
                    "relative_path": file_name,
                    "reason": first_chunk.get("exclude_reason"),
                })
                print(f"[EXCLUDED] {file_name}: {first_chunk.get('exclude_reason')}")
                continue

            content_hash = first_chunk.get("content_hash")
            if content_hash and content_hash in canonical_content_hashes:
                duplicate_of = canonical_content_hashes[content_hash]
                record["index_status"] = "duplicate"
                record["duplicate_of"] = duplicate_of
                manifest_records.append(record)
                duplicate_files.append({
                    "file_name": file_info["file_name"],
                    "relative_path": file_name,
                    "duplicate_of": duplicate_of,
                })
                print(f"[DUPLICATE] {file_name}: {duplicate_of}")
                continue
            if content_hash:
                canonical_content_hashes[content_hash] = file_name

            unique_chunks = []
            for chunk in chunks:
                chunk_hash = chunk.get("chunk_hash")
                if chunk_hash and chunk_hash in indexed_chunk_hashes:
                    continue
                if chunk_hash:
                    indexed_chunk_hashes.add(chunk_hash)
                unique_chunks.append(chunk)

            indexed_count = len(unique_chunks) if dry_run else index_chunks(unique_chunks)
            total_chunks += indexed_count
            record["index_status"] = "dry_run" if dry_run else "indexed"
            record["chunk_count"] = indexed_count
            manifest_records.append(record)
            indexed_files.append({
                "file_name": file_info["file_name"],
                "relative_path": file_name,
                "phong_ban": file_info.get("phong_ban"),
                "chunks": indexed_count,
            })
            print(f"[OK] {file_name}: {indexed_count} chunks")
        except Exception as exc:
            error = str(exc)
            manifest_records.append({
                "relative_path": file_name,
                "file_hash": None,
                "content_hash": None,
                "modified_time": file_info.get("updated_at"),
                "parser_version": None,
                "chunker_version": None,
                "index_status": "failed",
                "chunk_count": 0,
                "parse_status": "failed",
                "parse_error": error,
                "duplicate_of": None,
            })
            failed_files.append({
                "file_name": file_info["file_name"],
                "relative_path": file_name,
                "phong_ban": file_info.get("phong_ban"),
                "error": error,
            })
            print(f"[ERROR] {file_name}: {error}")

    vector_count = get_collection().count()
    report = {
        "index_version": LOCAL_DOCUMENTS_INDEX_VERSION,
        "dry_run": dry_run,
        "indexed_file_count": len(indexed_files),
        "discovered_document_count": len(files),
        "unsupported_file_count": len(unsupported_files),
        "excluded_file_count": len(excluded_files),
        "duplicate_file_count": len(duplicate_files),
        "parse_failed_file_count": len(failed_files),
        "total_chunks_indexed": total_chunks,
        "vector_count": vector_count,
        "unsupported_files": unsupported_files,
        "excluded_files": excluded_files,
        "duplicate_files": duplicate_files,
        "failed_files": failed_files,
    }
    _write_manifest(manifest_records, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse, classify and report without clearing or writing vectors.",
    )
    args = parser.parse_args()
    result = reindex_documents(dry_run=args.dry_run)

    print("\n=== Reindex summary ===")
    print(f"Dry run: {result['dry_run']}")
    print(f"Discovered documents: {result['discovered_document_count']}")
    print(f"Indexed files: {result['indexed_file_count']}")
    print(f"Unsupported files: {result['unsupported_file_count']}")
    print(f"Excluded files: {result['excluded_file_count']}")
    print(f"Duplicate files: {result['duplicate_file_count']}")
    print(f"Parse failed files: {result['parse_failed_file_count']}")
    print(f"Indexed chunks: {result['total_chunks_indexed']}")
    print(f"Vector count: {result['vector_count']}")

    if result["failed_files"]:
        print("\nFailed files:")
        for item in result["failed_files"]:
            print(f"- {item['relative_path']}: {item['error']}")


if __name__ == "__main__":
    main()
