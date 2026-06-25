import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.elasticsearch_client import clear_document_index_cache, search_documents


DEFAULT_QUERIES = [
    "đào tạo từ xa",
    "đối tượng được miễn giảm học phí",
    "đăng ký môn",
]


async def synthetic_parallel_benchmark(delay_ms: int = 100) -> dict:
    def simulated_branch():
        time.sleep(delay_ms / 1000)

    started = time.perf_counter()
    simulated_branch()
    simulated_branch()
    sequential_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    await asyncio.gather(
        asyncio.to_thread(simulated_branch),
        asyncio.to_thread(simulated_branch),
    )
    parallel_ms = (time.perf_counter() - started) * 1000
    return {
        "sequential_ms": round(sequential_ms, 3),
        "parallel_ms": round(parallel_ms, 3),
        "speedup": round(sequential_ms / parallel_ms, 3),
    }


async def benchmark(queries: list[str], rounds: int) -> dict:
    samples = []
    for query in queries:
        clear_document_index_cache()
        for round_index in range(rounds):
            debug = {}
            started = time.perf_counter()
            results = await search_documents(query, debug=debug)
            samples.append({
                "query": query,
                "round": round_index + 1,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "result_count": len(results),
                "expansion_reason": (debug.get("expanded_queries") or {}).get("reason"),
                "rerank_reason": (debug.get("reranking") or {}).get("reason"),
            })
    latencies = [sample["latency_ms"] for sample in samples]
    return {
        "samples": samples,
        "summary": {
            "count": len(samples),
            "mean_ms": round(statistics.mean(latencies), 3),
            "median_ms": round(statistics.median(latencies), 3),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
        },
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    result = (
        asyncio.run(synthetic_parallel_benchmark())
        if args.synthetic
        else asyncio.run(benchmark(args.queries or DEFAULT_QUERIES, args.rounds))
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
