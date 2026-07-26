import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.elasticsearch_client import normalize_text, search_documents
from app.data.gemini_client import get_gemini_call_count
from app.data.multi_aspect_query import (
    decompose_multi_aspect_query,
    filter_semantic_aspect_docs,
)


CASES = [
    {
        "question": "Điều kiện tốt nghiệp là gì và điều kiện tốt nghiệp loại giỏi là gì?",
        "aspects": [
            [["điểm trung bình tích lũy"], ["chứng chỉ", "chứng nhận"]],
            [["3,20", "3.20"], ["3,59", "3.59"]],
        ],
    },
    {
        "question": "Điều kiện chuyển trường là gì và hồ sơ chuyển trường cần những gì?",
        "aspects": [
            [["chuyển trường"], ["hiệu trưởng"], ["cùng ngành", "hoàn cảnh"]],
            None,
        ],
    },
    {
        "question": "Sinh viên được đăng ký học lại khi nào và cách đăng ký như thế nào?",
        "aspects": [
            [["học lại"], ["điểm f", "không đạt"]],
            [["đăng ký"], ["trực tuyến", "website", "hệ thống"], ["học lại", "chưa đạt"]],
        ],
    },
    {
        "question": "Điều kiện hoãn thi là gì và thủ tục xin hoãn thi thực hiện như thế nào?",
        "aspects": [
            [["hoãn thi", "điểm i"], ["ốm", "tai nạn", "khách quan"]],
            [["hoãn thi"], ["một cửa", "support.uneti"], ["gửi yêu cầu"]],
        ],
    },
    {
        "question": "Điểm trung bình tích lũy được tính như thế nào và bao nhiêu điểm thì được xếp loại giỏi?",
        "aspects": [
            [["ai là điểm", "ai"], ["ni là số tín chỉ", "ni"], ["làm tròn"]],
            [["3,20", "3.20"], ["3,59", "3.59"]],
        ],
    },
    {
        "question": "Sinh viên bị cảnh báo học tập khi nào và được đăng ký tối đa bao nhiêu tín chỉ?",
        "aspects": [
            [["cảnh báo kết quả học tập"], ["50%"], ["0,80", "0.80"]],
            [["16 tín chỉ"]],
        ],
    },
    {
        "question": "Điều kiện học chương trình thứ hai là gì? Thủ tục đăng ký cần những hồ sơ nào?",
        "aspects": [
            [["chương trình thứ hai"], ["điểm trung bình", "xếp hạng"]],
            [["đơn"], ["phòng đào tạo", "phòng chính trị", "ct&ctsv"]],
        ],
    },
    {
        "question": "Làm sao đăng ký thi lại; lệ phí thi lại được thanh toán như thế nào?",
        "aspects": [
            [["thi lại"], ["đăng ký"], ["support.uneti", "hệ thống"]],
            [["lệ phí"], ["học phí", "học kỳ tiếp theo"]],
        ],
    },
    {
        "question": "Sinh viên nghỉ học quá bao nhiêu tiết thì bị cấm thi và điểm chuyên cần được tính như thế nào?",
        "aspects": [
            [["50%"], ["cấm thi"]],
            [["điểm chuyên cần"], ["10 điểm"], ["8 điểm"], ["6 điểm"]],
        ],
    },
    {
        "question": "Cách gửi yêu cầu trên hệ thống và làm thế nào để kiểm tra trạng thái xử lý?",
        "clarification": True,
    },
    {
        "question": "Hồ sơ tốt nghiệp gồm những gì? Sinh viên nộp hồ sơ ở đâu? Thời hạn nộp khi nào?",
        "aspects": [
            [["đơn"], ["giấy xác nhận nhân sự"], ["khai sinh công chứng"]],
            [["bộ phận hành chính một cửa"], ["phòng đào tạo"]],
            None,
        ],
    },
    {
        "question": "Điều kiện được cấp bằng là gì, đồng thời trường hợp nào bị hạ hạng tốt nghiệp?",
        "aspects": [
            [["điều kiện xét tốt nghiệp"], ["điểm trung bình tích lũy"]],
            [["giảm đi một mức", "giảm một mức"], ["5%"], ["kỷ luật", "cảnh cáo"]],
        ],
    },
    {
        "question": "Học phần bắt buộc bị điểm F phải xử lý thế nào và học phần tự chọn không đạt có được học đổi không?",
        "aspects": [
            [["học phần bắt buộc"], ["điểm f"], ["học lại"]],
            [["học phần tự chọn"], ["học đổi", "tương đương"]],
        ],
    },
    {
        "question": "Sinh viên được rút bớt học phần trong trường hợp nào và hạn rút học phần là bao lâu?",
        "aspects": [
            [["rút bớt học phần", "rút học phần"], ["đơn", "chấp thuận"]],
            [["2 tuần", "hai tuần"], ["học kỳ chính"]],
        ],
    },
    {
        "question": "Tôi xem lịch thi ở đâu và kết quả thi được tra cứu như thế nào?",
        "aspects": [
            [["lịch thi"], ["support.uneti", "website", "hệ thống"]],
            [["kết quả thi", "kết quả học tập"], ["tra cứu", "xem"]],
        ],
    },
    {
        "question": "Làm sao báo hỏng thiết bị phòng học và kiểm tra tình trạng xử lý yêu cầu đó?",
        "aspects": [
            [["thiết bị", "báo hỏng"], ["phòng học"], ["gửi yêu cầu", "support.uneti"]],
            [["trạng thái", "tình trạng"], ["xử lý"]],
        ],
    },
    {
        "question": "Sinh viên tìm phiếu thủ tục cần đánh giá theo tiêu chí nào và phiếu ở trạng thái nào mới được đánh giá?",
        "aspects": [
            [["nhóm thủ tục"], ["tên thủ tục"], ["ngày đề nghị", "ngày gửi"]],
            [["đã xử lý"]],
        ],
    },
    {
        "question": "Học kỳ chính phải đăng ký tối thiểu bao nhiêu tín chỉ và sinh viên đang bị cảnh báo được đăng ký tối đa bao nhiêu?",
        "aspects": [
            [["tối thiểu 14 tín chỉ"], ["12 tín chỉ"]],
            [["16 tín chỉ"], ["cảnh báo"]],
        ],
    },
    {
        "question": "Làm sao để đăng ký thi lại và hoãn thi?",
        "aspects": [
            [["1.4. đăng ký thi lại", "đăng ký thi lại"], ["support.uneti"]],
            [["1.5. hoãn thi", "hoãn thi"], ["support.uneti"], ["mc-kt-05"]],
        ],
    },
    {
        "question": "Cách xem lịch thi và tra cứu kết quả học tập?",
        "aspects": [
            [["lịch thi"], ["support.uneti", "hệ thống"]],
            [["kết quả học tập"], ["tra cứu", "xem"]],
        ],
    },
    {
        "question": "Làm sao để hủy đăng ký thi lại và xem kết quả học tập?",
        "aspects": [
            [["hủy đăng ký thi lại"], ["support.uneti"]],
            [["kết quả học tập"], ["xem", "tra cứu"]],
        ],
    },
    {
        "question": "Cách đăng ký mượn thiết bị và báo hỏng thiết bị phòng học?",
        "aspects": [
            [["đăng ký mượn thiết bị"], ["phòng học", "lịch dạy"]],
            [["báo hỏng"], ["thiết bị"], ["phòng học"]],
        ],
    },
    {
        "question": "Cách xem thời khóa biểu và điểm danh?",
        "aspects": [
            [["thời khóa biểu"], ["xem", "lịch học"]],
            [["điểm danh"], ["xem", "theo dõi"]],
        ],
    },
    {
        "question": "Tôi kiểm tra lịch thi ở đâu và xem điểm học tập bằng cách nào?",
        "aspects": [
            [["lịch thi"], ["support.uneti", "hệ thống"]],
            [["kết quả học tập"], ["xem", "tra cứu"]],
        ],
    },
    {
        "question": "Điều kiện đăng ký học lại và học cải thiện điểm là gì?",
        "aspects": [
            [["học lại"], ["điểm f", "không đạt"]],
            [["học cải thiện"], ["điểm", "kết quả học tập"]],
        ],
    },
    {
        "question": "Khi nào được rút học phần và phải nộp đơn ở đâu?",
        "aspects": [
            [["rút bớt học phần", "rút học phần"], ["đơn", "chấp thuận"]],
            [["phòng đào tạo"], ["đơn"]],
        ],
    },
    {
        "question": "Sinh viên bị điểm F phải học lại thế nào và có được đổi học phần tự chọn không?",
        "aspects": [
            [["điểm f"], ["học lại"], ["học phần bắt buộc"]],
            [["học phần tự chọn"], ["học đổi", "tương đương"]],
        ],
    },
    {
        "question": "Điều kiện tốt nghiệp là gì và cách đăng ký tốt nghiệp trên hệ thống?",
        "aspects": [
            [["điều kiện xét tốt nghiệp"], ["điểm trung bình tích lũy"]],
            [["đăng ký tốt nghiệp"], ["bộ phận hành chính một cửa"], ["in đơn"]],
        ],
    },
    {
        "question": "Cách đánh giá thủ tục hành chính và xem trạng thái phiếu?",
        "aspects": [
            [["đánh giá"], ["thủ tục hành chính"], ["đã xử lý"]],
            [["trạng thái"], ["phiếu", "thủ tục"]],
        ],
    },
    {
        "question": "Làm sao để đăng ký thi lại; hủy đăng ký thi lại; và hoãn thi?",
        "aspects": [
            [["đăng ký thi lại"], ["support.uneti"]],
            [["hủy đăng ký thi lại"], ["support.uneti"]],
            [["hoãn thi"], ["support.uneti"], ["mc-kt-05"]],
        ],
    },
    {
        "question": "Cách báo hỏng thiết bị và kiểm tra tình trạng xử lý?",
        "aspects": [
            [["báo hỏng"], ["thiết bị"], ["gửi yêu cầu", "support.uneti"]],
            [["tình trạng", "trạng thái"], ["xử lý"]],
        ],
    },
    {
        "question": "Điều kiện hoãn thi là gì và cách gửi yêu cầu hoãn thi?",
        "aspects": [
            [["hoãn thi", "điểm i"], ["ốm", "tai nạn", "khách quan"]],
            [["hoãn thi"], ["support.uneti"], ["gửi yêu cầu"]],
        ],
    },
    {
        "question": "Điểm chữ được quy đổi ra sao và GPA được tính như thế nào?",
        "aspects": [
            [["thang điểm 4"], ["a tương ứng", "b+ tương ứng"]],
            [["ai là điểm", "ai"], ["ni là số tín chỉ", "ni"], ["làm tròn"]],
        ],
    },
]


def _document_text(docs: list[dict]) -> str:
    return normalize_text(
        " ".join(
            str(doc.get(field) or "")
            for doc in docs
            for field in ("title", "heading", "section_path", "content")
        )
    )


def _matches_expected(docs: list[dict], expected: list[list[str]] | None) -> bool:
    if expected is None:
        return not docs
    text = _document_text(docs)
    return all(
        any(normalize_text(alternative) in text for alternative in group)
        for group in expected
    )


def _source_summary(doc: dict) -> dict[str, Any]:
    return {
        "doc_name": doc.get("doc_name"),
        "title": doc.get("title"),
        "dieu": doc.get("dieu"),
        "chunk_index": doc.get("chunk_index"),
        "document_type": doc.get("document_type"),
    }


async def evaluate_case(
    index: int,
    case: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    decomposition = decompose_multi_aspect_query(case["question"])
    if case.get("clarification"):
        passed = decomposition.get("needs_clarification") is True
        return {
            "index": index,
            "question": case["question"],
            "passed": passed,
            "mode": "clarification",
            "reason": decomposition.get("clarification_reason"),
        }

    aspects = decomposition.get("aspects") or []
    expected_aspects = case["aspects"]
    if len(aspects) != len(expected_aspects):
        return {
            "index": index,
            "question": case["question"],
            "passed": False,
            "mode": "retrieval",
            "reason": f"aspect_count:{len(aspects)}!=expected:{len(expected_aspects)}",
            "decomposition": decomposition,
        }

    async def retrieve(aspect: dict, expected: list[list[str]] | None) -> dict:
        async def retrieve_query(retrieval_query: str) -> list[dict]:
            async with semaphore:
                return await search_documents(
                    retrieval_query,
                    ambiguity_decision={"action": "direct_retrieval"},
                    corpus_filter="local_documents",
                    rag_enabled_filter=True,
                )

        query_results = await asyncio.gather(
            *(
                retrieve_query(retrieval_query)
                for retrieval_query in aspect.get("retrieval_queries")
                or [aspect["retrieval_query"]]
            )
        )
        docs = []
        seen = set()
        max_rank = max((len(result) for result in query_results), default=0)
        for rank in range(max_rank):
            for result in query_results:
                if rank >= len(result):
                    continue
                doc = result[rank]
                key = (
                    doc.get("relative_path") or doc.get("doc_name"),
                    doc.get("chunk_index"),
                    doc.get("title"),
                )
                if key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
        filtered_docs, filter_reason = filter_semantic_aspect_docs(
            aspect.get("semantic_query") or aspect["retrieval_query"],
            docs,
        )
        selected_docs = filtered_docs[:3]
        return {
            "aspect_id": aspect["aspect_id"],
            "question": aspect["question"],
            "retrieval_query": aspect["retrieval_query"],
            "retrieval_queries": aspect.get("retrieval_queries"),
            "context_inherited": aspect.get("context_inherited", False),
            "passed": _matches_expected(selected_docs, expected),
            "expected_answerable": expected is not None,
            "filter_reason": filter_reason,
            "sources": [_source_summary(doc) for doc in selected_docs],
        }

    results = await asyncio.gather(
        *(
            retrieve(aspect, expected)
            for aspect, expected in zip(aspects, expected_aspects)
        )
    )
    return {
        "index": index,
        "question": case["question"],
        "passed": all(result["passed"] for result in results),
        "mode": "retrieval",
        "aspects": results,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--indices")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--failures-only", action="store_true")
    args = parser.parse_args()

    indexed_cases = list(enumerate(CASES, start=1))
    if args.indices:
        selected_indices = {
            int(value.strip())
            for value in args.indices.split(",")
            if value.strip()
        }
        indexed_cases = [
            item for item in indexed_cases if item[0] in selected_indices
        ]
    elif args.limit:
        indexed_cases = indexed_cases[: args.limit]
    gemini_before = get_gemini_call_count()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results = await asyncio.gather(
        *(
            evaluate_case(index, case, semaphore)
            for index, case in indexed_cases
        )
    )
    gemini_after = get_gemini_call_count()
    failed = [result for result in results if not result["passed"]]
    payload = {
        "summary": {
            "case_count": len(results),
            "passed": len(results) - len(failed),
            "failed": len(failed),
            "pass_rate": round((len(results) - len(failed)) / len(results), 4),
            "gemini_calls": gemini_after - gemini_before,
        },
        "results": failed if args.failures_only else results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
