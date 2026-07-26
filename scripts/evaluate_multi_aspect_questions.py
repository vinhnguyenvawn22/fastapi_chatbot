import concurrent.futures
import json
import sys
import uuid

import requests


QUESTIONS = [
    "Điều kiện tốt nghiệp là gì và điều kiện tốt nghiệp loại giỏi là gì?",
    "Điều kiện chuyển trường là gì và hồ sơ chuyển trường cần những gì?",
    "Sinh viên được đăng ký học lại khi nào và cách đăng ký như thế nào?",
    "Điều kiện hoãn thi là gì và thủ tục xin hoãn thi thực hiện như thế nào?",
    "Điểm trung bình tích lũy được tính như thế nào và bao nhiêu điểm thì được xếp loại giỏi?",
    "Sinh viên bị cảnh báo học tập khi nào và được đăng ký tối đa bao nhiêu tín chỉ?",
    "Điều kiện học chương trình thứ hai là gì? Thủ tục đăng ký cần những hồ sơ nào?",
    "Làm sao đăng ký thi lại; lệ phí thi lại được thanh toán như thế nào?",
    "Sinh viên nghỉ học quá bao nhiêu tiết thì bị cấm thi và điểm chuyên cần được tính như thế nào?",
    "Cách gửi yêu cầu trên hệ thống và làm thế nào để kiểm tra trạng thái xử lý?",
    "Hồ sơ tốt nghiệp gồm những gì? Sinh viên nộp hồ sơ ở đâu? Thời hạn nộp khi nào?",
    "Điều kiện được cấp bằng là gì, đồng thời trường hợp nào bị hạ hạng tốt nghiệp?",
]


def evaluate(base_url: str, index: int, question: str) -> dict:
    response = requests.post(
        f"{base_url}/api/chat/local-documents",
        json={
            "question": question,
            "request_id": f"multi-aspect-eval-{index}-{uuid.uuid4().hex[:8]}",
        },
        timeout=300,
    )
    response.raise_for_status()
    payload = response.json()
    trace = requests.get(
        f"{base_url}/api/chat/traces/{payload['trace_id']}",
        timeout=30,
    ).json()

    steps = {step["name"]: step.get("output") or {} for step in trace.get("steps", [])}
    decomposition = steps.get("multi_aspect_decomposition", {})
    retrieval = steps.get("multi_aspect_retrieval", {})
    return {
        "index": index,
        "question": question,
        "answer": payload.get("answer"),
        "sources": [
            {
                "title": source.get("title"),
                "doc_name": source.get("doc_name"),
                "dieu": source.get("dieu"),
            }
            for source in payload.get("sources", [])
        ],
        "trace_id": payload.get("trace_id"),
        "is_multi_aspect": decomposition.get("is_multi_aspect"),
        "aspects": [
            item.get("question") for item in decomposition.get("aspects", [])
        ],
        "coverage": retrieval.get("coverage"),
        "deterministic": "policy_deterministic_answer" in steps,
        "llm_called": "lcel_llm_call" in steps,
    }


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8005"
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(evaluate, base_url, index, question)
            for index, question in enumerate(QUESTIONS, start=1)
        ]
        results = [future.result() for future in futures]

    print(json.dumps(sorted(results, key=lambda item: item["index"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
