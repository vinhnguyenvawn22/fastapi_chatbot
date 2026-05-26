async def search_documents(question: str):
    fake_docs = [
        {
            "doc_name": "quy-che-dao-tao.pdf",
            "title": "Điều 2. Hoãn thi",
            "dieu": 2,
            "chunk_index": 1,
            "content": "Sinh viên có thể xin hoãn thi nếu có lý do chính đáng và cần nộp đơn trước kỳ thi.",
            "file_path": "uploads/ChatAI/quy-che-dao-tao.pdf",
        }
    ]

    if "hoãn thi" in question.lower() or "hoan thi" in question.lower():
        return fake_docs

    return []