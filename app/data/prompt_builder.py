def build_context(docs):
    context_parts = []

    for index, doc in enumerate(docs, start=1):
        file_name = doc.get("doc_name", "unknown")
        section_name = doc.get("title", "Không rõ mục")
        content = doc.get("content", "")

        block = (
            f'<NGUON id="{index}" ten_tai_lieu="{file_name}" '
            f'dieu_khoan="{section_name}">\n'
            f'{content}\n'
            f'</NGUON>'
        )

        context_parts.append(block)

    return "\n\n".join(context_parts)


def build_prompt(question, context):
    prompt = f"""
Bạn là trợ lý AI của nhà trường.
Chỉ trả lời dựa trên THÔNG TIN THAM KHẢO bên dưới.
Nếu không có dữ liệu phù hợp, hãy nói không tìm thấy nội dung phù hợp.
Cuối câu trả lời phải có dòng: (Nguồn: tên tài liệu)

THÔNG TIN THAM KHẢO:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
"""
    return prompt.strip()