from starlette.concurrency import run_in_threadpool
from app.data.elasticsearch_client import search_documents
from app.data.gemini_client import ask_gemini
from app.data.prompt_builder import build_context, build_prompt


async def handle_chat(request):
    question = request.question.strip()

    if not question:
        return {
            "question": request.question,
            "answer": "Vui lòng nhập câu hỏi.",
            "source": None,
        }

    docs = await search_documents(question)

    if not docs:
        return {
            "question": question,
            "answer": "Không tìm thấy nội dung phù hợp trong tài liệu.",
            "source": None,
        }

    context = build_context(docs)
    prompt = build_prompt(question, context)
    answer = await run_in_threadpool(ask_gemini, prompt)

    best_doc = docs[0]
    source = f'{best_doc.get("title")} - {best_doc.get("doc_name")}'

    return {
        "question": question,
        "answer": answer,
        "source": source,
    }
