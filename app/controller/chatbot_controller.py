from app.data.elasticsearch_client import search_documents
from app.data.prompt_builder import build_context, build_prompt
from app.data.gemini_client import ask_gemini


async def handle_chat(request):
    docs = await search_documents(request.question)

    if not docs:
        return {
            "question": request.question,
            "answer": "Không tìm thấy nội dung phù hợp trong tài liệu.",
            "source": None,
        }

    context = build_context(docs)
    prompt = build_prompt(request.question, context)
    answer = await ask_gemini(prompt)

    return {
        "question": request.question,
        "answer": answer,
        "source": docs[0].get("doc_name"),
    }