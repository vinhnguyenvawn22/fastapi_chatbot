from google import genai
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL

client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt: str) -> str:
    """
    Gửi prompt sang Gemini và nhận câu trả lời dạng text.
    """
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Lỗi khi gọi Gemini API: {str(e)}"