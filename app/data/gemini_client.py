from google import genai
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt: str) -> str:
    """
    Gửi prompt sang Gemini, nhận text trả lời và chuẩn hóa thông báo lỗi thường gặp.
    """
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        return response.text or "Không tìm thấy căn cứ đủ rõ trong tài liệu đã cung cấp."

    except Exception as e:
        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            return "Hệ thống AI tạm thời vượt giới hạn sử dụng. Vui lòng thử lại sau ít phút."

        if "503" in error_message or "UNAVAILABLE" in error_message:
            return "Hệ thống AI đang bận, vui lòng thử lại sau ít phút."

        return "Lỗi khi gọi Gemini API. Vui lòng thử lại sau."
