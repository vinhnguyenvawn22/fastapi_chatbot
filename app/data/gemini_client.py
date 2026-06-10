from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL


client = genai.Client(api_key=GEMINI_API_KEY)


def ask_gemini(prompt: str) -> str:
    """Send a text prompt to Gemini and normalize common API errors."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        return response.text or "Khong tim thay can cu du ro trong tai lieu da cung cap."

    except Exception as e:
        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            return "He thong AI tam thoi vuot gioi han su dung. Vui long thu lai sau it phut."

        if "503" in error_message or "UNAVAILABLE" in error_message:
            return "He thong AI dang ban, vui long thu lai sau it phut."

        return "Loi khi goi Gemini API. Vui long thu lai sau."


def ask_gemini_with_bytes(prompt: str, data: bytes, mime_type: str) -> str:
    """Send a small document/image to Gemini and return the model response."""
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=data, mime_type=mime_type),
            ],
        )

        return response.text or ""
    except Exception as e:
        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            return "He thong AI tam thoi vuot gioi han su dung. Vui long thu lai sau it phut."

        if "503" in error_message or "UNAVAILABLE" in error_message:
            return "He thong AI dang ban, vui long thu lai sau it phut."

        return "Loi khi goi Gemini API. Vui long thu lai sau."
