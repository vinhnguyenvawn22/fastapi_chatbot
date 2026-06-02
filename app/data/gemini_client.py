"""
Data layer: Gemini client.

File này phụ trách:
- Gọi Gemini API bằng thư viện google-generativeai.
- Có mock_gemini() để test khi chưa có API key.
- Đảm bảo câu trả lời cuối cùng luôn có dòng nguồn.
"""

from __future__ import annotations

import logging
import os
import re
from importlib import import_module
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    import google.generativeai as genai
except Exception:
    genai = None  # type: ignore[assignment]

if load_dotenv:
    load_dotenv()

logger = logging.getLogger(__name__)

NO_DATA_MESSAGE = "Không tìm thấy nội dung phù hợp"


def _get_config_value(*names: str, default: Any = None) -> Any:
    """
    Lấy cấu hình ưu tiên từ app.core.config, sau đó mới đến biến môi trường.
    """
    try:
        config = import_module("app.core.config")
    except Exception:
        config = None

    for name in names:
        if config is not None and hasattr(config, name):
            value = getattr(config, name)
            if value not in (None, ""):
                return value

        env_value = os.getenv(name)
        if env_value not in (None, ""):
            return env_value

    return default


def _extract_question_from_prompt(prompt: str) -> str:
    """Trích câu hỏi người dùng từ prompt để dùng cho mock."""
    patterns = [
        r'CÂU HỎI CỦA NGƯỜI DÙNG:\s*"(?P<question>.*?)"',
        r"CÂU HỎI:\s*(?P<question>.*?)(?:\n\n|TRẢ LỜI:)",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, flags=re.DOTALL | re.IGNORECASE)
        if match:
            question = match.group("question").strip().strip('"')
            if question:
                return " ".join(question.split())

    return "câu hỏi của người dùng"


def _extract_first_source_from_prompt(prompt: str) -> str:
    """
    Lấy tên tài liệu đầu tiên trong thuộc tính ten_tai_lieu của thẻ <NGUON>.
    """
    match = re.search(r'ten_tai_lieu="([^"]+)"', prompt)
    if match:
        return match.group(1).strip()

    return "tài_liệu_giả"


def _extract_first_source_from_answer(answer: str) -> str | None:
    """Lấy nguồn có sẵn từ câu trả lời nếu model đã trích dẫn."""
    match = re.search(r"\(Nguồn:\s*([^)]+)\)", answer, flags=re.IGNORECASE)
    if match:
        source = match.group(1).strip()
        return source or None
    return None


def _remove_duplicate_source_lines(answer: str) -> str:
    """
    Xóa các dòng nguồn trùng để chỉ giữ đúng một dòng nguồn ở cuối.
    """
    lines = [line.rstrip() for line in str(answer or "").strip().splitlines()]
    content_lines = [
        line
        for line in lines
        if not re.fullmatch(r"\(Nguồn:\s*[^)]+\)", line.strip(), flags=re.IGNORECASE)
    ]
    return "\n".join(content_lines).strip()


def ensure_source_line(answer: str, fallback_source: str = "Không có") -> str:
    """
    Đảm bảo câu trả lời luôn kết thúc bằng đúng một dòng:
    (Nguồn: tên_tài_liệu)
    """
    answer = str(answer or "").strip()
    source = _extract_first_source_from_answer(answer) or fallback_source or "Không có"
    clean_answer = _remove_duplicate_source_lines(answer)

    if not clean_answer:
        clean_answer = NO_DATA_MESSAGE

    return f"{clean_answer}\n\n(Nguồn: {source})"


def _extract_first_context_from_prompt(prompt: str) -> str:
    """
    Lấy nội dung trong thẻ <NGUON> đầu tiên để mock trả lời theo tài liệu.
    """
    match = re.search(
        r"<NGUON[^>]*>(.*?)</NGUON>",
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not match:
        return ""

    context = match.group(1)
    context = re.sub(r"\s+", " ", context).strip()
    return context


def mock_gemini(prompt: str) -> str:
    """
    Giả lập Gemini khi chưa có API key thật.

    Thay vì trả 'Đây là câu trả lời mẫu...',
    hàm này lấy nội dung tìm được trong tài liệu để trả lời chi tiết hơn.
    """
    question = _extract_question_from_prompt(prompt)
    source = _extract_first_source_from_prompt(prompt)
    context = _extract_first_context_from_prompt(prompt)

    if "<KHONG_CO_DU_LIEU>" in prompt or not context:
        return ensure_source_line(
            "Không tìm thấy nội dung phù hợp",
            fallback_source="Không có",
        )

    answer = (
        f"Dựa trên tài liệu tìm được, nội dung liên quan đến câu hỏi "
        f"\"{question}\" như sau:\n\n"
        f"{context[:1500]}"
    )

    return ensure_source_line(answer, fallback_source=source)
    """
    Giả lập Gemini để test khi chưa có API key.

    Args:
        prompt: Prompt đã build từ build_prompt().

    Returns:
        Câu trả lời mẫu luôn có nguồn.
    """
    question = _extract_question_from_prompt(prompt)
    source = _extract_first_source_from_prompt(prompt)

    if "<KHONG_CO_DU_LIEU>" in prompt:
        return ensure_source_line(NO_DATA_MESSAGE, fallback_source="Không có")

    return ensure_source_line(
        f"Đây là câu trả lời mẫu cho: {question}",
        fallback_source=source,
    )

def call_gemini(prompt: str) -> str:
    """
    Gọi Gemini API và trả về câu trả lời có trích dẫn nguồn.

    Nếu chưa cấu hình GEMINI_API_KEY hoặc chưa cài google-generativeai,
    hàm sẽ dùng mock_gemini(prompt) để nhóm vẫn test được API.

    Args:
        prompt: Prompt hoàn chỉnh từ prompt_builder.build_prompt().

    Returns:
        Text trả lời từ Gemini, luôn có dòng nguồn ở cuối.
    """
    api_key = _get_config_value("GEMINI_API_KEY", "GOOGLE_API_KEY", default=None)
    model_name = _get_config_value("GEMINI_MODEL", default="gemini-1.5-flash")
    fallback_source = _extract_first_source_from_prompt(prompt)

    if not api_key or str(api_key).strip().lower() in {"mock-api-key", "test", "dummy", "your_gemini_api_key"}:
        logger.warning("Chưa cấu hình GEMINI_API_KEY thật. Đang dùng mock_gemini().")
        return mock_gemini(prompt)

    if genai is None:
        logger.warning("Chưa cài google-generativeai. Đang dùng mock_gemini().")
        return mock_gemini(prompt)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(str(model_name))
        response = model.generate_content(prompt)
        answer = getattr(response, "text", "") or ""
        return ensure_source_line(answer, fallback_source=fallback_source)

    except Exception as exc:
        error_message = str(exc)
        logger.exception("Lỗi khi gọi Gemini API: %s", exc)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            return ensure_source_line(
                "Hệ thống AI tạm thời vượt giới hạn sử dụng. Vui lòng thử lại sau ít phút.",
                fallback_source=fallback_source,
            )

        if "503" in error_message or "UNAVAILABLE" in error_message:
            return ensure_source_line(
                "Hệ thống AI đang bận, vui lòng thử lại sau ít phút.",
                fallback_source=fallback_source,
            )

        return ensure_source_line(
            "Lỗi khi gọi Gemini API. Vui lòng thử lại sau.",
            fallback_source=fallback_source,
        )


def call_gemini(prompt: str) -> str:
    """
    Gọi Gemini API và trả về câu trả lời có trích dẫn nguồn.
    Nếu chưa cấu hình GEMINI_API_KEY thật thì dùng mock_gemini().
    """
    api_key = _get_config_value("GEMINI_API_KEY", "GOOGLE_API_KEY", default=None)
    model_name = _get_config_value("GEMINI_MODEL", default="gemini-1.5-flash")
    fallback_source = _extract_first_source_from_prompt(prompt)

    fake_keys = {"mock-api-key", "test", "dummy", "your_gemini_api_key"}

    if not api_key or str(api_key).strip().lower() in fake_keys:
        logger.warning("Chưa cấu hình GEMINI_API_KEY thật. Đang dùng mock_gemini().")
        return mock_gemini(prompt)

    if genai is None:
        logger.warning("Chưa cài google-generativeai. Đang dùng mock_gemini().")
        return mock_gemini(prompt)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(str(model_name))
        response = model.generate_content(prompt)
        answer = getattr(response, "text", "") or ""
        return ensure_source_line(answer, fallback_source=fallback_source)

    except Exception as exc:
        error_message = str(exc)
        logger.exception("Lỗi khi gọi Gemini API: %s", exc)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:
            return ensure_source_line(
                "Hệ thống AI tạm thời vượt giới hạn sử dụng. Vui lòng thử lại sau ít phút.",
                fallback_source=fallback_source,
            )

        if "503" in error_message or "UNAVAILABLE" in error_message:
            return ensure_source_line(
                "Hệ thống AI đang bận, vui lòng thử lại sau ít phút.",
                fallback_source=fallback_source,
            )

        return ensure_source_line(
            "Lỗi khi gọi Gemini API. Vui lòng thử lại sau.",
            fallback_source=fallback_source,
        )


def ask_gemini(prompt: str) -> str:
    """
    Hàm tương thích với controller hiện tại.
    Controller đang gọi ask_gemini(prompt) theo kiểu đồng bộ,
    nên hàm này phải là def, không phải async def.
    """
    return call_gemini(prompt)