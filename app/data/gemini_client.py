"""
Data layer: Gemini client.

File này phụ trách:
- Gọi Gemini API thật bằng thư viện google-genai.
- Có mock_gemini() để test khi chưa có API key.
- Đảm bảo câu trả lời cuối cùng luôn có dòng nguồn.
"""

from __future__ import annotations

import html
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
    from google import genai
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
    """
    Trích câu hỏi người dùng từ prompt.
    """
    patterns = [
        r'CÂU HỎI CỦA NGƯỜI DÙNG:\s*"(?P<question>.*?)"',
        r'CÂU HỎI:\s*"(?P<question>.*?)"',
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
    Lấy tên tài liệu đầu tiên từ thuộc tính ten_tai_lieu trong thẻ <NGUON>.
    """
    match = re.search(r'ten_tai_lieu="([^"]+)"', prompt, flags=re.IGNORECASE)
    if match:
        return html.unescape(match.group(1).strip())

    return "Không có"


def _extract_first_source_from_answer(answer: str) -> str | None:
    """
    Lấy nguồn có sẵn từ câu trả lời nếu model đã trích dẫn.
    """
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


def _clean_context_text(text: str) -> str:
    """
    Làm sạch nội dung lấy từ thẻ <NGUON>.
    Dùng cho mock khi chưa có API key thật.
    """
    text = html.unescape(str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_context_blocks_from_prompt(prompt: str) -> list[str]:
    """
    Lấy nội dung thật trong các thẻ <NGUON>.
    """
    matches = re.findall(
        r'<NGUON\b(?=[^>]*\bten_tai_lieu="[^"]+")[^>]*>(.*?)</NGUON>',
        prompt,
        flags=re.DOTALL | re.IGNORECASE,
    )

    cleaned_blocks: list[str] = []

    for item in matches:
        clean_item = _clean_context_text(item)
        if clean_item:
            cleaned_blocks.append(clean_item)

    return cleaned_blocks


def mock_gemini(prompt: str) -> str:
    """
    Mock Gemini khi chưa có API key thật.

    Chỉ dùng khi GEMINI_API_KEY là mock-api-key/test/dummy hoặc chưa có key.
    Nếu đã nhập API key thật mà Gemini lỗi, call_gemini() sẽ trả lỗi thật,
    không âm thầm fallback sang mock.
    """
    question = _extract_question_from_prompt(prompt)
    source = _extract_first_source_from_prompt(prompt)
    context_blocks = _extract_context_blocks_from_prompt(prompt)

    if "<KHONG_CO_DU_LIEU>" in prompt or not context_blocks:
        return ensure_source_line(
            NO_DATA_MESSAGE,
            fallback_source="Không có",
        )

    context = context_blocks[0]
    context = re.sub(r"Trang\s+\d+\s*-+", " ", context, flags=re.IGNORECASE)
    context = re.sub(r"TÀI LIỆU HƯỚNG DẪN[^.]*", " ", context, flags=re.IGNORECASE)
    context = re.sub(r"\s+", " ", context).strip()

    step_matches = re.findall(
        r"(Bước\s+\d+[:：.]?.*?)(?=Bước\s+\d+[:：.]?|$)",
        context,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if step_matches:
        steps: list[str] = []

        for step in step_matches[:6]:
            clean_step = re.sub(r"\s+", " ", step).strip()

            if len(clean_step) > 350:
                clean_step = clean_step[:350].rstrip() + "..."

            steps.append(f"- {clean_step}")

        answer = (
            f"Quy trình liên quan đến câu hỏi \"{question}\" gồm các bước chính:\n\n"
            + "\n".join(steps)
        )
    else:
        answer = (
            f"Nội dung liên quan đến câu hỏi \"{question}\" trong tài liệu là:\n\n"
            f"{context[:1200].rstrip()}"
        )

    return ensure_source_line(answer, fallback_source=source)


def call_gemini(prompt: str) -> str:
    """
    Gọi Gemini API thật và trả về câu trả lời có trích dẫn nguồn.

    Nếu không có API key thật thì dùng mock_gemini().
    Nếu có API key thật nhưng Gemini báo lỗi, hàm trả lỗi rõ ràng để biết
    vấn đề nằm ở API key/model/quota, không giả vờ trả mock.
    """
    api_key = _get_config_value("GEMINI_API_KEY", "GOOGLE_API_KEY", default=None)
    model_name = _get_config_value("GEMINI_MODEL", default="gemini-2.5-flash")
    fallback_source = _extract_first_source_from_prompt(prompt)

    fake_keys = {
        "mock-api-key",
        "test",
        "dummy",
        "your_gemini_api_key",
        "your-api-key",
    }

    if not api_key or str(api_key).strip().lower() in fake_keys:
        logger.warning("Chưa cấu hình GEMINI_API_KEY thật. Đang dùng mock_gemini().")
        return mock_gemini(prompt)

    if genai is None:
        return ensure_source_line(
            "Chưa cài thư viện google-genai. Hãy chạy: pip install google-genai",
            fallback_source=fallback_source,
        )

    try:
        client = genai.Client(api_key=str(api_key).strip())

        response = client.models.generate_content(
            model=str(model_name).strip(),
            contents=prompt,
        )

        answer = getattr(response, "text", "") or ""

        if not answer.strip():
            return ensure_source_line(
                NO_DATA_MESSAGE,
                fallback_source=fallback_source,
            )

        return ensure_source_line(answer, fallback_source=fallback_source)

    except Exception as exc:
        error_message = str(exc)
        logger.exception("Lỗi khi gọi Gemini API: %s", exc)

        if "403" in error_message or "PERMISSION_DENIED" in error_message:
            return ensure_source_line(
                "Lỗi Gemini API: API key hoặc project không có quyền truy cập Gemini. "
                "Hãy kiểm tra lại API key trong Google AI Studio.",
                fallback_source=fallback_source,
            )

        if "404" in error_message or "not found" in error_message.lower():
            return ensure_source_line(
                f"Lỗi Gemini API: model '{model_name}' không tồn tại hoặc không được hỗ trợ. "
                "Hãy thử GEMINI_MODEL=gemini-2.5-flash.",
                fallback_source=fallback_source,
            )

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
            f"Lỗi khi gọi Gemini API: {error_message}",
            fallback_source=fallback_source,
        )


def ask_gemini(prompt: str) -> str:
    """
    Hàm tương thích với controller hiện tại.

    Controller đang gọi ask_gemini(prompt) theo kiểu đồng bộ,
    nên hàm này phải là def, không phải async def.
    """
    return call_gemini(prompt)