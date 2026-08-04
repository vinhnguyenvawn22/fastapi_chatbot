"""Local Qwen client.

The module keeps its old filename and function names so existing imports keep
working while requests are served locally by Ollama instead of Gemini.
"""

import base64
from contextvars import ContextVar
import json
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import (
    LLM_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TIMEOUT_SECONDS,
)


_gemini_call_state: ContextVar[dict] = ContextVar(
    "llm_call_state",
    default={"count": 0},
)


def reset_gemini_call_count() -> None:
    _gemini_call_state.set({"count": 0})


def get_gemini_call_count() -> int:
    return int(_gemini_call_state.get().get("count", 0))


def _increment_gemini_call_count() -> None:
    state = _gemini_call_state.get()
    _gemini_call_state.set({"count": int(state.get("count", 0)) + 1})


def _error_message(exc: Exception) -> str:
    return str(exc)


def _is_quota_or_rate_limit(error_message: str) -> bool:
    lowered = error_message.lower()
    return "429" in error_message or "rate limit" in lowered or "quota" in lowered


def _is_unavailable(error_message: str) -> bool:
    lowered = error_message.lower()
    return (
        "503" in error_message
        or "unavailable" in lowered
        or "connection refused" in lowered
        or "urlopen error" in lowered
        or "khong ket noi duoc ollama" in lowered
    )


def _contents_to_prompt(contents) -> str:
    if isinstance(contents, str):
        return contents
    if isinstance(contents, (list, tuple)):
        return "\n\n".join(str(item) for item in contents if isinstance(item, str))
    return str(contents or "")


def _ollama_generate(model: str, prompt: str, images: list[str] | None = None):
    selected_model = model or LLM_MODEL
    payload = {
        "model": selected_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": OLLAMA_NUM_PREDICT,
            "temperature": 0.2,
        },
    }
    if images:
        payload["images"] = images

    request = Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Khong ket noi duoc Ollama. Hay mo Ollama va chay: "
            f"ollama run {selected_model}"
        ) from exc

    if result.get("error"):
        raise RuntimeError(f"Ollama error: {result['error']}")
    return SimpleNamespace(text=str(result.get("response") or ""))


def generate_content(model: str, contents):
    """Generate text with Qwen through the local Ollama HTTP API."""
    _increment_gemini_call_count()
    return _ollama_generate(model=model, prompt=_contents_to_prompt(contents))


def _normalized_error_answer(error_message: str) -> str:
    if _is_quota_or_rate_limit(error_message):
        return "He thong AI tam thoi vuot gioi han su dung. Vui long thu lai sau it phut."
    if _is_unavailable(error_message):
        return "Khong ket noi duoc Qwen local. Hay mo Ollama va tai model truoc."
    if "timed out" in error_message.lower() or "timeout" in error_message.lower():
        return "Qwen local phan hoi qua thoi gian cho phep."
    return "Loi khi goi mo hinh Qwen local. Vui long thu lai sau."


def ask_gemini(prompt: str) -> str:
    """Backward-compatible name: send a text prompt to local Qwen."""
    try:
        response = generate_content(model=LLM_MODEL, contents=prompt)
        return response.text or "Khong tim thay can cu du ro trong tai lieu da cung cap."
    except Exception as exc:
        return _normalized_error_answer(_error_message(exc))


def ask_gemini_with_bytes(prompt: str, data: bytes, mime_type: str) -> str:
    """Send an image to a vision-capable Qwen model through Ollama."""
    try:
        if not str(mime_type).lower().startswith("image/"):
            raise ValueError("Qwen local chi nhan bytes hinh anh; hay trich xuat van ban truoc")
        _increment_gemini_call_count()
        response = _ollama_generate(
            model=LLM_MODEL,
            prompt=prompt,
            images=[base64.b64encode(data).decode("ascii")],
        )
        return response.text or ""
    except Exception as exc:
        return _normalized_error_answer(_error_message(exc))


# Provider-neutral aliases for new code. Legacy imports continue to work.
ask_llm = ask_gemini
ask_llm_with_bytes = ask_gemini_with_bytes
reset_llm_call_count = reset_gemini_call_count
get_llm_call_count = get_gemini_call_count
