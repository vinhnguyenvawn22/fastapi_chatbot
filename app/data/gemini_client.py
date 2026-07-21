from contextvars import ContextVar
from itertools import count
from threading import Lock

from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEYS, GEMINI_MODEL


_clients: dict[str, genai.Client] = {}
_client_lock = Lock()
_key_counter = count()
_gemini_call_state: ContextVar[dict] = ContextVar(
    "gemini_call_state",
    default={"count": 0},
)


def reset_gemini_call_count() -> None:
    _gemini_call_state.set({"count": 0})


def get_gemini_call_count() -> int:
    return int(_gemini_call_state.get().get("count", 0))


def _increment_gemini_call_count() -> None:
    state = _gemini_call_state.get()
    _gemini_call_state.set({"count": int(state.get("count", 0)) + 1})


def _client_for_key(api_key: str) -> genai.Client:
    with _client_lock:
        client = _clients.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key)
            _clients[api_key] = client
        return client


def _ordered_api_keys() -> list[str]:
    if len(GEMINI_API_KEYS) <= 1:
        return list(GEMINI_API_KEYS)

    start = next(_key_counter) % len(GEMINI_API_KEYS)
    return GEMINI_API_KEYS[start:] + GEMINI_API_KEYS[:start]


def _error_message(exc: Exception) -> str:
    return str(exc)


def _is_quota_or_rate_limit(error_message: str) -> bool:
    return (
        "429" in error_message
        or "RESOURCE_EXHAUSTED" in error_message
        or "rate limit" in error_message.lower()
        or "quota" in error_message.lower()
    )


def _is_unavailable(error_message: str) -> bool:
    return "503" in error_message or "UNAVAILABLE" in error_message


def generate_content(model: str, contents):
    """Call Gemini with round-robin keys and fail over on quota/unavailable errors."""
    last_error = None

    for api_key in _ordered_api_keys():
        _increment_gemini_call_count()
        try:
            return _client_for_key(api_key).models.generate_content(
                model=model,
                contents=contents,
            )
        except Exception as exc:
            message = _error_message(exc)
            last_error = message
            if _is_quota_or_rate_limit(message) or _is_unavailable(message):
                continue
            raise

    raise RuntimeError(last_error or "Gemini API unavailable")


def _normalized_error_answer(error_message: str) -> str:
    if _is_quota_or_rate_limit(error_message):
        return "He thong AI tam thoi vuot gioi han su dung. Vui long thu lai sau it phut."

    if _is_unavailable(error_message):
        return "He thong AI dang ban, vui long thu lai sau it phut."

    return "Loi khi goi Gemini API. Vui long thu lai sau."


def ask_gemini(prompt: str) -> str:
    """Send a text prompt to Gemini and normalize common API errors."""
    try:
        response = generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        return response.text or "Khong tim thay can cu du ro trong tai lieu da cung cap."

    except Exception as exc:
        return _normalized_error_answer(_error_message(exc))


def ask_gemini_with_bytes(prompt: str, data: bytes, mime_type: str) -> str:
    """Send a small document/image to Gemini and return the model response."""
    try:
        response = generate_content(
            model=GEMINI_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=data, mime_type=mime_type),
            ],
        )

        return response.text or ""
    except Exception as exc:
        return _normalized_error_answer(_error_message(exc))
