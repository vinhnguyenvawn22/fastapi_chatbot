import json

from app.data import gemini_client


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_generate_content_calls_local_ollama(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(
            {
                "url": request.full_url,
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        return _FakeResponse({"response": "xin chao"})

    monkeypatch.setattr(gemini_client, "urlopen", fake_urlopen)
    response = gemini_client.generate_content("qwen3:4b", "prompt test")

    assert response.text == "xin chao"
    assert calls[0]["url"].endswith("/api/generate")
    assert calls[0]["body"] == {
        "model": "qwen3:4b",
        "prompt": "prompt test",
        "stream": False,
        "options": {
            "num_predict": gemini_client.OLLAMA_NUM_PREDICT,
            "temperature": 0.2,
        },
    }


def test_ask_gemini_is_backward_compatible_qwen_alias(monkeypatch):
    monkeypatch.setattr(
        gemini_client,
        "generate_content",
        lambda model, contents: _FakeTextResponse("tra loi local"),
    )

    assert gemini_client.ask_gemini("cau hoi") == "tra loi local"
    assert gemini_client.ask_llm("cau hoi") == "tra loi local"


class _FakeTextResponse:
    def __init__(self, text):
        self.text = text
