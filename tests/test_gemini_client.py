import itertools
import os
from types import SimpleNamespace


os.environ.setdefault("GEMINI_API_KEY", "test-gemini-api-key")

from app.data import gemini_client


class _FakeModels:
    def __init__(self, api_key, calls, failures):
        self.api_key = api_key
        self.calls = calls
        self.failures = failures

    def generate_content(self, model, contents):
        self.calls.append((self.api_key, model, contents))
        error = self.failures.get(self.api_key)
        if error:
            raise RuntimeError(error)
        return SimpleNamespace(text=f"ok:{self.api_key}")


class _FakeClient:
    def __init__(self, api_key, calls, failures):
        self.models = _FakeModels(api_key, calls, failures)


def test_generate_content_fails_over_to_next_key(monkeypatch):
    calls = []
    failures = {"key-1": "429 RESOURCE_EXHAUSTED"}

    monkeypatch.setattr(gemini_client, "GEMINI_API_KEYS", ["key-1", "key-2"])
    monkeypatch.setattr(gemini_client, "_key_counter", itertools.count())
    monkeypatch.setattr(
        gemini_client,
        "_client_for_key",
        lambda api_key: _FakeClient(api_key, calls, failures),
    )

    response = gemini_client.generate_content("model", "prompt")

    assert response.text == "ok:key-2"
    assert [call[0] for call in calls] == ["key-1", "key-2"]


def test_generate_content_rotates_starting_key(monkeypatch):
    calls = []

    monkeypatch.setattr(gemini_client, "GEMINI_API_KEYS", ["key-1", "key-2"])
    monkeypatch.setattr(gemini_client, "_key_counter", itertools.count())
    monkeypatch.setattr(
        gemini_client,
        "_client_for_key",
        lambda api_key: _FakeClient(api_key, calls, {}),
    )

    first = gemini_client.generate_content("model", "first")
    second = gemini_client.generate_content("model", "second")

    assert first.text == "ok:key-1"
    assert second.text == "ok:key-2"
    assert [call[0] for call in calls] == ["key-1", "key-2"]
