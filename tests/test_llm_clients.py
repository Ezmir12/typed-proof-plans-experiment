"""Native Gemini client + provider dispatch tests (no network).

HTTP is stubbed at urlopen; these pin URL shape, auth placement, response
parsing, and client selection.
"""
import io
import json
import urllib.error
from types import SimpleNamespace

from tpps.adapters.llm import (GeminiLlmClient, HttpLlmClient, LlmError,
                               client_for)


def _settings(**over):
    base = {"provider": "gemini",
            "model": "gemini-2.5-flash",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "api_key": "test-key",
            "timeout_s": 60.0,
            "max_retries": 2}
    base.update(over)
    return SimpleNamespace(**base)


class FakeResp:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._data


NATIVE_OK = {"candidates": [{"content": {"parts": [{"text": '{"a": 1}'}]}}]}


def _stub(monkeypatch, payload, captured):
    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        return FakeResp(payload)
    monkeypatch.setattr("tpps.adapters.llm.urllib.request.urlopen", fake_urlopen)


def test_native_url_normalized(monkeypatch):
    for base in ("https://generativelanguage.googleapis.com/v1beta/openai",
                 "https://generativelanguage.googleapis.com/v1beta",
                 "https://generativelanguage.googleapis.com/v1beta/"):
        c = GeminiLlmClient(_settings(base_url=base))
        assert c.url == ("https://generativelanguage.googleapis.com/v1beta"
                         "/models/gemini-2.5-flash:generateContent")


def test_native_chat_parses_and_auth_headers(monkeypatch):
    captured = {}
    _stub(monkeypatch, NATIVE_OK, captured)
    text = GeminiLlmClient(_settings()).chat("sys", "hi")
    assert text == '{"a": 1}'
    assert captured["headers"].get("X-goog-api-key") == "test-key"
    assert "key=" not in captured["url"]  # key never in the URL
    assert captured["body"]["systemInstruction"]["parts"][0]["text"] == "sys"
    assert captured["body"]["contents"][0]["parts"][0]["text"] == "hi"


def test_native_chat_json_requests_json_mime(monkeypatch):
    captured = {}
    _stub(monkeypatch, NATIVE_OK, captured)
    assert GeminiLlmClient(_settings()).chat_json("sys", "hi") == {"a": 1}
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_native_malformed_raises(monkeypatch):
    captured = {}
    _stub(monkeypatch, {"nope": []}, captured)
    try:
        GeminiLlmClient(_settings()).chat("sys", "hi")
        raise AssertionError("expected LlmError")
    except LlmError:
        pass


def test_client_for_dispatch():
    assert isinstance(client_for(_settings(provider="gemini")), GeminiLlmClient)
    assert isinstance(client_for(_settings(provider=" Gemini ")), GeminiLlmClient)
    assert isinstance(client_for(_settings(provider="openai-compatible")), HttpLlmClient)
    assert isinstance(client_for(_settings(provider="deepinfra")), HttpLlmClient)


def test_http_client_stays_openai_shaped(monkeypatch):
    """Regression: HttpLlmClient must never emit a hybrid OpenAI-body-to-
    native-URL request, whatever the provider string says."""
    captured = {}
    _stub(monkeypatch, {"choices": [{"message": {"content": "hi"}}]}, captured)
    text = HttpLlmClient(_settings(provider="gemini")).chat("sys", "hi")
    assert text == "hi"
    assert captured["url"].endswith("/chat/completions")
    assert ":generateContent" not in captured["url"]
    assert captured["headers"].get("Authorization") == "Bearer test-key"
    assert "x-goog-api-key" not in captured["headers"]
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["body"]["temperature"] == 0.1
    assert captured["body"]["max_tokens"] == 2000


def _http_error(code: int, body: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "reason", {},
                                  io.BytesIO(json.dumps(body).encode()))


def test_429_backoff_then_success(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        if len(calls) == 1:
            raise _http_error(429, {"error": {"details": [{"retryDelay": "0s"}]}})
        return FakeResp(NATIVE_OK)

    monkeypatch.setattr("tpps.adapters.llm.urllib.request.urlopen", fake_urlopen)
    assert GeminiLlmClient(_settings()).chat_json("sys", "hi") == {"a": 1}
    assert len(calls) == 2


def test_non_429_fails_fast_without_retry(monkeypatch):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)
        raise _http_error(400, {"error": {"message": "bad"}})

    monkeypatch.setattr("tpps.adapters.llm.urllib.request.urlopen", fake_urlopen)
    try:
        GeminiLlmClient(_settings()).chat("sys", "hi")
        raise AssertionError("expected LlmError")
    except LlmError as e:
        assert "HTTP 400" in str(e)
    assert len(calls) == 1
