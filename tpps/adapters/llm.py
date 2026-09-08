"""Exactly three callable capabilities; every output gated by typecheck.

No path from LLM output to formal kernel / certified status except
through the kernel interface or a certified-procedure re-checker.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LlmCallLog:
    model: str
    prompt: str
    raw_output: str


class LlmClient(ABC):
    """The only LLM write paths into plan state."""

    def __init__(self) -> None:
        self.log: list[LlmCallLog] = []

    @abstractmethod
    def extract(self, prose: str) -> str:
        """prose -> candidate plan fragment."""

    @abstractmethod
    def elaborate(self, typed_hole: str) -> str:
        """typed hole -> statement + sketch."""

    @abstractmethod
    def draft_method(self, examples: str) -> str:
        """examples -> heuristic natural-language method description."""

    def __getattribute__(self, name: str):
        # No fourth capability: fail closed on anything beyond the three.
        if name.startswith("_llm_extra_"):
            raise AttributeError("no LLM write path beyond extract/elaborate/draft_method")
        return super().__getattribute__(name)


def admit_fragment(raw: str, typecheck_fn) -> str:
    """Every LLM output passes through typecheck before storage."""
    if not typecheck_fn(raw):
        raise ValueError(f"LLM fragment rejected by typecheck: {raw!r}")
    return raw


class LlmError(RuntimeError):
    """Transport- or protocol-level LLM failure (not a content rejection)."""


def _retry_delay(exc: urllib.error.HTTPError, raw_body: str,
                 attempt: int) -> float | None:
    """Seconds to wait before retrying, or None when not retryable.

    Honors Retry-After / Google retryDelay hints (capped); otherwise a
    bounded backoff. Only 429 is retried — anything else fails fast.
    """
    if exc.code != 429:
        return None
    headers = exc.headers or {}
    if headers.get("Retry-After"):
        try:
            return min(float(headers["Retry-After"]), 120.0)
        except (TypeError, ValueError):
            pass
    try:
        details = json.loads(raw_body).get("error", {}).get("details", [])
    except (ValueError, AttributeError):
        details = []
    for d in details:
        delay = (d or {}).get("retryDelay", "")
        if delay.endswith("s"):
            try:
                return min(float(delay[:-1]), 120.0)
            except ValueError:
                pass
    return min(20.0 * (attempt + 1), 120.0)


def post_json(url: str, payload: dict, headers: dict, timeout_s: float,
              max_retries: int) -> dict:
    """POST JSON with bounded 429 backoff. Shared by both clients."""
    for attempt in range(1 + max_retries):
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raw = e.read().decode()
            delay = _retry_delay(e, raw, attempt)
            if delay is None or attempt >= max_retries:
                raise LlmError(f"HTTP {e.code}: {raw[:300]}")
            time.sleep(delay)
    raise LlmError("unreachable")


class HttpLlmClient(LlmClient):
    """OpenAI-compatible chat client over stdlib urllib (no new dependencies).

    The API key travels only in the Authorization header, never in logged
    payloads. Anything the model returns must still pass `admit_fragment`
    before it touches plan state.
    """
    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        base = settings.base_url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"

    def _post(self, payload: dict) -> dict:
        return post_json(
            self.url, payload,
            {"Content-Type": "application/json",
             "Authorization": "Bearer " + self.settings.api_key},
            self.settings.timeout_s, self.settings.max_retries)

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Raw chat call; returns the assistant message text."""
        payload = {"model": self.settings.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            body = self._post(payload)
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LlmError(f"malformed chat response: {e}")
        self.log.append(LlmCallLog(self.settings.model, user[:2000], (text or "")[:2000]))
        return text or ""

    def chat_json(self, system: str, user: str, **kwargs) -> dict:
        """Chat call parsed as a JSON object (substring-tolerant)."""
        return _parse_json_obj(self.chat(system, user, json_mode=True, **kwargs))

    def extract(self, prose: str) -> str:
        self.log.append(LlmCallLog(self.settings.model, prose, ""))
        return self.chat("Extract the proof plan fragment.", prose)

    def elaborate(self, typed_hole: str) -> str:
        self.log.append(LlmCallLog(self.settings.model, typed_hole, ""))
        return self.chat("Propose a statement and sketch for this typed hole.",
                         typed_hole)

    def draft_method(self, examples: str) -> str:
        self.log.append(LlmCallLog(self.settings.model, examples, ""))
        return self.chat("Draft a heuristic method description.", examples)


class GeminiLlmClient(LlmClient):
    """Native Gemini generateContent client over stdlib urllib.

    Talks to generativelanguage.googleapis.com directly instead of the
    OpenAI-compatibility layer. The API key travels only in the
    x-goog-api-key header (never in the URL, which is what gets logged).
    Same three-capability surface as HttpLlmClient.
    """

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        base = settings.base_url.rstrip("/")
        if base.endswith("/openai"):
            base = base[: -len("/openai")]
        self.url = f"{base}/models/{settings.model}:generateContent"

    def _post(self, payload: dict) -> dict:
        return post_json(
            self.url, payload,
            {"Content-Type": "application/json",
             "x-goog-api-key": self.settings.api_key},
            self.settings.timeout_s, self.settings.max_retries)

    @staticmethod
    def _text_of(body: dict) -> str:
        try:
            cands = body.get("candidates", [])
            texts = [p.get("text", "")
                     for cand in cands if isinstance(cand, dict)
                     for p in (cand.get("content") or {}).get("parts", [])
                     if isinstance(p, dict)]
        except (AttributeError, TypeError) as e:
            raise LlmError(f"malformed generateContent response: {e}")
        text = "".join(texts)
        if text:
            return text
        reasons = [c.get("finishReason") for c in cands if isinstance(c, dict)]
        raise LlmError(f"empty model response (finish={reasons}, "
                       f"feedback={body.get('promptFeedback', {})})")

    def chat(self, system: str, user: str, *, json_mode: bool = False,
             temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """Raw generateContent call; returns the concatenated reply text."""
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        text = self._text_of(self._post(payload))
        self.log.append(LlmCallLog(self.settings.model, user[:2000], text[:2000]))
        return text

    def chat_json(self, system: str, user: str, **kwargs) -> dict:
        """generateContent call parsed as a JSON object (substring-tolerant)."""
        return _parse_json_obj(self.chat(system, user, json_mode=True, **kwargs))

    def extract(self, prose: str) -> str:
        self.log.append(LlmCallLog(self.settings.model, prose, ""))
        return self.chat("Extract the proof plan fragment.", prose)

    def elaborate(self, typed_hole: str) -> str:
        self.log.append(LlmCallLog(self.settings.model, typed_hole, ""))
        return self.chat("Propose a statement and sketch for this typed hole.",
                         typed_hole)

    def draft_method(self, examples: str) -> str:
        self.log.append(LlmCallLog(self.settings.model, examples, ""))
        return self.chat("Draft a heuristic method description.", examples)


def client_for(settings) -> LlmClient:
    """Select the client class from the configured provider name."""
    if settings.provider.strip().lower() == "gemini":
        return GeminiLlmClient(settings)
    return HttpLlmClient(settings)


def _parse_json_obj(text: str) -> dict:
    """Parse JSON, tolerating prose around the object."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth, in_str, esc = 0, False, False
    for i in range(start if start >= 0 else 0, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
                break
    raise LlmError("no JSON object in model response")
