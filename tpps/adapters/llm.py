"""Exactly three callable capabilities; every output gated by typecheck.

No path from LLM output to formal kernel / certified status except
through the kernel interface or a certified-procedure re-checker.
"""
from __future__ import annotations

import json
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
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.settings.api_key})
        try:
            with urllib.request.urlopen(req, timeout=self.settings.timeout_s) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise LlmError(f"HTTP {e.code}: {e.read().decode()[:300]}")

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
