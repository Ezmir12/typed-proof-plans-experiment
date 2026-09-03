"""Exactly three callable capabilities; every output gated by typecheck.

No path from LLM output to formal kernel / certified status except
through the kernel interface or a certified-procedure re-checker.
"""
from __future__ import annotations

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
