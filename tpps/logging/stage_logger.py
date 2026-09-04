"""Stage-by-stage run logging: one JSONL file per run, streamable and replayable.

Every stage emits a start record on entry and exactly one end (or error)
record on exit. Secrets are redacted before anything hits disk.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

STAGES = ("extract", "typecheck", "evidence", "creative", "render", "roundtrip")

_REDACT_KEY = re.compile(r"api[_-]?key|authorization|bearer", re.IGNORECASE)
_REDACT_KEYVAL = re.compile(r"(?i)\b(api[_-]?key|authorization|bearer)\b\s*[:=]\s*\S+")
_REDACT_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/=]+")
REDACTED = "***REDACTED***"


def redact(obj):
    """Drop secrets: sensitive keys masked, key=value pairs and bearer
    tokens scrubbed even when embedded in larger strings."""
    if isinstance(obj, dict):
        return {k: (REDACTED if _REDACT_KEY.search(str(k)) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        obj = _REDACT_KEYVAL.sub(r"\1=***REDACTED***", obj)
        return _REDACT_BEARER.sub(f"Bearer {REDACTED}", obj)
    return obj


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"


class _StageCtx:
    def __init__(self, logger: StageLogger, stage: str, input_summary: dict):
        self._logger = logger
        self._stage = stage
        self._input = input_summary or {}
        self._output: dict | None = None
        self._debug: dict | None = None
        self._t0 = time.perf_counter()

    def set_output(self, output_summary: dict) -> None:
        self._output = output_summary

    def set_debug(self, debug_payload: dict) -> None:
        self._debug = debug_payload

    def note(self, key: str, value: object) -> None:
        """Add an input fact discovered mid-stage (e.g. trigger reason)."""
        self._input[key] = value


class StageLogger:
    """Writes one JSONL run log per run_id. Level from TPPS_LOG_LEVEL (INFO default).

    At INFO only the stage summaries are stored; at DEBUG the raw LLM
    prompt/response payloads passed via set_debug are stored too (redacted).
    """

    def __init__(self, log_dir: str | Path = "logs/runs", level: str | None = None,
                 run_id: str | None = None) -> None:
        self.run_id = run_id or make_run_id()
        self.path = Path(log_dir) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.level = (level or os.environ.get("TPPS_LOG_LEVEL", "INFO")).upper()
        self._debug_enabled = self.level == "DEBUG"

    def _write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(redact(record), default=str) + "\n")

    def _record(self, stage: str, event: str, input_summary: dict | None,
                output_summary: dict | None, duration_ms: float | None,
                debug_payload: dict | None = None,
                error: dict | None = None) -> None:
        record = {"run_id": self.run_id, "stage": stage, "event": event,
                  "timestamp": _now_iso(), "duration_ms": duration_ms,
                  "input_summary": input_summary or {},
                  "output_summary": output_summary,
                  "error": error}
        if event == "end" and self._debug_enabled and debug_payload:
            record["output_summary"] = {**(output_summary or {}),
                                        "llm_trace": debug_payload}
        self._write(record)

    @contextmanager
    def stage(self, stage: str, input_summary: dict | None = None):
        """With-block wrapper: logs start on entry, end (or error) on exit."""
        ctx = _StageCtx(self, stage, input_summary or {})
        self._record(stage, "start", ctx._input, None, None)
        try:
            yield ctx
        except Exception as exc:  # noqa: BLE001 - logged, then re-raised
            self._record(stage, "error", ctx._input, None,
                         (time.perf_counter() - ctx._t0) * 1000,
                         error={"type": type(exc).__name__, "message": str(exc)})
            raise
        self._record(stage, "end", ctx._input, ctx._output,
                     (time.perf_counter() - ctx._t0) * 1000, ctx._debug)
