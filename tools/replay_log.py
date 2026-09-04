#!/usr/bin/env python3
"""Replay a pipeline run: pretty-print its stage-by-stage trace.

Usage: replay_log.py <run_id> [--log-dir DIR] (defaults to the runs directory).
<run_id> may be a bare id, a file name, or a full path.
"""
import argparse
import json
import sys
from pathlib import Path

ORDER = ("extract", "typecheck", "evidence", "creative", "render", "roundtrip")


def load(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(value, indent=4):
    pad = " " * indent
    if isinstance(value, str) and "\n" not in value:
        return f"{pad}{value}"
    text = value if isinstance(value, str) else json.dumps(value, indent=2, default=str)
    return "\n".join(f"{pad}{line}" for line in text.splitlines())


def show_stage(records: list[dict]) -> None:
    stage = records[0]["stage"]
    starts = [r for r in records if r["event"] == "start"]
    ends = [r for r in records if r["event"] in ("end", "error")]
    head = ends[0] if ends else starts[0]
    dur = head.get("duration_ms")
    print(f"=== {stage} ({dur:.1f} ms)" if dur is not None else f"=== {stage}")
    for key, val in head.get("input_summary", {}).items():
        print(f"  in [{key}]:")
        print(fmt(val))
    if head["event"] == "error":
        err = head.get("error") or {}
        print(f"  ## ERROR {err.get('type')}: {err.get('message')}")
        return
    for key, val in (head.get("output_summary") or {}).items():
        if key == "llm_trace":
            continue
        print(f"  out [{key}]:")
        print(fmt(val))
    if stage == "roundtrip":
        out = head.get("output_summary") or {}
        if out.get("match"):
            print("  roundtrip: MATCH — re-extraction reproduced the plan")
        for d in out.get("diffs", []):
            print(f"  !! DIFF [{d.get('kind')}] {d}")
    trace = (head.get("output_summary") or {}).get("llm_trace")
    if trace:
        print("  out [llm_trace] (DEBUG):")
        print(fmt(trace))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay a TPPS pipeline run log.")
    ap.add_argument("run_id", help="run id, file name, or path to a .jsonl log")
    ap.add_argument("--log-dir", default="logs/runs")
    args = ap.parse_args()

    given = Path(args.run_id)
    if given.is_file():
        path = given
    elif (Path(args.log_dir) / args.run_id).is_file():
        path = Path(args.log_dir) / args.run_id
    elif (Path(args.log_dir) / f"{args.run_id}.jsonl").is_file():
        path = Path(args.log_dir) / f"{args.run_id}.jsonl"
    else:
        print(f"no log found for {args.run_id!r} under {args.log_dir}", file=sys.stderr)
        return 1

    records = load(path)
    by_stage: dict[str, list[dict]] = {}
    for r in records:
        by_stage.setdefault(r["stage"], []).append(r)
    print(f"RUN {records[0]['run_id']}  ({len(records)} events)  {path}\n")
    for stage in ORDER:
        if stage in by_stage:
            show_stage(by_stage[stage])
        else:
            print(f"=== {stage}\n  -- not reached (pipeline stopped early) --\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
