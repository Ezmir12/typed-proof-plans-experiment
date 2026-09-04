"""Extractor routing: the flag selects the extractor, unknown modes fail loud.

Retry policy (documented here, per the Phase 1 contract): retries are driven
by checker/transport rejection, never by the confidence number. A
low-confidence but well-typed candidate is accepted without retry.
"""
import pytest

from tpps.adapters.extract import CandidatePlan, ExtractedNode
from tpps.config import load_config
from tpps.core.plan import Plan, Provenance, StmtNode
from tpps.core.stmt import Shape, Stmt
from tpps.examples import example_registry
from tpps.pipeline import PipelineHooks, _resolve_extractor, run_pipeline


def _settings(monkeypatch, tmp_path, extractor):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TPPS_EXTRACTOR", extractor)
    return load_config()


def test_llm_flag_selects_llm_extractor(tmp_path, monkeypatch):
    mode, label, attempt, feedback, trace = _resolve_extractor(
        PipelineHooks(), example_registry(),
        _settings(monkeypatch, tmp_path, "llm"))
    assert mode == "llm"
    assert label == "LlmExtractor"
    assert feedback is True


def test_demo_flag_selects_demo_extractor(tmp_path, monkeypatch):
    mode, label, *_ = _resolve_extractor(
        PipelineHooks(), example_registry(),
        _settings(monkeypatch, tmp_path, "demo"))
    assert (mode, label) == ("demo", "DemoExtractor")


def test_explicit_hooks_win_over_flag(tmp_path, monkeypatch):
    hooks = PipelineHooks(extract_fn=lambda text: None)
    mode, label, *_ = _resolve_extractor(
        hooks, example_registry(), _settings(monkeypatch, tmp_path, "llm"))
    assert mode == "hooks"
    assert label.startswith("hooks:")


def test_unknown_mode_fails_before_logging(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="unknown extractor"):
        run_pipeline("anything", log_dir=str(tmp_path),
                     extractor="bogus",
                     settings=_settings(monkeypatch, tmp_path, "llm"))
    assert list(tmp_path.iterdir()) == []


def test_low_confidence_without_rejection_does_not_retry(tmp_path):
    """0.4 confidence, well-typed candidate: accepted, single attempt."""
    plan = Plan("lowconf")
    plan.add_stmt(StmtNode("claim", Stmt(shape=Shape("Claim", ())),
                           provenance=Provenance(source_span="t",
                                                 extraction_confidence=0.4)))
    fixed = CandidatePlan(
        [ExtractedNode(plan.stmts["claim"].stmt.shape, "t", 0.4,
                       "claim", "mock")], plan)
    hooks = PipelineHooks(extract_fn=lambda text: fixed,
                          registry_fn=example_registry,
                          creative_trigger_fn=lambda *args: None)
    res = run_pipeline("low confidence claim", log_dir=str(tmp_path),
                       hooks=hooks)
    assert res.accepted is True
    import json
    recs = [json.loads(line) for line in open(res.log_path) if line.strip()]
    extracts = [r for r in recs if r["stage"] == "extract"]
    assert len(extracts) == 2  # exactly one start+end pair
    end = next(r for r in extracts if r["event"] == "end")
    assert end["output_summary"]["extraction_confidence"] == 0.4
    assert end["output_summary"]["will_retry"] is False
