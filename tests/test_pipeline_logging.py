"""Pipeline logging tests: record order, rejection flow, secret redaction."""
import json

from tpps.adapters.extract import CandidatePlan, ExtractedNode
from tpps.core.plan import MethodInstance, Plan, Provenance, StmtNode
from tpps.core.schema import MethodSchema, SchemaRegistry
from tpps.core.stmt import Shape, Stmt, Var
from tpps.pipeline import PipelineHooks, run_pipeline

STAGES = ("extract", "typecheck", "evidence", "creative", "render", "roundtrip")


def _mock_registry() -> SchemaRegistry:
    reg = SchemaRegistry()
    reg.register(MethodSchema(
        id="Step", premises=[Shape("ClaimA", (Var("x"),))],
        conclusion=Shape("ClaimB", (Var("x"),)), conditions=[],
        checker=lambda ctx: True, elaborator="elab-Step",
        soundness_theorem="Thm-Step"))
    reg.promote("Step", kernel_verified=True)
    return reg


def _mock_candidate(conclusion: str = "ClaimB") -> CandidatePlan:
    plan = Plan("mock")
    for nid, head in (("a", "ClaimA"), ("b", conclusion)):
        plan.add_stmt(StmtNode(nid, Stmt(shape=Shape(head, (Var("x"),))),
                               provenance=Provenance(
                                   source_span="mock text",
                                   extraction_confidence=0.9)))
    plan.add_inst(MethodInstance("e", "Step", ["a"], "b"))
    nodes = [ExtractedNode(n.stmt.shape, "mock text", 0.9,
                           f"claim {n.stmt.shape.head}", "mock mapping")
             for n in plan.stmts.values()]
    return CandidatePlan(nodes, plan)


def _mock_hooks(conclusion: str = "ClaimB", trace: dict | None = None) -> PipelineHooks:
    fixed = _mock_candidate(conclusion)
    return PipelineHooks(
        extract_fn=lambda text: fixed,
        extract_trace_fn=(lambda: trace) if trace is not None else None,
        registry_fn=_mock_registry,
        creative_trigger_fn=lambda *args: "user_requested")


def _records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_one_start_end_pair_per_stage_in_order(tmp_path):
    res = run_pipeline("trivial input", log_dir=str(tmp_path),
                       hooks=_mock_hooks())
    recs = _records(res.log_path)
    assert [(r["stage"], r["event"]) for r in recs] == [
        (s, e) for s in STAGES for e in ("start", "end")]
    assert {r["run_id"] for r in recs} == {res.run_id}
    assert res.log_path.endswith(f"{res.run_id}.jsonl")
    for r in recs:
        assert (r["duration_ms"] is None) == (r["event"] == "start")
        assert r["error"] is None
    assert res.accepted and res.plan is not None


def test_ill_typed_plan_logs_rejection_and_continues(tmp_path):
    res = run_pipeline("bad input", log_dir=str(tmp_path),
                       hooks=_mock_hooks(conclusion="Bogus"))
    recs = _records(res.log_path)
    by_stage = {r["stage"]: r for r in recs if r["event"] == "end"}
    assert by_stage["typecheck"]["output_summary"]["well_typed"] is False
    assert by_stage["typecheck"]["output_summary"]["diagnostics"]
    # Rejection is logged downstream, never a silent stop.
    for stage in ("evidence", "render", "roundtrip"):
        assert stage in by_stage
    assert res.accepted is False and res.plan is not None


def test_api_key_never_reaches_disk(tmp_path, monkeypatch):
    secret = "sk-test-secret-123"
    trace = {"prompt": "prove it",
             "response": f"plan fragment; api_key={secret}"}
    monkeypatch.setenv("TPPS_LOG_LEVEL", "DEBUG")
    res = run_pipeline("sensitive", log_dir=str(tmp_path),
                       hooks=_mock_hooks(trace=trace))
    text = open(res.log_path, encoding="utf-8").read()
    assert secret not in text
    assert "***REDACTED***" in text

    monkeypatch.delenv("TPPS_LOG_LEVEL")
    res2 = run_pipeline("sensitive", log_dir=str(tmp_path),
                        hooks=_mock_hooks(trace=trace))
    recs = _records(res2.log_path)
    assert all("llm_trace" not in (r.get("output_summary") or {})
               for r in recs if r["event"] == "end")
