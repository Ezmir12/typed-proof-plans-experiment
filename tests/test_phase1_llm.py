"""Phase 1 graded acceptance: real LlmExtractor, no hand-built plans.

Skipped by default (slow, costs real money): set TPPS_RUN_LIVE_TESTS=1 with
a funded backend key to run. These are release gates, not unit tests.
The offline contract machinery is covered by scripted-client unit tests.
"""
import os
from pathlib import Path

import pytest

from tpps.adapters.llm import client_for
from tpps.adapters.llm_extract import LlmExtractor
from tpps.adapters.vocab import export_vocabulary
from tpps.config import load_config
from tpps.core import typecheck as T
from tpps.domains import rewriting as RW
from tpps.examples import example_registry
from tpps.pipeline import PipelineHooks, run_pipeline

FIX = Path(__file__).resolve().parent.parent / "files"
HAS_LLM = bool(load_config().llm.api_key)
RUN_LIVE = os.environ.get("TPPS_RUN_LIVE_TESTS") == "1" and HAS_LLM

pytestmark = pytest.mark.skipif(
    not RUN_LIVE,
    reason="live LLM tests skipped (TPPS_RUN_LIVE_TESTS=1 + funded key to run)")

T01_NOVEL = ("The cleanup sweeps finish on any finite directory tree, because "
             "each sweep deletes at least one temporary file, creates nothing "
             "new, and only runs while temporary files remain.")
NEGATIVE = "Remember to buy milk, eggs, and bread on the way home."


def _fixture_nl(name: str, section: str) -> str:
    """Pull the quoted NL block out of a fixture file (never pasted here)."""
    text = (FIX / name).read_text()
    lines = []
    in_section = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_section = line.strip("# ").lower().startswith(section.lower())
        elif in_section and line.startswith("> "):
            lines.append(line[2:])
    assert lines, f"no quoted NL under {section!r} in {name}"
    return "\n".join(lines)


def _live_hooks():
    settings = load_config()
    reg = example_registry()
    vocab = export_vocabulary(reg, RW.sort_library(), list(RW.ROLES))
    ext = LlmExtractor(client_for(settings.llm), reg, vocab)
    hooks = PipelineHooks(extract_fn=ext.extract,
                          feedback_fn=ext.extract_with_diagnostic,
                          registry_fn=example_registry)
    return hooks, reg


def _assert_taint_checklist(plan, reg, label: str) -> str:
    """Checklist items 1-7 from the reference fixture. Returns target id."""
    assert len(plan.insts) == 1, f"{label}: want exactly one instance"
    inst = next(iter(plan.insts.values()))
    assert inst.schema_ref == "WFMeasure", f"{label}: want WFMeasure"
    prem_heads = {plan.stmts[p].stmt.shape.head for p in inst.premises}
    assert {"Decreases", "WF"} <= prem_heads, f"{label}: both premises, got {prem_heads}"
    targets = [n for n in plan.stmts.values()
               if n.stmt.shape.head == "Terminates"]
    assert targets, f"{label}: want a Terminates target"
    assert any(r.role_name == "RankingFunction" for r in plan.roles), \
        f"{label}: want a RankingFunction role"
    obl_heads = {plan.stmts[o].stmt.shape.head for o in inst.obligations}
    assert obl_heads == {"Bounded", "NoRemove", "Productive"}, \
        f"{label}: three obligations, got {obl_heads}"
    assert all(plan.stmts[o].evidence.formal.name == "NONE" for o in inst.obligations), \
        f"{label}: obligations must stay open"
    assert T.check_plan(plan, reg).ok, f"{label}: must be well-typed"
    concl = plan.stmts[inst.conclusion]
    assert concl.stmt.shape.head == "Terminates"
    return inst.conclusion


def _assert_spans_grounded(plan, nl_text: str, label: str) -> None:
    for n in plan.stmts.values():
        span = n.provenance.source_span
        assert span and span in nl_text, \
            f"{label}: node {n.id} span not grounded in input"


def test_t01_novel_termination(tmp_path):
    hooks, reg = _live_hooks()
    res = run_pipeline(T01_NOVEL, log_dir=str(tmp_path), hooks=hooks)
    assert res.accepted, f"candidate rejected; see {res.log_path}"
    _assert_taint_checklist(res.plan, reg, "T0.1")
    _assert_spans_grounded(res.plan, T01_NOVEL, "T0.1")


def test_t11_reference_plan(tmp_path):
    from tpps.adapters.render import Register, render
    from tpps.pipeline import DEFAULT_REQUIREMENT
    hooks, reg = _live_hooks()
    nl = _fixture_nl("taint_propagation_reference.md", "Reference NL")
    res = run_pipeline(nl, log_dir=str(tmp_path), hooks=hooks)
    assert res.accepted, f"candidate rejected; see {res.log_path}"
    target = _assert_taint_checklist(res.plan, reg, "T1.1")
    _assert_spans_grounded(res.plan, nl, "T1.1")
    text = render(res.plan, target, DEFAULT_REQUIREMENT, Register.TEXTBOOK)
    assert "remains to show" in text  # open obligations honestly rendered


def test_t21_reworded_generalizes(tmp_path):
    hooks, reg = _live_hooks()
    nl = _fixture_nl("taint_propagation_reworded.md", "Reworded NL input")
    res = run_pipeline(nl, log_dir=str(tmp_path), hooks=hooks)
    assert res.accepted, f"candidate rejected; see {res.log_path}"
    inst = next(iter(res.plan.insts.values()))
    if (inst.schema_ref == "FiniteActiveDomainSaturation"
            and not inst.obligations and T.check_plan(res.plan, reg).ok):
        print("T2.1 preferred path: one-instance FiniteActiveDomainSaturation")
    else:
        _assert_taint_checklist(res.plan, reg, "T2.1-minimum")
        _assert_spans_grounded(res.plan, nl, "T2.1-minimum")
        print("T2.1 minimum path: WFMeasure skeleton")


def test_negative_declines_or_rejected(tmp_path):
    hooks, reg = _live_hooks()
    res = run_pipeline(NEGATIVE, log_dir=str(tmp_path), hooks=hooks)
    assert res.accepted is False
    if res.plan is not None:
        assert not any(n.stmt.shape.head == "Terminates"
                       for n in res.plan.stmts.values()), \
            "fabricated a Terminates target from a shopping list"
