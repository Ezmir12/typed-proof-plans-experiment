"""Tests for multi-register rendering and research evaluation protocols."""
import json
import pytest

from tpps.adapters.render import Register, lint_rendered, render
from tpps.core.evidence import Evidence, Formal
from tpps.core.plan import Layer, MethodInstance, Plan, StmtNode
from tpps.core.stmt import Shape, Stmt, Var
from tpps.eval.reports import export_metrics_json, generate_markdown_report
from tpps.eval.runners import (
    EvalReport,
    run_all_evaluations,
    run_auxiliary_synthesis_eval,
    run_drift_detection_eval,
    run_extraction_fidelity_eval,
    run_induction_ablation_eval,
    run_negative_transfer_eval,
    run_rendering_fidelity_eval,
)
from tpps.domains.rewriting import strategy_independence_cast_plan
from tpps.examples import fta_plan

CANONICAL_LEAN4_ENVIRONMENT_HASH = "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069"


# --- Rendering Tests ---


def test_render_textbook_vs_annotated_gating():
    plan = Plan("gated-register-derivation")
    plan.add_stmt(StmtNode("target", Stmt(shape=Shape("Theorem"))))
    requirement = Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH)

    # When target is NOT closed, textbook register is safely gated to annotated
    text_open = render(plan, "target", requirement, register=Register.TEXTBOOK)
    assert "Status of target: OPEN: obligations remain" in text_open

    # When target IS closed with required formal evidence, textbook register is allowed
    plan.stmts["target"].evidence = Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH)
    text_closed = render(plan, "target", requirement, register=Register.TEXTBOOK)
    assert "Status of target" not in text_closed
    assert "We show Theorem." in text_closed


def test_render_discourse_templates():
    plan = fta_plan()
    # Seed base axioms with kernel evidence so plan closes
    plan.stmts["poly"].evidence = Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH)
    plan.stmts["no-root"].evidence = Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH)
    plan.stmts["holo-g"].evidence = Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH)
    plan.stmts["bounded-g"].evidence = Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH)
    plan.stmts["const-g"].evidence = Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH)

    rendered = render(
        plan,
        "const-g",
        Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH),
        Register.ANNOTATED,
    )

    # Verify discourse elements appear
    assert "Consider the auxiliary function" in rendered
    assert "Suppose, for contradiction" in rendered


def test_render_bypass_cast_warning_never_smoothed():
    plan = strategy_independence_cast_plan()
    rendered = render(
        plan,
        "cast-Equiv",
        Evidence(formal=Formal.KERNEL, env_hash=CANONICAL_LEAN4_ENVIRONMENT_HASH),
        Register.ANNOTATED,
    )

    # Cast warning must explicitly state that the claim is not yet established
    assert "This does not yet establish the original claim" in rendered
    assert "remains open" in rendered

    # Lint must pass without eliding cast
    issues = lint_rendered(rendered, plan)
    assert "cast elided" not in issues


# --- Research Evaluation Protocol Tests ---


def test_run_all_evaluations_and_report_generation():
    reports = run_all_evaluations()
    assert len(reports) == 9

    # Check that all protocols passed
    for protocol_name, evaluation_report in reports.items():
        assert isinstance(evaluation_report, EvalReport)
        assert evaluation_report.passed, f"Protocol {protocol_name} failed: {evaluation_report.metrics}"

    # Verify Markdown report formatting
    markdown_report = generate_markdown_report(reports)
    assert "# Proof Planning System: Evaluation Benchmark Report" in markdown_report
    assert "| **extraction-fidelity** |" in markdown_report
    assert "| **rendering-fidelity** |" in markdown_report
    assert "| **drift-detection** |" in markdown_report
    assert "| **induction-ablation** |" in markdown_report
    assert "✓ PASS" in markdown_report

    # Verify JSON metrics export
    exported_json = export_metrics_json(reports)
    parsed_metrics = json.loads(exported_json)
    assert "extraction_fidelity" in parsed_metrics
    assert parsed_metrics["induction_ablation"]["metrics"]["strict_win"] is True


def test_eval_induction_ablation_strict_win():
    # Theorem 6 4-way ablation
    evaluation_report = run_induction_ablation_eval({
        "dep-graph": 0.20,
        "prose-embeddings": 0.35,
        "anti-unif-alone": 0.60,
        "anti-unif+completion": 0.95,
    })
    assert evaluation_report.passed
    assert evaluation_report.metrics["strict_win"] is True


def test_eval_drift_detection_catches_all_casts():
    evaluation_report = run_drift_detection_eval()
    assert evaluation_report.passed
    assert evaluation_report.metrics["cast_rate"] == 1.0


def test_eval_negative_transfer_blocks_all_false_analogies():
    evaluation_report = run_negative_transfer_eval()
    assert evaluation_report.passed
    assert evaluation_report.metrics["blocked_rate"] == 1.0

