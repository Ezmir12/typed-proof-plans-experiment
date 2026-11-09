"""Measurement protocols for typed proof-plan evaluation.

Stable public facade for the evaluation framework in `tpps.eval`.
"""
from __future__ import annotations

from tpps.eval.reports import generate_markdown_report
from tpps.eval.runners import (
    EvalReport,
    run_all_evaluations,
    run_auxiliary_synthesis_eval,
    run_drift_detection_eval,
    run_extraction_fidelity_eval,
    run_formal_closure_eval,
    run_induction_ablation_eval,
    run_negative_transfer_eval,
    run_rendering_fidelity_eval,
    run_residual_quality_eval,
    run_work_reduction_eval,
)


def eval_extraction_fidelity(corpus: list[dict]) -> EvalReport:
    return run_extraction_fidelity_eval(corpus)


def eval_rendering_fidelity(roundtrips: list[dict]) -> EvalReport:
    return run_rendering_fidelity_eval(roundtrips)


def eval_drift_detection(injections: list[dict]) -> EvalReport:
    return run_drift_detection_eval(injections)


def eval_induction_ablation(results: dict[str, float]) -> EvalReport:
    return run_induction_ablation_eval(results)


def eval_auxiliary_synthesis(cases: list[dict]) -> EvalReport:
    return run_auxiliary_synthesis_eval(cases)


def eval_negative_transfer(cases: list[dict]) -> EvalReport:
    return run_negative_transfer_eval(cases)


def eval_formal_closure(per_domain: dict[str, tuple[int, int]]) -> EvalReport:
    return run_formal_closure_eval(per_domain)


def eval_work_reduction(before: float, after: float) -> EvalReport:
    return run_work_reduction_eval(before, after)


def eval_residual_quality(ratings: list[float]) -> EvalReport:
    return run_residual_quality_eval(ratings)


__all__ = [
    "EvalReport",
    "eval_extraction_fidelity",
    "eval_rendering_fidelity",
    "eval_drift_detection",
    "eval_induction_ablation",
    "eval_auxiliary_synthesis",
    "eval_negative_transfer",
    "eval_formal_closure",
    "eval_work_reduction",
    "eval_residual_quality",
    "run_all_evaluations",
    "generate_markdown_report",
]
