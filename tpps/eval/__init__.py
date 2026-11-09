"""Evaluation framework for Typed Proof Plans.

Implements the seven evaluation protocols from Section 11 of
'Typed Proof Plans as the Intermediate Representation for Creative Mathematical Reasoning'
(Goertzel 2026).
"""
from .corpus import load_benchmark_corpus, load_injection_cases
from .reports import generate_markdown_report
from .runners import (
    run_all_evaluations,
    run_auxiliary_synthesis_eval,
    run_drift_detection_eval,
    run_extraction_fidelity_eval,
    run_formal_closure_eval,
    run_induction_ablation_eval,
    run_negative_transfer_eval,
    run_rendering_fidelity_eval,
    run_work_reduction_eval,
)

__all__ = [
    "load_benchmark_corpus",
    "load_injection_cases",
    "run_all_evaluations",
    "run_extraction_fidelity_eval",
    "run_rendering_fidelity_eval",
    "run_drift_detection_eval",
    "run_induction_ablation_eval",
    "run_auxiliary_synthesis_eval",
    "run_negative_transfer_eval",
    "run_formal_closure_eval",
    "run_work_reduction_eval",
    "generate_markdown_report",
]

