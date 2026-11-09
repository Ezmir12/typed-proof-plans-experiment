"""Research evaluation reporting and metric export."""
from __future__ import annotations

import json
from typing import Any

from .runners import EvalReport


def generate_markdown_report(reports: dict[str, EvalReport]) -> str:
    """Generate a publication-grade GitHub Flavored Markdown evaluation report."""
    lines: list[str] = [
        "# Proof Planning System: Evaluation Benchmark Report",
        "",
        "Evaluation results across automated proof-planning verification and reasoning metrics:",
        "",
        "| Evaluation Protocol | Key Metric | Result | Status |",
        "| :--- | :--- | :--- | :--- |",
    ]

    for protocol_key, evaluation_report in reports.items():
        protocol_name = evaluation_report.protocol
        passed_badge = "✓ PASS" if evaluation_report.passed else "✗ FAIL"
        # Extract primary metric
        primary_metric = "N/A"
        if "overall" in evaluation_report.metrics:
            primary_metric = f"Overall = {evaluation_report.metrics['overall']:.2%}"
        elif "clean_rate" in evaluation_report.metrics:
            primary_metric = f"Clean Rate = {evaluation_report.metrics['clean_rate']:.2%}"
        elif "cast_rate" in evaluation_report.metrics:
            primary_metric = f"Cast Rate = {evaluation_report.metrics['cast_rate']:.2%}"
        elif "strict_win" in evaluation_report.metrics:
            primary_metric = f"Strict Win = {evaluation_report.metrics['strict_win']}"
        elif "matching_rate" in evaluation_report.metrics:
            primary_metric = f"Matching Rate = {evaluation_report.metrics['matching_rate']:.2%}"
        elif "blocked_rate" in evaluation_report.metrics:
            primary_metric = f"Blocked Rate = {evaluation_report.metrics['blocked_rate']:.2%}"
        elif "reduction_ratio" in evaluation_report.metrics:
            primary_metric = f"Reduction = {evaluation_report.metrics['reduction_ratio']:.1%}"
        elif "mean_rating" in evaluation_report.metrics:
            primary_metric = f"Mean Rating = {evaluation_report.metrics['mean_rating']:.2f}/5.0"

        lines.append(f"| **{protocol_name}** | {primary_metric} | `{evaluation_report.metrics}` | **{passed_badge}** |")

    lines.extend([
        "",
        "## Evaluation Protocol Details",
        "",
    ])

    for protocol_key, evaluation_report in reports.items():
        lines.append(f"### {evaluation_report.protocol}")
        lines.append(f"- **Passed**: {evaluation_report.passed}")
        lines.append(f"- **Notes**: {evaluation_report.notes}")
        lines.append(f"- **Detailed Metrics**: `{json.dumps(evaluation_report.metrics)}`")
        lines.append("")

    return "\n".join(lines)


def export_metrics_json(reports: dict[str, EvalReport]) -> str:
    """Serialize all evaluation results into JSON."""
    payload = {protocol_identifier: report.to_dict() for protocol_identifier, report in reports.items()}
    return json.dumps(payload, indent=2)

