"""Measurement protocols for typed proof-plan evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalReport:
    protocol: str
    metrics: dict = field(default_factory=dict)
    notes: str = ""


def eval_extraction_fidelity(corpus: list[dict]) -> EvalReport:
    fields = ("target", "regime", "method", "roles", "dependencies")
    scores = {f: sum(1 for c in corpus if c.get(f)) / max(1, len(corpus)) for f in fields}
    return EvalReport("extraction-fidelity", scores)


def eval_rendering_fidelity(roundtrips: list[dict]) -> EvalReport:
    agree = sum(1 for r in roundtrips
                if r.get("shape_ok") and r.get("regime_ok") and r.get("obl_ok"))
    return EvalReport("rendering-fidelity",
                      {"agreement": agree / max(1, len(roundtrips))})


def eval_drift_detection(injections: list[dict]) -> EvalReport:
    caught = sum(1 for i in injections if i.get("recorded_as_cast"))
    return EvalReport("drift-detection",
                      {"cast_rate": caught / max(1, len(injections))})


def eval_induction_ablation(results: dict[str, float]) -> EvalReport:
    """Four-way ablation: dep-graph / prose-embeddings / anti-unif-alone /
    anti-unif+completion — the last must strictly win."""
    win = results.get("anti-unif+completion", 0) > max(
        results.get("dep-graph", 0), results.get("prose-embeddings", 0),
        results.get("anti-unif-alone", 0))
    return EvalReport("induction-ablation", dict(results, strict_win=win))


def eval_auxiliary_synthesis(cases: list[dict]) -> EvalReport:
    proposed = sum(1 for c in cases if c.get("construction_proposed"))
    matching = sum(1 for c in cases if c.get("obligations_match"))
    n = max(1, len(cases))
    return EvalReport("auxiliary-synthesis",
                      {"proposed": proposed / n, "obligations_match": matching / n})


def eval_negative_transfer(cases: list[dict]) -> EvalReport:
    blocked = sum(1 for c in cases if c.get("blocked"))
    return EvalReport("negative-transfer",
                      {"blocked_rate": blocked / max(1, len(cases))})


def eval_residual_quality(ratings: list[float]) -> EvalReport:
    return EvalReport("residual-quality",
                      {"mean_rating": sum(ratings) / max(1, len(ratings))})


def eval_formal_closure(per_domain: dict[str, tuple[int, int]]) -> EvalReport:
    return EvalReport("formal-closure",
                      {d: closed / max(1, total) for d, (closed, total) in per_domain.items()})


def eval_work_reduction(before: float, after: float) -> EvalReport:
    return EvalReport("work-reduction",
                      {"before": before, "after": after,
                       "reduction": (before - after) / max(1e-9, before)})
