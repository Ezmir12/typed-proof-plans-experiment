"""Prose → candidate plan → structural check → semantic review.

Extraction fidelity is an explicit obligation: structural
checking bounds malformed output but never mistranslation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .plan import Plan, Provenance, StmtNode
from .stmt import Shape, Stmt


@dataclass
class ExtractedNode:
    shape: Shape
    source_span: str
    extraction_confidence: float
    coverage: str  # which prose claim this accounts for
    entailment: str  # candidate entailment source-fragment ⇒ shape


@dataclass
class CandidatePlan:
    nodes: list[ExtractedNode]
    plan: Plan


@dataclass(frozen=True)
class SemanticIssue:
    node: str
    kind: str  # reversed-quantifier | dropped-regularity | pointwise-for-uniform | det-for-prob
    detail: str


MISTRANSLATION_KINDS = ("reversed-quantifier", "dropped-regularity",
                        "pointwise-for-uniform", "det-for-prob")


def build_candidate(plan_id: str, nodes: list[ExtractedNode]) -> CandidatePlan:
    plan = Plan(plan_id)
    for i, n in enumerate(nodes):
        plan.add_stmt(StmtNode(f"n{i}", Stmt(shape=n.shape),
                               provenance=Provenance(
                                   source_span=n.source_span,
                                   extraction_confidence=n.extraction_confidence)))
    return CandidatePlan(nodes, plan)


def semantic_review(candidate: CandidatePlan, flags: dict[str, str]) -> list[SemanticIssue]:
    """Semantic-consistency review.

    `flags` maps node id → suspected mistranslation kind (produced by a
    reviewer/critic; the prototype treats the reviewer as an input). A
    well-typed but mistranslated extraction is caught here, never accepted.
    """
    issues = []
    for i, n in enumerate(candidate.nodes):
        kind = flags.get(f"n{i}", "")
        if kind in MISTRANSLATION_KINDS:
            issues.append(SemanticIssue(f"n{i}", kind,
                                        f"{n.shape.head} mistranslated: {kind}"))
    return issues


def accept(candidate: CandidatePlan, issues: list[SemanticIssue]) -> bool:
    return not issues and all(
        n.source_span and n.coverage and n.entailment for n in candidate.nodes)
