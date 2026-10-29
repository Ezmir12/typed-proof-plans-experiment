"""Method schema registry with three-way dividing rule + trust gate.

`TrustStatus` and `TrustRecord` here are the canonical trust-state data model.
Policy-level promotion validation is layered on top.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .evidence import Evidence
from .stmt import Shape


class CondTag(Enum):
    S = "S"  # structural
    C = "C"  # certified
    OBL = "obligation"


@dataclass(frozen=True)
class Condition:
    tag: CondTag
    predicate_ref: str
    detail: str = ""


class TrustStatus(Enum):
    CANDIDATE = "candidate"
    REVIEWED = "reviewed"
    KERNEL_VERIFIED = "kernel-verified"


@dataclass(frozen=True)
class TrustRecord:
    status: TrustStatus = TrustStatus.CANDIDATE
    evidence: Evidence = field(default_factory=Evidence)
    # Bootstrap exception: two-independent review explicitly flagged.
    bootstrap_two_independent: bool = False
    version: int = 1


CheckerFn = Callable[[dict], bool]


@dataclass
class MethodSchema:
    id: str
    premises: list[Shape]
    conclusion: Shape
    conditions: list[Condition]
    obligation_templates: list[Shape] = field(default_factory=list)
    trust: TrustRecord = field(default_factory=TrustRecord)
    checker: CheckerFn | None = None
    elaborator: str = ""  # kernel term-builder ref
    soundness_theorem: str = ""  # kernel theorem ref
    # Plan-level transforms live in a distinct kind from schemas.
    is_transform: bool = False
    # Role slots this schema binds (e.g. RankingFunction on the measure).
    roles: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        """Seven-field check + dividing rule."""
        problems = []
        if not self.premises and self.conclusion is None:
            problems.append("missing premises/conclusion")
        if self.checker is None:
            problems.append("missing executable checker")
        if not self.elaborator:
            problems.append("missing elaborator")
        if not self.soundness_theorem:
            problems.append("missing soundness theorem reference")
        for c in self.conditions:
            if not isinstance(c.tag, CondTag):
                problems.append(f"untagged condition {c.predicate_ref!r}")
        return problems


def fuzz_terminate(schema: MethodSchema, trials: int = 1000, seed: int = 0) -> bool:
    """Bounded termination probe for schema checkers."""
    rng = random.Random(seed)
    assert schema.checker is not None
    for _ in range(trials):
        schema.checker({"fuzz": rng.randrange(10**6)})
    return True


class RegistryError(ValueError):
    pass


class SchemaRegistry:
    def __init__(self) -> None:
        self.schemas: dict[str, MethodSchema] = {}

    def register(self, schema: MethodSchema) -> MethodSchema:
        problems = schema.validate()
        if problems:
            raise RegistryError(f"schema {schema.id!r} rejected: {problems}")
        self.schemas[schema.id] = schema
        return schema

    def promote(self, schema_id: str, kernel_verified: bool | None = None,
                bootstrap_two_independent: bool = False,
                *, decision=None, model_originated: bool = False) -> MethodSchema:
        """Promote a schema through an explicit decision.

        New callers should pass ``decision``. The boolean arguments are
        retained for review-level compatibility; kernel verification requires
        an explicit artifact-backed decision.
        """
        s = self.schemas[schema_id]
        from .trust import TrustDecision, can_promote

        if decision is None:
            if kernel_verified:
                raise RegistryError(
                    "kernel verification requires an explicit artifact-backed "
                    "trust decision"
                )
            if bootstrap_two_independent:
                decision = TrustDecision(
                    TrustStatus.REVIEWED,
                    "compatibility bootstrap review",
                    reviewer_refs=("legacy-reviewer-1", "legacy-reviewer-2"),
                )
            else:
                raise RegistryError(
                    f"schema {schema_id!r} cannot be promoted without "
                    "an explicit trust decision"
                )

        if not can_promote(s.trust.status, decision,
                           model_originated=model_originated):
            raise RegistryError(
                f"schema {schema_id!r} cannot be promoted from "
                f"{s.trust.status.value!r} to {decision.status.value!r}"
            )

        s.trust = TrustRecord(
            decision.status,
            s.trust.evidence,
            bootstrap_two_independent=(
                decision.status is TrustStatus.REVIEWED
            ),
            version=s.trust.version + 1,
        )
        return s

    def usable_in_committed(self, schema_id: str) -> bool:
        t = self.schemas[schema_id].trust
        return t.status is TrustStatus.KERNEL_VERIFIED or (
            t.status is TrustStatus.REVIEWED and t.bootstrap_two_independent)
