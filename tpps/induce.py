"""EBG completion → schema, open-slot strategy, informative failure."""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Condition, MethodSchema, TrustRecord
from .stmt import Shape


@dataclass
class InducedSchema:
    schema: MethodSchema
    provenance: str  # "Compress(A, B); LGG + completion via <lemma>; θ_A, θ_B"


def complete_lgg(lgg: Shape, bindings: dict, via_lemma: str,
                 theta_a: dict, theta_b: dict, plan_a: str, plan_b: str,
                 conditions: list[Condition] | None = None,
                 obligation_templates: list[Shape] | None = None,
                 checker=None, elaborator: str = "", soundness: str = "") -> InducedSchema:
    """Background-theory completion resolves anti-unifier free vars.

    `bindings` maps each free `?U`/`?I` variable to its completed term.
    """
    resolved = lgg.instantiate(bindings)
    schema = MethodSchema(
        id=f"Induced-{via_lemma}",
        premises=[resolved],
        conclusion=resolved,
        conditions=list(conditions or []),
        obligation_templates=list(obligation_templates or []),
        trust=TrustRecord(),
        checker=checker or (lambda ctx: True),
        elaborator=elaborator,
        soundness_theorem=soundness,
    )
    prov = (f"Compress({plan_a}, {plan_b}); LGG + completion via {via_lemma}; "
            f"θ_A={theta_a} θ_B={theta_b}")
    return InducedSchema(schema, prov)


@dataclass
class Strategy:
    """Open method-slot strategy used when completion is unavailable."""
    id: str
    lgg: Shape
    open_slot: str
    known_fillers: list[str] = field(default_factory=list)

    def add_filler(self, filler: str) -> None:
        if filler not in self.known_fillers:
            self.known_fillers.append(filler)


@dataclass(frozen=True)
class InformativeFailure:
    plan_id: str
    schema_id: str
    failed_condition: str

    def critic_input(self) -> str:
        return (f"plan {self.plan_id} failed {self.schema_id} "
                f"at structural condition {self.failed_condition!r}")


def corrected_schema_name(base: str) -> str:
    return f"OneWay-{base}"
