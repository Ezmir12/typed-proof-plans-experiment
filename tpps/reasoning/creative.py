"""Abduction, analogy (align + anti-unify + score), blending,
representation change, obstruction inversion.

The LLM may only fill typed holes / narrate; typed structure
on both sides of such a call is produced and checked deterministically here.
PLN strength/confidence stays on annotations and never enters aggregation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tpps.core.plan import Plan
from tpps.core.stmt import App, BVar, Const, Shape, Term, Var, apply_subst, unify_exact


# PLN annotation quarantined from evidence aggregation.
@dataclass(frozen=True)
class PlnAnnotation:
    strength: float
    confidence: float
    note: str = ""


# --- Abduction ---

from .backward_chain import (
    CRITIC_CLASSES,
    TypedResidual,
    backward_chain,
    classify_residual,
)

# Premise table, extended by domain modules at import time.
PREMISE_TABLE: dict[str, list[Shape]] = {}


def _premises_of(schema_id: str) -> list[Shape]:
    return PREMISE_TABLE.get(schema_id, [])


# --- Analogy: align + anti-unify + score ---

@dataclass(frozen=True)
class Alignment:
    pairs: tuple[tuple[str, str], ...]  # (node_a, node_b)
    score: float = 0.0


def align_by_schema(plan_a: Plan, plan_b: Plan) -> Alignment:
    """Typed graph alignment seeded by schema identity."""
    by_schema_b: dict[str, list[str]] = {}
    for inst in plan_b.insts.values():
        by_schema_b.setdefault(inst.schema_ref, []).append(inst.id)
    pairs = []
    for inst in plan_a.insts.values():
        cands = by_schema_b.get(inst.schema_ref, [])
        if cands:
            pairs.append((inst.id, cands[0]))
    return Alignment(tuple(pairs), score=float(len(pairs)))


_counter = [0]


def _fresh_hole(prefix: str = "?U") -> Var:
    _counter[0] += 1
    return Var(f"{prefix}{_counter[0]}")


def anti_unify(a: Term, b: Term) -> tuple[Term, dict, dict]:
    """Least-general generalization; genuinely-different heads stay free
    variables: never silently guess a completion."""
    if a == b:
        return a, {}, {}
    if isinstance(a, Var) or isinstance(b, Var):
        hole = _fresh_hole()
        sa = {a.name: hole} if isinstance(a, Var) else {}
        sb = {b.name: hole} if isinstance(b, Var) else {}
        if not isinstance(a, Var):
            sa = {"?keep_a": a}
        if not isinstance(b, Var):
            sb = {"?keep_b": b}
        return hole, sa, sb
    if isinstance(a, App) and isinstance(b, App) and a.head == b.head \
            and len(a.args) == len(b.args):
        args, sa, sb = [], {}, {}
        for x, y in zip(a.args, b.args):
            g, s1, s2 = anti_unify(x, y)
            args.append(g)
            sa.update(s1)
            sb.update(s2)
        return App(a.head, tuple(args)), sa, sb
    if isinstance(a, (Const, BVar)) or isinstance(b, (Const, BVar)):
        hole = _fresh_hole()
        return hole, {"?A": a}, {"?B": b}
    hole = _fresh_hole()
    return hole, {"?A": a}, {"?B": b}


def anti_unify_shapes(a: Shape, b: Shape) -> tuple[Shape, dict, dict]:
    g, sa, sb = anti_unify(a.term(), b.term())
    head = g.head if isinstance(g, App) else "?U"
    args = g.args if isinstance(g, App) else ()
    return Shape(head, args, regime=a.regime or b.regime), sa, sb


def score_alignment(compression_gain: float, schema_complexity: float,
                    expected_obligation_cost: float) -> float:
    """score = compression gain − schema complexity − obligation cost."""
    return compression_gain - schema_complexity - expected_obligation_cost


# --- Blending as pushout ---

class BlendRejected(ValueError):
    pass


@dataclass
class BlendResult:
    merged: list[str]
    obligations: list[Shape]


def blend(common: list[str], only_a: list[str], only_b: list[str],
          sort_a: dict[str, str], sort_b: dict[str, str],
          equations: list[tuple[str, str, bool]]) -> BlendResult:
    """Pushout P1 ← K → P2; failed role laws become obligations, hard sort
    conflicts reject the blend outright."""
    shared = set(only_a) & set(only_b)
    for obj in shared:
        sa, sb = sort_a.get(obj, ""), sort_b.get(obj, "")
        if sa and sb and sa != sb:
            raise BlendRejected(f"hard sort conflict on {obj!r}: {sa} vs {sb}")
    obls = [Shape(f"law-holds-{o}", (), regime="blend") for o, _, holds in equations if not holds]
    return BlendResult(list(common) + list(only_a) + list(only_b), obls)


# --- Representation change as signature morphism ---

@dataclass(frozen=True)
class SignatureMorphism:
    sort_map: dict[str, str]
    op_map: dict[str, str]

    def transport_shape(self, shape: Shape) -> Shape:
        return Shape(self.op_map.get(shape.head, shape.head), shape.args,
                     regime=shape.regime,
                     sort=self.sort_map.get(shape.sort, shape.sort))

    def transport_schema(self, schema_id: str, premises: list[Shape],
                         conclusion: Shape) -> tuple[list[Shape], Shape, list[Shape]]:
        """Returns (transported premises, transported conclusion, preservation
        obligations for laws not automatically preserved)."""
        pre = [self.transport_shape(p) for p in premises]
        con = self.transport_shape(conclusion)
        obls = [Shape(f"preserves-{schema_id}", (), regime="morphism")]
        return pre, con, obls


# --- Obstruction inversion ---

@dataclass(frozen=True)
class FailureCertificate:
    attempted_schema: str
    failed_condition: str
    context: str
    witness_slot: str = ""


@dataclass(frozen=True)
class BarrierConjecture:
    shape: Shape
    witness_slot: str
    supporting_certificates: tuple[FailureCertificate, ...] = ()


def propose_barrier(certs: list[FailureCertificate]) -> BarrierConjecture | None:
    """Given a family sharing schema+condition across contexts, propose a
    barrier conjecture with an unfilled witness slot."""
    if not certs:
        return None
    first = certs[0]
    if not all(c.attempted_schema == first.attempted_schema
               and c.failed_condition == first.failed_condition for c in certs):
        return None
    return BarrierConjecture(
        Shape(f"Barrier-{first.attempted_schema}-{first.failed_condition}", (),
              regime="obstruction"),
        witness_slot=first.witness_slot or "witness",
        supporting_certificates=tuple(certs))
