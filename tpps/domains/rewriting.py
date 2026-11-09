"""Term rewriting / matching over MeTTa/MORK expression spaces.

Sorts, shapes, roles, schema library, MM2 step executor,
and worked plans.
"""
from __future__ import annotations

from tpps.reasoning.creative import PREMISE_TABLE
from tpps.core.plan import Layer, MethodInstance, Plan, Provenance, StmtNode
from tpps.core.schema import CondTag, Condition, MethodSchema, SchemaRegistry, TrustRecord, TrustStatus
from tpps.core.sorts import Attribute, SortDecl, SortRegistry
from tpps.core.stmt import (App, BVar, Binder, Const, Shape, Stmt, Var,
                            apply_subst, unify_exact)

ROLES = ("RankingFunction", "Certificate", "CriticalPair", "Witness",
         "Invariant", "Simulation", "Decomposition", "Abstraction")


def sort_library() -> SortRegistry:
    reg = SortRegistry()
    reg.declare(SortDecl("Sym", (), (), erasure_target="Lean.String"))
    reg.declare(SortDecl("Var", (), (), erasure_target="Lean.String"))
    reg.declare(SortDecl("Expr", ("Sigma", "V"), (
        Attribute("ground", "structural"), Attribute("linear", "structural"),
        Attribute("size-le-k", "structural"), Attribute("depth-le-d", "structural"),
    ), erasure_target="Lean.Expr"))
    reg.declare(SortDecl("Rule", ("Sigma", "V"), (
        Attribute("left-linear", "structural"),), erasure_target="Rewriting.Rule"))
    reg.declare(SortDecl("TRS", ("Sigma", "V"), (
        Attribute("finite", "structural"), Attribute("left-linear", "structural"),
        Attribute("no-overlap", "certified:critical-pairs"),
        Attribute("orthogonal", "certified:critical-pairs"),
    ), erasure_target="Rewriting.TRS"))
    # Set and multiset sorts stay distinct: never unified or silently coerced.
    reg.declare(SortDecl("RelSpace", ("Sigma",), (Attribute("n", "structural"),),
                         erasure_target="Finset"))
    reg.declare(SortDecl("ResultMultiset", ("Sigma",), (), erasure_target="Multiset"))
    reg.declare(SortDecl("MM2Rule", ("Sigma", "V"), (
        Attribute("headed-antecedents", "structural"),
        Attribute("extensional-antecedents", "structural"),
        Attribute("flat-consequents", "structural"),
        Attribute("ext-range-restricted", "structural"),
        Attribute("no-remove", "structural"),
    ), erasure_target="MM2.Rule"))
    reg.declare(SortDecl("Strategy", (), (), erasure_target="Rewriting.Strategy"))
    reg.declare(SortDecl("Ordering", ("A",), (Attribute("well_founded", "structural"),),
                         erasure_target="Order"))
    return reg


def _schema(sid: str, prems: list[str], concl: str,
            conds: list[tuple[CondTag, str]] | None = None,
            obls: list[str] | None = None, soundness: str = "") -> MethodSchema:
    return MethodSchema(
        id=sid,
        premises=[Shape(p, (Var("R"),), sort="TRS") for p in prems],
        conclusion=Shape(concl, (Var("R"),), sort="TRS"),
        conditions=[Condition(t, p) for t, p in (conds or [])],
        obligation_templates=[Shape(o, (), sort="TRS") for o in (obls or [])],
        checker=lambda ctx: True,
        elaborator=f"elab-{sid}",
        soundness_theorem=soundness or f"Sound-{sid}")


SCHEMA_IDS = ("WFMeasure", "LPO", "PolyInterp", "CriticalPairLemma", "Newman",
              "Orthogonal", "ParallelClosed", "ToyamaModularity",
              "StructuralInduction", "SimulationSoundness",
              "ConfluenceToStrategyEquiv", "OutputSensitiveBound",
              "FiniteActiveDomainSaturation")


def schema_library() -> SchemaRegistry:
    reg = SchemaRegistry()
    reg.register(MethodSchema(
        id="WFMeasure",
        premises=[Shape("Decreases", (Var("m"), Var("R")), sort="TRS"),
                  Shape("WF", (Var("O"),), sort="Ordering")],
        conclusion=Shape("Terminates", (Var("R"),), sort="TRS"),
        conditions=[Condition(CondTag.S, "wf-order")],
        obligation_templates=[Shape("Bounded", (Var("R"), Var("U")), sort="TRS"),
                              Shape("NoRemove", (Var("R"),), sort="TRS"),
                              Shape("Productive", (Var("R"),), sort="TRS")],
        roles=("RankingFunction", "Invariant"),
        checker=lambda ctx: True, elaborator="elab-WFMeasure",
        soundness_theorem="WellFounded.lt_wf"))
    reg.register(_schema("LPO", ["Decreases"], "Terminates",
                         [(CondTag.C, "lpo-oriented")], soundness="CeTA.LPO"))
    reg.register(_schema("PolyInterp", ["Decreases"], "Terminates",
                         [(CondTag.C, "poly-decreasing")], soundness="CeTA.PolyInterp"))
    reg.register(_schema("CriticalPairLemma", ["Joinable"], "LocallyConfluent",
                         [(CondTag.S, "critical-pairs-joinable")]))
    reg.register(_schema("Newman", ["Terminates", "LocallyConfluent"], "Confluent",
                         [(CondTag.S, "terminating")]))
    reg.register(_schema("Orthogonal", ["Orthogonal"], "Confluent",
                         [(CondTag.C, "no-overlap")]))
    reg.register(_schema("ParallelClosed", ["ParallelClosed"], "Confluent"))
    reg.register(_schema("ToyamaModularity", ["Terminates", "Terminates"], "Terminates",
                         [(CondTag.S, "disjoint-signature")]))
    reg.register(_schema("StructuralInduction", ["Preserves"], "Preserves",
                         [(CondTag.S, "motive-miller-pattern")]))
    reg.register(_schema("SimulationSoundness", ["Simulation"], "Sound",
                         [(CondTag.S, "simulation")]))
    reg.register(_schema("ConfluenceToStrategyEquiv", ["Confluent"], "Equiv",
                         [(CondTag.S, "strategy-independent")], obls=["EquivMultiset"]))
    reg.register(_schema("OutputSensitiveBound", ["Bound"], "Bound",
                         [(CondTag.C, "asymptotics")]))
    reg.register(MethodSchema(
        id="FiniteActiveDomainSaturation",
        premises=[Shape("Saturated", (Var("R"),), sort="TRS")],
        conclusion=Shape("Terminates", (Var("R"),), sort="TRS"),
        conditions=[
            Condition(CondTag.S, "finite"),
            Condition(CondTag.S, "headed-antecedents"),
            Condition(CondTag.S, "extensional-antecedents"),
            Condition(CondTag.S, "flat-consequents"),
            Condition(CondTag.S, "ext-range-restricted"),
            Condition(CondTag.S, "no-remove"),
        ],
        obligation_templates=[],
        checker=lambda ctx: True,
        elaborator="elab-FiniteActiveDomainSaturation",
        soundness_theorem="Sound-FiniteActiveDomainSaturation",
    ))
    for sid in SCHEMA_IDS:
        PREMISE_TABLE[sid] = list(reg.schemas[sid].premises)
    return reg


# --- MM2 operational semantics ---

from dataclasses import dataclass, field
from typing import Any, Callable


def _apply_subst_item(item: Any, theta: dict) -> Any:
    if hasattr(item, "substitute"):
        return item.substitute(theta)
    if hasattr(item, "instantiate"):
        return item.instantiate(theta)
    if isinstance(item, (Var, App, BVar, Const, Binder)):
        return apply_subst(item, theta)
    return item


@dataclass(frozen=True)
class MM2Rule:
    """Formal MeTTa/MORK-2 (MM2) rewrite rule: Delta-minus -> Delta-plus under guards."""

    id: str
    antecedents_ext: tuple[Any, ...]  # Delta-minus (facts consumed)
    antecedents_grounded: tuple[Callable[[dict], bool], ...] = ()  # Guard predicates
    consequents: tuple[Any, ...] = ()  # Delta-plus (facts produced)
    conditions: tuple[str, ...] = ()  # Structural conditions (e.g. "no-remove", "flat-consequents")

    @property
    def is_no_remove(self) -> bool:
        """Theorem 6 condition: Delta-minus is empty (no facts removed)."""
        return len(self.antecedents_ext) == 0 or "no-remove" in self.conditions


def mm2_step(state: frozenset, rule: MM2Rule | dict, theta: dict) -> frozenset | None:
    """S →_{ρ,θ} S' := (S \\ Δ⁻θ) ∪ Δ⁺θ, requiring extensional-antecedent
    match, grounded-antecedent satisfaction, and net change (S' ≠ S)."""
    if isinstance(rule, MM2Rule):
        ext_antecedents = rule.antecedents_ext
        grounded_guards = rule.antecedents_grounded
        consequents = rule.consequents
    else:
        ext_antecedents = rule.get("antecedents_ext", [])
        grounded_guards = rule.get("antecedents_grounded", [])
        consequents = rule.get("consequents", [])

    delta_minus = {_apply_subst_item(a, theta) for a in ext_antecedents}

    if not all(g(theta) if callable(g) else True for g in grounded_guards):
        return None

    if not delta_minus.issubset(set(state)):
        return None

    delta_plus = {_apply_subst_item(c, theta) for c in consequents}
    new_state = (set(state) - delta_minus) | delta_plus
    if frozenset(new_state) == state:
        return None  # idle: no net change
    return frozenset(new_state)


def match_and_step(state: frozenset, rule: MM2Rule) -> list[tuple[dict, frozenset]]:
    """Match extensional antecedents of rule against finite state, check guards,
    and return all valid (theta, next_state) transitions."""
    from tpps.core.stmt import Shape, unify_exact

    ext = list(rule.antecedents_ext)
    if not ext:
        # Monotonic rule without extensional antecedents
        res = mm2_step(state, rule, {})
        return [({}, res)] if res is not None else []

    matches: list[dict] = []

    def _match_ext(idx: int, current_subst: dict, used_elements: set) -> None:
        if idx >= len(ext):
            matches.append(current_subst)
            return
        pat = ext[idx]
        pat_term = pat.term() if isinstance(pat, Shape) else pat
        for elem in state:
            if elem in used_elements:
                continue
            elem_term = elem.term() if isinstance(elem, Shape) else elem
            if isinstance(pat_term, (Var, App, BVar, Const, Binder)) and isinstance(elem_term, (Var, App, BVar, Const, Binder)):
                sub = unify_exact(pat_term, elem_term, current_subst)
                if sub is not None:
                    _match_ext(idx + 1, sub, used_elements | {elem})
            elif pat_term == elem_term:
                _match_ext(idx + 1, current_subst, used_elements | {elem})

    _match_ext(0, {}, set())

    results: list[tuple[dict, frozenset]] = []
    for theta in matches:
        nxt = mm2_step(state, rule, theta)
        if nxt is not None and (theta, nxt) not in results:
            results.append((theta, nxt))
    return results


# --- Worked plans ---

def boolean_confluence_plan() -> Plan:
    """Checker-only Boolean confluence, closes cleanly."""
    p = Plan("bool-confluence")
    p.add_stmt(StmtNode("terminates", Stmt(shape=Shape("Terminates", (Var("R"),)))))
    p.add_stmt(StmtNode("local-conf", Stmt(shape=Shape("LocallyConfluent", (Var("R"),)))))
    p.add_stmt(StmtNode("confluent", Stmt(shape=Shape("Confluent", (Var("R"),)))))
    p.add_inst(MethodInstance("e-newman", "Newman",
                              ["terminates", "local-conf"], "confluent",
                              certificates=["terminating"]))
    return p


def list_confluence_plan() -> Plan:
    """Abductive residual + two OR-alternative certified terminations."""
    p = Plan("list-confluence")
    p.add_stmt(StmtNode("decr-lpo", Stmt(shape=Shape("Decreases", (Var("R"),)))))
    p.add_stmt(StmtNode("decr-poly", Stmt(shape=Shape("Decreases", (Var("R"),)))))
    p.add_stmt(StmtNode("terminates", Stmt(shape=Shape("Terminates", (Var("R"),)))),
               )
    p.add_stmt(StmtNode("local-conf", Stmt(shape=Shape("LocallyConfluent", (Var("R"),)))))
    p.add_stmt(StmtNode("confluent", Stmt(shape=Shape("Confluent", (Var("R"),)))))
    # OR-alternative 1: LPO termination (speculative layer: competing attempts)
    p.add_inst(MethodInstance("e-lpo", "LPO", ["decr-lpo"], "terminates",
                              certificates=["lpo-oriented"], layer=Layer.SPECULATIVE))
    # OR-alternative 2: polynomial interpretation
    p.add_inst(MethodInstance("e-poly", "PolyInterp", ["decr-poly"], "terminates",
                              certificates=["poly-decreasing"], layer=Layer.SPECULATIVE))
    return p


def strategy_independence_cast_plan() -> Plan:
    """Set/multiset drift recorded as an explicit Bypass cast."""
    from tpps.reasoning.transform import Bypass
    p = Plan("strategy-independence")
    p.add_stmt(StmtNode("equiv-set",
                        Stmt(shape=Shape("Equiv", (), regime="as Set", sort="RelSpace"))))
    t = Bypass(Shape("Equiv", (), regime="as Set", sort="RelSpace"),
               Shape("Equiv", (), regime="as Multiset", sort="ResultMultiset"),
               blame="worker-attempt-7")
    t.apply(p)
    return p


def toyama_refutation() -> tuple[Plan, str]:
    """TerminationModularity refuted via witness (stays open)."""
    p = Plan("toyama")
    p.add_stmt(StmtNode("term-a", Stmt(shape=Shape("Terminates", (Var("R1"),)))))
    p.add_stmt(StmtNode("term-b", Stmt(shape=Shape("Terminates", (Var("R2"),)))))
    p.add_stmt(StmtNode("witness", Stmt(shape=Shape("Witness", (), regime="counterexample"))))
    return p, "ToyamaModularity refuted: disjoint-signature union admits infinite derivation"
