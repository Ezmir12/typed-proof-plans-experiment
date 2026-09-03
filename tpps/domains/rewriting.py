"""Term rewriting / matching over MeTTa/MORK expression spaces.

Sorts, shapes, roles, schema library, MM2 step executor,
and worked plans.
"""
from __future__ import annotations

from tpps.reasoning.creative import PREMISE_TABLE
from tpps.core.plan import Layer, MethodInstance, Plan, Provenance, StmtNode
from tpps.core.schema import CondTag, Condition, MethodSchema, SchemaRegistry, TrustRecord, TrustStatus
from tpps.core.sorts import Attribute, SortDecl, SortRegistry
from tpps.core.stmt import Shape, Stmt, Var

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
    reg.register(_schema("WFMeasure", ["Decreases"], "Terminates",
                         [(CondTag.S, "wf-order")], soundness="WellFounded.lt_wf"))
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
    reg.register(_schema("FiniteActiveDomainSaturation", ["Saturated"], "Confluent",
                         [(CondTag.S, "no-remove")],
                         obls=["ActiveDomainInvariant"]))
    for sid in SCHEMA_IDS:
        PREMISE_TABLE[sid] = list(reg.schemas[sid].premises)
    return reg


# --- MM2 operational semantics ---
def mm2_step(state: frozenset, rule: dict, theta: dict) -> frozenset | None:
    """S →_{ρ,θ} S' := (S \\ Δ⁻θ) ∪ Δ⁺θ, requiring extensional-antecedent
    match, grounded-antecedent satisfaction, and net change (S' ≠ S)."""
    delta_minus = {a.substitute(theta) if hasattr(a, "substitute") else a
                   for a in rule.get("antecedents_ext", [])}
    grounded = rule.get("antecedents_grounded", [])
    if not all(g(theta) if callable(g) else True for g in grounded):
        return None
    if not delta_minus.issubset(set(state)):
        # θ must match every extensional antecedent against S
        matched = {a for a in delta_minus if a in set(state)}
        if matched != delta_minus:
            return None
    delta_plus = {c.substitute(theta) if hasattr(c, "substitute") else c
                  for c in rule.get("consequents", [])}
    new_state = (set(state) - delta_minus) | delta_plus
    if frozenset(new_state) == state:
        return None  # idle: no net change
    return frozenset(new_state)


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
