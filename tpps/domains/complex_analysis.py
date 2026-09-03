"""Elementary complex analysis — sorts, shapes, roles, schemas,
transforms, certified procedures, worked plans,
CircleExhaustion strategy + AuxiliaryDivision.
"""
from __future__ import annotations

from tpps.reasoning.creative import PREMISE_TABLE
from tpps.reasoning.induce import Strategy
from tpps.adapters.kernel import Certificate
from tpps.core.plan import MethodInstance, Plan, StmtNode
from tpps.core.schema import CondTag, Condition, MethodSchema, SchemaRegistry
from tpps.core.sorts import Attribute, SortDecl, SortRegistry
from tpps.core.stmt import Shape, Stmt, Var

ROLES = ("Bound", "TestPoint", "ExhaustingParameter", "AuxiliaryFunction",
         "Decomposition", "Witness", "ExceptionalSet")


def sort_library() -> SortRegistry:
    reg = SortRegistry()
    reg.declare(SortDecl("Point", (), (), erasure_target="Complex"))
    reg.declare(SortDecl("Radius", (), (), erasure_target="Real"))
    reg.declare(SortDecl("Bound", (), (), erasure_target="Real"))
    reg.declare(SortDecl("Domain", (), (
        Attribute("open", "certified:domain-calculus"),
        Attribute("connected", "certified:domain-calculus"),
        Attribute("simply-connected", "certified:domain-calculus"),
        Attribute("named", "structural"),
    ), erasure_target="Complex.Set"))
    reg.declare(SortDecl("Function", (), (
        Attribute("continuous", "certified:symbolic-regularity"),
        Attribute("holomorphic", "certified:symbolic-regularity"),
        Attribute("entire", "certified:symbolic-regularity"),
        Attribute("explicit", "structural"),
    ), erasure_target="Complex.Function"))
    return reg


def _schema(sid: str, prems: list[str], concl: str,
            conds: list[tuple[CondTag, str]] | None = None,
            obls: list[str] | None = None, soundness: str = "") -> MethodSchema:
    return MethodSchema(
        id=sid,
        premises=[Shape(p, (Var("f"),), sort="Function") for p in prems],
        conclusion=Shape(concl, (Var("f"),), sort="Function"),
        conditions=[Condition(t, p) for t, p in (conds or [])],
        obligation_templates=[Shape(o, ()) for o in (obls or [])],
        checker=lambda ctx: True,
        elaborator=f"elab-{sid}",
        soundness_theorem=soundness or f"Complex.{sid}")


SCHEMA_IDS = ("CauchyEstimate", "CoefficientEstimate", "ZeroDerivativeConstant",
              "Liouville", "MaxModulus", "FactorLemma", "RemovableSingularity",
              "QuotientHolo", "ContinuousOnCompactBounded", "PolynomialGrowth",
              "TaylorTruncation", "LimitBound")


def schema_library() -> SchemaRegistry:
    reg = SchemaRegistry()
    reg.register(_schema("CauchyEstimate", ["Holo", "Bounded"], "Estimate",
                         [(CondTag.C, "domain-calculus")], soundness="Complex.cauchy_estimate"))
    reg.register(_schema("CoefficientEstimate", ["Holo", "Bounded"], "Estimate",
                         [(CondTag.C, "domain-calculus")]))
    reg.register(_schema("ZeroDerivativeConstant", ["Estimate"], "Constant",
                         [(CondTag.S, "vanishing-estimate")], soundness="Complex.zero_deriv_constant"))
    reg.register(_schema("Liouville", ["Entire", "Bounded"], "Constant",
                         [(CondTag.S, "entire"), (CondTag.C, "asymptotics")],
                         obls=["DerivativeBound"],
                         soundness="Complex.liouville"))
    reg.register(_schema("MaxModulus", ["Holo"], "Estimate",
                         [(CondTag.C, "domain-calculus")],
                         soundness="Complex.maximum_modulus"))
    reg.register(_schema("FactorLemma", ["Holo", "Zero"], "Holo",
                         [(CondTag.C, "symbolic-regularity")]))
    reg.register(_schema("RemovableSingularity", ["Holo", "Bounded"], "Holo",
                         [(CondTag.C, "symbolic-regularity")],
                         soundness="Complex.removable_singularity"))
    reg.register(_schema("QuotientHolo", ["Holo", "Holo"], "Holo",
                         [(CondTag.C, "symbolic-regularity")]))
    reg.register(_schema("ContinuousOnCompactBounded", ["Continuous"], "Bounded",
                         [(CondTag.S, "compact-domain")]))
    reg.register(_schema("PolynomialGrowth", ["Polynomial"], "Bounded",
                         [(CondTag.C, "asymptotics")]))
    reg.register(_schema("TaylorTruncation", ["Holo"], "Estimate",
                         [(CondTag.C, "domain-calculus")]))
    reg.register(_schema("LimitBound", ["Estimate"], "Estimate",
                         [(CondTag.C, "asymptotics")]))
    for sid in SCHEMA_IDS:
        PREMISE_TABLE[sid] = list(reg.schemas[sid].premises)
    return reg


# --- Certified procedures, each emitting re-checkable certs ---
def domain_containment(a: str, r: float, b: str, R: float) -> Certificate | None:
    """D(a,r) ⊆ D(b,R) iff |a−b|+r ≤ R (names parsed as floats in prototype)."""
    try:
        if abs(float(a) - float(b)) + r <= R:
            return Certificate("domain-calculus",
                               {"containment_witness": f"D({a},{r}) ⊆ D({b},{R})"})
    except ValueError:
        return None
    return None


def regularity_zero_order(formula: str) -> Certificate | None:
    if "z**" in formula or "z^" in formula:
        return Certificate("symbolic-regularity",
                           {"order": 1, "formula": formula})
    return None


def asymptotic_limit(n: int, k: int) -> Certificate | None:
    """(A+Br^k)/r^n → 0 for n > k."""
    if n > k:
        return Certificate("asymptotics", {"limit_witness": f"r^{k}/r^{n} → 0"})
    return None


# --- Worked plans ---

def liouville_plan() -> Plan:
    p = Plan("liouville")
    p.add_stmt(StmtNode("entire", Stmt(shape=Shape("Holo", (Var("f"),), regime="entire"))))
    p.add_stmt(StmtNode("bounded", Stmt(shape=Shape("Bounded", (Var("f"),)))))
    p.add_stmt(StmtNode("deriv-bound", Stmt(shape=Shape("Estimate", (Var("f"),)))))
    p.add_stmt(StmtNode("limit-est", Stmt(shape=Shape("Estimate", (Var("f"),)))))
    p.add_stmt(StmtNode("constant", Stmt(shape=Shape("Constant", (Var("f"),)))))
    p.add_inst(MethodInstance("e-cauchy", "CauchyEstimate",
                              ["entire", "bounded"], "deriv-bound",
                              certificates=["domain-calculus"]))
    p.add_inst(MethodInstance("e-limit", "LimitBound", ["deriv-bound"], "limit-est",
                              certificates=["asymptotics"]))
    p.add_inst(MethodInstance("e-const", "ZeroDerivativeConstant", ["limit-est"],
                              "constant", certificates=["vanishing-estimate"]))
    return p


def schwarz_plan() -> Plan:
    """Schwarz lemma with two OR-alternative Holo(g|D) derivations."""
    from tpps.core.plan import Layer
    p = Plan("schwarz")
    p.add_stmt(StmtNode("holo-f", Stmt(shape=Shape("Holo", (Var("f"),)))))
    p.add_stmt(StmtNode("f0-zero", Stmt(shape=Shape("Zero", (Var("f"),)))))
    p.add_stmt(StmtNode("bounded-g", Stmt(shape=Shape("Bounded", (Var("g"),)))))
    p.add_stmt(StmtNode("holo-g-alt1", Stmt(shape=Shape("Holo", (Var("g"),)))))
    p.add_stmt(StmtNode("holo-g-alt2", Stmt(shape=Shape("Holo", (Var("g"),)))))
    p.add_stmt(StmtNode("bound-g", Stmt(shape=Shape("Estimate", (Var("g"),)))))
    p.add_inst(MethodInstance("e-factor", "FactorLemma", ["holo-f", "f0-zero"],
                              "holo-g-alt1", certificates=["symbolic-regularity"],
                              layer=Layer.SPECULATIVE))
    p.add_inst(MethodInstance("e-removable", "RemovableSingularity",
                              ["holo-f", "bounded-g"], "holo-g-alt2",
                              certificates=["symbolic-regularity"],
                              layer=Layer.SPECULATIVE))
    p.add_inst(MethodInstance("e-maxmod", "MaxModulus", ["holo-g-alt1"], "bound-g",
                              certificates=["domain-calculus"]))
    return p


def circle_exhaustion_strategy() -> Strategy:
    """Open-slot strategy, NOT a schema."""
    return Strategy(
        id="CircleExhaustion",
        lgg=Shape("Estimate", (Var("f"), Var("r")), regime="circle-r"),
        open_slot="?EstimateMethod",
        known_fillers=["CauchyEstimate", "MaxModulus", "CoefficientEstimate"])


def auxiliary_division_residual() -> Shape:
    """g := f/h residual with Holo + boundedness obligations."""
    return Shape("Holo", (Var("g"),), regime="aux-division")
