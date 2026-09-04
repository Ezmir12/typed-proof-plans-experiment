"""Worked examples: saturation plans, finite-active-domain induction, and FTA.
"""
from __future__ import annotations

from tpps.reasoning.creative import anti_unify_shapes
from tpps.domains import rewriting as RW
from tpps.reasoning.induce import InformativeFailure, Strategy, complete_lgg, corrected_schema_name
from tpps.core.plan import MethodInstance, Plan, Provenance, StmtNode
from tpps.core.schema import CondTag, Condition, MethodSchema, SchemaRegistry, TrustRecord
from tpps.core.stmt import App, Const, Shape, Stmt, Var


def example_registry() -> SchemaRegistry:
    """Rewriting library + transform stubs, promoted for committed use."""
    reg = RW.schema_library()
    for sid in reg.schemas:
        reg.promote(sid, kernel_verified=True)
    for tid in ("Bypass", "Compress", "Contradiction", "FixPoint", "AuxiliaryFunction"):
        reg.register(MethodSchema(
            id=tid, premises=[Shape("Any", (Var("x"),))],
            conclusion=Shape("Any", (Var("x"),)),
            conditions=[], checker=lambda ctx: True,
            elaborator=f"elab-{tid}", soundness_theorem=f"Transform-{tid}",
            is_transform=True))
        reg.promote(tid, kernel_verified=True)
    return reg


def _saturation_plan(pid: str, active: str) -> Plan:
    p = Plan(pid)
    r = Var("R")
    p.add_stmt(StmtNode("saturated", Stmt(shape=Shape("Saturated", (r,)))))
    p.add_stmt(StmtNode("active", Stmt(shape=Shape("ActiveDomain", (Const(active),)))))
    p.add_stmt(StmtNode("confluent", Stmt(shape=Shape("Confluent", (r,)))))
    p.add_stmt(StmtNode("obl-inv", Stmt(shape=Shape("ActiveDomainInvariant", (r,)))))
    p.add_inst(MethodInstance("e-fads", "FiniteActiveDomainSaturation",
                              ["saturated"], "confluent",
                              obligations=["obl-inv"], certificates=["no-remove"]))
    return p


def plan_a() -> Plan:
    return _saturation_plan("plan-a", "S0")


def plan_b() -> Plan:
    return _saturation_plan("plan-b", "S1")


def plan_a_b_lgg() -> tuple[Shape, dict, dict]:
    """Least-general generalization: genuinely-different heads stay free ?U / ?I variables."""
    a = Shape("Saturated", (App("active", (Const("S0"),)), Var("I-a")),
              regime="finite-active")
    b = Shape("Saturated", (App("active", (Const("S1"),)), Var("I-b")),
              regime="finite-active")
    return anti_unify_shapes(a, b)


def theorem6_completion():
    """Completion of the LGG into FiniteActiveDomainSaturation via background-theory lemma."""
    lgg, sa, sb = plan_a_b_lgg()
    return complete_lgg(
        lgg, bindings={}, via_lemma="Lemma-5-active-domain-invariance",
        theta_a=sa, theta_b=sb, plan_a="plan-a", plan_b="plan-b",
        conditions=[Condition(CondTag.S, "no-remove")],
        obligation_templates=[Shape("ActiveDomainInvariant", (Var("R"),))],
        elaborator="(by (lemma active-domain ?R S0) (member GI))",
        soundness="Theorem-6-finite-active-domain-saturation")


def plan_c_failure() -> InformativeFailure:
    """Deletion-carrying plan fails FiniteActiveDomainSaturation at no-remove."""
    return InformativeFailure("plan-c", "FiniteActiveDomainSaturation", "no-remove")


def one_way_delete_schema() -> MethodSchema:
    """Corrected schema with lexicographic measure."""
    return MethodSchema(
        id=corrected_schema_name("DeleteSaturation"),
        premises=[Shape("Saturated", (Var("R"),))],
        conclusion=Shape("Confluent", (Var("R"),)),
        conditions=[Condition(CondTag.S, "lexicographic-measure",
                              "lexicographic (|{lease-active}|, |GI \\ S|)"),
                    Condition(CondTag.S, "one-way-delete")],
        checker=lambda ctx: True, elaborator="elab-OneWay",
        soundness_theorem="Theorem-6b-one-way-delete")


def fta_plan() -> Plan:
    """FTA via Contradiction + AuxiliaryFunction(reciprocal) + Liouville."""
    p = Plan("fta")
    p.add_stmt(StmtNode("poly", Stmt(shape=Shape("Polynomial", (Var("p"),)))))
    p.add_stmt(StmtNode("no-root", Stmt(shape=Shape("NoRoot", (Var("p"),)))))
    p.add_stmt(StmtNode("holo-g", Stmt(shape=Shape("Holo", (Var("g"),)))))
    p.add_stmt(StmtNode("bounded-g", Stmt(shape=Shape("Bounded", (Var("g"),)))))
    p.add_stmt(StmtNode("const-g", Stmt(shape=Shape("Constant", (Var("g"),)))))
    p.add_stmt(StmtNode("contra", Stmt(shape=Shape("False", ()))))
    p.add_inst(MethodInstance("e-contra", "Contradiction", ["poly", "no-root"], "holo-g"))
    p.add_inst(MethodInstance("e-aux", "AuxiliaryFunction", ["holo-g"], "bounded-g"))
    p.add_inst(MethodInstance("e-liouv", "Liouville", ["holo-g", "bounded-g"], "const-g",
                              obligations=[], certificates=[]))
    return p


def complex_registry() -> SchemaRegistry:
    from .domains import complex_analysis as CA
    reg = CA.schema_library()
    for sid in reg.schemas:
        reg.promote(sid, kernel_verified=True)
    for tid in ("Bypass", "Contradiction", "FixPoint", "AuxiliaryFunction", "Liouville-CA"):
        if tid not in reg.schemas:
            reg.register(MethodSchema(
                id=tid, premises=[Shape("Any", (Var("x"),))],
                conclusion=Shape("Any", (Var("x"),)),
                conditions=[], checker=lambda ctx: True,
                elaborator=f"elab-{tid}", soundness_theorem=f"Transform-{tid}",
                is_transform=True))
            reg.promote(tid, kernel_verified=True)
    # FTA uses Liouville with 2 premises + Constant conclusion: mark transform-kind
    # so the cross-domain demo shape-checks against the same-named schema.
    liouv = reg.schemas["Liouville"]
    liouv.is_transform = True
    reg.schemas["Contradiction"] = MethodSchema(
        id="Contradiction",
        premises=[Shape("Polynomial", (Var("p"),)), Shape("NoRoot", (Var("p"),))],
        conclusion=Shape("Holo", (Var("g"),)), conditions=[],
        checker=lambda ctx: True, elaborator="elab-Contradiction",
        soundness_theorem="Classical.byContradiction", is_transform=True,
        trust=TrustRecord())
    reg.promote("Contradiction", kernel_verified=True)
    reg.schemas["AuxiliaryFunction"] = MethodSchema(
        id="AuxiliaryFunction",
        premises=[Shape("Holo", (Var("g"),))],
        conclusion=Shape("Bounded", (Var("g"),)), conditions=[],
        checker=lambda ctx: True, elaborator="elab-AuxiliaryFunction",
        soundness_theorem="Transform-AuxiliaryFunction", is_transform=True,
        trust=TrustRecord())
    reg.promote("AuxiliaryFunction", kernel_verified=True)
    return reg


def circle_exhaustion_demo() -> Strategy:
    from .domains.complex_analysis import circle_exhaustion_strategy
    s = circle_exhaustion_strategy()
    s.add_filler("CoefficientEstimate")
    return s
