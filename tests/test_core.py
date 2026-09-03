"""Core tests: evidence, sorts, stmt, schema, plan, typecheck, transform."""
import pytest

from tpps import evidence as E
from tpps import typecheck as T
from tpps.evidence import (BOTTOM, Certified, Empirical, Evidence, Formal, Fresh,
                           Reviewed, aggregate, check_support_consistent, conjoin,
                           is_closed, is_kernel_refuted)
from tpps.plan import Layer, MethodInstance, Plan, Provenance, StmtNode
from tpps.schema import CondTag, Condition, MethodSchema, SchemaRegistry, TrustRecord
from tpps.sorts import Attribute, SortDecl, SortRegistry
from tpps.stmt import (App, CheckKind, Const, Shape, Stmt, Var, alpha_eq, bind,
                       run_check, unify_exact)
from tpps.transform import (AuxiliaryFunction, Bypass, Compress, Contradiction,
                            FixPoint, GeneralizeSpecialize, SortClashError,
                            cast_evidence, elaboration_ready,
                            transformation_sound)


def K(**kw):
    return Evidence(formal=Formal.KERNEL, kernel_system="lean4+mathlib",
                    env_hash="abc123", **kw)


# --- Evidence ---
def test_evidence_ordering_componentwise():
    a = Evidence(reviewed=Reviewed.INDEPENDENT)
    b = Evidence(reviewed=Reviewed.AUTHOR)
    assert a.ge(b) and not b.ge(a)
    assert not Evidence(formal=Formal.SKETCHED).ge(Evidence(certified=Certified.CERTIFIED))


def test_evidence_meet_join():
    a = K(reviewed=Reviewed.INDEPENDENT)
    b = Evidence(reviewed=Reviewed.TWO_INDEPENDENT)
    assert a.meet(b).reviewed == Reviewed.INDEPENDENT
    assert a.join(b).reviewed == Reviewed.TWO_INDEPENDENT
    assert a.join(b).formal == Formal.KERNEL


def test_kernel_requires_env_hash():
    with pytest.raises(ValueError):
        Evidence(formal=Formal.KERNEL, kernel_system="lean")


def test_aggregation_and_or():
    tau = Evidence(reviewed=Reviewed.INDEPENDENT)
    prem = K(reviewed=Reviewed.TWO_INDEPENDENT)
    obl = Evidence(reviewed=Reviewed.INDEPENDENT)
    deriv = conjoin([tau, prem, obl])
    # premises at kernel + schema at reviewed:independent caps at independent
    assert deriv.formal == Formal.NONE
    assert deriv.reviewed == Reviewed.INDEPENDENT
    assert aggregate(Evidence(), [deriv]).ge(deriv)


def test_closure_and_invalidation():
    store = E.EvidenceStore()
    rho = Evidence(reviewed=Reviewed.INDEPENDENT)
    store.set("v", Evidence(reviewed=Reviewed.AUTHOR))
    assert not store.closure("v", rho)
    store.set("v", Evidence(reviewed=Reviewed.TWO_INDEPENDENT))
    assert store.closure("v", rho)  # recomputed after change


def test_refutation_inconsistency():
    nu = Evidence(formal=Formal.KERNEL, kernel_system="lean", env_hash="h")
    assert is_kernel_refuted(nu)
    assert not check_support_consistent([Evidence(), nu])
    assert check_support_consistent([Evidence()])


# --- Sorts ---
def test_sort_registry_routes_attributes():
    reg = SortRegistry()
    reg.declare(SortDecl("TRS", (), (Attribute("finite", "structural"),
                                     Attribute("orthogonal", "certified:critical-pairs")),
                         erasure_target="Rewriting.TRS"))
    assert reg.route_attribute("TRS", "finite") == "structural"
    assert reg.route_attribute("TRS", "orthogonal") == "critical-pairs"
    rt = reg.roundtrip(reg.get("TRS"))
    assert rt == reg.get("TRS")
    reg.regimes.add("general", "specific")
    assert reg.regimes.subsumes("general", "specific")
    assert reg.regimes.subsumes("x", "x")
    assert not reg.regimes.subsumes("specific", "general")


# --- Statements ---
def test_alpha_equivalence():
    t1 = bind("forall", "x", App("P", (Var("x"),)))
    t2 = bind("forall", "y", App("P", (Var("y"),)))
    assert alpha_eq(t1, t2)
    assert not alpha_eq(t1, bind("forall", "y", App("Q", (Var("y"),))))


def test_four_checks_distinct_and_logged():
    from tpps.sorts import RegimeOrder
    regimes = RegimeOrder()
    regimes.add("weak", "strong")
    pat = Shape("Bound", (Var("x"),), regime="weak")
    tgt = Shape("Bound", (Const("c"),), regime="strong")
    r_exact = run_check(CheckKind.EXACT, pat, tgt, regimes)
    r_sub = run_check(CheckKind.SUBSUME, pat, tgt, regimes)
    assert r_exact.kind != r_sub.kind
    assert r_sub.ok and r_sub.subst.get("x") == Const("c")
    assert run_check(CheckKind.ALPHA, pat, pat, regimes).ok
    assert run_check(CheckKind.SPECIALIZE, pat, tgt, regimes).ok
    # regime mismatch fails subsumption one way
    assert not run_check(CheckKind.SUBSUME, tgt, pat, regimes).ok


def test_unify_occurs_check():
    assert unify_exact(Var("x"), App("f", (Var("x"),))) is None


# --- Schemas ---
def _good_schema():
    return MethodSchema(id="S", premises=[Shape("A", ())], conclusion=Shape("B", ()),
                        conditions=[Condition(CondTag.S, "p")],
                        checker=lambda ctx: True, elaborator="e", soundness_theorem="t")


def test_registry_rejects_incomplete_and_promotes():
    reg = SchemaRegistry()
    with pytest.raises(Exception):
        reg.register(MethodSchema(id="Bad", premises=[], conclusion=Shape("B", ()),
                                  conditions=[], checker=None,
                                  elaborator="", soundness_theorem=""))
    reg.register(_good_schema())
    with pytest.raises(Exception):
        reg.promote("S", kernel_verified=False)  # no bootstrap flag
    assert not reg.usable_in_committed("S")
    reg.promote("S", kernel_verified=False, bootstrap_two_independent=True)
    assert reg.usable_in_committed("S")
    assert reg.schemas["S"].trust.bootstrap_two_independent


def test_checker_termination_probe():
    from tpps.schema import fuzz_terminate
    assert fuzz_terminate(_good_schema(), trials=100)


# --- Plans ---
def _chain_plan(layer=Layer.COMMITTED):
    p = Plan("c")
    for nid, head in (("a", "A"), ("b", "B")):
        p.add_stmt(StmtNode(nid, Stmt(shape=Shape(head, (Var("R"),)))))
    return p


def test_committed_cycle_detected_speculative_allowed():
    p = _chain_plan()
    p.add_inst(MethodInstance("e1", "S", ["a"], "b"))
    p.add_inst(MethodInstance("e2", "S", ["b"], "a"))
    assert p.committed_cycle() is not None
    assert p.committed_support("a") == {"a", "b"}
    for i in p.insts.values():
        i.layer = Layer.SPECULATIVE
    assert p.committed_cycle() is None


def test_atoms_roundtrip():
    from tpps.examples import example_registry, plan_a
    p = plan_a()
    shapes = {nid: n.stmt.shape for nid, n in p.stmts.items()}
    q = Plan.from_atoms(p.id, p.to_atoms(), shapes)
    assert set(q.stmts) == set(p.stmts) and set(q.insts) == set(p.insts)
    inst = q.insts["e-fads"]
    assert inst.premises == ["saturated"] and inst.conclusion == "confluent"
    assert inst.obligations == ["obl-inv"]


# --- Typecheck ---
def test_typecheck_pass_fail_idempotent():
    from tpps.examples import example_registry, plan_a
    reg = example_registry()
    rep = T.check_plan(plan_a(), reg)
    assert rep.ok, rep.diagnostic
    rep2 = T.check_plan(plan_a(), reg)
    assert rep == rep2  # idempotent, pure


def test_typecheck_four_failing_subconditions():
    from tpps.examples import example_registry
    reg = example_registry()
    base = Plan("t")
    base.add_stmt(StmtNode("ok-a", Stmt(shape=Shape("Saturated", (Var("R"),)))))
    base.add_stmt(StmtNode("ok-c", Stmt(shape=Shape("Confluent", (Var("R"),)))))
    base.add_stmt(StmtNode("ok-o", Stmt(shape=Shape("ActiveDomainInvariant", (Var("R"),)))))
    # (1) premises fail
    p1 = Plan("t1")
    for nid in ("ok-a", "ok-c", "ok-o"):
        p1.add_stmt(base.stmts[nid])
    p1.add_stmt(StmtNode("wrong", Stmt(shape=Shape("Bogus", ()))))
    p1.add_inst(MethodInstance("e", "FiniteActiveDomainSaturation",
                               ["wrong"], "ok-c", ["ok-o"], ["no-remove"]))
    assert not T.check_instance(p1, p1.insts["e"], reg).premises_instantiate
    # (3) missing certificate
    p3 = Plan("t3")
    for nid in ("ok-a", "ok-c", "ok-o"):
        p3.add_stmt(base.stmts[nid])
    p3.add_stmt(StmtNode("need-cert", Stmt(shape=Shape("Terminates", (Var("R"),)))))
    p3.add_inst(MethodInstance("e", "LPO", ["ok-a"], "need-cert", [], []))
    r3 = T.check_instance(p3, p3.insts["e"], reg)
    assert not r3.premises_instantiate or not r3.side_conditions_certified
    # (4) missing obligation
    p4 = Plan("t4")
    for nid in ("ok-a", "ok-c"):
        p4.add_stmt(base.stmts[nid])
    p4.add_inst(MethodInstance("e", "FiniteActiveDomainSaturation",
                               ["ok-a"], "ok-c", [], ["no-remove"]))
    assert not T.check_instance(p4, p4.insts["e"], reg).obligations_present
    # unpromoted schema rejected in committed layer
    reg2 = SchemaRegistry()
    reg2.register(_good_schema())
    pu = Plan("u")
    pu.add_stmt(StmtNode("a", Stmt(shape=Shape("A", (Var("R"),)))))
    pu.add_stmt(StmtNode("b", Stmt(shape=Shape("B", (Var("R"),)))))
    pu.add_inst(MethodInstance("e", "S", ["a"], "b"))
    assert not T.check_plan(pu, reg2).ok


def test_eigenvar_escape_rejected():
    from tpps.transform import FixPoint
    p = Plan("fx")
    p.add_stmt(StmtNode("t", Stmt(shape=Shape("P", ()))))
    FixPoint(target="t", eigenvar="x0").apply(p)
    assert T.check_well_scoped(p) == []
    p.add_stmt(StmtNode("evil", Stmt(
        context=(__import__("tpps.stmt", fromlist=["CtxEntry"]).CtxEntry(
            "Eigenvar", "x0", "FixPoint-t"),),
        shape=Shape("Q", ()))))
    errs = T.check_well_scoped(p)
    assert any("escapes" in e for e in errs)


# --- Transforms ---
def test_bypass_cast_evidence_cap_and_blame():
    mu_phi = K(reviewed=Reviewed.TWO_INDEPENDENT)
    mu_obl = Evidence()
    capped = cast_evidence(mu_phi, mu_obl)
    assert capped.formal == Formal.NONE  # open obligation caps kernel
    p = Plan("cast")
    p.add_stmt(StmtNode("s", Stmt(shape=Shape("Equiv", (), regime="as Set",
                                              sort="RelSpace"))))
    t = Bypass(Shape("Equiv", (), regime="as Set", sort="RelSpace"),
               Shape("Equiv", (), regime="as Multiset", sort="ResultMultiset"),
               blame="worker-attempt-7")
    res = t.apply(p)
    assert res is not None and res.obligations and "worker-attempt-7" in res.soundness_note
    assert p.insts["cast-Equiv-Equiv"].obligations == res.obligations


def test_bypass_sort_clash_rejected_at_construction():
    t = Bypass(Shape("A", (), sort="Expr"), Shape("B", (), sort="Domain"))
    with pytest.raises(SortClashError):
        t.apply(Plan("x"))


def test_other_transforms_emit_obligations():
    p = Plan("m")
    p.add_stmt(StmtNode("t", Stmt(shape=Shape("Goal", ()))))
    assert Contradiction(target="t").apply(p).obligations == ["witness-t"]
    assert FixPoint(target="t").apply(p).obligations == ["testpoint-t"]
    r = AuxiliaryFunction(target="t", construction="g := f/z",
                          new_shape=Shape("GoalG", ())).apply(p)
    assert r.obligations == ["wf-t", "transfer-t"]
    GeneralizeSpecialize(target="t", subst={"x": Const("c")}).apply(p)
    assert Compress(schema_ref="S", premises=["t"], conclusion="t").apply(p) is not None


def test_soundness_gates():
    assert elaboration_ready(True, True, True, True)
    assert not elaboration_ready(True, True, True, False)  # needs all legs
    assert transformation_sound(True)
    assert not transformation_sound(False)  # no closure before discharge
