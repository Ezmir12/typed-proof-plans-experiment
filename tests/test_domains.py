"""Domain + eval + end-to-end tests."""
from tpps import eval_harness as EV
from tpps import typecheck as T
from tpps.domains import complex_analysis as CA
from tpps.domains import rewriting as RW
from tpps.domains.rewriting import mm2_step


def test_rewriting_sorts_shapes_roles():
    reg = RW.sort_library()
    assert reg.get("RelSpace").erasure_target != reg.get("ResultMultiset").erasure_target
    lib = RW.schema_library()
    assert set(RW.SCHEMA_IDS) <= set(lib.schemas)
    assert set(RW.ROLES) >= {"RankingFunction", "Witness"}


def test_mm2_step_semantics():
    rule = {"antecedents_ext": [{"a"}], "antecedents_grounded": [],
            "consequents": [{"b"}]}

    class Atom(str):
        def substitute(self, theta):
            return self

    s = frozenset({Atom("a"), Atom("c")})
    assert mm2_step(s, {"antecedents_ext": [Atom("a")],
                        "antecedents_grounded": [], "consequents": [Atom("b")]}, {}) \
        == frozenset({Atom("c"), Atom("b")})
    # idle: no net change
    assert mm2_step(s, {"antecedents_ext": [Atom("a")],
                        "antecedents_grounded": [], "consequents": [Atom("a")]}, {}) is None
    # antecedent absent
    assert mm2_step(s, {"antecedents_ext": [Atom("zzz")],
                        "antecedents_grounded": [], "consequents": [Atom("b")]}, {}) is None


def _promote_all(lib):
    for sid in lib.schemas:
        lib.promote(sid, kernel_verified=True)
    return lib


def test_rewriting_worked_plans_check():
    from tpps.examples import example_registry
    reg = example_registry()
    assert T.check_plan(RW.boolean_confluence_plan(), reg).ok
    assert T.check_plan(RW.list_confluence_plan(), reg).ok
    cast = RW.strategy_independence_cast_plan()
    assert any(i.schema_ref == "Bypass" for i in cast.insts.values())
    assert "worker-attempt-7" in cast.insts["cast-Equiv-Equiv"].provenance.producing_attempt
    _, note = RW.toyama_refutation()
    assert "refuted" in note


def test_complex_library_procedures_and_plans():
    reg = CA.sort_library()
    assert reg.route_attribute("Domain", "open") == "domain-calculus"
    assert reg.route_attribute("Function", "holomorphic") == "symbolic-regularity"
    lib = _promote_all(CA.schema_library())
    assert set(CA.SCHEMA_IDS) <= set(lib.schemas)
    assert CA.domain_containment("0", 1.0, "0", 2.0) is not None
    assert CA.domain_containment("0", 5.0, "0", 2.0) is None
    assert CA.regularity_zero_order("z**2") is not None
    assert CA.asymptotic_limit(3, 1) is not None
    assert CA.asymptotic_limit(1, 3) is None
    assert T.check_plan(CA.liouville_plan(), lib).ok
    assert T.check_plan(CA.schwarz_plan(), lib).ok
    strat = CA.circle_exhaustion_strategy()
    assert strat.open_slot == "?EstimateMethod" and "MaxModulus" in strat.known_fillers
    assert CA.auxiliary_division_residual().head == "Holo"


def test_section7_end_to_end():
    from tpps.examples import (example_registry, one_way_delete_schema, plan_a,
                               plan_a_b_lgg, plan_b, plan_c_failure,
                               theorem6_completion)
    reg = example_registry()
    assert T.check_plan(plan_a(), reg).ok
    assert T.check_plan(plan_b(), reg).ok
    lgg, _, _ = plan_a_b_lgg()
    assert any(str(a).startswith("?U") or str(a).startswith("Var")
               for a in lgg.args) or lgg.head.startswith("?U")
    induced = theorem6_completion()
    reg.register(induced.schema)
    fail = plan_c_failure()
    assert fail.failed_condition == "no-remove"
    assert fail.schema_id == "FiniteActiveDomainSaturation"
    fixed = one_way_delete_schema()
    assert any("lexicographic" in c.detail for c in fixed.conditions)


def test_fta_plan_checks_and_renders():
    from tpps.examples import complex_registry, fta_plan
    from tpps.evidence import Evidence
    from tpps.render import Register, render
    reg = complex_registry()
    plan = fta_plan()
    assert T.check_plan(plan, reg).ok
    text = render(plan, "const-g", Evidence(), Register.ANNOTATED)
    assert "contradiction" in text.lower()


def test_nine_eval_protocols_run():
    assert EV.eval_extraction_fidelity(
        [{"target": 1, "regime": 1, "method": 1, "roles": 0, "dependencies": 1}]).metrics
    assert EV.eval_rendering_fidelity(
        [{"shape_ok": True, "regime_ok": True, "obl_ok": False}]).metrics["agreement"] == 0
    assert EV.eval_drift_detection(
        [{"recorded_as_cast": True}, {"recorded_as_cast": False}]).metrics["cast_rate"] == 0.5
    abl = EV.eval_induction_ablation({"dep-graph": 0.2, "prose-embeddings": 0.3,
                                      "anti-unif-alone": 0.5, "anti-unif+completion": 0.8})
    assert abl.metrics["strict_win"]
    assert EV.eval_auxiliary_synthesis(
        [{"construction_proposed": True, "obligations_match": True}]).metrics["proposed"] == 1
    assert EV.eval_negative_transfer([{"blocked": True}]).metrics["blocked_rate"] == 1
    assert EV.eval_residual_quality([4.0, 5.0]).metrics["mean_rating"] == 4.5
    assert EV.eval_formal_closure({"rewriting": (3, 4)}).metrics["rewriting"] == 0.75
    assert EV.eval_work_reduction(100, 20).metrics["reduction"] == pytest.approx(0.8)


import pytest  # noqa: E402
