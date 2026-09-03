"""End-to-end coverage: full pipelines chained across modules.

Unit tests pin single-module behavior; these tests pin the cross-module
flows: build → check → aggregate → transform/induce → render/extract →
commit. Each test fails if any stage of its pipeline breaks.
"""
from tpps import eval_harness as EV
from tpps import typecheck as T
from tpps.domains import complex_analysis as CA
from tpps.domains import rewriting as RW
from tpps.evidence import (Evidence, EvidenceStore, Formal, Reviewed,
                           aggregate, check_support_consistent, conjoin,
                           is_closed)
from tpps.examples import (complex_registry, example_registry, fta_plan,
                           one_way_delete_schema, plan_a, plan_b,
                           plan_c_failure, theorem6_completion)
from tpps.extract import (ExtractedNode, accept, build_candidate,
                          semantic_review)
from tpps.kernel import KernelInterface
from tpps.plan import Plan
from tpps.render import Register, lint_rendered, render
from tpps.stmt import Shape, Var
from tpps.store import MockStore
from tpps.transform import elaboration_ready


def test_rewriting_induction_lifecycle():
    """Plan A checks → induction completes → corrected schema unblocks Plan C."""
    reg = example_registry()
    store = MockStore()

    for plan in (plan_a(), plan_b()):
        report = T.check_plan(plan, reg)
        assert report.ok, report.diagnostic
        rec = store.commit_gate_typing_pass(
            plan.to_atoms(),
            lambda p=plan: ((r := T.check_plan(p, reg)).ok, r.diagnostic))
        assert rec.check_ok

    induced = theorem6_completion()
    assert "Compress(plan-a, plan-b)" in induced.provenance
    reg.register(induced.schema)
    reg.promote(induced.schema.id, kernel_verified=True)

    failure = plan_c_failure()
    assert failure.failed_condition == "no-remove"
    assert "no-remove" in failure.critic_input()

    corrected = one_way_delete_schema()
    reg.register(corrected)
    reg.promote(corrected.id, kernel_verified=True)
    assert reg.usable_in_committed(corrected.id)


def test_evidence_closure_drives_render_and_commit():
    """Evidence aggregation → closure → register choice → clean lint → commit."""
    reg = example_registry()
    plan = plan_a()
    assert T.check_plan(plan, reg).ok

    store = EvidenceStore()
    requirement = Evidence(reviewed=Reviewed.INDEPENDENT)
    target_mu = aggregate(
        Evidence(reviewed=Reviewed.INDEPENDENT),
        [conjoin([Evidence(reviewed=Reviewed.INDEPENDENT),
                  Evidence(formal=Formal.KERNEL, kernel_system="lean4+mathlib",
                           env_hash="h", reviewed=Reviewed.TWO_INDEPENDENT)])])
    store.set("confluent", target_mu)
    assert is_closed(store.mus["confluent"], requirement)

    plan.stmts["confluent"].evidence = target_mu
    text = render(plan, "confluent", requirement)  # closed → textbook by default
    assert "Status of" not in text
    assert lint_rendered(text, plan) == []

    rec = store_commit(plan, reg)
    assert rec.check_ok


def store_commit(plan: Plan, reg) -> object:
    store = MockStore()
    return store.commit_gate_typing_pass(
        plan.to_atoms(),
        lambda: ((r := T.check_plan(plan, reg)).ok, r.diagnostic))


def test_kernel_discharge_closes_obligation():
    """Mock kernel verification promotes an obligation and enables elaboration."""
    kernel = KernelInterface(theorems={"WellFounded.lt_wf": True})
    ok, artifact = kernel.check_theorem("WellFounded.lt_wf")
    assert ok and artifact.env_hash

    discharged = Evidence(formal=Formal.KERNEL, kernel_system=artifact.system,
                          env_hash=artifact.env_hash,
                          reviewed=Reviewed.TWO_INDEPENDENT)
    requirement = Evidence(formal=Formal.KERNEL, kernel_system="lean4+mathlib",
                           env_hash="h", reviewed=Reviewed.INDEPENDENT)
    assert is_closed(discharged, requirement)
    assert elaboration_ready(True, True, True, True)


def test_cast_and_refutation_pipeline():
    """Cast renders explicitly and stays capped; refutation poisons support."""
    plan = RW.strategy_independence_cast_plan()
    text = render(plan, "cast-Equiv", Evidence(), Register.TEXTBOOK)
    assert "cast" in text.lower()
    assert "remains to show" in text.lower()

    _, note = RW.toyama_refutation()
    assert "refuted" in note
    kernel_refuted = Evidence(formal=Formal.KERNEL, kernel_system="lean4+mathlib",
                              env_hash="h")
    assert not check_support_consistent([Evidence(), kernel_refuted])

    blocked = EV.eval_negative_transfer([{"blocked": True}, {"blocked": False}])
    assert blocked.metrics["blocked_rate"] == 0.5


def test_complex_analysis_prose_roundtrip():
    """Liouville plan renders, extracts, reviews clean, and commits."""
    reg = complex_registry()
    plan = CA.liouville_plan()
    assert T.check_plan(plan, reg).ok

    requirement = Evidence(reviewed=Reviewed.INDEPENDENT)
    annotated = render(plan, "constant", requirement, Register.ANNOTATED)
    assert "Status of constant" in annotated
    assert lint_rendered(annotated, plan) == []

    candidate = build_candidate("liouville-prose", [ExtractedNode(
        Shape("Holo", (Var("f"),)), "sent 1", 0.95,
        "f is entire", "sentence entails holomorphy shape")])
    issues = semantic_review(candidate, {})
    assert issues == [] and accept(candidate, issues)
    assert not accept(candidate, semantic_review(candidate, {"n0": "dropped-regularity"}))

    fta = fta_plan()
    assert T.check_plan(fta, reg).ok
    assert "contradiction" in render(fta, "const-g", Evidence(),
                                     Register.ANNOTATED).lower()

    assert store_commit(plan, reg).check_ok


def test_bypass_drift_recorded_not_silently_accepted():
    """Set/multiset drift lands as an explicit cast with blame attached."""
    plan = RW.strategy_independence_cast_plan()
    inst = plan.insts["cast-Equiv-Equiv"]
    assert inst.schema_ref == "Bypass"
    assert inst.obligations, "drift must leave the implication obligation open"
    assert "worker-attempt-7" in inst.provenance.producing_attempt
    # The cast target can never outrank its open obligation.
    from tpps.transform import cast_evidence
    capped = cast_evidence(
        Evidence(formal=Formal.KERNEL, kernel_system="lean4+mathlib", env_hash="h"),
        Evidence())
    assert capped.formal == Formal.NONE
