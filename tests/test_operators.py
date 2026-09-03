"""Operator tests: creative, induce, extract, render, kernel, llm, store."""
import pytest

from tpps.reasoning import creative as C
from tpps.adapters import render as R
from tpps.reasoning.creative import (CRITIC_CLASSES, Alignment, FailureCertificate,
                                     SignatureMorphism, align_by_schema, anti_unify_shapes,
                                     backward_chain, blend, BlendRejected, classify_residual,
                                     propose_barrier, score_alignment)
from tpps.core.evidence import Evidence, Formal, Reviewed
from tpps.adapters.extract import (MISTRANSLATION_KINDS, ExtractedNode, accept,
                                   build_candidate, semantic_review)
from tpps.reasoning.induce import (Strategy, complete_lgg, corrected_schema_name)
from tpps.adapters.kernel import Certificate, KernelInterface, recheck
from tpps.adapters.llm import LlmClient, admit_fragment
from tpps.core.plan import MethodInstance, Plan, StmtNode
from tpps.adapters.render import Register, lint_rendered, render, smooth_with_llm
from tpps.core.schema import CondTag, Condition
from tpps.core.stmt import App, Const, Shape, Stmt, Var
from tpps.adapters.store import MockStore


# --- Creative ---
def test_abduction_residual_and_seven_critic_classes():
    from tpps.domains import rewriting as RW
    RW.schema_library()  # populates PREMISE_TABLE
    lib = RW.schema_library()
    target = Shape("Terminates", (Var("R"),))
    index = [(sid, s.conclusion) for sid, s in lib.schemas.items()]
    res = backward_chain(target, index, known_heads={"Terminates"},
                         plan_heads=set())
    assert any(r.shape.head == "Decreases" for r in res)
    assert any(r.origin_schema in ("LPO", "PolyInterp") for r in res)
    seen = {classify_residual(res[0], hint) for hint in
            ("missing lemma", "library generalization", "bad decomposition",
             "refuted candidate", "overstrong target", "representation mismatch",
             "auxiliary construction")}
    assert seen == set(CRITIC_CLASSES)


def test_analogy_lgg_leaves_free_vars_and_scores():
    a = Plan("a")
    b = Plan("b")
    for pid, plan in (("a", a), ("b", b)):
        plan.add_stmt(StmtNode("s", Stmt(shape=Shape("Saturated", (Var("R"),)))))
        plan.add_stmt(StmtNode("c", Stmt(shape=Shape("Confluent", (Var("R"),)))))
        plan.add_inst(MethodInstance(f"e-{pid}", "FiniteActiveDomainSaturation",
                                     ["s"], "c"))
    al = align_by_schema(a, b)
    assert al.pairs == (("e-a", "e-b"),)
    lgg, sa, sb = anti_unify_shapes(Shape("Confluent", (Var("R"),)),
                                    Shape("Terminates", (Var("R"),)))
    assert lgg.head.startswith("?U") and sa and sb  # never guessed
    assert score_alignment(10.0, 3.0, 2.0) == 5.0


def test_blend_obligations_and_rejection():
    ok = blend(["k"], ["a1"], ["b1"], {"x": "A"}, {"x": "A"},
               [("law1", "r", False), ("law2", "r", True)])
    assert [o.head for o in ok.obligations] == ["law-holds-law1"]
    with pytest.raises(BlendRejected):
        blend(["k"], ["x"], ["x"], {"x": "Set"}, {"x": "Multiset"}, [])


def test_representation_change_transport():
    m = SignatureMorphism({"Set": "Multiset"}, {"Equiv": "EquivM"})
    pre, con, obls = m.transport_schema("S", [Shape("Equiv", ())], Shape("Equiv", ()))
    assert con.head == "EquivM" and len(obls) == 1


def test_obstruction_inversion_barrier():
    certs = [FailureCertificate("ToyamaModularity", "disjoint-signature", f"ctx{i}",
                                witness_slot="w")
             for i in range(2)]
    barrier = propose_barrier(certs)
    assert barrier is not None and barrier.witness_slot == "w"
    assert barrier.supporting_certificates == tuple(certs)
    assert propose_barrier([]) is None
    assert propose_barrier(certs[:1] + [FailureCertificate("Other", "c", "x")]) is None


# --- Induce ---
def test_completion_provenance_and_strategy():
    from tpps.examples import plan_a_b_lgg, theorem6_completion
    lgg, _, _ = plan_a_b_lgg()
    assert "Compress(plan-a, plan-b)" in theorem6_completion().provenance
    assert "Lemma-5" in theorem6_completion().provenance
    assert theorem6_completion().schema.obligation_templates
    s = Strategy("S", lgg, "?EstimateMethod", ["CauchyEstimate"])
    s.add_filler("MaxModulus")
    s.add_filler("MaxModulus")
    assert s.known_fillers == ["CauchyEstimate", "MaxModulus"]
    assert corrected_schema_name("DeleteSaturation").startswith("OneWay-")


# --- Extract ---
def _candidate():
    return build_candidate("ex", [ExtractedNode(
        Shape("Holo", (Var("f"),)), "Lie 1-3", 0.9, "claim 1", "frag ⇒ shape")])


def test_extraction_pipeline_catches_mistranslation():
    cand = _candidate()
    assert cand.plan.stmts["n0"].provenance.source_span == "Lie 1-3"
    for kind in MISTRANSLATION_KINDS:
        issues = semantic_review(cand, {"n0": kind})
        assert issues and issues[0].kind == kind
        assert not accept(cand, issues)
    assert accept(cand, semantic_review(cand, {}))


# --- Render ---
def _render_plan():
    from tpps.examples import example_registry, plan_a
    return plan_a(), example_registry()


def test_renderer_deterministic_registers_and_default():
    plan, reg = _render_plan()
    req = Evidence(reviewed=Reviewed.INDEPENDENT)
    t1 = render(plan, "confluent", req, Register.TEXTBOOK)
    assert render(plan, "confluent", req, Register.TEXTBOOK) == t1  # byte-identical
    ann = render(plan, "confluent", req, Register.ANNOTATED)
    assert "Status of confluent" in ann
    assert "Three facts" not in t1  # shared linearization, merged sentences not required here
    # open target defaults to annotated
    assert "Status of" in render(plan, "confluent", req)
    # closed target defaults to textbook
    plan.stmts["confluent"].evidence = Evidence(
        formal=Formal.KERNEL, kernel_system="lean4+mathlib", env_hash="h",
        reviewed=Reviewed.TWO_INDEPENDENT)
    assert "Status of" not in render(plan, "confluent", req)
    assert lint_rendered(t1, plan) == []


def test_renderer_never_smooths_casts_obligations_regimes():
    from tpps.domains.rewriting import strategy_independence_cast_plan
    p = strategy_independence_cast_plan()
    req = Evidence()
    text = render(p, "cast-Equiv", req, Register.TEXTBOOK)
    assert "cast" in text.lower() and "remains to show" in text.lower()
    assert lint_rendered(text, p) == []
    # smoothing that drops the obligation sentence is rejected
    out = smooth_with_llm(text, lambda t: "Hence the theorem.",
                           p, lambda s: ([], [], []))
    assert out == text


# --- Kernel ---
def test_certificates_recheck_without_search():
    assert recheck(Certificate("lpo", {"precedence": ["f", "g"], "status": "oriented"}))
    assert recheck(Certificate("polyinterp", {"coefficients": [1], "decreasing": True}))
    assert recheck(Certificate("domain-calculus", {"containment_witness": "w"}))
    assert recheck(Certificate("symbolic-regularity", {"order": 1}))
    assert recheck(Certificate("asymptotics", {"limit_witness": "w"}))
    assert not recheck(Certificate("unknown-proc", {}))
    assert not recheck(Certificate("lpo", {"precedence": []}))


def test_kernel_interface_pins_env():
    k = KernelInterface(theorems={"WellFounded.lt_wf": True})
    ok, art = k.check_theorem("WellFounded.lt_wf")
    assert ok and art.env_hash == k.env_hash
    assert k.elaborate("Newman", ["p1"], ["o1"], []).startswith("(by Newman")
    assert k.discharge_batch(["o1"]) == {"o1": False}


# --- LLM ---
class _StubLlm(LlmClient):
    def extract(self, prose: str) -> str:
        self.log.append(__import__("tpps.adapters.llm", fromlist=["LlmCallLog"]).LlmCallLog(
            "stub", prose, "frag"))
        return "frag"

    def elaborate(self, typed_hole: str) -> str:
        return "stmt+sketch"

    def draft_method(self, examples: str) -> str:
        return "heuristic description"


def test_llm_boundary_three_paths_gated():
    llm = _StubLlm()
    assert llm.extract("proof") == "frag"
    assert llm.elaborate("?H") == "stmt+sketch"
    assert llm.draft_method("ex") == "heuristic description"
    assert len(llm.log) == 1  # model identity + prompt + output logged
    assert admit_fragment("good", lambda raw: True) == "good"
    with pytest.raises(ValueError):
        admit_fragment("bad-sort", lambda raw: False)  # malformed: rejected pre-storage
    assert not hasattr(llm, "rewrite_plan")  # no fourth write path


# --- Store ---
def test_commit_gate_typing_pass_and_retrieval():
    from tpps.examples import example_registry, plan_a
    from tpps.core import typecheck as T
    reg = example_registry()
    plan = plan_a()
    store = MockStore()
    rec = store.commit_gate_typing_pass(
        plan.to_atoms(), lambda: ((r := T.check_plan(plan, reg)).ok, r.diagnostic))
    assert rec.check_ok and rec.content_hash
    assert store.retrieve_by_schema("FiniteActiveDomainSaturation")
    n_atoms = len(store.atoms)
    rec2 = store.commit_gate_typing_pass(
        plan.to_atoms(), lambda: ((r := T.check_plan(plan, reg)).ok, r.diagnostic))
    assert rec2.check_ok and len(store.atoms) == n_atoms  # content-addressed
    assert rec2.content_hash == rec.content_hash
    other = store.commit_gate_typing_pass(
        ["(plan other stmt n (seq [] Z))"], lambda: (True, ""))
    assert other.content_hash != rec.content_hash
