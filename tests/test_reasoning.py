"""Tests for equational rewriting, recursive backward chaining, and version spaces."""
import pytest

from tpps.core.plan import Plan, StmtNode
from tpps.core.schema import CondTag, Condition, MethodSchema, SchemaRegistry, TrustRecord
from tpps.core.stmt import App, Const, Shape, Stmt, Var
from tpps.domains.rewriting import MM2Rule, match_and_step, mm2_step
from tpps.reasoning.backward_chain import (
    BackwardChainer,
    CRITIC_CLASSES,
    TypedResidual,
    backward_chain,
    classify_residual,
)
from tpps.reasoning.version_space import SortClashError, VersionSpace


# --- MM2 Rewriting Tests ---


def test_mm2_rule_step_consumes_delta_minus_and_adds_delta_plus():
    rule = MM2Rule(
        id="rewrite-eval",
        antecedents_ext=(Shape("Redex", (Var("x"),)),),
        consequents=(Shape("Val", (Var("x"),)),),
    )

    state = frozenset({
        Shape("Redex", (Const("42"),)),
        Shape("Untouched", (Const("keep-me"),)),
    })

    theta = {"x": Const("42")}
    next_state = mm2_step(state, rule, theta)

    assert next_state is not None
    # Delta-minus consumed
    assert Shape("Redex", (Const("42"),)) not in next_state
    # Delta-plus added
    assert Shape("Val", (Const("42"),)) in next_state
    # Untouched state elements preserved
    assert Shape("Untouched", (Const("keep-me"),)) in next_state


def test_mm2_rule_rejects_when_antecedent_absent_or_idle():
    rule = MM2Rule(
        id="rewrite-step",
        antecedents_ext=(Shape("Need", (Const("missing"),)),),
        consequents=(Shape("Produced", ()),),
    )
    state = frozenset({Shape("Need", (Const("present"),))})
    # Missing antecedent
    assert mm2_step(state, rule, {}) is None

    # Idle step (no net change) returns None
    idle_rule = MM2Rule(
        id="idle-step",
        antecedents_ext=(Shape("Keep", ()),),
        consequents=(Shape("Keep", ()),),
    )
    assert mm2_step(frozenset({Shape("Keep", ())}), idle_rule, {}) is None


def test_mm2_grounded_guards_block_transition():
    rule = MM2Rule(
        id="guarded-rule",
        antecedents_ext=(Shape("Num", (Var("n"),)),),
        antecedents_grounded=(lambda theta: int(theta["n"].value) > 10,),
        consequents=(Shape("BigNum", (Var("n"),)),),
    )
    state = frozenset({Shape("Num", (Const("5"),))})
    assert mm2_step(state, rule, {"n": Const("5")}) is None

    state_ok = frozenset({Shape("Num", (Const("25"),))})
    res = mm2_step(state_ok, rule, {"n": Const("25")})
    assert res is not None
    assert Shape("BigNum", (Const("25"),)) in res


def test_match_and_step_finds_substitutions_automatically():
    rule = MM2Rule(
        id="auto-step",
        antecedents_ext=(Shape("Edge", (Var("a"), Var("b"))),),
        consequents=(Shape("Path", (Var("a"), Var("b"))),),
    )
    state = frozenset({
        Shape("Edge", (Const("node1"), Const("node2"))),
        Shape("Edge", (Const("node2"), Const("node3"))),
    })

    transitions = match_and_step(state, rule)
    assert len(transitions) == 2
    # Verify substitution produced for one match
    substs = [theta for theta, _ in transitions]
    assert any(s["a"] == Const("node1") and s["b"] == Const("node2") for s in substs)


# --- Backward Chaining Tests ---


def _chainer_registry() -> SchemaRegistry:
    reg = SchemaRegistry()
    # Schema 1: Goal <- Premise1 AND Premise2
    reg.register(MethodSchema(
        id="rule-and",
        premises=[Shape("Premise1", (Var("x"),)), Shape("Premise2", (Var("x"),))],
        conclusion=Shape("Goal", (Var("x"),)),
        conditions=[],
        checker=lambda ctx: True,
        elaborator="elab-and",
        soundness_theorem="sound",
    ))
    # Schema 2: Premise1 <- BaseFact
    reg.register(MethodSchema(
        id="rule-base",
        premises=[Shape("BaseFact", (Var("x"),))],
        conclusion=Shape("Premise1", (Var("x"),)),
        conditions=[],
        checker=lambda ctx: True,
        elaborator="elab-base",
        soundness_theorem="sound",
    ))
    # Schema 3: Goal <- AltFact (OR-alternative)
    reg.register(MethodSchema(
        id="rule-alt",
        premises=[Shape("AltFact", (Var("x"),))],
        conclusion=Shape("Goal", (Var("x"),)),
        conditions=[],
        checker=lambda ctx: True,
        elaborator="elab-alt",
        soundness_theorem="sound",
    ))
    return reg


def test_recursive_backward_chain_and_or_derivation():
    reg = _chainer_registry()
    chainer = BackwardChainer(reg, max_depth=4)

    target = Shape("Goal", (Const("foo"),))
    # BaseFact and Premise2 are known axioms
    known = {"BaseFact", "Premise2"}

    goal_node = chainer.search(target, known_heads=known)
    assert goal_node.is_closed
    # No residuals when derivation closes cleanly
    assert goal_node.collect_residuals() == []


def test_recursive_backward_chain_residual_on_gap():
    reg = _chainer_registry()
    chainer = BackwardChainer(reg, max_depth=4)

    target = Shape("Goal", (Const("foo"),))
    # Only BaseFact is known; Premise2 and AltFact remain unproducible
    known = {"BaseFact"}

    goal_node = chainer.search(target, known_heads=known)
    assert not goal_node.is_closed
    residuals = goal_node.collect_residuals()
    assert len(residuals) >= 1
    # Gaps reported with their origin schemas
    res_origins = {r.origin_schema for r in residuals}
    assert "rule-and" in res_origins or "rule-alt" in res_origins


def test_backward_chain_budget_and_cycle_limits():
    reg = SchemaRegistry()
    # Cyclic rule: Loop(x) <- Loop(x)
    reg.register(MethodSchema(
        id="cycle-rule",
        premises=[Shape("Loop", (Var("x"),))],
        conclusion=Shape("Loop", (Var("x"),)),
        conditions=[],
        checker=lambda ctx: True,
        elaborator="elab-cycle",
        soundness_theorem="sound",
    ))
    chainer = BackwardChainer(reg, max_depth=3, max_nodes=10)
    target = Shape("Loop", (Const("val"),))
    goal = chainer.search(target)

    assert not goal.is_closed
    residuals = goal.collect_residuals()
    assert any(r.origin_schema in ("cycle", "budget-exhausted") for r in residuals)


# --- Version Space Refinement Tests ---


def test_version_space_candidate_specific_refinement():
    vs = VersionSpace()
    s1 = MethodSchema(
        id="candidate-saturation",
        premises=[Shape("Saturated", (Var("R"),))],
        conclusion=Shape("Terminates", (Var("R"),)),
        conditions=[Condition(CondTag.S, "finite")],
        checker=lambda ctx: True,
        elaborator="elab-sat",
        soundness_theorem="sound",
    )
    s2 = MethodSchema(
        id="candidate-lpo",
        premises=[Shape("Decreases", (Var("R"),))],
        conclusion=Shape("Terminates", (Var("R"),)),
        conditions=[],
        checker=lambda ctx: True,
        elaborator="elab-lpo",
        soundness_theorem="sound",
    )
    vs.register_candidate(s1)
    vs.register_candidate(s2)

    # Refine only candidate-saturation on negative counter-evidence
    refined = vs.refine_on_negative(
        failed_schema_id="candidate-saturation",
        failed_condition="no-remove",
        new_condition_tag=CondTag.S,
        new_condition_predicate="no-remove",
        new_schema_id="OneWay-Saturation",
    )

    # Failing schema produced a refined candidate with the restricted condition
    assert refined.id == "OneWay-Saturation"
    assert any(c.predicate_ref == "no-remove" for c in refined.conditions)

    # Original s2 is untouched and unaffected
    assert "candidate-lpo" in vs.candidates
    assert vs.candidates["candidate-lpo"] == s2

