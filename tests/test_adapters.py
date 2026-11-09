"""Tests for external adapters: Mathematics-ai-atp and persistent-atp commit gate."""
import pytest

from tpps.adapters.mathematics_ai_atp import (
    ConvertedSearchResult,
    build_search_request,
    convert_search_result,
)
from tpps.adapters.persistent_atp import (
    CommitGateRejectionError,
    apply_commit_provenance,
    build_plan_proposal,
    plan_to_inert_ops,
    submit_to_commit_gate,
)
from tpps.core.certificates import CertificateStatus
from tpps.core.evidence import Evidence, Formal
from tpps.core.plan import Layer, MethodInstance, Plan, StmtNode
from tpps.core.stmt import Shape, Stmt


# --- Mathematics-ai-atp Search Adapter Tests ---


def test_build_search_request_validates_required_fields():
    # Valid request
    req = build_search_request(
        proof_id="proof-001",
        claim_id="claim-001",
        formal_declaration_id="decl-001",
        run_id="run-001",
        base_revision=3,
        lease_id="lease-001",
        fencing_token=42,
        lean_source_artifact="art-source-123",
        environment_id="env-lean4",
        environment_hash="env-hash-abc",
        goal=Stmt(shape=Shape("Terminates")),
    )
    assert req["proof_id"] == "proof-001"
    assert req["run_id"] == "run-001"
    assert req["goal_text"] == "Terminates"

    # Missing required field raises ValueError
    with pytest.raises(ValueError, match="missing required search request fields"):
        build_search_request(
            proof_id="proof-001",
            claim_id="claim-001",
            formal_declaration_id="decl-001",
            run_id="run-001",
            base_revision=3,
            lease_id="lease-001",
            fencing_token=42,
            lean_source_artifact="art-source-123",
            environment_id="env-lean4",
            environment_hash=None,  # missing environment_hash!
            goal="Goal",
        )


def test_convert_search_result_enforces_speculative_layer_and_empty_formal_evidence():
    fake_result = {
        "proof_id": "proof-100",
        "run_id": "run-100",
        "declaration_id": "decl-100",
        "environment_hash": "env-100",
        "disposition": "proved-pending-replay",
        "root_state_id": "proof-100/fs-1",
        "states": [
            {"state_id": "proof-100/fs-1", "goal_text": "EvenSum"},
        ],
        "tactic_edges": [
            {
                "tactic_id": "proof-100/ta-1",
                "source_state_id": "proof-100/fs-1",
                "executor_result": "lean-accepted",
                "tactic_label": "exact",
                "produced_goal_ids": [],
            }
        ],
        "certificate": {
            "artifact_hash": "cert-hash-xyz",
            "status": "candidate",
        },
        "obstructions": [],
        "budget_used": {"tactic_steps": 1},
    }

    converted = convert_search_result(fake_result)
    assert isinstance(converted, ConvertedSearchResult)
    assert converted.is_proved_pending_replay
    assert converted.root_state_id == "proof-100/fs-1"

    plan = converted.plan
    # Converted state must be strictly speculative and have NO positive formal evidence
    root_node = plan.stmts["proof-100/fs-1"]
    assert root_node.evidence.formal == Formal.NONE
    assert root_node.evidence.certified.name == "NONE"

    # Converted tactic instance must be marked Layer.SPECULATIVE
    inst = plan.insts["proof-100/ta-1"]
    assert inst.layer is Layer.SPECULATIVE

    # Candidate certificate must remain in CANDIDATE status
    assert converted.candidate_certificate is not None
    assert converted.candidate_certificate.status is CertificateStatus.CANDIDATE
    assert not converted.candidate_certificate.is_verified


def test_convert_search_result_handles_counterexample():
    fake_counterexample = {
        "proof_id": "proof-counter",
        "run_id": "run-counter",
        "environment_hash": "env-hash-1",
        "disposition": "counterexample",
        "root_state_id": "proof-counter/fs-1",
        "states": [
            {"state_id": "proof-counter/fs-1", "goal_text": "Fails"},
        ],
        "tactic_edges": [],
        "obstructions": [{"kind": "likely-false"}],
    }

    converted = convert_search_result(fake_counterexample)
    assert converted.is_counterexample
    root_node = converted.plan.stmts["proof-counter/fs-1"]
    assert root_node.counter_evidence.empirical.name != "NONE"


def test_convert_search_result_handles_budget_exhausted_and_stagnated():
    budget_result = {
        "proof_id": "proof-budget",
        "run_id": "run-budget",
        "environment_hash": "env-hash-2",
        "disposition": "budget-exhausted",
        "root_state_id": "proof-budget/fs-1",
        "states": [{"state_id": "proof-budget/fs-1", "goal_text": "OpenGoal"}],
        "tactic_edges": [],
        "budget_used": {"tactic_steps": 100, "wall_clock_ms": 5000},
    }
    converted = convert_search_result(budget_result)
    assert converted.disposition == "budget-exhausted"
    assert converted.budget_used["tactic_steps"] == 100
    assert converted.plan.stmts["proof-budget/fs-1"].evidence.formal == Formal.NONE


def test_submit_to_commit_gate_fencing_and_lease_rejections():
    plan = _sample_plan()
    proposal = build_plan_proposal(
        plan,
        proof_id="proof-abc",
        actor="tpps-worker-1",
        base_revision=2,
        lease_id="expired-lease",
        fencing_token=10,
    )
    gate = MockCommitGate(should_accept=False, rejections=("fencing-mismatch", "lease-loss"))
    with pytest.raises(CommitGateRejectionError) as exc_info:
        submit_to_commit_gate(proposal, gate)
    assert "fencing-mismatch" in exc_info.value.rejections
    assert "lease-loss" in exc_info.value.rejections


# --- Persistent-atp Commit Gate Adapter Tests ---


class MockCommitGate:
    """Test double for external CommitGate."""

    def __init__(self, *, should_accept: bool = True, rejections: tuple = ()) -> None:
        self.should_accept = should_accept
        self.rejections = rejections
        self.committed_proposals = []

    def commit(self, proposal):
        self.committed_proposals.append(proposal)
        if self.should_accept:
            return type("CommitResult", (), {
                "accepted": True,
                "rejections": (),
                "event_hash": "evt-hash-999",
                "revision": 7,
            })()
        return type("CommitResult", (), {
            "accepted": False,
            "rejections": self.rejections or ("lease-expired",),
            "event_hash": None,
            "revision": None,
        })()


def _sample_plan() -> Plan:
    plan = Plan("test-plan")
    plan.add_stmt(StmtNode("s1", Stmt(shape=Shape("Axiom"))))
    plan.add_stmt(StmtNode("s2", Stmt(shape=Shape("Theorem"))))
    plan.add_inst(MethodInstance("i1", "Rule", ["s1"], "s2", layer=Layer.COMMITTED))
    return plan


def test_plan_to_inert_ops():
    plan = _sample_plan()
    ops = plan_to_inert_ops(plan)
    assert len(ops) >= 3
    # Check that ops follow the commit_gate algebra
    labels = {op.get("label") for op in ops if op.get("op") == "upsert_node"}
    assert "Statement" in labels
    assert "MethodInstance" in labels
    assert any(op.get("op") == "add_edge" for op in ops)


def test_submit_to_commit_gate_success_and_provenance():
    plan = _sample_plan()
    proposal = build_plan_proposal(
        plan,
        proof_id="proof-abc",
        actor="tpps-worker-1",
        base_revision=6,
        lease_id="lease-xyz",
    )
    assert proposal.proof_id == "proof-abc"
    assert proposal.base_revision == 6

    gate = MockCommitGate(should_accept=True)
    result = submit_to_commit_gate(proposal, gate)
    assert result.accepted
    assert result.revision == 7

    # Apply provenance back to plan
    updated_plan = apply_commit_provenance(plan, result)
    assert "rev-7" in updated_plan.stmts["s1"].provenance.producing_attempt
    assert "evt-hash-999" in updated_plan.insts["i1"].provenance.producing_attempt


def test_submit_to_commit_gate_surfaces_rejection():
    plan = _sample_plan()
    proposal = build_plan_proposal(
        plan,
        proof_id="proof-abc",
        actor="tpps-worker-1",
    )
    gate = MockCommitGate(should_accept=False, rejections=("stale-base-revision",))

    with pytest.raises(CommitGateRejectionError, match="stale-base-revision") as exc_info:
        submit_to_commit_gate(proposal, gate)
    assert "stale-base-revision" in exc_info.value.rejections

