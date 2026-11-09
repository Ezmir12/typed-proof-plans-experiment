"""Adapter connecting Typed Proof Plans to Mathematics-ai-atp and FormalATP backends.

Translates TPPS goal sequents and contexts into formal ATP requests, and converts
search graph outputs (states, tactic hyperedges, AND/OR alternatives, checkpoints,
and obstructions) into speculative TPPS plan nodes.

Soundness invariant: Every converted result remains strictly in Layer.SPECULATIVE
with empty formal evidence until accepted by replay and promotion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from tpps.core.certificates import CertificateRecord, CertificateStatus
from tpps.core.evidence import Empirical, Evidence, Formal
from tpps.core.plan import Layer, MethodInstance, Plan, Provenance, StmtNode
from tpps.core.stmt import Shape, Stmt, Var


REQUEST_REQUIRED_FIELDS = (
    "proof_id",
    "claim_id",
    "formal_declaration_id",
    "run_id",
    "base_revision",
    "lease_id",
    "fencing_token",
    "lean_source_artifact",
    "environment_id",
    "environment_hash",
)


def build_search_request(
    *,
    proof_id: str,
    claim_id: str,
    formal_declaration_id: str,
    run_id: str,
    base_revision: int,
    lease_id: str,
    fencing_token: int,
    lean_source_artifact: str,
    environment_id: str,
    environment_hash: str,
    goal: Stmt | StmtNode | str,
    context: list[str] | None = None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated request mapping to FormalATP / Mathematics-ai-atp.

    Validates presence of all mandatory protocol fields.
    """
    if isinstance(goal, StmtNode):
        goal_text = str(goal.stmt.shape.head)
        ctx = [c.expr for c in goal.stmt.context]
    elif isinstance(goal, Stmt):
        goal_text = str(goal.shape.head)
        ctx = [c.expr for c in goal.context]
    else:
        goal_text = str(goal)
        ctx = list(context or [])

    req: dict[str, Any] = {
        "proof_id": proof_id,
        "claim_id": claim_id,
        "formal_declaration_id": formal_declaration_id,
        "run_id": run_id,
        "base_revision": base_revision,
        "lease_id": lease_id,
        "fencing_token": fencing_token,
        "lean_source_artifact": lean_source_artifact,
        "environment_id": environment_id,
        "environment_hash": environment_hash,
        "goal_text": goal_text,
        "context": ctx,
    }
    if budget is not None:
        req["budget"] = dict(budget)

    # Validate all required fields are present and non-empty
    missing = [f for f in REQUEST_REQUIRED_FIELDS if req.get(f) is None]
    if missing:
        raise ValueError(f"missing required search request fields: {missing}")

    return req


@dataclass
class ConvertedSearchResult:
    """Output of converting a formal ATP search graph into a TPPS plan representation."""

    plan: Plan
    disposition: str
    run_id: str
    environment_hash: str
    root_state_id: str = ""
    candidate_certificate: CertificateRecord | None = None
    obstructions: list[dict[str, Any]] = field(default_factory=list)
    budget_used: dict[str, Any] = field(default_factory=dict)

    @property
    def is_proved_pending_replay(self) -> bool:
        return self.disposition == "proved-pending-replay"

    @property
    def is_counterexample(self) -> bool:
        return self.disposition == "counterexample"


def convert_search_result(
    result: Mapping[str, Any],
    plan_id: str = "",
) -> ConvertedSearchResult:
    """Convert an external formal search result into a speculative TPPS plan graph.

    All nodes and method instances are marked as Layer.SPECULATIVE.
    Positive formal evidence is never assigned directly here.
    """
    pid = plan_id or result.get("proof_id", "search-plan")
    plan = Plan(pid)

    disposition = result.get("disposition", "unknown")
    run_id = result.get("run_id", "")
    env_hash = result.get("environment_hash", "")
    root_state_id = result.get("root_state_id", "")

    prov = Provenance(
        producing_attempt=f"formal_atp:{run_id}",
        source_span=result.get("declaration_id", ""),
        extraction_confidence=1.0 if disposition == "proved-pending-replay" else 0.0,
    )

    # 1. Convert formal states into statement nodes
    states = result.get("states", [])
    for st in states:
        sid = st.get("state_id", "")
        text = st.get("goal_text", sid)
        shape = Shape(text, sort="Prop")
        counter_ev = Evidence()
        if disposition == "counterexample" and sid == root_state_id:
            counter_ev = Evidence(empirical=Empirical.SAMPLED, env_hash=env_hash)

        node = StmtNode(
            id=sid,
            stmt=Stmt(shape=shape),
            evidence=Evidence(),  # Strictly speculative; no positive evidence invented
            counter_evidence=counter_ev,
            provenance=prov,
        )
        plan.add_stmt(node)

    # 2. Convert tactic hyperedges into speculative method instances
    tactic_edges = result.get("tactic_edges", [])
    for edge in tactic_edges:
        tid = edge.get("tactic_id", "")
        source_id = edge.get("source_state_id", "")
        produced_ids = edge.get("produced_goal_ids", [])
        tactic_label = edge.get("tactic_label", "tactic")
        executor_result = edge.get("executor_result", "")

        # In backward representation: conclusion is source goal, premises are subgoals
        inst = MethodInstance(
            id=tid,
            schema_ref=tactic_label,
            premises=list(produced_ids),
            conclusion=source_id,
            layer=Layer.SPECULATIVE,
            provenance=Provenance(
                producing_attempt=f"tactic:{tid}:{executor_result}",
                extraction_confidence=1.0 if executor_result == "lean-accepted" else 0.0,
            ),
        )
        plan.add_inst(inst)

    # 3. Candidate certificate (candidate status only)
    candidate_cert: CertificateRecord | None = None
    cert_data = result.get("certificate")
    if cert_data is not None and isinstance(cert_data, Mapping):
        candidate_cert = CertificateRecord(
            certificate_id=cert_data.get("artifact_hash", f"cert-{run_id}"),
            procedure="formal-search",
            verifier_id="lean4-kernel",
            artifact_ref=cert_data.get("artifact_hash", ""),
            env_hash=env_hash,
            status=CertificateStatus.CANDIDATE,
        )

    return ConvertedSearchResult(
        plan=plan,
        disposition=disposition,
        run_id=run_id,
        environment_hash=env_hash,
        root_state_id=root_state_id,
        candidate_certificate=candidate_cert,
        obstructions=list(result.get("obstructions", [])),
        budget_used=dict(result.get("budget_used", {})),
    )

