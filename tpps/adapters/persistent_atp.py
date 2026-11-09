"""Adapter connecting Typed Proof Plans to the persistent-atp commit gate.

Translates TPPS plan changes into inert operation sequences, constructs
validated Proposal objects, submits them through the external CommitGate,
and propagates accepted commit provenance back into the plan.

Soundness invariant: The commit gate is the sole authority for creating
committed state. This adapter never writes to local journals directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from tpps.core.plan import Layer, Plan, Provenance


class CommitGateRejectionError(RuntimeError):
    """Raised when the external CommitGate rejects a plan proposal."""

    def __init__(self, message: str, rejections: list[Any] | None = None) -> None:
        super().__init__(message)
        self.rejections = list(rejections or [])


@dataclass(frozen=True)
class InertOp:
    """Inert graph operation descriptor matching commit_gate op algebra."""

    op: str  # "upsert_node" | "set_field" | "add_edge" | "remove_edge"
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        d = {"op": self.op}
        d.update(self.payload)
        return d


@dataclass(frozen=True)
class PlanProposal:
    """Inert proposal offered to the external CommitGate."""

    proof_id: str
    actor: str
    worker_class: str
    ops: tuple[dict[str, Any], ...]
    base_revision: int | None = None
    lease_id: str | None = None
    fencing_token: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "actor": self.actor,
            "worker_class": self.worker_class,
            "base_revision": self.base_revision,
            "lease_id": self.lease_id,
            "fencing_token": self.fencing_token,
            "ops": list(self.ops),
        }


def plan_to_inert_ops(plan: Plan) -> list[dict[str, Any]]:
    """Convert statement nodes, instances, and transforms into inert operations."""
    ops: list[dict[str, Any]] = []

    # 1. Statement nodes
    for nid, node in plan.stmts.items():
        ops.append({
            "op": "upsert_node",
            "label": "Statement",
            "id": nid,
            "fields": {
                "shape_head": node.stmt.shape.head,
                "sort": node.stmt.shape.sort,
                "regime": node.stmt.regime,
                "has_counter_evidence": node.counter_evidence.formal.name != "NONE",
            },
        })

    # 2. Method instances & structural edges
    for inst in plan.insts.values():
        ops.append({
            "op": "upsert_node",
            "label": "MethodInstance",
            "id": inst.id,
            "fields": {
                "schema_ref": inst.schema_ref,
                "layer": inst.layer.value,
                "is_committed": inst.layer is Layer.COMMITTED,
            },
        })
        # Edges from premises to instance
        for idx, prem_id in enumerate(inst.premises):
            ops.append({
                "op": "add_edge",
                "source": prem_id,
                "target": inst.id,
                "kind": "premise",
                "fields": {"position": idx},
            })
        # Edge from instance to conclusion
        ops.append({
            "op": "add_edge",
            "source": inst.id,
            "target": inst.conclusion,
            "kind": "derives",
            "fields": {},
        })

    # 3. Transform records
    for tr in plan.transforms:
        ops.append({
            "op": "upsert_node",
            "label": "TransformRecord",
            "id": f"tr-{tr.kind}-{tr.source}->{tr.target}",
            "fields": {
                "kind": tr.kind,
                "source": tr.source,
                "target": tr.target,
                "obligations": list(tr.obligations),
            },
        })

    return ops


def build_plan_proposal(
    plan: Plan,
    *,
    proof_id: str,
    actor: str = "tpps-worker",
    worker_class: str = "formal-prover",
    base_revision: int | None = None,
    lease_id: str | None = None,
    fencing_token: int | None = None,
) -> PlanProposal:
    """Build an inert proposal containing the plan's operations."""
    ops = plan_to_inert_ops(plan)
    return PlanProposal(
        proof_id=proof_id,
        actor=actor,
        worker_class=worker_class,
        ops=tuple(ops),
        base_revision=base_revision,
        lease_id=lease_id,
        fencing_token=fencing_token,
    )


def submit_to_commit_gate(
    proposal: PlanProposal,
    gate: Any,
) -> Any:
    """Submit an inert proposal through the external commit gate.

    Never writes to a local journal directly. Raises CommitGateRejectionError
    if the gate rejects the proposal.
    """
    if not hasattr(gate, "commit"):
        raise TypeError(f"expected CommitGate object with .commit() method, got {type(gate)!r}")

    # Convert to gate Proposal object if gate expects its own dataclass
    payload = proposal
    try:
        from commit_gate.proposal import Proposal
        if hasattr(gate, "_store"):  # Real CommitGate instance
            payload = Proposal.from_dict(proposal.to_dict())
    except ImportError:
        pass

    result = gate.commit(payload)

    # Check acceptance
    accepted = getattr(result, "accepted", False)
    if isinstance(result, Mapping):
        accepted = result.get("accepted", False)

    if not accepted:
        rejections = getattr(result, "rejections", ())
        if isinstance(result, Mapping):
            rejections = result.get("rejections", ())
        raise CommitGateRejectionError(
            f"CommitGate rejected proposal for proof {proposal.proof_id!r}: {rejections}",
            rejections=list(rejections),
        )

    return result


def apply_commit_provenance(plan: Plan, commit_result: Any) -> Plan:
    """Update plan provenance with the durable revision and event hash."""
    rev = getattr(commit_result, "revision", None)
    event_hash = getattr(commit_result, "event_hash", "")
    if isinstance(commit_result, Mapping):
        rev = commit_result.get("revision", rev)
        event_hash = commit_result.get("event_hash", event_hash)

    prov_str = f"persistent-atp:rev-{rev}:{event_hash}"
    for node in plan.stmts.values():
        node.provenance = Provenance(
            producing_attempt=prov_str,
            source_span=node.provenance.source_span,
            extraction_confidence=node.provenance.extraction_confidence,
            kernel_artifact=node.provenance.kernel_artifact,
        )
    for inst in plan.insts.values():
        inst.provenance = Provenance(
            producing_attempt=prov_str,
            source_span=inst.provenance.source_span,
            extraction_confidence=inst.provenance.extraction_confidence,
            kernel_artifact=inst.provenance.kernel_artifact,
        )
    return plan

