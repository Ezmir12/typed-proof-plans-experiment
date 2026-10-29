"""Trust-state promotion policy for method schemas.

The trust-state data model remains in :mod:`tpps.core.schema`; this module
validates promotion decisions and keeps model output outside the authority
boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .schema import TrustStatus


@dataclass(frozen=True)
class TrustDecision:
    status: TrustStatus
    reason: str
    artifact_ref: str = ""
    reviewer_refs: tuple[str, ...] = ()


def can_promote(
    current: TrustStatus,
    decision: TrustDecision,
    *,
    model_originated: bool,
) -> bool:
    """Return whether a decision may move a schema to a stronger trust state."""
    if model_originated or not decision.reason.strip():
        return False
    if current is TrustStatus.KERNEL_VERIFIED:
        return False

    if decision.status is TrustStatus.REVIEWED:
        refs = {ref.strip() for ref in decision.reviewer_refs if ref.strip()}
        return (
            current is TrustStatus.CANDIDATE
            and len(refs) >= 2
        )
    if decision.status is TrustStatus.KERNEL_VERIFIED:
        return (
            current in (TrustStatus.CANDIDATE, TrustStatus.REVIEWED)
            and bool(decision.artifact_ref.strip())
        )
    return False


def reject_model_authority(payload: object) -> None:
    """Reject payloads that attempt to make model output authoritative."""
    if not isinstance(payload, Mapping):
        return
    prohibited = {
        "trust_decision",
        "trust_status",
        "promotion",
        "kernel_verified",
        "artifact_ref",
    }
    if prohibited.intersection(payload):
        raise ValueError("model-originated payload cannot authorize trust promotion")
