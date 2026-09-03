"""Five-dimensional product evidence, AND/OR aggregation, refutation.

Dimensions aggregate componentwise; never collapsed to a scalar for
trust decisions. PLN annotations live outside this module (see creative.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Formal(IntEnum):
    NONE = 0
    SKETCHED = 1
    KERNEL = 2


class Certified(IntEnum):
    NONE = 0
    CERTIFIED = 1


class Reviewed(IntEnum):
    NONE = 0
    AUTHOR = 1
    INDEPENDENT = 2
    TWO_INDEPENDENT = 3


class Empirical(IntEnum):
    NONE = 0
    SAMPLED = 1
    EXHAUSTIVE_FINITE = 2


class Fresh(IntEnum):
    STALE = 0
    FRESH = 1


@dataclass(frozen=True)
class Evidence:
    """µ(v) = (formal, certified, reviewed, empirical, fresh)."""

    formal: Formal = Formal.NONE
    certified: Certified = Certified.NONE
    reviewed: Reviewed = Reviewed.NONE
    empirical: Empirical = Empirical.NONE
    fresh: Fresh = Fresh.FRESH
    # Kernel provenance; required non-empty whenever formal == KERNEL.
    kernel_system: str = ""
    env_hash: str = ""
    # Certified-procedure provenance.
    procedure: str = ""
    procedure_version: str = ""

    def __post_init__(self) -> None:
        if self.formal == Formal.KERNEL and not self.env_hash:
            raise ValueError("formal=kernel requires a non-empty env_hash")

    def ge(self, other: Evidence) -> bool:
        return (
            self.formal >= other.formal
            and self.certified >= other.certified
            and self.reviewed >= other.reviewed
            and self.empirical >= other.empirical
            and self.fresh >= other.fresh
        )

    def meet(self, other: Evidence) -> Evidence:
        return Evidence(
            formal=min(self.formal, other.formal),
            certified=min(self.certified, other.certified),
            reviewed=min(self.reviewed, other.reviewed),
            empirical=min(self.empirical, other.empirical),
            fresh=min(self.fresh, other.fresh),
            kernel_system=self.kernel_system if self.formal <= other.formal else other.kernel_system,
            env_hash=self.env_hash if self.formal <= other.formal else other.env_hash,
            procedure=self.procedure or other.procedure,
            procedure_version=self.procedure_version or other.procedure_version,
        )

    def join(self, other: Evidence) -> Evidence:
        return Evidence(
            formal=max(self.formal, other.formal),
            certified=max(self.certified, other.certified),
            reviewed=max(self.reviewed, other.reviewed),
            empirical=max(self.empirical, other.empirical),
            fresh=max(self.fresh, other.fresh),
            kernel_system=self.kernel_system if self.formal >= other.formal else other.kernel_system,
            env_hash=self.env_hash if self.formal >= other.formal else other.env_hash,
            procedure=self.procedure or other.procedure,
            procedure_version=self.procedure_version or other.procedure_version,
        )


BOTTOM = Evidence(fresh=Fresh.STALE)


def aggregate(direct: Evidence, derivations: list[Evidence]) -> Evidence:
    """µ(v) = µ_direct(v) ∨ ⋁ derivations."""
    out = direct
    for d in derivations:
        out = out.join(d)
    return out


def conjoin(parts: list[Evidence]) -> Evidence:
    """τ ∧ ⋀ premises ∧ ⋀ obligations (componentwise meet)."""
    if not parts:
        return Evidence(fresh=Fresh.FRESH)
    out = parts[0]
    for p in parts[1:]:
        out = out.meet(p)
    return out


def is_closed(mu: Evidence, requirement: Evidence) -> bool:
    """Node closed at ρ iff µ(v) ≥ ρ componentwise."""
    return mu.ge(requirement)


def is_kernel_refuted(nu: Evidence) -> bool:
    """A node refuted at kernel level poisons committed derivations."""
    return nu.formal == Formal.KERNEL


def check_support_consistent(support_nus: list[Evidence]) -> bool:
    """False (inconsistent) if any committed-support node is kernel-refuted."""
    return not any(is_kernel_refuted(nu) for nu in support_nus)


@dataclass
class EvidenceStore:
    """Cached-but-invalidatable closure facts."""

    mus: dict[str, Evidence] = field(default_factory=dict)
    _closure_cache: dict[tuple[str, Evidence], bool] = field(default_factory=dict)

    def set(self, node_id: str, mu: Evidence) -> None:
        self.mus[node_id] = mu
        self._closure_cache = {k: v for k, v in self._closure_cache.items() if k[0] != node_id}

    def closure(self, node_id: str, requirement: Evidence) -> bool:
        key = (node_id, requirement)
        if key not in self._closure_cache:
            self._closure_cache[key] = is_closed(self.mus.get(node_id, BOTTOM), requirement)
        return self._closure_cache[key]
