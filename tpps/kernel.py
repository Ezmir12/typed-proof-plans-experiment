"""Lean 4 integration boundary (mock for the prototype).

Real kernel calls need a pinned Lean 4 + mathlib revision; this module pins
the interface shape (env-hash, artifact refs, certificate re-checking) so the
mock can be swapped for RPC/CLI later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

ENV_HASH = "mock-env-hash-0000"  # prototype pin; real runs record the Lean revision


@dataclass(frozen=True)
class KernelArtifact:
    system: str  # e.g. "lean4+mathlib"
    env_hash: str
    ref: str


@dataclass(frozen=True)
class Certificate:
    procedure: str  # lpo | polyinterp | domain-calculus | symbolic-regularity | asymptotics
    payload: dict
    version: str = "1"


RECHECKERS: dict[str, Callable[[Certificate], bool]] = {}


def register_rechecker(procedure: str):
    def deco(fn: Callable[[Certificate], bool]):
        RECHECKERS[procedure] = fn
        return fn
    return deco


@register_rechecker("lpo")
def _recheck_lpo(cert: Certificate) -> bool:
    p = cert.payload
    return bool(p.get("precedence")) and p.get("status") in ("oriented", "equivalent")


@register_rechecker("polyinterp")
def _recheck_polyinterp(cert: Certificate) -> bool:
    p = cert.payload
    return bool(p.get("coefficients")) and p.get("decreasing") is True


@register_rechecker("domain-calculus")
def _recheck_domain(cert: Certificate) -> bool:
    return bool(cert.payload.get("containment_witness"))


@register_rechecker("symbolic-regularity")
def _recheck_regularity(cert: Certificate) -> bool:
    return "order" in cert.payload


@register_rechecker("asymptotics")
def _recheck_asymptotics(cert: Certificate) -> bool:
    return bool(cert.payload.get("limit_witness"))


def recheck(cert: Certificate) -> bool:
    """Independent re-check without re-running the search."""
    fn = RECHECKERS.get(cert.procedure)
    if fn is None:
        return False
    return fn(cert)


@dataclass
class KernelInterface:
    """Mock kernel: pass/fail + artifact hash."""

    env_hash: str = ENV_HASH
    theorems: dict[str, bool] = field(default_factory=dict)
    elaborators: dict[str, Callable] = field(default_factory=dict)
    erasure_map: dict[str, str] = field(default_factory=dict)  # schema -> kernel decl

    def check_theorem(self, ref: str) -> tuple[bool, KernelArtifact]:
        ok = self.theorems.get(ref, False)
        return ok, KernelArtifact("lean4+mathlib", self.env_hash, ref)

    def elaborate(self, schema_id: str, prem_proofs: list[str],
                  obl_proofs: list[str], certs: list[Certificate]) -> str:
        fn = self.elaborators.get(schema_id)
        if fn is None:
            return f"(by {schema_id} {' '.join(prem_proofs)} {' '.join(obl_proofs)})"
        return fn(prem_proofs, obl_proofs, certs)

    def discharge_batch(self, stmts: list[str]) -> dict[str, bool]:
        """Batch obligation discharge."""
        return {s: False for s in stmts}
