"""Deterministic test double for proof store and commit-gate typing passes.

NOTE: This is an in-memory mock store intended exclusively for local, deterministic
unit testing. It is NOT the durable external store. Durable persistence and journal
authority are provided by `persistent-atp` (see `tpps.adapters.persistent_atp`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class CommitRecord:
    owner: str
    atoms: list[str]
    check_ok: bool
    check_diagnostic: str
    content_hash: str


class MockStore:
    """In-memory deterministic test double for storage and commit validation."""
    def __init__(self, owner: str = "typed-proof-plans") -> None:
        self.owner = owner
        self.atoms: dict[str, str] = {}  # content-hash -> atom
        self.commits: list[CommitRecord] = []
        self._leases: set[str] = set()

    @staticmethod
    def _hash(atom: str) -> str:
        return hashlib.sha256(atom.encode()).hexdigest()[:16]

    def lease(self, key: str) -> str:
        self._leases.add(key)
        return key

    def release(self, key: str) -> None:
        self._leases.discard(key)

    def commit_gate_typing_pass(self, atoms: list[str], check_fn) -> CommitRecord:
        """Typecheck + evidence aggregation run before admission."""
        ok, diagnostic = check_fn()
        hashes = [self._hash(a) for a in atoms]
        for h, a in zip(hashes, atoms):
            self.atoms[h] = a  # append-only: never overwritten
        rec = CommitRecord(self.owner, atoms, ok, diagnostic,
                           hashlib.sha256("".join(hashes).encode()).hexdigest()[:16])
        self.commits.append(rec)
        return rec

    def retrieve_by_schema(self, schema_ref: str) -> list[str]:
        """Context-compiler retrieval over plan atoms."""
        return [a for a in self.atoms.values() if f" {schema_ref} " in a]
