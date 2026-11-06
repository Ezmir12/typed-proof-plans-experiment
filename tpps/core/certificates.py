"""First-class certificates and verification results for external procedures.

Certificates represent structured claims from external decision procedures,
solvers, or computer algebra systems. Only a verified certificate object
can produce Certified evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .evidence import Certified, Evidence


class CertificateStatus(Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class VerifiedCertificate:
    """A verified certificate producing certified evidence."""

    procedure: str
    version: str = ""
    evidence_ref: str = ""
    env_hash: str = ""

    def to_evidence(self) -> Evidence:
        """Produce certified evidence from a verified certificate."""
        return Evidence(
            certified=Certified.CERTIFIED,
            procedure=self.procedure,
            procedure_version=self.version,
        )


@dataclass(frozen=True)
class CertificateRecord:
    """Full lifecycle certificate record including verifier identity and disposition."""

    certificate_id: str
    procedure: str
    verifier_id: str
    verifier_version: str = ""
    artifact_ref: str = ""
    env_hash: str = ""
    status: CertificateStatus = CertificateStatus.CANDIDATE
    rejection_reason: str = ""

    @property
    def is_verified(self) -> bool:
        return self.status is CertificateStatus.VERIFIED

    def to_verified(self) -> VerifiedCertificate:
        if not self.is_verified:
            raise ValueError(
                f"certificate {self.certificate_id!r} is not verified: "
                f"status={self.status.value}, reason={self.rejection_reason!r}"
            )
        return VerifiedCertificate(
            procedure=self.procedure,
            version=self.verifier_version,
            evidence_ref=self.artifact_ref,
            env_hash=self.env_hash,
        )

    def to_evidence(self) -> Evidence:
        if not self.is_verified:
            return Evidence()
        return self.to_verified().to_evidence()

