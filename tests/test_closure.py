"""Tests for recursive evidence closure, refutation propagation, and certificate replay."""
import pytest

from tpps.adapters.kernel import KernelArtifact, replay_evidence
from tpps.core.certificates import CertificateRecord, CertificateStatus, VerifiedCertificate
from tpps.core.closure import EvidenceClosure
from tpps.core.evidence import Certified, Evidence, Formal, Reviewed
from tpps.core.plan import Layer, MethodInstance, Plan, StmtNode
from tpps.core.schema import Condition, CondTag, MethodSchema, SchemaRegistry, TrustRecord, TrustStatus
from tpps.core.stmt import Shape, Stmt
from tpps.core.trust import TrustDecision
from tpps.reasoning.transform import Bypass


def _registry(*schema_ids: str) -> SchemaRegistry:
    registry = SchemaRegistry()
    for schema_id in schema_ids:
        registry.register(MethodSchema(
            id=schema_id,
            premises=[Shape("Premise")],
            conclusion=Shape("Conclusion"),
            conditions=[Condition(CondTag.S, "shape")],
            checker=lambda _ctx: True,
            elaborator="elaborate",
            soundness_theorem="sound",
            trust=TrustRecord(
                status=TrustStatus.KERNEL_VERIFIED,
                evidence=Evidence(
                    formal=Formal.KERNEL,
                    reviewed=Reviewed.TWO_INDEPENDENT,
                    env_hash="env-001"))))
    return registry


def _plan(*, include_alternative: bool = True) -> Plan:
    plan = Plan("closure")
    plan.add_stmt(StmtNode("premise", Stmt(shape=Shape("Premise")),
                           evidence=Evidence(
                               certified=Certified.CERTIFIED,
                               reviewed=Reviewed.INDEPENDENT)))
    plan.add_stmt(StmtNode("alternative", Stmt(shape=Shape("Premise")),
                           evidence=Evidence(
                               formal=Formal.KERNEL,
                               reviewed=Reviewed.INDEPENDENT,
                               env_hash="env-001")))
    plan.add_stmt(StmtNode("target", Stmt(shape=Shape("Conclusion"))))
    plan.add_inst(MethodInstance("derive", "derive", ["premise"], "target"))
    if include_alternative:
        plan.add_inst(MethodInstance("alternative-derive", "derive",
                                     ["alternative"], "target"))
    return plan


def test_closure_conjoins_schema_and_premise_evidence():
    closure = EvidenceClosure(_plan(), _registry("derive"))
    result = closure.close_plan()

    assert result.evidence["target"].reviewed == Reviewed.INDEPENDENT
    assert result.evidence["target"].certified == Certified.NONE
    assert result.support["target"] == {"target", "premise", "alternative"}


def test_kernel_refuted_support_is_poisoned_but_alternative_remains():
    plan = _plan()
    plan.stmts["premise"].counter_evidence = Evidence(
        formal=Formal.KERNEL, env_hash="env-001")
    closure = EvidenceClosure(plan, _registry("derive"))

    result = closure.close_plan()

    assert result.evidence["target"].formal == Formal.KERNEL
    assert "alternative" in result.support["target"]
    assert "premise" not in result.support["target"]


def test_inconsistent_flagged_when_support_is_refuted():
    plan = _plan(include_alternative=False)
    plan.stmts["premise"].counter_evidence = Evidence(
        formal=Formal.KERNEL, env_hash="env-001")
    closure = EvidenceClosure(plan, _registry("derive"))

    result = closure.close_plan()

    assert result.evidence["target"].formal == Formal.NONE
    assert "premise" in result.refuted
    assert "target" in result.inconsistent


def test_invalidation_recomputes_dependent_conclusions():
    plan = _plan(include_alternative=False)
    plan.stmts["premise"].evidence = Evidence(certified=Certified.CERTIFIED)
    closure = EvidenceClosure(plan, _registry("derive"))
    assert closure.close_node("target")[0].reviewed == Reviewed.NONE

    plan.stmts["premise"].evidence = Evidence(reviewed=Reviewed.INDEPENDENT)
    closure.invalidate("premise")
    assert closure.close_node("target")[0].reviewed == Reviewed.INDEPENDENT


def test_speculative_instances_do_not_contribute_closure():
    plan = _plan()
    plan.insts["derive"].layer = Layer.SPECULATIVE
    plan.insts["alternative-derive"].layer = Layer.SPECULATIVE
    closure = EvidenceClosure(plan, _registry("derive"))

    assert closure.close_node("target")[0] == plan.stmts["target"].evidence


def test_obligation_evidence_is_required_by_derivation():
    plan = _plan(include_alternative=False)
    plan.add_stmt(StmtNode("obligation", Stmt(shape=Shape("Premise"))))
    plan.insts["derive"].obligations.append("obligation")
    closure = EvidenceClosure(plan, _registry("derive"))

    result = closure.close_plan()

    assert result.evidence["target"].reviewed == Reviewed.NONE
    assert "obligation" in result.support["target"]


def test_cycle_is_reported_and_remains_open():
    plan = _plan(include_alternative=False)
    plan.insts["derive"].premises = ["target"]
    closure = EvidenceClosure(plan, _registry("derive"))

    result = closure.close_plan()

    assert result.evidence["target"].formal == Formal.NONE
    assert any("cycle" in diagnostic for diagnostic in result.diagnostics)


def test_closure_reads_trust_written_by_promote():
    reg = SchemaRegistry()
    reg.register(MethodSchema(
        id="sound-rule",
        premises=[Shape("Premise")],
        conclusion=Shape("Conclusion"),
        conditions=[Condition(CondTag.S, "shape")],
        checker=lambda _ctx: True,
        elaborator="elaborate",
        soundness_theorem="sound",
    ))
    # Promote via explicit TrustDecision
    reg.promote(
        "sound-rule",
        decision=TrustDecision(
            TrustStatus.KERNEL_VERIFIED,
            "verified in Lean 4",
            artifact_ref="artifact-lean4-123",
        ),
    )
    # Verify promote populated trust evidence tau automatically
    assert reg.schemas["sound-rule"].trust.evidence.formal == Formal.KERNEL
    assert reg.schemas["sound-rule"].trust.evidence.env_hash == "artifact-lean4-123"

    plan = Plan("test-promote-closure")
    plan.add_stmt(StmtNode("premise", Stmt(shape=Shape("Premise")),
                           evidence=Evidence(formal=Formal.KERNEL, env_hash="artifact-lean4-123")))
    plan.add_stmt(StmtNode("target", Stmt(shape=Shape("Conclusion"))))
    plan.add_inst(MethodInstance("derive", "sound-rule", ["premise"], "target"))

    closure = EvidenceClosure(plan, reg)
    result = closure.close_plan()
    assert result.evidence["target"].formal == Formal.KERNEL
    assert result.evidence["target"].env_hash == "artifact-lean4-123"


def test_closure_through_bypass_transform():
    plan = Plan("bypass-plan")
    plan.add_stmt(StmtNode("src", Stmt(shape=Shape("From", ())),
                           evidence=Evidence(formal=Formal.KERNEL, env_hash="env-001")))
    bp = Bypass(Shape("From", ()), Shape("To", ()), blame="claim-drift")
    res = bp.apply(plan)
    assert res is not None

    reg = SchemaRegistry()
    closure = EvidenceClosure(plan, reg)
    result = closure.close_plan()

    # Cast target remains open while implication obligation is open
    assert result.evidence["cast-To"].formal == Formal.NONE

    # Discharging the obligation closes the cast target
    plan.stmts[res.obligations[0]].evidence = Evidence(
        formal=Formal.KERNEL, env_hash="env-001")
    closure.invalidate(res.obligations[0])
    ev, sup = closure.close_node("cast-To")
    assert ev.formal == Formal.KERNEL
    assert "src" in sup


def test_certificates_data_model_and_evidence():
    candidate_cert = CertificateRecord(
        certificate_id="cert-1",
        procedure="polyinterp",
        verifier_id="poly-verifier",
        status=CertificateStatus.CANDIDATE,
    )
    assert not candidate_cert.is_verified
    assert candidate_cert.to_evidence().certified == Certified.NONE

    verified_cert = CertificateRecord(
        certificate_id="cert-2",
        procedure="polyinterp",
        verifier_id="poly-verifier",
        verifier_version="1.0.0",
        artifact_ref="art-poly",
        status=CertificateStatus.VERIFIED,
    )
    assert verified_cert.is_verified
    ev = verified_cert.to_evidence()
    assert ev.certified == Certified.CERTIFIED
    assert ev.procedure == "polyinterp"
    assert ev.procedure_version == "1.0.0"


def test_replay_evidence_keeps_formal_and_certified_independent():
    # Procedure replay produces certified evidence only
    cert = CertificateRecord(
        certificate_id="cert-lpo",
        procedure="lpo",
        verifier_id="lpo-verifier",
        status=CertificateStatus.VERIFIED,
    )
    ev_cert = replay_evidence(cert_record=cert)
    assert ev_cert.certified == Certified.CERTIFIED
    assert ev_cert.formal == Formal.NONE

    # Kernel artifact replay produces formal evidence only
    art = KernelArtifact("lean4+mathlib", "env-lean", "thm-1")
    ev_art = replay_evidence(artifact=art, recheck_ok=True)
    assert ev_art.formal == Formal.KERNEL
    assert ev_art.certified == Certified.NONE
    assert ev_art.env_hash == "env-lean"

