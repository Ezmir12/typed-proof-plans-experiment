import pytest

from tpps.core.schema import Condition, CondTag, MethodSchema, SchemaRegistry, TrustStatus
from tpps.core.stmt import Shape
from tpps.core.trust import TrustDecision, can_promote, reject_model_authority


def _schema() -> MethodSchema:
    return MethodSchema(
        id="trust-test",
        premises=[Shape("A")],
        conclusion=Shape("B"),
        conditions=[Condition(CondTag.S, "structural")],
        checker=lambda ctx: True,
        elaborator="candidate-elaborator",
        soundness_theorem="candidate-theorem",
    )


def test_promotion_policy_requires_explicit_review_or_artifact():
    reviewed = TrustDecision(
        TrustStatus.REVIEWED,
        "two independent reviews",
        reviewer_refs=("reviewer-a", "reviewer-b"),
    )
    kernel = TrustDecision(
        TrustStatus.KERNEL_VERIFIED,
        "kernel artifact replayed",
        artifact_ref="lean://artifact/1",
    )
    assert can_promote(TrustStatus.CANDIDATE, reviewed, model_originated=False)
    assert can_promote(TrustStatus.REVIEWED, kernel, model_originated=False)
    assert not can_promote(TrustStatus.CANDIDATE, reviewed, model_originated=True)
    assert not can_promote(
        TrustStatus.CANDIDATE,
        TrustDecision(TrustStatus.KERNEL_VERIFIED, "missing artifact"),
        model_originated=False,
    )


def test_model_authority_payload_is_rejected():
    with pytest.raises(ValueError, match="cannot authorize trust promotion"):
        reject_model_authority({"trust_status": "kernel-verified"})
    reject_model_authority({"confidence": 0.99})


def test_registry_accepts_explicit_decision_and_preserves_compatibility():
    reg = SchemaRegistry()
    reg.register(_schema())
    reg.promote(
        "trust-test",
        decision=TrustDecision(
            TrustStatus.REVIEWED,
            "reviewed by two people",
            reviewer_refs=("a", "b"),
        ),
    )
    assert reg.schemas["trust-test"].trust.status is TrustStatus.REVIEWED
    reg.promote(
        "trust-test",
        decision=TrustDecision(
            TrustStatus.KERNEL_VERIFIED,
            "replayed artifact",
            artifact_ref="lean://artifact/2",
        ),
    )
    assert reg.usable_in_committed("trust-test")


def test_registry_rejects_model_originated_decision():
    reg = SchemaRegistry()
    reg.register(_schema())
    with pytest.raises(ValueError):
        reg.promote(
            "trust-test",
            decision=TrustDecision(
                TrustStatus.KERNEL_VERIFIED,
                "model says verified",
                artifact_ref="model://artifact",
            ),
            model_originated=True,
        )


def test_review_requires_two_nonempty_distinct_references():
    decision = TrustDecision(
        TrustStatus.REVIEWED,
        "reviewed",
        reviewer_refs=(" ", "reviewer-a"),
    )
    assert not can_promote(TrustStatus.CANDIDATE, decision,
                           model_originated=False)


def test_legacy_kernel_promotion_requires_explicit_artifact_decision():
    reg = SchemaRegistry()
    reg.register(_schema())
    with pytest.raises(ValueError, match="explicit artifact-backed"):
        reg.promote("trust-test", kernel_verified=True)
