import pytest

from tpps.core.evidence import Evidence, Formal, Reviewed
from tpps.core.plan import Layer, MethodInstance, Plan, Provenance, Role, StmtNode
from tpps.core.serialization import (
    SERIALIZATION_VERSION,
    SerializationError,
    decode_plan,
    encode_plan,
    structural_diff,
)
from tpps.core.stmt import App, Binder, Const, CtxEntry, Shape, Stmt, Var


def _plan() -> Plan:
    plan = Plan("serialization")
    plan.add_stmt(StmtNode(
        "premise",
        Stmt(
            context=(CtxEntry("Assumption", "finite(R)"),),
            shape=Shape(
                "Goal",
                (Binder("forall", App("P", (Var("x"),))), Const("R", "TRS")),
                regime="finite",
                sort="TRS",
            ),
            regime="finite",
        ),
        provenance=Provenance("lines 1-2", 0.8, "attempt-1", "artifact-0"),
        evidence=Evidence(reviewed=Reviewed.INDEPENDENT),
        counter_evidence=Evidence(),
        layer=Layer.SPECULATIVE,
    ))
    plan.add_stmt(StmtNode(
        "conclusion",
        Stmt(shape=Shape("Result", (Var("R"),))),
    ))
    plan.add_stmt(StmtNode(
        "obligation",
        Stmt(shape=Shape("Bounded", (Var("R"),))),
    ))
    plan.add_inst(MethodInstance(
        "instance",
        "Schema",
        ["premise"],
        "conclusion",
        obligations=["obligation"],
        certificates=["certificate"],
        substitution={"R": Const("R", "TRS"), "meta": ("x", 1)},
        layer=Layer.COMMITTED,
        provenance=Provenance(producing_attempt="attempt-2"),
    ))
    plan.roles.append(Role("premise", "Invariant", "finite state"))
    return plan


def test_plan_round_trip_preserves_structure_and_metadata():
    original = _plan()
    restored = decode_plan(encode_plan(original))
    assert structural_diff(original, restored) == []
    assert restored.insts["instance"].premises == ["premise"]
    assert restored.insts["instance"].obligations == ["obligation"]
    assert restored.roles == [Role("premise", "Invariant", "finite state")]


def test_encoding_is_versioned_and_deterministic():
    payload = encode_plan(_plan())
    assert payload["version"] == SERIALIZATION_VERSION
    assert payload == encode_plan(_plan())


def test_unknown_version_is_rejected():
    payload = encode_plan(_plan())
    payload["version"] = SERIALIZATION_VERSION + 1
    with pytest.raises(SerializationError, match="unsupported serialization version"):
        decode_plan(payload)


def test_malformed_references_and_duplicate_ids_are_rejected():
    payload = encode_plan(_plan())
    payload["insts"][0]["premises"] = ["missing"]
    with pytest.raises(SerializationError):
        decode_plan(payload)

    duplicate = encode_plan(_plan())
    duplicate["stmts"].append(dict(duplicate["stmts"][0]))
    with pytest.raises(SerializationError, match="duplicate statement id"):
        decode_plan(duplicate)


def test_unsupported_term_payload_is_rejected():
    payload = encode_plan(_plan())
    payload["stmts"][0]["stmt"]["shape"]["args"][0] = {"kind": "Unknown"}
    with pytest.raises(SerializationError, match="unknown term kind"):
        decode_plan(payload)
