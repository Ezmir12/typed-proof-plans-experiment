"""Versioned, lossless serialization for typed proof plans.

The format contains data only. Executable checkers, elaborators, and other
callables are deliberately excluded; schema references are resolved by the
registry at validation time.
"""
from __future__ import annotations

from enum import Enum, IntEnum
from typing import Any

from .evidence import Certified, Empirical, Evidence, Formal, Fresh, Reviewed
from .plan import Layer, MethodInstance, Plan, Provenance, Role, StmtNode
from .stmt import App, BVar, Binder, Const, CtxEntry, Shape, Stmt, Var

SERIALIZATION_VERSION = 1


class SerializationError(ValueError):
    """Malformed or unsupported serialized plan."""


def _record(kind: str, **fields: Any) -> dict[str, Any]:
    return {"kind": kind, **fields}


def _encode_term(term: object) -> dict[str, Any]:
    if isinstance(term, Var):
        return _record("Var", name=term.name)
    if isinstance(term, BVar):
        return _record("BVar", index=term.index)
    if isinstance(term, Const):
        return _record("Const", value=term.value, sort=term.sort)
    if isinstance(term, App):
        return _record("App", head=term.head,
                       args=[_encode_term(a) for a in term.args])
    if isinstance(term, Binder):
        if term.body is None:
            raise SerializationError("binder body cannot be null")
        return _record("Binder", binder_kind=term.kind,
                       body=_encode_term(term.body))
    raise SerializationError(f"unsupported term type: {type(term).__name__}")


def _decode_term(value: object) -> Var | BVar | Const | App | Binder:
    data = _object(value, "term")
    kind = _string(data, "kind")
    if kind == "Var":
        return Var(_string(data, "name"))
    if kind == "BVar":
        return BVar(_integer(data, "index"))
    if kind == "Const":
        return Const(_string(data, "value"), _string(data, "sort"))
    if kind == "App":
        return App(_string(data, "head"),
                   tuple(_decode_term(a) for a in _list(data, "args")))
    if kind == "Binder":
        return Binder(_string(data, "binder_kind"),
                      _decode_term(data.get("body")))
    raise SerializationError(f"unknown term kind: {kind!r}")


def _encode_evidence(evidence: Evidence) -> dict[str, Any]:
    return {
        "formal": int(evidence.formal),
        "certified": int(evidence.certified),
        "reviewed": int(evidence.reviewed),
        "empirical": int(evidence.empirical),
        "fresh": int(evidence.fresh),
        "kernel_system": evidence.kernel_system,
        "env_hash": evidence.env_hash,
        "procedure": evidence.procedure,
        "procedure_version": evidence.procedure_version,
    }


def _decode_evidence(value: object) -> Evidence:
    data = _object(value, "evidence")
    try:
        return Evidence(
            formal=Formal(_integer(data, "formal")),
            certified=Certified(_integer(data, "certified")),
            reviewed=Reviewed(_integer(data, "reviewed")),
            empirical=Empirical(_integer(data, "empirical")),
            fresh=Fresh(_integer(data, "fresh")),
            kernel_system=_string(data, "kernel_system"),
            env_hash=_string(data, "env_hash"),
            procedure=_string(data, "procedure"),
            procedure_version=_string(data, "procedure_version"),
        )
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"invalid evidence: {exc}") from exc


def _encode_provenance(provenance: Provenance) -> dict[str, Any]:
    return {
        "source_span": provenance.source_span,
        "extraction_confidence": provenance.extraction_confidence,
        "producing_attempt": provenance.producing_attempt,
        "kernel_artifact": provenance.kernel_artifact,
    }


def _decode_provenance(value: object) -> Provenance:
    data = _object(value, "provenance")
    confidence = data.get("extraction_confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise SerializationError("provenance.extraction_confidence must be numeric")
    return Provenance(
        source_span=_string(data, "source_span"),
        extraction_confidence=float(confidence),
        producing_attempt=_string(data, "producing_attempt"),
        kernel_artifact=_string(data, "kernel_artifact"),
    )


def _encode_shape(shape: Shape) -> dict[str, Any]:
    return {
        "head": shape.head,
        "args": [_encode_term(a) for a in shape.args],
        "regime": shape.regime,
        "sort": shape.sort,
    }


def _decode_shape(value: object) -> Shape:
    data = _object(value, "shape")
    return Shape(
        head=_string(data, "head"),
        args=tuple(_decode_term(a) for a in _list(data, "args")),
        regime=_string(data, "regime"),
        sort=_string(data, "sort"),
    )


def _encode_stmt(stmt: Stmt) -> dict[str, Any]:
    return {
        "context": [
            {"kind": entry.kind, "expr": entry.expr,
             "introduced_by": entry.introduced_by}
            for entry in stmt.context
        ],
        "shape": _encode_shape(stmt.shape),
        "regime": stmt.regime,
    }


def _decode_stmt(value: object) -> Stmt:
    data = _object(value, "statement")
    context = []
    for entry_value in _list(data, "context"):
        entry = _object(entry_value, "context entry")
        context.append(CtxEntry(
            _string(entry, "kind"),
            _string(entry, "expr"),
            _string(entry, "introduced_by"),
        ))
    return Stmt(tuple(context), _decode_shape(data.get("shape")),
                _string(data, "regime"))


def _encode_value(value: object) -> object:
    if isinstance(value, (Var, BVar, Const, App, Binder)):
        return {"term": _encode_term(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, tuple):
        return {"tuple": [_encode_value(item) for item in value]}
    if isinstance(value, dict):
        return {
            "dict": [[_encode_value(key), _encode_value(item)]
                     for key, item in value.items()]
        }
    raise SerializationError(f"unsupported substitution value: {type(value).__name__}")


def _decode_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    data = _object(value, "value")
    keys = set(data)
    if keys == {"term"}:
        return _decode_term(data["term"])
    if keys == {"tuple"}:
        return tuple(_decode_value(item) for item in _list(data, "tuple"))
    if keys == {"dict"}:
        result = {}
        for pair in _list(data, "dict"):
            items = _list_value(pair, "dictionary entry")
            if len(items) != 2:
                raise SerializationError("dictionary entry must contain two values")
            result[_decode_value(items[0])] = _decode_value(items[1])
        return result
    raise SerializationError("unknown serialized value wrapper")


def _encode_node(node: StmtNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "stmt": _encode_stmt(node.stmt),
        "provenance": _encode_provenance(node.provenance),
        "evidence": _encode_evidence(node.evidence),
        "counter_evidence": _encode_evidence(node.counter_evidence),
        "layer": node.layer.value,
    }


def _decode_node(value: object) -> StmtNode:
    data = _object(value, "statement node")
    try:
        layer = Layer(_string(data, "layer"))
    except ValueError as exc:
        raise SerializationError(f"unknown layer: {data.get('layer')!r}") from exc
    return StmtNode(
        _string(data, "id"),
        _decode_stmt(data.get("stmt")),
        _decode_provenance(data.get("provenance")),
        _decode_evidence(data.get("evidence")),
        _decode_evidence(data.get("counter_evidence")),
        layer,
    )


def _encode_instance(instance: MethodInstance) -> dict[str, Any]:
    return {
        "id": instance.id,
        "schema_ref": instance.schema_ref,
        "premises": list(instance.premises),
        "conclusion": instance.conclusion,
        "obligations": list(instance.obligations),
        "certificates": list(instance.certificates),
        "substitution": _encode_value(instance.substitution),
        "layer": instance.layer.value,
        "provenance": _encode_provenance(instance.provenance),
    }


def _decode_instance(value: object) -> MethodInstance:
    data = _object(value, "method instance")
    try:
        layer = Layer(_string(data, "layer"))
    except ValueError as exc:
        raise SerializationError(f"unknown layer: {data.get('layer')!r}") from exc
    substitution = _decode_value(data.get("substitution"))
    if not isinstance(substitution, dict):
        raise SerializationError("instance substitution must decode to a dictionary")
    return MethodInstance(
        _string(data, "id"),
        _string(data, "schema_ref"),
        _strings(data, "premises"),
        _string(data, "conclusion"),
        _strings(data, "obligations"),
        _strings(data, "certificates"),
        substitution,
        layer,
        _decode_provenance(data.get("provenance")),
    )


def encode_plan(plan: Plan) -> dict[str, Any]:
    """Encode a plan deterministically without executable fields."""
    return {
        "version": SERIALIZATION_VERSION,
        "kind": "Plan",
        "id": plan.id,
        "stmts": [_encode_node(plan.stmts[key]) for key in sorted(plan.stmts)],
        "insts": [_encode_instance(plan.insts[key]) for key in sorted(plan.insts)],
        "roles": [
            {"object_ref": role.object_ref, "role_name": role.role_name,
             "annotation": role.annotation}
            for role in sorted(plan.roles, key=lambda item: (
                item.object_ref, item.role_name, item.annotation))
        ],
    }


def decode_plan(payload: dict) -> Plan:
    """Decode a versioned plan and reject malformed references."""
    if not isinstance(payload, dict):
        raise SerializationError("plan payload must be an object")
    if payload.get("version") != SERIALIZATION_VERSION:
        raise SerializationError(
            f"unsupported serialization version: {payload.get('version')!r}")
    if payload.get("kind") != "Plan":
        raise SerializationError("payload kind must be 'Plan'")
    plan = Plan(_string(payload, "id"))
    for value in _list(payload, "stmts"):
        node = _decode_node(value)
        if node.id in plan.stmts:
            raise SerializationError(f"duplicate statement id: {node.id!r}")
        plan.add_stmt(node)
    for value in _list(payload, "insts"):
        instance = _decode_instance(value)
        if instance.id in plan.insts:
            raise SerializationError(f"duplicate instance id: {instance.id!r}")
        try:
            plan.add_inst(instance)
        except KeyError as exc:
            raise SerializationError(str(exc)) from exc
    for value in _list(payload, "roles"):
        data = _object(value, "role")
        plan.roles.append(Role(
            _string(data, "object_ref"),
            _string(data, "role_name"),
            _string(data, "annotation"),
        ))
    return plan


def structural_diff(left: Plan, right: Plan) -> list[str]:
    """Return deterministic paths whose serialized values differ."""
    differences: list[str] = []

    def compare(a: object, b: object, path: str) -> None:
        if type(a) is not type(b):
            differences.append(path)
            return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    differences.append(f"{path}.{key}")
                else:
                    compare(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                differences.append(path)
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                compare(item_a, item_b, f"{path}[{index}]")
        elif a != b:
            differences.append(path)

    compare(encode_plan(left), encode_plan(right), "$")
    return differences


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SerializationError(f"{label} must be an object")
    return value


def _list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list):
        raise SerializationError(f"{key} must be a list")
    return value


def _list_value(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SerializationError(f"{label} must be a list")
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise SerializationError(f"{key} must be a string")
    return value


def _strings(data: dict[str, Any], key: str) -> list[str]:
    values = _list(data, key)
    if not all(isinstance(value, str) for value in values):
        raise SerializationError(f"{key} must contain only strings")
    return values


def _integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SerializationError(f"{key} must be an integer")
    return value
