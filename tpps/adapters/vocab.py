"""Typed-vocabulary export: serialize a schema registry (plus sorts and role
slots) into the compact, fixed vocabulary an LLM extractor must stay inside.

The model may only emit schema names, shape heads, and role names appearing
here; anything else is rejected before typechecking ever sees it.
"""
from __future__ import annotations

from tpps.core.schema import SchemaRegistry
from tpps.core.sorts import SortRegistry


def _term_shape(shape) -> dict:
    return {"head": shape.head,
            "arity": len(shape.args),
            "regime": shape.regime,
            "sort": shape.sort}


def export_vocabulary(registry: SchemaRegistry, sorts: SortRegistry | None = None,
                      roles: list[str] | None = None) -> dict:
    """Serialize the registry's typed vocabulary to plain data."""
    return {
        "schemas": [
            {"name": sid,
             "premises": [_term_shape(p) for p in s.premises],
             "conclusion": _term_shape(s.conclusion),
             "conditions": [{"class": c.tag.value, "predicate": c.predicate_ref}
                            for c in s.conditions],
             "obligation_templates": [_term_shape(t) for t in s.obligation_templates],
             "roles": list(s.roles)}
            for sid, s in registry.schemas.items() if not s.is_transform
        ],
        "sorts": [
            {"name": name, "params": list(d.params),
             "attributes": [a.name for a in d.attributes],
             "erasure": d.erasure_target}
            for name, d in (sorts.sorts.items() if sorts else [])
        ],
        "roles": list(roles or []),
    }


def vocab_prompt(vocab: dict) -> str:
    """Render the vocabulary as compact prompt text."""
    lines = ["SCHEMAS (name | premises => conclusion | obligations | roles):"]
    for s in vocab["schemas"]:
        prems = ", ".join(f"{p['head']}/{p['arity']}" for p in s["premises"])
        obls = ", ".join(o["head"] for o in s["obligation_templates"]) or "-"
        roles = ", ".join(s["roles"]) or "-"
        lines.append(f"- {s['name']}: {prems} => {s['conclusion']['head']} "
                     f"| obls: {obls} | roles: {roles}")
    if vocab["sorts"]:
        lines.append("SORTS: " + ", ".join(
            f"{d['name']}{list(d['params'])}" for d in vocab["sorts"]))
    if vocab["roles"]:
        lines.append("ROLES: " + ", ".join(vocab["roles"]))
    return "\n".join(lines)


def allowed_names(vocab: dict) -> tuple[set[str], set[str], set[str]]:
    """Return (schema names, shape heads, role names) the model may emit."""
    schemas = {s["name"] for s in vocab["schemas"]}
    heads = {s["conclusion"]["head"] for s in vocab["schemas"]}
    for s in vocab["schemas"]:
        heads.update(p["head"] for p in s["premises"])
        heads.update(o["head"] for o in s["obligation_templates"])
    return schemas, heads, set(vocab["roles"])
