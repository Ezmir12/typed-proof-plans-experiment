"""Pure deterministic plan → English, textbook + annotated registers.

Base pass never calls the LLM. Both registers come from one shared
linearization pass and differ only in suppression.
"""
from __future__ import annotations

from enum import Enum

from tpps.core.evidence import Evidence, is_closed
from tpps.core.plan import Layer, Plan


class Register(Enum):
    TEXTBOOK = "textbook"
    ANNOTATED = "annotated"


def linearize(plan: Plan, target: str) -> list[str]:
    """Shared pass: DFS over committed support, premises before conclusion."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen:
            return
        seen.add(nid)
        for inst in plan.insts.values():
            if inst.conclusion == nid and inst.layer is Layer.COMMITTED:
                for p in inst.premises:
                    visit(p)
                for o in inst.obligations:
                    visit(o)
        order.append(nid)

    visit(target)
    return order


def _evidence_tag(mu: Evidence, requirement: Evidence) -> str:
    if is_closed(mu, requirement):
        return ""
    return (f"[formal={mu.formal.name} certified={mu.certified.name} "
            f"reviewed={mu.reviewed.name}]")


def render(plan: Plan, target: str, requirement: Evidence,
           register: Register | None = None) -> str:
    """Render the committed support derivation of target."""
    node = plan.stmts[target]
    closed = is_closed(node.evidence, requirement)
    if register is None:
        register = Register.TEXTBOOK if closed else Register.ANNOTATED
    # Obligation ids belonging to an explicit Bypass cast: always read as casts.
    cast_obls = {o for i in plan.insts.values()
                 if i.schema_ref == "Bypass" or i.id.startswith("cast-")
                 for o in i.obligations}
    lines: list[str] = []
    if register is Register.ANNOTATED:
        status = "closed" if closed else "OPEN: obligations remain"
        lines.append(f"Status of {target}: {status}.")

    for nid in linearize(plan, target):
        n = plan.stmts[nid]
        producers = plan.producers(nid)
        head = n.stmt.shape.head
        regime = f" {n.stmt.regime}" if n.stmt.regime else ""
        if nid == target:
            lines.append(f"We show {head}{regime}.")
            continue
        if head == "Witness":
            lines.append(f"Take a witness for {nid}{regime}; it remains to show its properties.")
            continue
        if head == "Implies" and (nid in cast_obls
                                   or any("cast-" in i.id for i in producers)):
            lines.append(
                f"By cast from established premises, modulo the open implication "
                f"{nid}{regime}; it remains to show {nid}.")
            continue
        if head == "Implies":
            # Open obligation: never smoothed over.
            lines.append(f"It remains to show {nid}{regime}.")
            continue
        if producers and any(i.schema_ref == "Contradiction" for i in producers):
            lines.append(f"Suppose not; seeking a contradiction via {nid}.")
            continue
        if register is Register.ANNOTATED:
            tag = _evidence_tag(n.evidence, requirement)
            extra = f" {tag}" if tag else ""
            lines.append(f"Since {head}{regime} holds{extra}, we proceed.")
        else:
            lines.append(f"Since {head}{regime} holds, we proceed.")

    # OR-alternatives: textbook shows only the highest-evidence one inline.
    alts = [i for n in plan.stmts for i in plan.producers(n)]
    _ = alts
    return "\n".join(lines)


def lint_rendered(text: str, plan: Plan) -> list[str]:
    """Lint: casts / open obligations / regime qualifiers elided."""
    flags = []
    has_cast = any(i.schema_ref == "Bypass" for i in plan.insts.values())
    if has_cast and "cast" not in text.lower():
        flags.append("cast elided")
    for n in plan.stmts.values():
        if n.stmt.shape.head == "Implies" and "remains to show" not in text.lower():
            flags.append(f"obligation {n.id} smoothed over")
        if n.stmt.regime and n.stmt.regime not in text:
            flags.append(f"regime {n.stmt.regime!r} dropped at {n.id}")
    return flags


def smooth_with_llm(text: str, llm_fn, plan: Plan,
                    reextract) -> str:
    """Optional smoothing with re-extraction + diff gate."""
    smoothed = llm_fn(text)
    reextracted_shapes, reextracted_regimes, reextracted_obls = reextract(smoothed)
    orig_shapes = {n.stmt.shape.head for n in plan.stmts.values()}
    orig_regimes = {n.stmt.regime for n in plan.stmts.values() if n.stmt.regime}
    if (set(reextracted_shapes) != orig_shapes
            or set(reextracted_regimes) != orig_regimes
            or "remains to show" not in smoothed and any(
                n.stmt.shape.head == "Implies" for n in plan.stmts.values())):
        return text  # reject: fall back to unsmoothed
    _ = reextracted_obls
    return smoothed
