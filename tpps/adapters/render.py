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
        for tr in plan.transforms:
            if tr.target == nid:
                visit(tr.source)
                for o in tr.obligations:
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
    """Render the committed support derivation of target into natural mathematical discourse.

    Register Gating:
    - Textbook register: Generates clean, publication-ready mathematical exposition.
      Gated strictly to closed derivations meeting formal evidence thresholds; certified
      lemmas are cited and intermediate bookkeeping obligations are folded.
    - Annotated register: Exposes explicit step statuses, 5D evidence markers, role
      bindings, and representation cast warnings. Open, speculative, or refuted targets
      are automatically directed to this register to prevent silent proof-gap smoothing.
    """
    node = plan.stmts[target]
    closed = is_closed(node.evidence, requirement)
    if register is None:
        register = Register.TEXTBOOK if closed else Register.ANNOTATED
    elif register is Register.TEXTBOOK and not closed:
        # Safety gate: open, refuted, or unclosed targets cannot be rendered as textbook
        register = Register.ANNOTATED

    # Obligation ids belonging to an explicit Bypass cast: always read as casts.
    cast_obls = {o for tr in plan.transforms if tr.kind == "Bypass" for o in tr.obligations} | {
        o for i in plan.insts.values()
        if i.schema_ref == "Bypass" or i.id.startswith("cast-")
        for o in i.obligations
    }
    # Every other obligation-linked node is open until discharged: it must
    # read as remaining work, never as an established premise.
    obl_ids = {o for i in plan.insts.values() for o in i.obligations} - cast_obls
    lines: list[str] = []
    if register is Register.ANNOTATED:
        status = "closed" if closed else "OPEN: obligations remain"
        lines.append(f"Status of {target}: {status}.")

    for node_id in linearize(plan, target):
        node = plan.stmts[node_id]
        producers = plan.producers(node_id)
        head = node.stmt.shape.head
        regime = f" {node.stmt.regime}" if node.stmt.regime else ""

        if node_id == target:
            if any(inst.schema_ref == "Contradiction" for inst in producers):
                lines.append(f"Hence {head}{regime}, a contradiction.")
            else:
                lines.append(f"We show {head}{regime}.")
            continue

        if head == "Witness":
            lines.append(f"Take a witness for {node_id}{regime}; it remains to show its properties.")
            continue

        # Explicit representation cast warning
        if head == "Implies" and (node_id in cast_obls
                                   or any("cast-" in inst.id for inst in producers)):
            lines.append(
                f"We introduce an explicit representation cast for {node_id}{regime}; "
                f"it remains to show the cast condition, and the passage from {node_id} remains open. "
                f"This does not yet establish the original claim.")
            continue

        if head == "Implies":
            # Open obligation: never smoothed over.
            lines.append(f"It remains to show that {node_id}{regime}.")
            continue

        if node_id in obl_ids:
            evidence_tag = ""
            if register is Register.ANNOTATED:
                formatted_tag = _evidence_tag(node.evidence, requirement)
                evidence_tag = f" {formatted_tag}" if formatted_tag else ""
            lines.append(f"It remains to show that {node_id}{regime}{evidence_tag}.")
            continue

        # Schema-driven discourse templates
        if producers:
            primary_producer = producers[0]
            schema_identifier = primary_producer.schema_ref

            if schema_identifier == "Contradiction":
                lines.append(f"Suppose, for contradiction, that {head}{regime} fails.")
                continue
            if schema_identifier == "AuxiliaryFunction":
                lines.append(f"Consider the auxiliary function for {head}{regime}; since this holds, it provides the required bound.")
                continue
            if schema_identifier == "Decomposition":
                lines.append(f"We treat the cases for {head}{regime} separately; combining the bounds completes the step.")
                continue
            if schema_identifier == "FixPoint":
                lines.append(f"Fix an arbitrary parameter for {head}{regime}.")
                continue

        if register is Register.ANNOTATED:
            formatted_tag = _evidence_tag(node.evidence, requirement)
            extra_annotation = f" {formatted_tag}" if formatted_tag else ""
            lines.append(f"Since {head}{regime} holds{extra_annotation}, we proceed.")
        else:
            lines.append(f"Since {head}{regime} holds, we proceed.")

    return "\n".join(lines)


def lint_rendered(text: str, plan: Plan) -> list[str]:
    """Lint: casts / open obligations / regime qualifiers elided."""
    flags = []
    has_cast = any(i.schema_ref == "Bypass" for i in plan.insts.values())
    if has_cast and "cast" not in text.lower():
        flags.append("cast elided")
    obl_ids = {o for i in plan.insts.values() for o in i.obligations}
    lowered = text.lower()
    for n in plan.stmts.values():
        if (n.stmt.shape.head == "Implies" or n.id in obl_ids) \
                and "remains to show" not in lowered:
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
    obl_ids = {o for i in plan.insts.values() for o in i.obligations}
    if (set(reextracted_shapes) != orig_shapes
            or set(reextracted_regimes) != orig_regimes
            or "remains to show" not in smoothed and bool(obl_ids)):
        return text  # reject: fall back to unsmoothed
    _ = reextracted_obls
    return smoothed
