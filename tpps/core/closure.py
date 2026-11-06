"""Recursive AND/OR evidence closure and refutation propagation for typed proof plans."""
from __future__ import annotations

from dataclasses import dataclass, field

from .evidence import (BOTTOM, Evidence, aggregate, conjoin,
                       is_kernel_refuted)
from .plan import Layer, Plan
from .schema import SchemaRegistry


@dataclass
class ClosureResult:
    """Evidence and support computed for one closure evaluation."""

    evidence: dict[str, Evidence] = field(default_factory=dict)
    support: dict[str, set[str]] = field(default_factory=dict)
    refuted: set[str] = field(default_factory=set)
    inconsistent: set[str] = field(default_factory=set)
    open_nodes: set[str] = field(default_factory=set)
    diagnostics: list[str] = field(default_factory=list)


class EvidenceClosure:
    """Compute recursive evidence closure over a proof plan without mutating it."""

    def __init__(self, plan: Plan, registry: SchemaRegistry) -> None:
        self.plan = plan
        self.registry = registry
        self._cache: dict[str, tuple[Evidence, set[str]]] = {}
        self._diagnostics: list[str] = []
        self._inconsistent: set[str] = set()
        self._dependents = self._build_dependents()

    def _build_dependents(self) -> dict[str, set[str]]:
        dependents: dict[str, set[str]] = {node_id: set() for node_id in self.plan.stmts}
        for inst in self.plan.insts.values():
            if inst.layer is not Layer.COMMITTED:
                continue
            for source in inst.premises + inst.obligations:
                dependents.setdefault(source, set()).add(inst.conclusion)
        for tr in self.plan.transforms:
            dependents.setdefault(tr.source, set()).add(tr.target)
            for obl in tr.obligations:
                dependents.setdefault(obl, set()).add(tr.target)
        return dependents

    def close_node(self, node_id: str, _active: set[str] | None = None
                   ) -> tuple[Evidence, set[str]]:
        if node_id not in self.plan.stmts:
            self._diagnostics.append(f"unknown statement node {node_id!r}")
            return BOTTOM, set()
        if node_id in self._cache:
            return self._cache[node_id]

        active = set() if _active is None else _active
        if node_id in active:
            self._diagnostics.append(
                f"evidence closure cycle at statement {node_id!r}")
            return BOTTOM, {node_id}
        active.add(node_id)

        node = self.plan.stmts[node_id]
        direct = node.evidence
        derivations: list[Evidence] = []
        support: set[str] = {node_id}

        # 1. Derivations from committed method instances
        for inst in self.plan.producers(node_id):
            if inst.layer is not Layer.COMMITTED:
                continue
            schema = self.registry.schemas.get(inst.schema_ref)
            if schema is None:
                self._diagnostics.append(
                    f"instance {inst.id!r} references unknown schema "
                    f"{inst.schema_ref!r}")
                continue
            parts = [schema.trust.evidence]
            derivation_support: set[str] = {node_id}
            poisoned = False
            for dependency in inst.premises + inst.obligations:
                dependency_node = self.plan.stmts.get(dependency)
                if dependency_node is None:
                    poisoned = True
                    self._diagnostics.append(
                        f"instance {inst.id!r} references missing node "
                        f"{dependency!r}")
                    continue
                if is_kernel_refuted(dependency_node.counter_evidence):
                    poisoned = True
                dep_evidence, dep_support = self.close_node(
                    dependency, active)
                if is_kernel_refuted(
                        self.plan.stmts[dependency].counter_evidence):
                    poisoned = True
                parts.append(dep_evidence)
                if not poisoned:
                    derivation_support |= dep_support
            if is_kernel_refuted(node.counter_evidence):
                poisoned = True
            if poisoned:
                self._inconsistent.add(node_id)
            else:
                derivations.append(conjoin(parts))
                support |= derivation_support

        # 2. Derivations from meta-level transforms (e.g. Bypass casts)
        for tr in self.plan.transforms:
            if tr.target != node_id:
                continue
            src_node = self.plan.stmts.get(tr.source)
            if src_node is None:
                continue
            tr_parts = []
            tr_support: set[str] = {node_id, tr.source}
            poisoned = is_kernel_refuted(src_node.counter_evidence)
            src_ev, src_sup = self.close_node(tr.source, active)
            tr_parts.append(src_ev)
            tr_support |= src_sup

            for obl_id in tr.obligations:
                obl_node = self.plan.stmts.get(obl_id)
                if obl_node is None:
                    poisoned = True
                    continue
                if is_kernel_refuted(obl_node.counter_evidence):
                    poisoned = True
                obl_ev, obl_sup = self.close_node(obl_id, active)
                tr_parts.append(obl_ev)
                tr_support |= obl_sup

            if poisoned:
                self._inconsistent.add(node_id)
            else:
                derivations.append(conjoin(tr_parts))
                support |= tr_support

        active.remove(node_id)
        result = (aggregate(direct, derivations), support)
        self._cache[node_id] = result
        return result

    def close_plan(self) -> ClosureResult:
        result = ClosureResult()
        for node_id in self.plan.stmts:
            evidence, support = self.close_node(node_id)
            result.evidence[node_id] = evidence
            result.support[node_id] = support

            if is_kernel_refuted(self.plan.stmts[node_id].counter_evidence):
                result.refuted.add(node_id)
                result.inconsistent.add(node_id)

            if node_id in self._inconsistent:
                result.inconsistent.add(node_id)

            # Support-level inconsistency: any node in committed support is kernel-refuted
            if any(is_kernel_refuted(self.plan.stmts[s].counter_evidence)
                   for s in support if s in self.plan.stmts):
                result.inconsistent.add(node_id)

            if (
                self.plan.stmts[node_id].evidence == Evidence()
                and not any(
                    inst.layer is Layer.COMMITTED
                    and inst.schema_ref in self.registry.schemas
                    for inst in self.plan.producers(node_id)
                )
                and not any(tr.target == node_id for tr in self.plan.transforms)
            ):
                result.open_nodes.add(node_id)
        result.diagnostics = list(dict.fromkeys(self._diagnostics))
        return result

    def invalidate(self, node_id: str) -> None:
        """Invalidate a node and all committed conclusions depending on it."""
        affected = {node_id}
        stack = [node_id]
        while stack:
            current = stack.pop()
            for dependent in self._dependents.get(current, ()):
                if dependent not in affected:
                    affected.add(dependent)
                    stack.append(dependent)
        for affected_id in affected:
            self._cache.pop(affected_id, None)


def close_node(plan: Plan, registry: SchemaRegistry,
               node_id: str) -> tuple[Evidence, set[str]]:
    """Compute evidence and support for one statement node."""
    return EvidenceClosure(plan, registry).close_node(node_id)


def close_plan(plan: Plan, registry: SchemaRegistry) -> ClosureResult:
    """Compute recursive evidence closure for every statement node."""
    return EvidenceClosure(plan, registry).close_plan()


def invalidate(closure: EvidenceClosure, node_id: str) -> None:
    """Invalidate cached closure facts affected by a source node."""
    closure.invalidate(node_id)

