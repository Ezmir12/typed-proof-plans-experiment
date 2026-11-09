"""Recursive AND/OR backward chaining and typed residual generation.

Searches the schema registry backward from a target shape, decomposing
goals into AND-groups (premises required together) and OR-alternatives
(competing schema applications), bounded by depth and node limits.
Unclosed subgoals yield structured TypedResidual records classified
for creative critics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tpps.core.schema import MethodSchema, SchemaRegistry
from tpps.core.stmt import Shape, Subst, Term, unify_exact


CRITIC_CLASSES = (
    "missing lemma",
    "missing library entry",
    "bad decomposition",
    "impossible branch",
    "overstrong target",
    "representation mismatch",
    "auxiliary-construction request",
)


@dataclass(frozen=True)
class TypedResidual:
    """Canonical residual type representing an open subgoal or derivation gap."""

    shape: Shape
    origin_schema: str  # schema whose premise could not be produced
    context: str = ""
    critic_class: str = ""

    @property
    def producer_schema(self) -> str:
        """Alias for origin_schema for backwards compatibility."""
        return self.origin_schema


def classify_residual(residual: TypedResidual, hint: str = "") -> str:
    """Critic classification of a residual into one of the seven fixed classes."""
    h = (hint + " " + residual.shape.head + " " + residual.context).lower()
    if "librar" in h or "generaliz" in h:
        return "missing library entry"
    if "decompos" in h:
        return "bad decomposition"
    if "impossible" in h or "refut" in h:
        return "impossible branch"
    if "overstrong" in h:
        return "overstrong target"
    if "represent" in h or "mismatch" in h:
        return "representation mismatch"
    if "auxiliar" in h or "construction" in h:
        return "auxiliary-construction request"
    return "missing lemma"


@dataclass
class AndBranch:
    """One candidate schema derivation step requiring all premises (AND-group)."""

    schema_id: str
    subst: Subst
    subgoals: list[GoalNode] = field(default_factory=list)
    obligations: list[Shape] = field(default_factory=list)

    @property
    def is_closed(self) -> bool:
        """All premise subgoals must be closed for this branch to succeed."""
        return all(g.is_closed for g in self.subgoals)


@dataclass
class GoalNode:
    """A target goal shape with competing candidate derivation branches (OR-node)."""

    target: Shape
    depth: int = 0
    alternatives: list[AndBranch] = field(default_factory=list)
    residuals: list[TypedResidual] = field(default_factory=list)
    is_axiom: bool = False
    exhausted_budget: bool = False

    @property
    def is_closed(self) -> bool:
        """A goal is closed if it is an axiom or at least one OR-branch is closed."""
        if self.is_axiom:
            return True
        return any(alt.is_closed for alt in self.alternatives)

    def collect_residuals(self) -> list[TypedResidual]:
        """Collect all unresolved residuals from open branches."""
        if self.is_closed:
            return []
        res = list(self.residuals)
        for alt in self.alternatives:
            if not alt.is_closed:
                for sg in alt.subgoals:
                    res.extend(sg.collect_residuals())
        # Deduplicate while preserving order
        seen = set()
        out = []
        for r in res:
            key = (r.shape.head, tuple(str(a) for a in r.shape.args), r.origin_schema)
            if key not in seen:
                seen.add(key)
                out.append(r)
        return out


class BackwardChainer:
    """Recursive AND/OR backward chainer over a schema registry."""

    def __init__(
        self,
        registry: SchemaRegistry,
        *,
        max_depth: int = 5,
        max_nodes: int = 100,
    ) -> None:
        self.registry = registry
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._node_count = 0

    def search(
        self,
        target: Shape,
        known_heads: set[str] | None = None,
        plan_heads: set[str] | None = None,
    ) -> GoalNode:
        """Search for derivations of target up to depth and node budgets."""
        self._node_count = 0
        known = set() if known_heads is None else known_heads
        plan = set() if plan_heads is None else plan_heads
        return self._search_node(
            target,
            depth=0,
            active_shapes=set(),
            known=known,
            plan=plan,
            parent_schema="",
        )

    def _search_node(
        self,
        target: Shape,
        depth: int,
        active_shapes: set[str],
        known: set[str],
        plan: set[str],
        parent_schema: str = "",
    ) -> GoalNode:
        self._node_count += 1
        node = GoalNode(target=target, depth=depth)

        # 1. Base cases: known theorem, already present in plan, or cycle
        if target.head in known or target.head in plan:
            node.is_axiom = True
            return node

        target_key = f"{target.head}({','.join(str(a) for a in target.args)})"
        if target_key in active_shapes:
            # Cycle detected along derivation path
            r = TypedResidual(
                shape=target,
                origin_schema="cycle",
                context="derivation cycle",
                critic_class="bad decomposition",
            )
            node.residuals.append(r)
            return node

        if depth >= self.max_depth or self._node_count >= self.max_nodes:
            node.exhausted_budget = True
            r = TypedResidual(
                shape=target,
                origin_schema="budget-exhausted",
                context=f"depth={depth}, nodes={self._node_count}",
                critic_class="overstrong target",
            )
            node.residuals.append(r)
            return node

        # 2. Find matching schemas in registry
        matching_schemas: list[tuple[MethodSchema, Subst]] = []
        for schema in self.registry.schemas.values():
            # Check conclusion compatibility
            sub = unify_exact(schema.conclusion.term(), target.term())
            if sub is not None:
                # Check regime / sort compatibility if specified
                if schema.conclusion.regime and target.regime:
                    if schema.conclusion.regime != target.regime:
                        continue
                matching_schemas.append((schema, sub))

        if not matching_schemas:
            # No schema produces this shape
            r = TypedResidual(
                shape=target,
                origin_schema=parent_schema,
                context="no producing schema",
                critic_class=classify_residual(TypedResidual(target, parent_schema)),
            )
            node.residuals.append(r)
            return node

        # 3. Explore each schema as an OR-alternative
        new_active = active_shapes | {target_key}
        for schema, sub in matching_schemas:
            # Instantiate premises and obligations under substitution
            inst_premises = [p.instantiate(sub) for p in schema.premises]
            inst_obligations = [o.instantiate(sub) for o in schema.obligation_templates]

            branch = AndBranch(
                schema_id=schema.id,
                subst=sub,
                obligations=inst_obligations,
            )

            # Search each premise as an AND-subgoal
            for prem in inst_premises:
                subgoal = self._search_node(
                    prem,
                    depth=depth + 1,
                    active_shapes=new_active,
                    known=known,
                    plan=plan,
                    parent_schema=schema.id,
                )
                branch.subgoals.append(subgoal)

            node.alternatives.append(branch)

            # If this alternative is closed, we found a derivation!
            if branch.is_closed:
                break

        return node


def backward_chain(
    target: Shape,
    registry_or_index: SchemaRegistry | list[tuple[str, Shape]],
    known_heads: set[str] | None = None,
    plan_heads: set[str] | None = None,
    *,
    max_depth: int = 5,
    max_nodes: int = 100,
) -> list[TypedResidual]:
    """Backward chaining entrypoint supporting both SchemaRegistry and conclusion index."""
    from tpps.core.stmt import Shape

    known = set() if known_heads is None else known_heads
    plan = set() if plan_heads is None else plan_heads

    if isinstance(registry_or_index, SchemaRegistry):
        chainer = BackwardChainer(registry_or_index, max_depth=max_depth, max_nodes=max_nodes)
        goal = chainer.search(target, known_heads=known, plan_heads=plan)
        return goal.collect_residuals()

    # Legacy index format: list of (schema_id, conclusion_shape)
    from tpps.reasoning.creative import _premises_of

    residuals: list[TypedResidual] = []
    producers = {
        sid
        for sid, concl in registry_or_index
        if unify_exact(concl.term(), target.term()) is not None
    }
    for sid, _concl in registry_or_index:
        if sid not in producers:
            continue
        for prem in _premises_of(sid):
            if prem.head not in known and prem.head not in plan:
                r = TypedResidual(prem, sid)
                residuals.append(r)
    return residuals

