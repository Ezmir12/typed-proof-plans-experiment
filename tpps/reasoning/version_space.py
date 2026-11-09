"""Version space learning and candidate-specific refinement for method schemas.

Maintains general and specific hypotheses over candidate schemas, refining
only the failing candidate when negative counter-evidence arrives while
preserving regimes, sorts, and unaffected schemas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tpps.core.plan import Plan
from tpps.core.schema import Condition, CondTag, MethodSchema, TrustRecord
from tpps.core.stmt import Shape


class SortClashError(ValueError):
    """Hard sort incompatibility rejected outright."""


@dataclass
class VersionSpace:
    """Version space managing candidate schemas and boundary contraction."""

    candidates: dict[str, MethodSchema] = field(default_factory=dict)
    negative_history: list[dict[str, Any]] = field(default_factory=list)

    def register_candidate(self, schema: MethodSchema) -> None:
        self.candidates[schema.id] = schema

    def refine_on_negative(
        self,
        failed_schema_id: str,
        failed_condition: str,
        new_condition_tag: CondTag = CondTag.S,
        new_condition_predicate: str = "",
        *,
        new_schema_id: str = "",
    ) -> MethodSchema:
        """Candidate-specific refinement: specializes ONLY the failing schema
        without modifying or discarding unaffected candidates."""
        schema = self.candidates.get(failed_schema_id)
        if schema is None:
            raise KeyError(f"candidate schema {failed_schema_id!r} not in version space")

        pred = new_condition_predicate or f"restricted-{failed_condition}"
        new_cond = Condition(new_condition_tag, pred)
        # Avoid duplicate conditions
        existing_preds = {c.predicate_ref for c in schema.conditions}
        refined_conditions = list(schema.conditions)
        if pred not in existing_preds:
            refined_conditions.append(new_cond)

        target_id = new_schema_id or f"Refined-{schema.id}"
        refined_schema = MethodSchema(
            id=target_id,
            premises=list(schema.premises),
            conclusion=schema.conclusion,
            conditions=refined_conditions,
            obligation_templates=list(schema.obligation_templates),
            trust=TrustRecord(),
            roles=schema.roles,
            checker=schema.checker,
            elaborator=schema.elaborator,
            soundness_theorem=schema.soundness_theorem,
            is_transform=schema.is_transform,
        )

        self.candidates[target_id] = refined_schema
        self.negative_history.append({
            "failed_schema_id": failed_schema_id,
            "failed_condition": failed_condition,
            "refined_schema_id": target_id,
            "added_condition": pred,
        })
        return refined_schema

    def prune_inconsistent(self, check_fn) -> list[str]:
        """Prune candidate schemas that fail a validation predicate."""
        removed = []
        for sid, schema in list(self.candidates.items()):
            if not check_fn(schema):
                removed.append(sid)
                del self.candidates[sid]
        return removed

