"""Plan transforms (partial P ⇒ P' ▷ O) + explicit Bypass casts.

PlanTransform is a distinct type from MethodSchema: meta-level
induction over proofs is never confused with object-level deduction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from tpps.core.evidence import Evidence
from tpps.core.plan import Layer, MethodInstance, Plan, Provenance, StmtNode
from tpps.core.stmt import CtxEntry, Shape, Stmt


# Sort pairs that are bridgeable via an explicit cast (the
# RelSpace (set) vs ResultMultiset (multiset) drift is caught *as* a cast,
# not rejected). Anything else with clashing sorts is a hard mismatch.
_BRIDGEABLE_SORTS = {("RelSpace", "ResultMultiset"), ("ResultMultiset", "RelSpace")}


class SortClashError(ValueError):
    """Hard type/sort/binder mismatch: rejected at construction."""


@dataclass
class TransformResult:
    plan: Plan
    obligations: list[str]  # new stmt-node ids
    soundness_note: str = ""


class PlanTransform(ABC):
    id: str = "transform"

    @abstractmethod
    def apply(self, plan: Plan) -> TransformResult | None:
        """Partial function T : P ⇒ P' ▷ O. None = inapplicable."""


def cast_evidence(mu_phi: Evidence, mu_obl: Evidence) -> Evidence:
    """ψ's record = meet of φ's record and the obligation's."""
    return mu_phi.meet(mu_obl)


@dataclass
class Bypass(PlanTransform):
    """Explicit cast φ ⇒ ψ with obligation + blame."""
    from_shape: Shape
    to_shape: Shape
    blame: str = ""
    id: str = "Bypass"

    def check_admissible(self) -> None:
        fs, ts = self.from_shape.sort, self.to_shape.sort
        if fs and ts and fs != ts and (fs, ts) not in _BRIDGEABLE_SORTS:
            raise SortClashError(
                f"sort clash {fs!r} vs {ts!r}: "
                "rejected at construction, not an obligation")

    def apply(self, plan: Plan) -> TransformResult | None:
        self.check_admissible()
        src = next((n for n in plan.stmts.values()
                    if n.stmt.shape.head == self.from_shape.head), None)
        if src is None:
            return None
        obl_id = f"obl-{self.from_shape.head}-implies-{self.to_shape.head}"
        tgt_id = f"cast-{self.to_shape.head}"
        plan.add_stmt(StmtNode(obl_id, Stmt(shape=Shape(
            "Implies", (), regime=src.stmt.regime),
            ), provenance=Provenance(producing_attempt=self.blame)))
        plan.add_stmt(StmtNode(
            tgt_id, Stmt(shape=self.to_shape, regime=self.to_shape.regime),
            provenance=Provenance(producing_attempt=self.blame)))
        plan.add_inst(MethodInstance(
            f"cast-{self.from_shape.head}-{self.to_shape.head}", "Bypass",
            [src.id], tgt_id, obligations=[obl_id],
            provenance=Provenance(producing_attempt=self.blame)))
        return TransformResult(plan, [obl_id],
                               f"cast {self.from_shape.head} ⇒ {self.to_shape.head}, "
                               f"blame={self.blame}")


@dataclass
class Compress(PlanTransform):
    """Collapse a multi-instance derivation into one instance."""
    schema_ref: str
    premises: list[str]
    conclusion: str
    certificates: list[str] = None  # type: ignore[assignment]
    id: str = "Compress"

    def apply(self, plan: Plan) -> TransformResult | None:
        inst = MethodInstance(f"compress-{self.schema_ref}", self.schema_ref,
                              list(self.premises), self.conclusion,
                              certificates=list(self.certificates or []))
        plan.add_inst(inst)
        return TransformResult(plan, [], f"compressed via {self.schema_ref}")


@dataclass
class GeneralizeSpecialize(PlanTransform):
    """Apply a substitution to a target node's shape."""
    target: str
    subst: dict
    generalize: bool = False
    id: str = "GeneralizeSpecialize"

    def apply(self, plan: Plan) -> TransformResult | None:
        node = plan.stmts.get(self.target)
        if node is None:
            return None
        node.stmt = node.stmt.instantiate(self.subst)
        return TransformResult(plan, [], "generalize" if self.generalize else "specialize")


@dataclass
class Contradiction(PlanTransform):
    """Context-growing transform opening a Witness role slot."""
    target: str
    id: str = "Contradiction"

    def apply(self, plan: Plan) -> TransformResult | None:
        node = plan.stmts.get(self.target)
        if node is None:
            return None
        ctx = node.stmt.context + (CtxEntry("Assumption", f"Not({node.stmt.shape.head})",
                                            introduced_by=f"{self.id}-{self.target}"),)
        node.stmt = Stmt(ctx, node.stmt.shape, node.stmt.regime)
        wit = f"witness-{self.target}"
        plan.add_stmt(StmtNode(wit, Stmt(ctx, Shape("Witness", (), regime=node.stmt.regime))))
        return TransformResult(plan, [wit], "contradiction block with Witness slot")


@dataclass
class FixPoint(PlanTransform):
    """Eigenvariable-introducing transform opening a TestPoint slot."""
    target: str
    eigenvar: str = "x0"
    id: str = "FixPoint"

    def apply(self, plan: Plan) -> TransformResult | None:
        node = plan.stmts.get(self.target)
        if node is None:
            return None
        ctx = node.stmt.context + (CtxEntry("Eigenvar", self.eigenvar,
                                            introduced_by=f"{self.id}-{self.target}"),)
        node.stmt = Stmt(ctx, node.stmt.shape, node.stmt.regime)
        tp = f"testpoint-{self.target}"
        plan.add_stmt(StmtNode(tp, Stmt(ctx, Shape("TestPoint", (), regime=node.stmt.regime))))
        return TransformResult(plan, [tp], f"fixpoint block, eigenvar {self.eigenvar}")


@dataclass
class AuxiliaryFunction(PlanTransform):
    """Target change Γ ⊢ φ(f) → Γ,g:=c(f) ⊢ φ'(g) + obligations."""
    target: str
    construction: str
    new_shape: Shape
    wellformed_shape: Shape = None  # type: ignore[assignment]
    id: str = "AuxiliaryFunction"

    def apply(self, plan: Plan) -> TransformResult | None:
        node = plan.stmts.get(self.target)
        if node is None:
            return None
        ctx = node.stmt.context + (CtxEntry("Definition", self.construction,
                                            introduced_by=f"{self.id}-{self.target}"),)
        node.stmt = Stmt(ctx, self.new_shape, node.stmt.regime)
        wf = self.wellformed_shape or Shape(f"Holo", (), regime=node.stmt.regime)
        wf_id, tr_id = f"wf-{self.target}", f"transfer-{self.target}"
        plan.add_stmt(StmtNode(wf_id, Stmt(ctx, wf)))
        plan.add_stmt(StmtNode(tr_id, Stmt(ctx, Shape("Implies", (), regime=node.stmt.regime))))
        return TransformResult(plan, [wf_id, tr_id],
                               f"auxiliary {self.construction} with well-formedness + transfer")


# Soundness gates: elaborate only when every leg holds; P' closure only after obligations discharge.
def elaboration_ready(report_ok: bool, all_kernel: bool, certs_valid: bool,
                       obligations_kernel: bool) -> bool:
    """Elaborate to a kernel proof only when every leg holds."""
    return report_ok and all_kernel and certs_valid and obligations_kernel


def transformation_sound(obligations_discharged: bool) -> bool:
    """P' closure follows from P only after O is discharged."""
    return obligations_discharged
