"""Plan transforms (partial P ⇒ P' ▷ O) + explicit Bypass casts.

PlanTransform is a distinct type from MethodSchema: meta-level
induction over proofs is never confused with object-level deduction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from tpps.core.evidence import Evidence
from tpps.core.plan import (Layer, MethodInstance, Plan, Provenance,
                            StmtNode, TransformRecord)
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
    applicable: bool = True
    blame: str = ""
    provenance: Provenance = field(default_factory=Provenance)
    soundness_reference: str = ""


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
        prov = Provenance(producing_attempt=self.blame)
        # Explicit open implication obligation phi => psi.
        # Target node starts open (default Evidence(), not proven by itself).
        plan.add_stmt(StmtNode(
            obl_id,
            Stmt(
                src.stmt.context,
                Shape(
                    "Implies",
                    (self.from_shape.term(), self.to_shape.term()),
                    regime=src.stmt.regime,
                ),
                regime=src.stmt.regime,
            ),
            provenance=prov,
            layer=src.layer,
        ))
        plan.add_stmt(StmtNode(
            tgt_id,
            Stmt(src.stmt.context, self.to_shape, regime=self.to_shape.regime),
            provenance=prov,
            layer=src.layer,
        ))
        plan.transforms.append(TransformRecord(
            id=f"cast-{self.from_shape.head}-{self.to_shape.head}",
            kind=self.id,
            source=src.id,
            target=tgt_id,
            obligations=(obl_id,),
            provenance=prov,
        ))
        note = f"cast {self.from_shape.head} ⇒ {self.to_shape.head}, blame={self.blame}"
        return TransformResult(
            plan=plan,
            obligations=[obl_id],
            soundness_note=note,
            applicable=True,
            blame=self.blame,
            provenance=prov,
            soundness_reference="Bypass-Soundness",
        )


@dataclass
class Compress(PlanTransform):
    """Collapse a multi-instance derivation into one instance."""
    schema_ref: str
    premises: list[str]
    conclusion: str
    certificates: list[str] | None = None
    id: str = "Compress"

    def apply(self, plan: Plan) -> TransformResult | None:
        if (self.conclusion not in plan.stmts
                or any(premise not in plan.stmts for premise in self.premises)):
            return None
        prov = Provenance(producing_attempt=f"compress-{self.schema_ref}")
        obligation_id = f"obl-uniform-{self.schema_ref}"
        plan.add_stmt(StmtNode(
            obligation_id,
            Stmt(shape=Shape("UniformDerivation",
                             tuple(plan.stmts[p].stmt.shape.term()
                                   for p in self.premises))),
            provenance=prov,
            layer=Layer.COMMITTED,
        ))
        inst = MethodInstance(
            f"compress-{self.schema_ref}",
            self.schema_ref,
            list(self.premises),
            self.conclusion,
            obligations=[obligation_id],
            certificates=list(self.certificates or []),
            provenance=prov,
            layer=Layer.COMMITTED,
        )
        plan.add_inst(inst)
        plan.transforms.append(TransformRecord(
            id=f"compress-{self.schema_ref}",
            kind=self.id,
            source=self.premises[0] if self.premises else self.conclusion,
            target=self.conclusion,
            obligations=(obligation_id,),
            provenance=prov,
        ))
        return TransformResult(
            plan=plan,
            obligations=[obligation_id],
            soundness_note=f"compressed via {self.schema_ref}",
            applicable=True,
            provenance=prov,
            soundness_reference="Compress-UniformDerivation",
        )


@dataclass
class GeneralizeSpecialize(PlanTransform):
    """Apply a substitution to a target node's shape without in-place mutation."""
    target: str
    subst: dict
    generalize: bool = False
    id: str = "GeneralizeSpecialize"

    def apply(self, plan: Plan) -> TransformResult | None:
        node = plan.stmts.get(self.target)
        if node is None:
            return None
        action = "generalize" if self.generalize else "specialize"
        prov = Provenance(producing_attempt=f"{action}-{self.target}")
        specialized_shape = node.stmt.shape.instantiate(self.subst)
        new_target_id = f"{self.target}-{action}"
        obligation_id = f"obl-{action}-{self.target}"

        obligation_shape = (
            Shape("Implies", (specialized_shape.term(), node.stmt.shape.term()),
                  regime=node.stmt.regime)
            if self.generalize
            else Shape("Implies", (node.stmt.shape.term(), specialized_shape.term()),
                       regime=node.stmt.regime)
        )
        plan.add_stmt(StmtNode(
            obligation_id,
            Stmt(node.stmt.context, obligation_shape, regime=node.stmt.regime),
            provenance=prov,
            layer=node.layer,
        ))
        plan.add_stmt(StmtNode(
            new_target_id,
            Stmt(node.stmt.context, specialized_shape, regime=node.stmt.regime),
            provenance=prov,
            layer=node.layer,
        ))
        plan.transforms.append(TransformRecord(
            id=f"{action}-{self.target}",
            kind=self.id,
            source=self.target,
            target=new_target_id,
            obligations=(obligation_id,),
            provenance=prov,
        ))
        return TransformResult(
            plan=plan,
            obligations=[obligation_id],
            soundness_note=action,
            applicable=True,
            provenance=prov,
            soundness_reference=f"{action.capitalize()}-Soundness",
        )


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
        prov = Provenance(producing_attempt=f"{self.id}-{self.target}")
        plan.add_stmt(StmtNode(
            wit,
            Stmt(ctx, Shape("Witness", (), regime=node.stmt.regime)),
            provenance=prov,
            layer=node.layer,
        ))
        plan.transforms.append(TransformRecord(
            id=f"{self.id.lower()}-{self.target}",
            kind=self.id,
            source=self.target,
            target=self.target,
            obligations=(wit,),
            provenance=prov,
        ))
        return TransformResult(
            plan=plan,
            obligations=[wit],
            soundness_note="contradiction block with Witness slot",
            applicable=True,
            provenance=prov,
            soundness_reference="Classical.byContradiction",
        )


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
        prov = Provenance(producing_attempt=f"{self.id}-{self.target}")
        plan.add_stmt(StmtNode(
            tp,
            Stmt(ctx, Shape("TestPoint", (), regime=node.stmt.regime)),
            provenance=prov,
            layer=node.layer,
        ))
        plan.transforms.append(TransformRecord(
            id=f"{self.id.lower()}-{self.target}",
            kind=self.id,
            source=self.target,
            target=self.target,
            obligations=(tp,),
            provenance=prov,
        ))
        return TransformResult(
            plan=plan,
            obligations=[tp],
            soundness_note=f"fixpoint block, eigenvar {self.eigenvar}",
            applicable=True,
            provenance=prov,
            soundness_reference="FixPoint-Eigenvar",
        )


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
        prov = Provenance(producing_attempt=f"{self.id}-{self.target}")
        plan.add_stmt(StmtNode(wf_id, Stmt(ctx, wf), provenance=prov, layer=node.layer))
        plan.add_stmt(StmtNode(tr_id, Stmt(ctx, Shape("Implies", (), regime=node.stmt.regime)),
                               provenance=prov, layer=node.layer))
        plan.transforms.append(TransformRecord(
            id=f"{self.id.lower()}-{self.target}",
            kind=self.id,
            source=self.target,
            target=self.target,
            obligations=(wf_id, tr_id),
            provenance=prov,
        ))
        return TransformResult(
            plan=plan,
            obligations=[wf_id, tr_id],
            soundness_note=f"auxiliary {self.construction} with well-formedness + transfer",
            applicable=True,
            provenance=prov,
            soundness_reference="AuxiliaryFunction-Transfer",
        )


# Soundness gates: elaborate only when every leg holds; P' closure only after obligations discharge.
def elaboration_ready(report_ok: bool, all_kernel: bool, certs_valid: bool,
                       obligations_kernel: bool) -> bool:
    """Elaborate to a kernel proof only when every leg holds."""
    return report_ok and all_kernel and certs_valid and obligations_kernel


def transformation_sound(obligations_discharged: bool) -> bool:
    """P' closure follows from P only after O is discharged."""
    return obligations_discharged
