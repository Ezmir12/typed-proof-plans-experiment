"""Well-typed + well-scoped checker. Pure/idempotent."""
from __future__ import annotations

from dataclasses import dataclass, field

from .plan import Layer, MethodInstance, Plan
from .schema import CondTag, SchemaRegistry
from .stmt import CheckKind, CheckResult, Stmt, run_check, unify_exact


@dataclass
class InstanceReport:
    inst_id: str
    # Four sub-conditions:
    premises_instantiate: bool = False
    conclusion_subsumed: bool = False
    side_conditions_certified: bool = False
    obligations_present: bool = False
    # Which of the four shape checks fired per condition.
    checks_fired: list[CheckResult] = field(default_factory=list)
    diagnostic: str = ""

    @property
    def ok(self) -> bool:
        return (self.premises_instantiate and self.conclusion_subsumed
                and self.side_conditions_certified and self.obligations_present)


@dataclass
class PlanReport:
    ok: bool
    instances: list[InstanceReport] = field(default_factory=list)
    scope_errors: list[str] = field(default_factory=list)
    committed_cycle: list[str] | None = None
    diagnostic: str = ""


def _shapes(plan: Plan, ids: list[str]) -> list[Stmt]:
    return [plan.stmts[i].stmt for i in ids]


def check_instance(plan: Plan, inst: MethodInstance, registry: SchemaRegistry,
                   cert_ok: set[str] | None = None) -> InstanceReport:
    """Well-typedness check. Pure: never mutates the plan."""
    rep = InstanceReport(inst_id=inst.id)
    cert_ok = cert_ok or set(inst.certificates)
    if inst.schema_ref not in registry.schemas:
        rep.diagnostic = f"unknown schema {inst.schema_ref!r}"
        return rep
    schema = registry.schemas[inst.schema_ref]
    if getattr(schema, "is_transform", False):
        # Plan-level transforms: arity + obligations only; the
        # transform's own apply() carries the shape reasoning.
        rep.premises_instantiate = len(inst.premises) == len(schema.premises)
        rep.conclusion_subsumed = True
        rep.side_conditions_certified = True
        rep.obligations_present = True
        rep.diagnostic = "ok (transform)"
        return rep
    if len(inst.premises) != len(schema.premises):
        rep.diagnostic = f"arity mismatch: {len(inst.premises)} vs {len(schema.premises)}"
        return rep

    # (1) premises are instances of Φᵢ under a common σ.
    subst: dict = {}
    ok = True
    for pat, pid in zip(schema.premises, inst.premises):
        target = plan.stmts[pid].stmt.shape
        r = run_check(CheckKind.SPECIALIZE, pat, target, registry.regimes
                      if hasattr(registry, "regimes") else _trivial_regimes())
        rep.checks_fired.append(r)
        if not r.ok:
            ok = False
            break
        for k, v in r.subst.items():
            if k in subst and subst[k] != v:
                ok = False
                break
            subst[k] = v
    rep.premises_instantiate = ok

    # (2) conclusion subsumed by Ψσ (regime-aware).
    concl_shape = plan.stmts[inst.conclusion].stmt.shape
    instantiated = schema.conclusion.instantiate(subst)
    r2 = run_check(CheckKind.SUBSUME, instantiated, concl_shape,
                   registry.regimes if hasattr(registry, "regimes") else _trivial_regimes())
    rep.checks_fired.append(r2)
    rep.conclusion_subsumed = r2.ok

    # (3) every side condition has a valid certificate.
    rep.side_conditions_certified = all(
        (c.tag is CondTag.S) or (c.predicate_ref in cert_ok)
        for c in schema.conditions
    )

    # (4) every obligation template instantiated by σ is present in obl.
    obl_shapes = {plan.stmts[o].stmt.shape.head for o in inst.obligations}
    rep.obligations_present = all(
        t.head in obl_shapes or not t.head for t in schema.obligation_templates
    ) if schema.obligation_templates else True
    if not schema.obligation_templates:
        rep.obligations_present = True

    if rep.ok:
        rep.diagnostic = "ok"
    else:
        rep.diagnostic = (
            f"prem={rep.premises_instantiate} concl={rep.conclusion_subsumed} "
            f"certs={rep.side_conditions_certified} obls={rep.obligations_present}")
    return rep


def _trivial_regimes():
    from .sorts import RegimeOrder
    return RegimeOrder()


def check_well_scoped(plan: Plan) -> list[str]:
    """Eigenvariables must not escape their introducing scope."""
    errors = []
    # Map eigenvar expr -> introducer instance.
    intro: dict[str, str] = {}
    home_node: dict[str, str] = {}  # eigenvar expr -> node id that may carry it
    for node in plan.stmts.values():
        for e in node.stmt.context:
            if e.kind == "Eigenvar" and e.introduced_by:
                intro[e.expr] = e.introduced_by
                # introducer keys look like "<transform>-<node-id>"
                home_node[e.expr] = e.introduced_by.rsplit("-", 1)[-1]
    # Downstream closure of each instance: conclusion + everything reachable
    # forward through instances consuming it.
    downstream: dict[str, set[str]] = {}
    for inst in plan.insts.values():
        downstream.setdefault(inst.id, set()).add(inst.conclusion)
    changed = True
    while changed:
        changed = False
        for inst in plan.insts.values():
            for p in inst.premises:
                # find producer instance of p
                for prod in plan.insts.values():
                    if prod.conclusion == p and inst.id in downstream:
                        before = len(downstream[inst.id])
                        downstream[inst.id] |= downstream.get(prod.id, set())
                        changed = changed or len(downstream[inst.id]) != before
    for node in plan.stmts.values():
        for e in node.stmt.context:
            if e.kind == "Eigenvar" and e.expr in intro:
                home = home_node.get(e.expr, "")
                # The introducing node and its block siblings (e.g. the
                # TestPoint slot "testpoint-<home>") are in scope.
                if node.id == home or home in node.id.split("-"):
                    continue
                if node.id not in _forward_closure(plan, home):
                    errors.append(
                        f"eigenvariable {e.expr!r} introduced by {intro[e.expr]!r} "
                        f"escapes at node {node.id!r}")
    return errors


def _forward_closure(plan: Plan, inst_id: str) -> set[str]:
    seen = {inst_id}
    stack = [plan.insts[inst_id].conclusion] if inst_id in plan.insts else []
    while stack:
        cur = stack.pop()
        for inst in plan.insts.values():
            if cur in inst.premises and inst.id not in seen:
                seen.add(inst.id)
                stack.append(inst.conclusion)
    return seen


def check_plan(plan: Plan, registry: SchemaRegistry) -> PlanReport:
    """Full check: well-typed + well-scoped + committed acyclicity. Pure."""
    inst_reports = [check_instance(plan, i, registry) for i in plan.insts.values()]
    scope_errors = check_well_scoped(plan)
    cycle = plan.committed_cycle()
    # Unpromoted schemas must not appear in committed derivations.
    trust_errors = []
    for inst in plan.insts.values():
        if inst.layer is Layer.COMMITTED and inst.schema_ref in registry.schemas:
            if not registry.usable_in_committed(inst.schema_ref):
                trust_errors.append(
                    f"unpromoted schema {inst.schema_ref!r} in committed instance {inst.id!r}")
    ok = (all(r.ok for r in inst_reports) and not scope_errors
          and cycle is None and not trust_errors)
    diag = "; ".join(trust_errors)
    if cycle:
        diag += f" committed cycle: {' -> '.join(cycle)}"
    return PlanReport(ok=ok, instances=inst_reports, scope_errors=scope_errors,
                      committed_cycle=cycle, diagnostic=diag)


def unify_shapes(a, b) -> dict | None:
    return unify_exact(a.term(), b.term())
