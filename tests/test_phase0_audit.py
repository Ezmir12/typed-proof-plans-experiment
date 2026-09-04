"""Phase 0 audit: the checker must enforce schema signatures.

For every rewriting schema: a missing premise, a wrong-shaped premise, a
missing obligation node, or a missing certificate must be REJECTED. A full
contract — including obligation nodes that are present but open and
undischarged — must be ACCEPTED (well-typed and closed are different things).
"""
import pytest

from tpps.core import typecheck as T
from tpps.core.plan import MethodInstance, Plan, StmtNode
from tpps.core.schema import CondTag
from tpps.core.stmt import Shape, Stmt, Var
from tpps.domains import rewriting as RW
from tpps.examples import example_registry

REG = example_registry()

WITH_CERTS = [sid for sid in RW.SCHEMA_IDS
              if any(c.tag is CondTag.C for c in REG.schemas[sid].conditions)]
WITH_OBLS = [sid for sid in RW.SCHEMA_IDS
             if REG.schemas[sid].obligation_templates]


def _contract_plan(sid: str, drop_premise=False, wrong_premise=False,
                   drop_obls=False, drop_certs=False) -> Plan:
    schema = REG.schemas[sid]
    plan = Plan(f"audit-{sid}")
    prems = schema.premises if not drop_premise else schema.premises[:-1]
    prem_ids = []
    for i, shape in enumerate(prems):
        if wrong_premise and i == 0:
            shape = Shape("Bogus", shape.args)
        plan.add_stmt(StmtNode(f"p{i}", Stmt(shape=shape)))
        prem_ids.append(f"p{i}")
    plan.add_stmt(StmtNode("c", Stmt(shape=schema.conclusion)))
    obl_ids = []
    for i, shape in enumerate([] if drop_obls else schema.obligation_templates):
        plan.add_stmt(StmtNode(f"o{i}", Stmt(shape=shape)))
        obl_ids.append(f"o{i}")
    certs = [] if drop_certs else [
        c.predicate_ref for c in schema.conditions if c.tag is CondTag.C]
    plan.add_inst(MethodInstance("e", sid, prem_ids, "c",
                                 obligations=obl_ids, certificates=certs))
    return plan


def test_wfmeasure_rejects_decreases_only():
    """Decreases without WF(O) must not satisfy WFMeasure, even though the
    old single-premise registration wrongly accepted exactly this plan."""
    plan = Plan("wfmeasure-decreases-only")
    plan.add_stmt(StmtNode("d", Stmt(shape=Shape("Decreases", (Var("R"),)))))
    plan.add_stmt(StmtNode("t", Stmt(shape=Shape("Terminates", (Var("R"),)))))
    plan.add_inst(MethodInstance("e", "WFMeasure", ["d"], "t"))
    assert T.check_plan(plan, REG).ok is False


def test_wfmeasure_full_contract_accepted():
    plan = _contract_plan("WFMeasure")
    assert T.check_plan(plan, REG).ok, T.check_plan(plan, REG).diagnostic


@pytest.mark.parametrize("sid", RW.SCHEMA_IDS)
def test_missing_premise_rejected(sid):
    assert T.check_plan(_contract_plan(sid, drop_premise=True), REG).ok is False


@pytest.mark.parametrize("sid", RW.SCHEMA_IDS)
def test_wrong_premise_shape_rejected(sid):
    assert T.check_plan(_contract_plan(sid, wrong_premise=True), REG).ok is False


@pytest.mark.parametrize("sid", WITH_OBLS)
def test_missing_obligation_rejected(sid):
    assert T.check_plan(_contract_plan(sid, drop_obls=True), REG).ok is False


@pytest.mark.parametrize("sid", WITH_CERTS)
def test_missing_certificate_rejected(sid):
    assert T.check_plan(_contract_plan(sid, drop_certs=True), REG).ok is False


@pytest.mark.parametrize("sid", RW.SCHEMA_IDS)
def test_full_contract_accepted(sid):
    plan = _contract_plan(sid)
    assert T.check_plan(plan, REG).ok, T.check_plan(plan, REG).diagnostic


def test_open_obligations_are_still_well_typed():
    """Obligation nodes present but undischarged: accepted as well-typed."""
    plan = _contract_plan("FiniteActiveDomainSaturation")
    assert T.check_plan(plan, REG).ok
    assert plan.stmts["o0"].evidence.formal.name == "NONE"
