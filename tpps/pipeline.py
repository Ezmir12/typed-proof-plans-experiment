"""NL-to-rendered-proof pipeline with stage-by-stage logging.

run_pipeline threads one NL string through extract → typecheck → evidence →
creative (when triggered) → render → roundtrip, logging every stage to a
single JSONL file. Stage modules are only *called* here, never modified:
all stage functions are injectable via PipelineHooks so tests can use mocks.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tpps.adapters.extract import CandidatePlan, ExtractedNode
from tpps.core.evidence import (BOTTOM, Evidence, Reviewed, aggregate, conjoin,
                                is_closed)
from tpps.core.plan import Plan, Provenance, StmtNode
from tpps.core.schema import SchemaRegistry
from tpps.core.stmt import Shape, Stmt, Var
from tpps.core.typecheck import PlanReport
from tpps.adapters.render import Register, render
from tpps.core import typecheck as T
from tpps.logging.stage_logger import StageLogger
from tpps.reasoning.creative import backward_chain, classify_residual

DEFAULT_REQUIREMENT = Evidence(reviewed=Reviewed.INDEPENDENT)


@dataclass
class PipelineHooks:
    """Injectable stage functions. An explicit extract_fn always wins;
    otherwise the TPPS_EXTRACTOR flag selects the llm (default) or demo
    extractor. feedback_fn re-prompts with a checker diagnostic for retries."""

    extract_fn: Callable[[str], CandidatePlan] | None = None
    feedback_fn: Callable[[str, str], CandidatePlan] | None = None
    extract_trace_fn: Callable[[], dict] | None = None
    registry_fn: Callable[[], SchemaRegistry] | None = None
    requirement: Evidence | None = None
    register: str | None = None  # requested register; None = automatic
    creative_trigger_fn: Callable[[Plan, PlanReport, dict, bool], str | None] | None = None
    induce_fn: Callable[[list], list] | None = None
    creative_trace_fn: Callable[[], dict] | None = None


@dataclass
class RunResult:
    run_id: str
    log_path: str
    plan: Plan | None
    accepted: bool = False
    target_closed: bool = False
    evidence_by_node: dict = field(default_factory=dict)
    diffs: list = field(default_factory=list)


# --- Demo defaults (deterministic stand-ins, not stage logic) ---

def _demo_registry() -> SchemaRegistry:
    from tpps.examples import complex_registry, example_registry
    base = example_registry()
    for sid, schema in complex_registry().schemas.items():
        if sid not in base.schemas:
            base.schemas[sid] = schema
    return base


def _demo_extract(nl_text: str) -> CandidatePlan:
    """DEV/DEMO ONLY: keyword-mapped candidates, not real extraction.

    Reachable only via TPPS_EXTRACTOR=demo or an explicit hooks.extract_fn.
    Anything else becomes a single open claim."""
    from tpps.domains import complex_analysis as CA
    from tpps.domains import rewriting as RW
    lowered = nl_text.lower()
    if "entire" in lowered or "liouville" in lowered:
        plan, confidence = CA.liouville_plan(), 0.85
    elif "schwarz" in lowered:
        plan, confidence = CA.schwarz_plan(), 0.85
    elif "confluen" in lowered:
        plan, confidence = RW.boolean_confluence_plan(), 0.85
    else:
        plan = Plan("nl-claim")
        plan.add_stmt(StmtNode("claim", Stmt(shape=Shape("Claim", ()))))
        confidence = 0.4
    nodes = [ExtractedNode(n.stmt.shape, nl_text, confidence,
                           f"claim: {n.stmt.shape.head}",
                           "demo keyword mapping ⇒ shape")
             for n in plan.stmts.values()]
    for n in plan.stmts.values():
        n.provenance = Provenance(source_span=nl_text,
                                  extraction_confidence=confidence,
                                  producing_attempt="demo-extractor")
    candidate = CandidatePlan(nodes, plan)
    return candidate


def _default_trigger(plan: Plan, report: PlanReport, evidence_by_node: dict,
                     target_closed: bool) -> str | None:
    if not report.ok:
        return "typecheck_failed"
    if not target_closed:
        return "open_obligations"
    return None


def _primary_target(plan: Plan) -> str:
    conclusions = {i.conclusion for i in plan.insts.values()}
    consumed = ({p for i in plan.insts.values() for p in i.premises}
                | {o for i in plan.insts.values() for o in i.obligations})
    sinks = [nid for nid in plan.stmts if nid not in consumed]
    cands = [n for n in sinks if n in conclusions] or sinks or sorted(plan.stmts)
    return sorted(cands)[0]


def _aggregate_plan(plan: Plan, registry: SchemaRegistry) -> dict[str, object]:
    stored = {nid: n.evidence for nid, n in plan.stmts.items()}
    out: dict[str, object] = {}
    for nid in plan.stmts:
        derivs = []
        for inst in plan.producers(nid):
            schema = registry.schemas.get(inst.schema_ref)
            tau = schema.trust.evidence if schema else Evidence()
            parts = ([tau]
                     + [stored.get(p, BOTTOM) for p in inst.premises]
                     + [stored.get(o, BOTTOM) for o in inst.obligations])
            derivs.append(conjoin(parts))
        out[nid] = aggregate(stored[nid], derivs)
    return out


def _diagnostics(report) -> list[str]:
    return ([r.diagnostic for r in report.instances if not r.ok]
            + report.scope_errors
            + ([report.diagnostic] if report.diagnostic else []))


def replace_settings_extractor(settings, extractor: str):
    """Override the configured extractor for one run (leaves files alone)."""
    from dataclasses import replace
    return replace(settings, app=replace(settings.app, extractor=extractor))


def _resolve_extractor(hooks: PipelineHooks, registry: SchemaRegistry,
                       settings):
    """Return (mode, label, attempt_fn, feedback_available, trace_fn).

    attempt_fn(nl_text, diagnostic) runs one extraction attempt. label names
    the concrete extractor class for the log. Unknown modes raise ValueError.
    """
    from tpps.adapters.llm import HttpLlmClient
    from tpps.adapters.llm_extract import LlmExtractor
    from tpps.adapters.vocab import export_vocabulary
    from tpps.domains import rewriting as RW

    if hooks.extract_fn is not None:
        base = hooks.extract_fn
        feedback = hooks.feedback_fn
        label = f"hooks:{getattr(base, '__name__', type(base).__name__)}"
        def attempt(nl_text: str, diagnostic: str) -> CandidatePlan:
            if diagnostic and feedback is not None:
                return feedback(nl_text, diagnostic)
            return base(nl_text)
        return "hooks", label, attempt, feedback is not None, hooks.extract_trace_fn

    mode = settings.app.extractor
    if mode == "demo":
        return "demo", "DemoExtractor", lambda nl, diag: _demo_extract(nl), False, None
    if mode == "llm":
        vocab = export_vocabulary(registry, RW.sort_library(), list(RW.ROLES))
        ext = LlmExtractor(HttpLlmClient(settings.llm), registry, vocab)
        def llm_trace() -> dict:
            return {"prompt": ext.last_prompt, "response": ext.last_response}
        return "llm", type(ext).__name__, ext.extract, True, llm_trace
    raise ValueError(f"unknown extractor {mode!r} (want llm|demo|hooks)")


def _diff_plans(first: Plan, second: Plan) -> list[dict]:
    """Shape/regime/obligation deltas only, fully enumerated, sorted."""
    diffs = []
    kinds = ("shape", "regime")
    for kind in kinds:
        key = (lambda n: n.stmt.shape.head) if kind == "shape" else (
            lambda n: (n.stmt.shape.head, n.stmt.regime))
        ca, cb = Counter(map(key, first.stmts.values())), Counter(map(key, second.stmts.values()))
        for item in sorted(set(ca) | set(cb), key=str):
            if ca[item] != cb[item]:
                diffs.append({"kind": kind, "item": str(item),
                              "expected": ca[item], "found": cb[item]})
    obls_a = sum(len(i.obligations) for i in first.insts.values())
    obls_b = sum(len(i.obligations) for i in second.insts.values())
    impl_a = sum(1 for n in first.stmts.values() if n.stmt.shape.head == "Implies")
    impl_b = sum(1 for n in second.stmts.values() if n.stmt.shape.head == "Implies")
    if (obls_a, impl_a) != (obls_b, impl_b):
        diffs.append({"kind": "obligation",
                      "expected": {"obligations": obls_a, "open": impl_a},
                      "found": {"obligations": obls_b, "open": impl_b}})
    return sorted(diffs, key=lambda d: (d["kind"], str(d.get("item", ""))))


# --- The pipeline ---

def run_pipeline(nl_text: str, log_dir: str | Path = "logs/runs",
                 hooks: PipelineHooks | None = None,
                 level: str | None = None,
                 extractor: str | None = None,
                 settings=None) -> RunResult:
    """Run NL input to rendered output, logging every stage to one JSONL file.

    Extractor selection: explicit hooks.extract_fn wins; else the `extractor`
    argument or the configured TPPS_EXTRACTOR flag (llm by default, demo only
    on request). Rejected candidates are re-prompted with the checker's
    diagnostic (bounded retries); a clean decline stops the run gracefully.
    Soft failures (ill-typed plan, unclosed target) are logged and rendered,
    not skipped; only a raised exception stops the pipeline early.
    """
    from tpps.adapters.llm_extract import DeclinedExtraction, VocabularyViolation
    from tpps.adapters.llm import LlmError
    from tpps.config import load_config

    hooks = hooks or PipelineHooks()
    settings = settings or load_config()
    if extractor is not None:
        settings = replace_settings_extractor(settings, extractor)
    registry_fn = hooks.registry_fn or _demo_registry
    registry = registry_fn()
    requirement = hooks.requirement or DEFAULT_REQUIREMENT
    trigger_fn = hooks.creative_trigger_fn or _default_trigger
    # Resolve before creating the logger: an unknown mode fails explicitly
    # with no log file and no stage records at all.
    mode, extractor_label, attempt, feedback, trace_fn = _resolve_extractor(
        hooks, registry, settings)
    logger = StageLogger(log_dir, level)
    plan: Plan | None = None

    def done(accepted=False, target_closed=False, evidence_by_node=None, diffs=None):
        return RunResult(logger.run_id, str(logger.path), plan, accepted,
                         target_closed, evidence_by_node or {}, diffs or [])

    single_shot = lambda text: attempt(text, "")
    max_attempts = 1 + (settings.app.extract_retries if feedback else 0)

    # 1. extract (one stage event per attempt; diagnostics fed back verbatim)
    candidate: CandidatePlan | None = None
    report = None
    diagnostic = ""
    last_error: Exception | None = None
    for attempt_no in range(1, max_attempts + 1):
        with logger.stage("extract", {"nl_text": nl_text, "attempt": attempt_no,
                                      "mode": mode,
                                      "extractor": extractor_label}) as st:
            try:
                candidate = attempt(nl_text, diagnostic)
            except DeclinedExtraction as exc:
                st.set_output({"extractor": extractor_label,
                               "declined": True, "reason": str(exc)})
                return done()
            except (VocabularyViolation, LlmError) as exc:
                last_error = exc
                diagnostic = f"{type(exc).__name__}: {exc}"
                st.set_output({"extractor": extractor_label,
                               "well_typed": False, "attempt": attempt_no,
                               "diagnostics": [diagnostic],
                               "will_retry": attempt_no < max_attempts})
                continue
            plan = candidate.plan
            report = T.check_plan(plan, registry)
            diagnostics = _diagnostics(report)
            confs = [n.extraction_confidence for n in candidate.nodes] or [0.0]
            st.set_output({"extractor": extractor_label,
                           "candidate_plan": plan.to_atoms(),
                           "extraction_confidence": sum(confs) / len(confs),
                           "node_count": len(plan.stmts),
                           "attempt": attempt_no,
                           "well_typed": all(r.ok for r in report.instances),
                           "diagnostics": diagnostics,
                           "will_retry": not report.ok and attempt_no < max_attempts})
            if trace_fn:
                st.set_debug(trace_fn())
        if report.ok or attempt_no >= max_attempts:
            break
        diagnostic = "; ".join(diagnostics) or "candidate rejected"
    if candidate is None or report is None:
        raise last_error if last_error is not None else RuntimeError(
            "extraction produced no candidate")

    # 2. typecheck
    with logger.stage("typecheck", {"candidate_plan_id": plan.id}) as st:
        diagnostics = _diagnostics(report)
        st.set_output({"well_typed": all(r.ok for r in report.instances),
                       "well_scoped": not report.scope_errors,
                       "diagnostics": diagnostics})
    accepted = report.ok

    # 3. evidence
    with logger.stage("evidence", {"plan_id": plan.id}) as st:
        mus = _aggregate_plan(plan, registry)
        per_node = {nid: {"formal": mu.formal.name, "certified": mu.certified.name,
                          "reviewed": mu.reviewed.name, "empirical": mu.empirical.name,
                          "fresh": mu.fresh.name} for nid, mu in mus.items()}
        closure = {nid: is_closed(mu, requirement) for nid, mu in mus.items()}
        target = _primary_target(plan)
        target_closed = closure.get(target, False)
        st.set_output({"per_node_evidence": per_node, "closure_status": closure,
                       "target": target, "target_closed": target_closed})

    # 4. creative (always logged; runs only when triggered)
    with logger.stage("creative", {"plan_id": plan.id}) as st:
        reason = trigger_fn(plan, report, mus, target_closed)
        residuals: list[dict] = []
        induced: list[dict] = []
        if reason is not None:
            st.note("trigger_reason", reason)
            index = [(sid, s.conclusion) for sid, s in registry.schemas.items()]
            known = {s.conclusion.head for s in registry.schemas.values()}
            heads = {n.stmt.shape.head for n in plan.stmts.values()}
            for nid, node in plan.stmts.items():
                if plan.producers(nid):
                    continue
                for res in backward_chain(node.stmt.shape, index, known, heads):
                    residuals.append({"shape": res.shape.head,
                                      "origin_schema": res.origin_schema,
                                      "critic_class": classify_residual(res)})
            if hooks.induce_fn:
                induced = hooks.induce_fn(residuals)
        st.set_output({"trigger_reason": reason, "triggered": reason is not None,
                       "residuals": residuals, "induced_candidates": induced})
        if hooks.creative_trace_fn:
            st.set_debug(hooks.creative_trace_fn())

    # 5. render (always runs; rejected/open plans render annotated)
    with logger.stage("render", {"plan_id": plan.id, "target": target,
                                 "register": hooks.register or "auto"}) as st:
        st.set_output({"textbook_text": render(plan, target, requirement,
                                                Register.TEXTBOOK),
                       "annotated_text": render(plan, target, requirement,
                                                Register.ANNOTATED)})

    # 6. roundtrip (the same real extractor, run on rendered output)
    with logger.stage("roundtrip", {}) as st:
        textbook = render(plan, target, requirement, Register.TEXTBOOK)
        st.note("rendered_text", textbook)
        try:
            re_extracted = single_shot(textbook)
        except DeclinedExtraction as exc:
            diffs = [{"kind": "extraction",
                      "detail": f"re-extraction declined: {exc}"}]
            st.set_output({"match": False, "diffs": diffs})
            return done(accepted, target_closed, mus, diffs)
        diffs = _diff_plans(plan, re_extracted.plan)
        st.set_output({"match": not diffs, "diffs": diffs})

    return done(accepted, target_closed, mus, diffs)
