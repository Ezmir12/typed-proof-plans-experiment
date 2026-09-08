"""Registry-constrained LLM extractor: the model proposes, the checker decides.

The model sees only the registry's typed vocabulary (schema names, shape
heads, role names) and must reply in a fixed JSON contract. Names outside
the vocabulary are rejected here, before typechecking ever runs; anything
that passes this gate must still pass check_plan before it is trusted.
"""
from __future__ import annotations

from tpps.adapters.extract import CandidatePlan, ExtractedNode
from tpps.adapters.vocab import allowed_names, vocab_prompt
from tpps.core.plan import MethodInstance, Plan, Provenance, Role, StmtNode
from tpps.core.schema import SchemaRegistry
from tpps.core.stmt import Const, Shape, Stmt, Var


class VocabularyViolation(ValueError):
    """The model emitted a name outside the registry vocabulary."""


class DeclinedExtraction(ValueError):
    """The model reports no applicable schema for this input."""


SYSTEM_TEMPLATE = """\
You translate a natural-language proof sketch into a typed proof-plan fragment.

VOCABULARY (the ONLY schema names, statement heads, and role names you may emit):
{vocab}

OUTPUT CONTRACT (reply with exactly one JSON object, nothing else):
{{
  "nodes": [{{"id": str, "shape": str, "args": [{{"var": str}} | {{"const": str}}],
              "regime": str, "source_span": str, "confidence": number}}],
  "instances": [{{"id": str, "schema": str, "premises": [str],
                  "conclusion": str, "obligations": [str], "certificates": [str]}}],
  "roles": [{{"object": str, "role": str}}]
}}
or, when no schema fits: {{"no_applicable_schema": true, "reason": str}}

RULES:
- Every shape head, schema name, and role name must come from VOCABULARY.
- If a schema lists obligation templates, emit one node per template and
  list them in the instance's obligations. If it lists roles, bind each.
- source_span must be an exact substring of the proof motivating the node.
- confidence is your own 0..1 estimate per node.
- Never invent definitions, names, or rule semantics absent from the proof.
  When a definition is missing, emit the best-fitting skeleton anyway.
"""


class LlmExtractor:
    """Extracts CandidatePlans via a JSON-speaking chat client.

    `client` needs only `chat_json(system, user) -> dict`, so tests can
    substitute a scripted double; production passes HttpLlmClient.
    """

    def __init__(self, client, registry: SchemaRegistry, vocab: dict,
                 plan_id: str = "extracted", max_tokens: int = 8000) -> None:
        self.client = client
        self.registry = registry
        self.vocab = vocab
        self.schemas, self.heads, self.roles = allowed_names(vocab)
        self.system = SYSTEM_TEMPLATE.format(vocab=vocab_prompt(vocab))
        self.plan_id = plan_id
        self.max_tokens = max_tokens
        self.last_prompt = ""
        self.last_response: dict | None = None

    def extract(self, nl_text: str, diagnostic: str = "") -> CandidatePlan:
        """Derive a CandidatePlan from the NL text, not from a human plan."""
        user = f"PROOF:\n{nl_text}"
        if diagnostic:
            user += ("\nThe previous candidate was REJECTED by the typechecker:\n"
                     f"{diagnostic}\nFix exactly what the diagnostic says; "
                     "change nothing else.")
        self.last_prompt = user
        raw = self.client.chat_json(self.system, user,
                                    max_tokens=self.max_tokens)
        self.last_response = raw
        return self._build(nl_text, raw)

    def extract_with_diagnostic(self, nl_text: str, diagnostic: str) -> CandidatePlan:
        return self.extract(nl_text, diagnostic)

    # -- contract enforcement --
    def _arg(self, spec: dict):
        if "var" in spec:
            return Var(str(spec["var"]))
        if "const" in spec:
            return Const(str(spec["const"]))
        raise ValueError(f"malformed arg (want var/const): {spec!r}")

    def _build(self, nl_text: str, raw: dict) -> CandidatePlan:
        if raw.get("no_applicable_schema"):
            raise DeclinedExtraction(str(raw.get("reason", "")))
        nodespecs = raw.get("nodes", [])
        instspecs = raw.get("instances", [])
        if not isinstance(nodespecs, list) or not isinstance(instspecs, list):
            raise ValueError("malformed extraction: nodes/instances must be lists")
        for n in nodespecs:
            if n.get("shape") not in self.heads:
                raise VocabularyViolation(
                    f"shape {n.get('shape')!r} outside registry vocabulary")
        for i in instspecs:
            if i.get("schema") not in self.schemas:
                raise VocabularyViolation(
                    f"schema {i.get('schema')!r} outside registry vocabulary")
        for r in raw.get("roles", []):
            if r.get("role") not in self.roles:
                raise VocabularyViolation(
                    f"role {r.get('role')!r} outside registry vocabulary")

        plan = Plan(self.plan_id)
        for n in nodespecs:
            span = n.get("source_span", "")
            if not isinstance(span, str) or span not in nl_text:
                span = nl_text
            try:
                conf = float(n.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = min(1.0, max(0.0, conf))
            shape = Shape(n["shape"],
                          tuple(self._arg(a) for a in n.get("args", [])),
                          regime=str(n.get("regime", "")))
            plan.add_stmt(StmtNode(
                str(n["id"]), Stmt(shape=shape),
                provenance=Provenance(source_span=span,
                                      extraction_confidence=conf,
                                      producing_attempt="llm-extract")))
        for i in instspecs:
            prem, concl = list(i.get("premises", [])), str(i.get("conclusion", ""))
            obls = list(i.get("obligations", []))
            for nid in prem + [concl] + obls:
                if nid not in plan.stmts:
                    raise ValueError(f"instance {i.get('id')!r} "
                                     f"references unknown node {nid!r}")
            plan.add_inst(MethodInstance(
                str(i.get("id", "e")), str(i["schema"]), prem, concl,
                obligations=obls, certificates=[str(c) for c in
                                                i.get("certificates", [])]))
        for r in raw.get("roles", []):
            plan.roles.append(Role(str(r.get("object", "")),
                                   str(r["role"]), ""))
        extracted = [ExtractedNode(plan.stmts[str(n["id"])].stmt.shape,
                                   plan.stmts[str(n["id"])].provenance.source_span,
                                   plan.stmts[str(n["id"])].provenance.extraction_confidence,
                                   f"claim: {n['shape']}",
                                   "llm extraction ⇒ shape")
                     for n in nodespecs]
        return CandidatePlan(extracted, plan)
