"""LlmExtractor unit tests with a scripted client (no network).

These pin the contract machinery — vocabulary export, hard rejects,
candidate construction — not model behavior. Graded model behavior is
gated separately and needs a live backend.
"""
import pytest

from tpps.adapters.llm_extract import (DeclinedExtraction, LlmExtractor,
                                       VocabularyViolation)
from tpps.adapters.vocab import allowed_names, export_vocabulary, vocab_prompt
from tpps.core import typecheck as T
from tpps.domains import rewriting as RW
from tpps.examples import example_registry


def _registry():
    return example_registry()


def _vocab():
    return export_vocabulary(_registry(), RW.sort_library(), list(RW.ROLES))


class ScriptedClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat_json(self, system, user, **kwargs):
        self.calls.append((system, user))
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


NL = "The rules finish on any finite space because each step adds something new."

WF_PAYLOAD = {
    "nodes": [
        {"id": "t", "shape": "Terminates", "args": [{"var": "R"}],
         "regime": "", "source_span": "rules finish", "confidence": 0.8},
        {"id": "d", "shape": "Decreases",
         "args": [{"var": "m"}, {"var": "R"}], "regime": "",
         "source_span": "each step adds something new", "confidence": 0.7},
        {"id": "w", "shape": "WF", "args": [{"var": "O"}],
         "regime": "", "source_span": "finite space", "confidence": 0.7},
        {"id": "o1", "shape": "Bounded",
         "args": [{"var": "R"}, {"var": "U"}], "regime": "",
         "source_span": "finite space", "confidence": 0.6},
        {"id": "o2", "shape": "NoRemove", "args": [{"var": "R"}],
         "regime": "", "source_span": "adds something new", "confidence": 0.6},
        {"id": "o3", "shape": "Productive", "args": [{"var": "R"}],
         "regime": "", "source_span": "adds something new", "confidence": 0.6},
    ],
    "instances": [{"id": "e", "schema": "WFMeasure", "premises": ["d", "w"],
                   "conclusion": "t", "obligations": ["o1", "o2", "o3"],
                   "certificates": []}],
    "roles": [{"object": "m", "role": "RankingFunction"}],
}


def test_vocab_export_covers_wfmeasure_contract():
    vocab = _vocab()
    wf = next(s for s in vocab["schemas"] if s["name"] == "WFMeasure")
    assert [p["head"] for p in wf["premises"]] == ["Decreases", "WF"]
    assert wf["conclusion"]["head"] == "Terminates"
    assert [o["head"] for o in wf["obligation_templates"]] == [
        "Bounded", "NoRemove", "Productive"]
    assert "RankingFunction" in wf["roles"]
    schemas, heads, roles = allowed_names(vocab)
    assert {"WFMeasure", "Newman"} <= schemas
    assert {"Terminates", "Decreases", "WF", "Bounded"} <= heads
    assert "RankingFunction" in roles
    assert "WFMeasure" in vocab_prompt(vocab)


def test_valid_payload_builds_checkable_plan():
    ext = LlmExtractor(ScriptedClient([WF_PAYLOAD]), _registry(), _vocab())
    cand = ext.extract(NL)
    assert T.check_plan(cand.plan, _registry()).ok
    inst = cand.plan.insts["e"]
    assert sorted(inst.obligations) == ["o1", "o2", "o3"]
    assert cand.plan.roles and cand.plan.roles[0].role_name == "RankingFunction"
    assert cand.plan.stmts["t"].provenance.source_span == "rules finish"
    assert cand.nodes[0].extraction_confidence == 0.8


def test_unknown_names_rejected_before_typecheck():
    import copy
    bad = copy.deepcopy(WF_PAYLOAD)
    bad["instances"][0]["schema"] = "MagicOracle"
    ext = LlmExtractor(ScriptedClient([bad]), _registry(), _vocab())
    try:
        ext.extract(NL)
        raise AssertionError("expected VocabularyViolation")
    except VocabularyViolation:
        pass
    bad2 = copy.deepcopy(WF_PAYLOAD)
    bad2["nodes"][0]["shape"] = "Teleportates"
    ext2 = LlmExtractor(ScriptedClient([bad2]), _registry(), _vocab())
    try:
        ext2.extract(NL)
        raise AssertionError("expected VocabularyViolation")
    except VocabularyViolation:
        pass


def test_decline_and_malformed():
    ext = LlmExtractor(
        ScriptedClient([{"no_applicable_schema": True, "reason": "shopping"}]),
        _registry(), _vocab())
    try:
        ext.extract("buy milk")
        raise AssertionError("expected DeclinedExtraction")
    except DeclinedExtraction:
        pass
    ext2 = LlmExtractor(ScriptedClient([{"nodes": "not-a-list", "instances": []}]),
                        _registry(), _vocab())
    try:
        ext2.extract(NL)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_invalid_spans_are_rejected_and_confidence_is_clamped():
    import copy
    payload = copy.deepcopy(WF_PAYLOAD)
    payload["nodes"][0]["source_span"] = "not in the input at all"
    payload["nodes"][0]["confidence"] = 7.5
    ext = LlmExtractor(ScriptedClient([payload]), _registry(), _vocab())
    with pytest.raises(ValueError, match="source_span"):
        ext.extract(NL)

    payload = copy.deepcopy(WF_PAYLOAD)
    payload["nodes"][0]["confidence"] = 7.5
    cand = LlmExtractor(ScriptedClient([payload]), _registry(), _vocab()).extract(NL)
    assert cand.plan.stmts["t"].provenance.extraction_confidence == 1.0


def test_no_fixture_words_in_extractor():
    """T2.1 fail condition, enforced offline: fixture-specific vocabulary
    must not appear in extractor code or prompt templates."""
    import pathlib
    for name in ("llm_extract.py", "vocab.py"):
        src = (pathlib.Path(__file__).resolve().parent.parent
               / "tpps" / "adapters" / name).read_text().lower()
        for word in ("stale", "superseded", "outdated"):
            assert word not in src, f"{name} contains fixture word {word!r}"


def test_extractor_rejects_model_authority_and_forbidden_keys():
    import copy
    import pytest
    from tpps.core.plan import Layer
    for forbidden_key in ("trust_status", "trust_decision", "kernel_verified", "artifact_ref"):
        bad_top = copy.deepcopy(WF_PAYLOAD)
        bad_top[forbidden_key] = "unauthorized"
        ext = LlmExtractor(ScriptedClient([bad_top]), _registry(), _vocab())
        with pytest.raises(ValueError):
            ext.extract(NL)

    for forbidden_field in ("checker", "soundness_theorem", "elaborator", "kernel_artifact", "evidence"):
        bad_node = copy.deepcopy(WF_PAYLOAD)
        bad_node["nodes"][0][forbidden_field] = "hack"
        ext = LlmExtractor(ScriptedClient([bad_node]), _registry(), _vocab())
        with pytest.raises(ValueError):
            ext.extract(NL)

        bad_inst = copy.deepcopy(WF_PAYLOAD)
        bad_inst["instances"][0][forbidden_field] = "hack"
        ext = LlmExtractor(ScriptedClient([bad_inst]), _registry(), _vocab())
        with pytest.raises(ValueError):
            ext.extract(NL)


def test_extracted_nodes_and_instances_are_speculative_with_open_evidence():
    from tpps.core.plan import Layer
    from tpps.core.evidence import Formal
    ext = LlmExtractor(ScriptedClient([WF_PAYLOAD]), _registry(), _vocab())
    cand = ext.extract(NL)
    for node in cand.plan.stmts.values():
        assert node.layer is Layer.SPECULATIVE
        assert node.evidence.formal is Formal.NONE
    for inst in cand.plan.insts.values():
        assert inst.layer is Layer.SPECULATIVE


def test_extracted_certificate_names_do_not_grant_certification():
    import copy
    payload = copy.deepcopy(WF_PAYLOAD)
    payload["instances"] = [{
        "id": "e",
        "schema": "LPO",
        "premises": ["d"],
        "conclusion": "t",
        "obligations": [],
        "certificates": ["lpo-oriented"],
    }]
    ext = LlmExtractor(ScriptedClient([payload]), _registry(), _vocab())
    candidate = ext.extract(NL)
    assert not T.check_plan(candidate.plan, _registry()).ok
