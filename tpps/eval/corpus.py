"""Benchmark problem corpus and evaluation fixtures.

Loads versioned mathematical proofs, extraction examples, auxiliary synthesis
cases, and drift-injection benchmarks across rewriting, complex analysis, and algebra.
"""
from __future__ import annotations

from typing import Any

from tpps.core.plan import Plan
from tpps.domains.rewriting import (
    boolean_confluence_plan,
    list_confluence_plan,
    strategy_independence_cast_plan,
    toyama_refutation,
)
from tpps.examples import (
    fta_plan,
    plan_a,
    plan_b,
    plan_c_failure,
)


def load_benchmark_corpus() -> list[dict[str, Any]]:
    """Load canonical benchmark proof cases with ground-truth fields."""
    return [
        {
            "id": "taint-propagation-plan-a",
            "domain": "rewriting",
            "target": "Terminates",
            "target_node_id": "terminates",
            "regime": "finite-active",
            "method": "WFMeasure",
            "roles": ["RankingFunction", "Invariant"],
            "dependencies": ["Decreases", "WF", "NoRemove"],
            "plan_builder": plan_a,
        },
        {
            "id": "dependency-closure-plan-b",
            "domain": "rewriting",
            "target": "Terminates",
            "target_node_id": "terminates",
            "regime": "finite-active",
            "method": "FiniteActiveDomainSaturation",
            "roles": ["Certificate"],
            "dependencies": ["Saturated"],
            "plan_builder": plan_b,
        },
        {
            "id": "fundamental-theorem-algebra",
            "domain": "complex-analysis",
            "target": "Constant",
            "target_node_id": "const-g",
            "regime": "entire",
            "method": "Liouville",
            "roles": ["AuxiliaryFunction", "Decomposition"],
            "dependencies": ["Holo", "Bounded"],
            "plan_builder": fta_plan,
        },
        {
            "id": "list-confluence",
            "domain": "rewriting",
            "target": "Confluent",
            "target_node_id": "confluent",
            "regime": "",
            "method": "Newman",
            "roles": ["CriticalPair"],
            "dependencies": ["Terminates", "LocallyConfluent"],
            "plan_builder": list_confluence_plan,
        },
        {
            "id": "boolean-confluence",
            "domain": "rewriting",
            "target": "Confluent",
            "target_node_id": "confluent",
            "regime": "",
            "method": "Newman",
            "roles": [],
            "dependencies": ["Terminates", "LocallyConfluent"],
            "plan_builder": boolean_confluence_plan,
        },
    ]


def load_injection_cases() -> list[dict[str, Any]]:
    """Cases with injected unannounced representation shifts to test drift detection."""
    return [
        {
            "name": "set-to-multiset-drift",
            "from_regime": "as Set",
            "to_regime": "as Multiset",
            "recorded_as_cast": True,
            "blame": "confluence-to-strategy-equiv",
        },
        {
            "name": "pointwise-to-uniform-drift",
            "from_regime": "pointwise",
            "to_regime": "uniform",
            "recorded_as_cast": True,
            "blame": "limit-exchange",
        },
        {
            "name": "standard-derivation",
            "from_regime": "finite-active",
            "to_regime": "finite-active",
            "recorded_as_cast": False,
            "blame": "",
        },
    ]


def load_auxiliary_synthesis_cases() -> list[dict[str, Any]]:
    """Theorems where abduction must propose the correct auxiliary function."""
    return [
        {
            "theorem": "fundamental-theorem-of-algebra",
            "construction_proposed": True,
            "auxiliary_form": "1/p(z)",
            "obligations_match": True,
            "obligations": ["Holo", "Bounded"],
        },
        {
            "theorem": "schwarz-lemma",
            "construction_proposed": True,
            "auxiliary_form": "f(z)/z",
            "obligations_match": True,
            "obligations": ["ZeroAtOrigin", "Bounded"],
        },
    ]


def load_negative_transfer_cases() -> list[dict[str, Any]]:
    """Negative examples that must be blocked by sort/regime checks or witnesses."""
    return [
        {
            "name": "toyama-modularity-counterexample",
            "attempted_schema": "ToyamaModularity",
            "blocked": True,
            "reason": "witness refutation",
        },
        {
            "name": "fads-on-deletion",
            "attempted_schema": "FiniteActiveDomainSaturation",
            "blocked": True,
            "reason": "failed condition no-remove",
        },
        {
            "name": "schwarz-without-zero-hypothesis",
            "attempted_schema": "AuxiliaryDivision",
            "blocked": True,
            "reason": "witness counterexample",
        },
    ]

