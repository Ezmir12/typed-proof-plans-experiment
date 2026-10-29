"""Parametrized many-sorted types with structural/certified attributes.

`SortDecl` is the canonical sort-declaration type for this repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Attribute:
    name: str
    # "structural" or "certified:<procedure_ref>"
    klass: str

    @property
    def is_structural(self) -> bool:
        return self.klass == "structural"

    @property
    def procedure_ref(self) -> str | None:
        if self.klass.startswith("certified:"):
            return self.klass.split(":", 1)[1]
        return None


@dataclass(frozen=True)
class SortDecl:
    name: str
    params: tuple[str, ...] = ()
    attributes: tuple[Attribute, ...] = ()
    erasure_target: str = ""

    def attribute(self, name: str) -> Attribute:
        for a in self.attributes:
            if a.name == name:
                return a
        raise KeyError(f"sort {self.name} has no attribute {name!r}")


@dataclass
class RegimeOrder:
    """Decidable ⊑ over regimes. subsumes(general, specific)."""

    pairs: set[tuple[str, str]] = field(default_factory=set)

    def add(self, general: str, specific: str) -> None:
        self.pairs.add((general, specific))

    def subsumes(self, general: str, specific: str) -> bool:
        if general == specific:
            return True
        return (general, specific) in self.pairs

    def explain(self, general: str, specific: str) -> str:
        """Explain a known relationship or explicitly report uncertainty."""
        if general == specific:
            return f"{general!r} subsumes itself"
        if (general, specific) in self.pairs:
            return f"{general!r} subsumes {specific!r}"
        return f"unknown regime relationship: {general!r} vs {specific!r}"


class SortRegistry:
    def __init__(self) -> None:
        self.sorts: dict[str, SortDecl] = {}
        self.regimes = RegimeOrder()

    def declare(self, decl: SortDecl) -> SortDecl:
        self.sorts[decl.name] = decl
        return decl

    def get(self, name: str) -> SortDecl:
        return self.sorts[name]

    def route_attribute(self, sort: str, attr: str) -> str:
        """Return 'structural' or the certified procedure ref."""
        a = self.sorts[sort].attribute(attr)
        if a.is_structural:
            return "structural"
        return a.procedure_ref or "certified:unknown"

    def roundtrip(self, decl: SortDecl) -> SortDecl:
        """Storage round-trip preserves all fields."""
        import copy

        return copy.deepcopy(decl)
