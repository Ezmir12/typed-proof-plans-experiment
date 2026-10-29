"""Bipartite hypergraph (stmt nodes + method-instance nodes).

Committed vs speculative layers; S-expression atom
serialization for the MORK-style store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .evidence import BOTTOM, Evidence
from .stmt import Shape, Stmt


class Layer(Enum):
    COMMITTED = "committed"
    SPECULATIVE = "speculative"


@dataclass(frozen=True)
class Provenance:
    source_span: str = ""
    extraction_confidence: float = 0.0
    producing_attempt: str = ""
    kernel_artifact: str = ""


@dataclass
class StmtNode:
    id: str
    stmt: Stmt
    provenance: Provenance = field(default_factory=Provenance)
    evidence: Evidence = field(default_factory=Evidence)
    counter_evidence: Evidence = field(default_factory=Evidence)
    layer: Layer = Layer.COMMITTED


@dataclass(frozen=True)
class Role:
    object_ref: str
    role_name: str
    annotation: str = ""


@dataclass
class MethodInstance:
    id: str
    schema_ref: str
    premises: list[str]  # ordered stmt-node ids
    conclusion: str  # exactly one stmt-node id
    obligations: list[str] = field(default_factory=list)
    certificates: list[str] = field(default_factory=list)
    substitution: dict = field(default_factory=dict)
    layer: Layer = Layer.COMMITTED
    provenance: Provenance = field(default_factory=Provenance)


class Plan:
    def __init__(self, plan_id: str) -> None:
        self.id = plan_id
        self.stmts: dict[str, StmtNode] = {}
        self.insts: dict[str, MethodInstance] = {}
        self.roles: list[Role] = []

    # -- construction --
    def add_stmt(self, node: StmtNode) -> StmtNode:
        if node.id in self.stmts:
            raise ValueError(f"duplicate statement id {node.id!r}")
        self.stmts[node.id] = node
        return node

    def add_inst(self, inst: MethodInstance) -> MethodInstance:
        if inst.id in self.insts:
            raise ValueError(f"duplicate instance id {inst.id!r}")
        if inst.conclusion not in self.stmts:
            raise KeyError(f"conclusion {inst.conclusion!r} not in plan")
        for p in inst.premises + inst.obligations:
            if p not in self.stmts:
                raise KeyError(f"node {p!r} not in plan")
        self.insts[inst.id] = inst
        return inst

    def graph_errors(self) -> list[str]:
        """Return deterministic structural errors without mutating the plan."""
        errors: list[str] = []
        for inst_id in sorted(self.insts):
            inst = self.insts[inst_id]
            if inst.conclusion not in self.stmts:
                errors.append(
                    f"instance {inst.id!r} conclusion {inst.conclusion!r} "
                    "does not reference a statement"
                )
            for position, premise in enumerate(inst.premises):
                if premise not in self.stmts:
                    errors.append(
                        f"instance {inst.id!r} premise[{position}] {premise!r} "
                        "does not reference a statement"
                    )
            for position, obligation in enumerate(inst.obligations):
                if obligation not in self.stmts:
                    errors.append(
                        f"instance {inst.id!r} obligation[{position}] "
                        f"{obligation!r} does not reference a statement"
                    )
        return errors

    # -- committed support derivation --
    def committed_support(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for inst in self.insts.values():
                if inst.conclusion == cur and inst.layer is Layer.COMMITTED:
                    stack.extend(inst.premises)
                    stack.extend(inst.obligations)
        return seen

    def committed_cycle(self) -> list[str] | None:
        """Return a committed-layer cycle if one exists, else None."""
        # DFS over committed edges premise -> conclusion.
        succ: dict[str, list[str]] = {nid: [] for nid in self.stmts}
        for inst in self.insts.values():
            if inst.layer is Layer.COMMITTED:
                for p in inst.premises:
                    succ[p].append(inst.conclusion)
        color: dict[str, int] = {}
        path: list[str] = []

        def dfs(u: str) -> list[str] | None:
            color[u] = 1
            path.append(u)
            for v in succ.get(u, []):
                if color.get(v) == 1:
                    return path[path.index(v):] + [v]
                if color.get(v) is None:
                    hit = dfs(v)
                    if hit:
                        return hit
            path.pop()
            color[u] = 2
            return None

        for nid in self.stmts:
            if color.get(nid) is None:
                hit = dfs(nid)
                if hit:
                    return hit
        return None

    def producers(self, node_id: str) -> list[MethodInstance]:
        return [i for i in self.insts.values() if i.conclusion == node_id]

    # -- atom serialization --
    def to_atoms(self) -> list[str]:
        atoms = []
        for n in self.stmts.values():
            atoms.append(
                f"(plan {self.id} stmt {n.id} (seq {list(n.stmt.context)!r} "
                f"{n.stmt.shape.head}{list(n.stmt.shape.args)!r}@{n.stmt.regime}))")
        for i in self.insts.values():
            atoms.append(
                f"(plan {self.id} inst {i.id} {i.schema_ref} "
                f"({' '.join(i.premises)}) {i.conclusion} (sigma {i.substitution!r}))")
            for o in i.obligations:
                atoms.append(f"(plan {self.id} obl {i.id} {o})")
        for r in self.roles:
            atoms.append(f"(plan {self.id} role {r.object_ref} "
                         f"{r.role_name} {r.annotation})")
        return atoms

    @staticmethod
    def from_atoms(plan_id: str, atoms: list[str],
                   shape_of: dict[str, Shape] | None = None) -> Plan:
        """Deserialize; isomorphic round-trip given the same shape table."""
        import re

        plan = Plan(plan_id)
        shape_of = shape_of or {}
        for a in atoms:
            toks = a.strip("()").split()
            # (plan PID stmt NID (seq ...))
            if len(toks) >= 4 and toks[2] == "stmt":
                nid = toks[3]
                shape = shape_of.get(nid, Shape("Unknown"))
                plan.add_stmt(StmtNode(nid, Stmt(shape=shape)))
            elif len(toks) >= 7 and toks[2] == "inst":
                iid, schema = toks[3], toks[4]
                # premises are the parenthesized group; parse ids between parens
                m = re.search(r"\(([^()]*)\)\s+(\S+)\s+\(sigma", a)
                prems = m.group(1).split() if m and m.group(1).strip() else []
                concl = m.group(2) if m else ""
                plan.add_inst(MethodInstance(iid, schema, prems, concl))
            # obl atoms carry no new structure beyond inst.obligations linkage,
            # which is restored here:
            elif len(toks) >= 5 and toks[2] == "obl" and toks[3] in plan.insts:
                if toks[4] not in plan.insts[toks[3]].obligations:
                    plan.insts[toks[3]].obligations.append(toks[4])
            elif len(toks) >= 5 and toks[2] == "role":
                annotation = " ".join(toks[5:]).strip("() ") if len(toks) > 5 else ""
                plan.roles.append(Role(toks[3], toks[4], annotation))
        return plan


def empty_evidence(node: StmtNode) -> StmtNode:
    node.evidence = BOTTOM
    return node
