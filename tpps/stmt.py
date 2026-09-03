"""Sequents Γ ⊢ φ with de Bruijn binders and four distinct checks."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# --- Terms (sorted signature, de Bruijn binders) ---

@dataclass(frozen=True)
class Var:
    name: str  # free / pattern variable; "?U" style = anti-unification hole


@dataclass(frozen=True)
class BVar:
    index: int  # de Bruijn index for bound variables


@dataclass(frozen=True)
class Const:
    value: str
    sort: str = ""


@dataclass(frozen=True)
class App:
    head: str
    args: tuple[Term, ...] = ()


@dataclass(frozen=True)
class Binder:
    kind: str  # "forall" | "exists" | "lambda"
    body: Term = None  # type: ignore[assignment]


Term = Var | BVar | Const | App | Binder


def alpha_normalize(t: Term, depth: int = 0, env: dict[str, int] | None = None) -> Term:
    """Reify named binders to de Bruijn indices."""
    env = {} if env is None else env
    if isinstance(t, Var):
        return BVar(depth - 1 - env[t.name]) if t.name in env else t
    if isinstance(t, (BVar, Const)):
        return t
    if isinstance(t, App):
        return App(t.head, tuple(alpha_normalize(a, depth, env) for a in t.args))
    if isinstance(t, Binder):
        # Binder introduces one anonymous level; named Vars are resolved
        # positionally only if they shadow via env pushed by callers using
        # bind(name, body) helper below. Bare Binder bodies are normalized as-is.
        assert t.body is not None
        return Binder(t.kind, alpha_normalize(t.body, depth + 1, env))
    raise TypeError(f"unknown term {t!r}")


def bind(kind: str, var_name: str, body: Term) -> Binder:
    """Helper: bind var_name occurrences in body as de Bruijn level 0."""
    def go(t: Term, depth: int) -> Term:
        if isinstance(t, Var):
            return BVar(depth) if t.name == var_name else t
        if isinstance(t, (BVar, Const)):
            return t
        if isinstance(t, App):
            return App(t.head, tuple(go(a, depth) for a in t.args))
        if isinstance(t, Binder):
            assert t.body is not None
            return Binder(t.kind, go(t.body, depth + 1))
        raise TypeError(f"unknown term {t!r}")

    return Binder(kind, go(body, 0))


def alpha_eq(a: Term, b: Term) -> bool:
    return alpha_normalize(a) == alpha_normalize(b)


# --- Substitution / unification (first-order + Miller-pattern holes) ---

Subst = dict[str, Term]


def apply_subst(t: Term, s: Subst) -> Term:
    if isinstance(t, Var):
        return s.get(t.name, t)
    if isinstance(t, (BVar, Const)):
        return t
    if isinstance(t, App):
        return App(t.head, tuple(apply_subst(a, s) for a in t.args))
    if isinstance(t, Binder):
        assert t.body is not None
        return Binder(t.kind, apply_subst(t.body, s))
    raise TypeError(f"unknown term {t!r}")


def _occurs(name: str, t: Term) -> bool:
    if isinstance(t, Var):
        return t.name == name
    if isinstance(t, (BVar, Const)):
        return False
    if isinstance(t, App):
        return any(_occurs(name, a) for a in t.args)
    if isinstance(t, Binder):
        assert t.body is not None
        return _occurs(name, t.body)
    raise TypeError(f"unknown term {t!r}")


def unify_exact(pattern: Term, target: Term, subst: Subst | None = None) -> Subst | None:
    """First-order unification; Var in pattern acts as a match variable.

    Miller-pattern fragment: a `?P`-style second-order variable applied to
    distinct bound variables is treated as an opaque match variable here;
    full higher-order search is out of scope for the prototype.
    """
    subst = {} if subst is None else dict(subst)
    if isinstance(pattern, Var):
        if pattern.name in subst:
            return subst if subst[pattern.name] == target else None
        if isinstance(target, Var) and target.name == pattern.name:
            return subst  # X unifies with itself; no binding needed
        if _occurs(pattern.name, target):
            return None
        subst[pattern.name] = target
        return subst
    if isinstance(pattern, BVar):
        return subst if pattern == target else None
    if isinstance(pattern, Const):
        return subst if pattern == target else None
    if isinstance(pattern, App):
        if not isinstance(target, App) or pattern.head != target.head:
            return None
        if len(pattern.args) != len(target.args):
            return None
        for p, q in zip(pattern.args, target.args):
            subst = unify_exact(p, q, subst)
            if subst is None:
                return None
        return subst
    if isinstance(pattern, Binder):
        if not isinstance(target, Binder) or pattern.kind != target.kind:
            return None
        assert pattern.body is not None and target.body is not None
        return unify_exact(pattern.body, target.body, subst)
    raise TypeError(f"unknown term {pattern!r}")


# --- Shapes, contexts, sequents ---

@dataclass(frozen=True)
class Shape:
    head: str
    args: tuple[Term, ...] = ()
    regime: str = ""
    sort: str = ""  # head sort, used for cast clash detection

    def term(self) -> Term:
        return App(self.head, self.args)

    def instantiate(self, s: Subst) -> Shape:
        return Shape(self.head, tuple(apply_subst(a, s) for a in self.args), self.regime, self.sort)


@dataclass(frozen=True)
class CtxEntry:
    kind: str  # "Assumption" | "Definition" | "Eigenvar"
    expr: str
    introduced_by: str = ""  # instance id that introduced this entry


@dataclass(frozen=True)
class Stmt:
    context: tuple[CtxEntry, ...] = ()
    shape: Shape = field(default_factory=lambda: Shape("True"))
    regime: str = ""

    def instantiate(self, s: Subst) -> Stmt:
        return Stmt(self.context, self.shape.instantiate(s), self.regime)


class CheckKind(Enum):
    EXACT = "exact"
    ALPHA = "alpha"
    SUBSUME = "subsume"
    SPECIALIZE = "specialize"


@dataclass(frozen=True)
class CheckResult:
    kind: CheckKind
    ok: bool
    subst: Subst = field(default_factory=dict)


def run_check(kind: CheckKind, pattern: Shape, target: Shape, regimes) -> CheckResult:
    """Four separately invocable checks; never conflated."""
    if kind is CheckKind.EXACT:
        s = unify_exact(pattern.term(), target.term())
        return CheckResult(kind, s is not None and pattern.regime == target.regime, s or {})
    if kind is CheckKind.ALPHA:
        return CheckResult(kind, alpha_eq(pattern.term(), target.term()))
    if kind is CheckKind.SUBSUME:
        s = unify_exact(pattern.term(), target.term())
        ok = s is not None and regimes.subsumes(pattern.regime, target.regime)
        return CheckResult(kind, ok, s or {})
    if kind is CheckKind.SPECIALIZE:
        s = unify_exact(pattern.term(), target.term())
        return CheckResult(kind, s is not None, s or {})
    raise ValueError(kind)
