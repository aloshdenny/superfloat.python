"""Mix tests over the whole Stage 1.0 grammar (spec sections 5 and 10).

Random expression trees over + - * / unary - abs() sf.fma() and ( ), with
SF values and host literals as leaves.  Each tree is rendered as Python
source and evaluated two ways:

1. ``eval`` on real sf objects, so Python's own parsing and precedence are
   what is being tested;
2. a step-by-step walk of the ``ast`` with the Fraction oracle, in the
   same evaluation order (left operand, right operand, then the op).

The step-by-step SF result and *the first raised error* (its type and the
op that raised it) must match the oracle.  Underflow is tested both as a
warning (default policy) and as a raise (``under="raise"``).
"""
from __future__ import annotations

import ast
import random
from fractions import Fraction

import pytest

import sf
from conftest import EVENT_ERROR, SEED, dtype_of
from oracle import Fmt

BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


class Gen:
    """Random expression source over the 1.0 grammar."""

    def __init__(self, bits: int, seed: int, literal_prob: float = 0.3) -> None:
        self.rnd = random.Random(seed)
        self.bits = bits
        self.m = 2 ** (bits - 1) - 1
        self.literal_prob = literal_prob
        self.env: dict[str, object] = {}

    def leaf(self) -> str:
        r = self.rnd.random()
        name = f"v{len(self.env)}"
        if r < self.literal_prob:
            # host literal: mostly in range, sometimes out of range / tiny / int
            kind = self.rnd.random()
            if kind < 0.7:
                v = self.rnd.uniform(-1, 1)
            elif kind < 0.8:
                v = self.rnd.choice([1, -1, 2, 1.0, -1.5, 0, 0.0])
            elif kind < 0.9:
                v = self.rnd.uniform(-1, 1) * 2 ** -self.bits
            else:
                v = Fraction(self.rnd.randint(-self.m, self.m), 2 ** (self.bits - 1))
            self.env[name] = v
            return name
        k = self.rnd.randint(-self.m, self.m)
        if self.rnd.random() < 0.1:
            k = self.rnd.choice([0, 1, -1, self.m, -self.m])
        self.env[name] = sf.from_code(k, dtype_of(self.bits))
        return name

    def expr(self, depth: int) -> str:
        if depth == 0 or self.rnd.random() < 0.15:
            return self.leaf()
        r = self.rnd.random()
        if r < 0.08:
            return f"-{self.expr(depth - 1)}"
        if r < 0.14:
            return f"abs({self.expr(depth - 1)})"
        if r < 0.2:
            return f"sf.fma({self.expr(depth - 1)}, {self.expr(depth - 1)}, {self.expr(depth - 1)})"
        op = self.rnd.choice("+-*/")
        left, right = self.expr(depth - 1), self.expr(depth - 1)
        # parenthesise at random so precedence and explicit grouping both get exercised
        if self.rnd.random() < 0.5:
            left = f"({left})"
        if self.rnd.random() < 0.5:
            right = f"({right})"
        return f"{left} {op} {right}"


class Lit:
    """A host literal that has not been quantized yet."""

    def __init__(self, v) -> None:
        self.v = v


class OracleWalk:
    """Evaluate an ast with the Fraction oracle, stopping at the first event
    the policy would raise on.  Mirrors Python's evaluation order: both
    operands are evaluated first, then host literals are quantized (left to
    right) by the op that consumes them, then the op runs."""

    class Stop(Exception):
        def __init__(self, event: str, op_index: int) -> None:
            self.event, self.op_index = event, op_index

    def __init__(self, fmt: Fmt, env: dict, raise_on: set[str]) -> None:
        self.fmt, self.env, self.raise_on = fmt, env, raise_on
        self.ops = 0
        self.events: list[tuple[int, str]] = []

    def _event(self, ev):
        self.ops += 1
        if ev is not None:
            self.events.append((self.ops, ev))
            if ev in self.raise_on:
                raise self.Stop(ev, self.ops)

    def fix(self, x):
        """Quantize a pending literal; SF values pass through."""
        if isinstance(x, Lit):
            v = x.v
            r, ev = self.fmt.cast(v if isinstance(v, float) else Fraction(v))
            self._event(ev)
            return r
        return x

    def eval(self, node):
        if isinstance(node, ast.Expression):
            return self.eval(node.body)
        if isinstance(node, ast.Name):
            v = self.env[node.id]
            return v.exact if isinstance(v, sf.SFScalar) else Lit(v)
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand)
            assert isinstance(node.op, ast.USub)
            return Lit(-v.v) if isinstance(v, Lit) else -v
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "abs":
                v = self.eval(node.args[0])
                return Lit(abs(v.v)) if isinstance(v, Lit) else abs(v)
            assert node.func.attr == "fma"
            a, b, c = (self.eval(x) for x in node.args)
            a, b, c = self.fix(a), self.fix(b), self.fix(c)
            r, ev = self.fmt.fma(a, b, c)
            self._event(ev)
            return r
        if isinstance(node, ast.BinOp):
            a = self.eval(node.left)
            b = self.eval(node.right)
            a, b = self.fix(a), self.fix(b)
            r, ev = getattr(self.fmt, BIN[type(node.op)])(a, b)
            self._event(ev)
            return r
        raise AssertionError(node)


def _oracle_order_matches(bits: int, seed: int, under_raise: bool):
    gen = Gen(bits, seed)
    src = gen.expr(depth=gen.rnd.randint(1, 5))
    tree = ast.parse(src, mode="eval")
    env = dict(gen.env, sf=sf)
    sf_names = {k for k, v in gen.env.items() if isinstance(v, sf.SFScalar)}

    # skip trees where some sub-expression is pure host math (not SF semantics)
    def has_sf(n):
        return any(isinstance(x, ast.Name) and x.id in sf_names for x in ast.walk(n))
    for n in ast.walk(tree):
        if isinstance(n, ast.BinOp) and not (has_sf(n.left) or has_sf(n.right)):
            return None
        if isinstance(n, ast.UnaryOp) and not has_sf(n.operand):
            return None
        if isinstance(n, ast.Call) and not any(has_sf(a) for a in n.args):
            return None
    if not has_sf(tree):
        return None

    raise_on = {"overflow", "divide", "invalid"} | ({"underflow"} if under_raise else set())
    walk = OracleWalk(Fmt(bits), gen.env, raise_on)
    try:
        expect = walk.eval(tree)
        expect_err = None
    except OracleWalk.Stop as s:
        expect, expect_err = None, EVENT_ERROR[s.event]

    policy = sf.errstate(under="raise" if under_raise else "ignore")
    with policy:
        try:
            got = eval(src, {"__builtins__": {"abs": abs}}, env)
            got_err = None
        except sf.SFError as e:
            got, got_err = None, type(e)
    return src, expect, expect_err, got, got_err


@pytest.mark.parametrize("bits", [4, 6, 8, 10, 12, 16])
@pytest.mark.parametrize("under_raise", [False, True], ids=["under=ignore", "under=raise"])
def test_random_bodmas_trees(bits, under_raise):
    n_trees, checked = 1500, 0
    for i in range(n_trees):
        res = _oracle_order_matches(bits, SEED + i * 7 + bits, under_raise)
        if res is None:
            continue
        src, expect, expect_err, got, got_err = res
        checked += 1
        assert got_err is expect_err, f"sf{bits}: {src!r} raised {got_err} but oracle says {expect_err}"
        if expect_err is None:
            assert got.exact == expect, f"sf{bits}: {src!r} = {got} but oracle says {expect}"
            assert type(got) is dtype_of(bits)
    assert checked > n_trees // 2


def test_first_error_is_at_the_right_op():
    """x / (y - y) raises at the division; (x + 1) / 0 raises at the literal."""
    x, y = sf.sf16(0.5), sf.sf16(0.25)
    with pytest.raises(sf.SFZeroDivisionError):
        x / (y - y)
    with pytest.raises(sf.SFOverflowError):          # the literal 1 fails first
        (x + 1) / 0
    with sf.errstate(over="ignore"):
        with pytest.raises(sf.SFZeroDivisionError):  # then the division
            (x + 1) / 0
    with pytest.raises(sf.SFOverflowError):          # x / y = 2.0 overflows before + 0
        x / y + 0
    with sf.errstate(over="ignore"):
        assert (x / y + 0).code == 32767


def test_mixed_width_trees():
    a, b, c = sf.sf16(0.5), sf.sf8(0.25), sf.sf4(0.125)
    r = a * b + c - 0.25
    assert type(r) is sf.sf16 and r.code == 0x1000 + 0x1000 - 0x2000
    assert type(b + c) is sf.sf8 and (b + c).code == 0x30
    assert type((b + c) * a) is sf.sf16
    assert type(sf.fma(c, b, a)) is sf.sf16
