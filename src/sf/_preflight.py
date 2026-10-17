"""Pre-flight gate for heavy ops (NOTES.md "decide on the host").

Before a launch the host decides what it can:

1. static checks (dtype support, launch limits): raised by the device
   ``check``; shape errors are raised by the array code before it gets here;
2. known answers: divide-by-zero and domain errors are read straight off the
   inputs, so ``div0`` / ``invalid`` in the status are exact;
3. range bound: carry operand magnitudes through the op's rule.  ``safe``
   means no element can saturate (the post-op scan is skipped and
   ``status.exact`` is True); ``certain`` means every element must saturate
   (the launch is refused under ``sf.config(preflight=...)``);
4. golden-vector probe: for the undecidable band, run a sample of the real
   inputs through the host reference.  If every probed element saturates,
   treat it as ``certain``; otherwise launch and scan.

``Verdict.events`` carries the host-known event counts so the runtime can
build the status without a second pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _config, _errstate, _linalg, _vectorized as vec
from ._errors import SFOverflowError, SFOverflowWarning
from ._format import SFFormat


@dataclass
class Verdict:
    kind: str                                  # "safe" | "unknown" | "certain"
    reason: str = ""
    events: dict = field(default_factory=dict)  # exact host-known counts: div0, invalid


def _magnitudes(*codes):
    return [int(np.abs(np.asarray(c)).max()) if np.asarray(c).size else 0 for c in codes]


def _minmags(*codes):
    return [int(np.abs(np.asarray(c)).min()) if np.asarray(c).size else 0 for c in codes]


def check(op: str, fmt: SFFormat, *operands, **kw) -> Verdict:
    M, S = fmt.max_code, fmt.scale
    ev: dict = {}
    if op == "div":
        ev["div0"] = int(np.count_nonzero(np.broadcast_to(operands[1] == 0,
                                          np.broadcast_shapes(*(np.shape(o) for o in operands)))))
    if op == "sqrt":
        ev["invalid"] = int(np.count_nonzero(operands[0] < 0))

    if op in ("add", "sub"):
        a, b = _magnitudes(*operands)
        if a + b <= M:
            return Verdict("safe", "max|a| + max|b| <= max", ev)
        lo_a, lo_b = _minmags(*operands)
        # certain only when even the smallest magnitudes overflow in the same direction
        pa, pb = np.asarray(operands[0]), np.asarray(operands[1])
        sb = -pb if op == "sub" else pb
        if lo_a + lo_b > M and (np.all(pa > 0) and np.all(sb > 0) or np.all(pa < 0) and np.all(sb < 0)):
            return Verdict("certain", "every element of a and b overflows", ev)
        return Verdict("unknown", "", ev)
    if op in ("mul", "neg", "abs"):
        return Verdict("safe", "cannot overflow", ev)
    if op == "fma":
        a, b, c = _magnitudes(*operands)
        if (a * b) // S + c <= M:
            return Verdict("safe", "trunc(max|a| max|b|) + max|c| <= max", ev)
        return Verdict("unknown", "", ev)
    if op == "div":
        a, b = np.asarray(operands[0]), np.asarray(operands[1])
        shape = np.broadcast_shapes(a.shape, b.shape)
        a, b = np.broadcast_to(a, shape), np.broadcast_to(b, shape)
        nz = b != 0
        if nz.any() and np.all(np.abs(a[nz]) >= np.abs(b[nz])) and np.all(a[nz] != 0):
            return Verdict("certain", "|a| >= |b| for every element", ev)
        if nz.all() and np.all(np.abs(a) < np.abs(b)):
            return Verdict("safe", "|a| < |b| everywhere", ev)
        return Verdict("unknown", "", ev)
    if op == "sqrt":
        return Verdict("safe", "sqrt cannot overflow", ev)
    if op == "pow":
        return Verdict("safe", "powers cannot overflow", ev)
    if op == "matmul":
        if _linalg.matmul_bound_safe(fmt, operands[0], operands[1]):
            return Verdict("safe", "K * max|A| * max|B| <= max", ev)
        return Verdict("unknown", "", ev)
    if op == "sum":
        a = np.asarray(operands[0])
        K = a.shape[kw.get("axis", 0)] if a.ndim else 1
        if K * _magnitudes(a)[0] <= M:
            return Verdict("safe", "K * max|a| <= max", ev)
        return Verdict("unknown", "", ev)
    return Verdict("unknown", "", ev)


def probe(op: str, fmt: SFFormat, *operands, **kw) -> bool | None:
    """Golden-vector probe for the undecidable band.

    Returns True if every probed element saturates (treat as certain), False
    if the probe is clean or mixed, None if the op has no probe.
    """
    n = _config.get("probe_size")
    rng = np.random.default_rng(0)
    if op in ("add", "sub", "fma"):
        shape = np.broadcast_shapes(*(np.shape(o) for o in operands))
        ops = [np.broadcast_to(np.asarray(o), shape).reshape(-1) for o in operands]
        total = ops[0].size
        if total == 0:
            return False
        idx = np.unique(np.concatenate([
            rng.integers(0, total, min(n, total)),
            np.argsort(np.abs(ops[0]))[-8:], np.argsort(np.abs(ops[1]))[-8:],
            np.argsort(np.abs(ops[0]))[:8],
        ]))
        _, ev = getattr(vec, op)(fmt, *(o[idx] for o in ops))
        return bool(ev["overflow"].all())
    if op == "matmul":
        a, b = np.asarray(operands[0]), np.asarray(operands[1])
        a2 = a.reshape(-1, a.shape[-2], a.shape[-1])[0]
        b2 = b.reshape(-1, b.shape[-2], b.shape[-1])[0]
        if a2.shape[0] == 0 or b2.shape[1] == 0:
            return False
        # a random sample of rows and columns plus the largest-magnitude ones;
        # "certain" only if every probed output saturates
        rows = min(8, a2.shape[0])
        cols = min(8, b2.shape[1])
        ra = np.unique(np.concatenate([rng.integers(0, a2.shape[0], rows),
                                       np.argsort(np.abs(a2).max(axis=1))[-2:]]))
        cb = np.unique(np.concatenate([rng.integers(0, b2.shape[1], cols),
                                       np.argsort(np.abs(b2).max(axis=0))[-2:]]))
        _, ev = _linalg.matmul_codes(fmt, a2[ra], b2[:, cb])
        return bool(ev["overflow"].all())
    return None


def gate(op: str, fmt: SFFormat, *operands, **kw) -> Verdict:
    """Run the gate.  Raises / warns per ``sf.config(preflight=...)`` when the
    op is known to saturate everywhere, so the chip is never asked."""
    return decide(check(op, fmt, *operands, **kw), op, fmt, *operands, **kw)


def decide(v: Verdict, op: str, fmt: SFFormat, *operands, **kw) -> Verdict:
    """Second half of the gate: probe the undecidable band and apply the
    pre-flight policy to a ``certain`` verdict."""
    if v.kind == "unknown" and _config.get("probe"):
        p = probe(op, fmt, *operands, **kw)
        if p:
            v = Verdict("certain", "golden-vector probe: every probed element saturates", v.events)
    if v.kind == "certain":
        policy = _config.get("preflight")
        msg = (f"pre-flight: {op} on {fmt.name} would saturate on every element "
               f"({v.reason}); not launched")
        if policy == "raise":
            raise SFOverflowError(msg)
        if policy == "warn":
            import warnings
            warnings.warn(msg, SFOverflowWarning, stacklevel=_errstate._user_stacklevel())
    return v
