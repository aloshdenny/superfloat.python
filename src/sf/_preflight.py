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
   inputs through the host reference.  If every probed element saturates the
   op is ``suspected`` (a sample is not a proof): warned about by default,
   refused or verified by exact replay under ``sf.config(suspect=...)``.

``Verdict.events`` carries the host-known event counts so the runtime can
build the status without a second pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import _config, _errstate, _linalg, _vectorized as vec
from ._errors import SFOverflowError, SFOverflowWarning, SFPreflightWarning
from ._format import SFFormat


@dataclass
class Verdict:
    kind: str                                  # "safe" | "unknown" | "suspected" | "certain"
    reason: str = ""
    events: dict = field(default_factory=dict)  # exact host-known counts: div0, invalid


def _magnitudes(*codes):
    return [int(np.abs(np.asarray(c)).max()) if np.asarray(c).size else 0 for c in codes]


def _minmags(*codes):
    return [int(np.abs(np.asarray(c)).min()) if np.asarray(c).size else 0 for c in codes]


def _live(operands, kw):
    """Restrict broadcast operands to the live elements of a mask (masked ops)."""
    live = kw.get("live")
    if live is None:
        return operands
    shape = np.broadcast_shapes(*(np.shape(o) for o in operands), np.shape(live))
    live = np.broadcast_to(live, shape)
    return tuple(np.broadcast_to(np.asarray(o), shape)[live] for o in operands)


def check(op: str, fmt: SFFormat, *operands, **kw) -> Verdict:
    M, S = fmt.max_code, fmt.scale
    ev: dict = {}
    operands = _live(operands, kw)
    if any(np.size(o) == 0 for o in operands):
        return Verdict("safe", "no live elements", ev)
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
            return Verdict("safe", "L1 bound: no partial sum can exceed max", ev)
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
    operands = _live(operands, kw)
    if op in ("add", "sub", "fma", "div"):
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


def replay_all_saturate(op: str, fmt: SFFormat, *operands, **kw) -> bool:
    """Exact host replay: does every (live) output saturate?  Costs the whole op."""
    if op not in ("matmul", "sum"):
        operands = _live(operands, kw)
    kw = {k: v for k, v in kw.items() if k != "live"}
    codes, events = host_events(op, fmt, *operands, **kw)
    over = events.get("overflow")
    return over is not None and np.size(over) > 0 and bool(np.all(over))


def host_events(op: str, fmt: SFFormat, *operands, **kw):
    """Run the op on the host reference and return (codes, event masks)."""
    kw = {k: v for k, v in kw.items() if k != "live"}
    if op == "matmul":
        return _linalg.matmul_codes(fmt, operands[0], operands[1])
    if op == "sum":
        return _linalg.sum_codes(fmt, operands[0], kw.get("axis", 0))
    fn = {"abs": vec.abs_, "pow": vec.pow_int}.get(op) or getattr(vec, op)
    return fn(fmt, *operands, **kw)


def decide(v: Verdict, op: str, fmt: SFFormat, *operands, **kw) -> Verdict:
    """Second half of the gate.

    A ``certain`` verdict is a proof over every output and is refused under
    ``sf.config(preflight=...)``.  A probe hit is only a *suspicion* (the
    probe is a sample): under ``sf.config(suspect=...)`` it warns (default),
    raises, is ignored, or is verified by an exact host replay that can turn
    it into ``certain``.  A sample never by itself justifies "every element".
    """
    import warnings
    if v.kind == "unknown" and _config.get("probe"):
        if probe(op, fmt, *operands, **kw):
            v = Verdict("suspected", "golden-vector probe: every probed output saturates", v.events)
    if v.kind == "suspected":
        policy = _config.get("suspect")
        if policy == "verify":
            if replay_all_saturate(op, fmt, *operands, **kw):
                v = Verdict("certain", "host replay: every output saturates", v.events)
            else:
                v = Verdict("unknown", "", v.events)
        elif policy == "raise":
            raise SFOverflowError(f"pre-flight: {op} on {fmt.name} is suspected to saturate on "
                                  f"every element ({v.reason}); not launched "
                                  f"(sf.config(suspect=...) controls this)")
        elif policy == "warn":
            warnings.warn(f"pre-flight: {op} on {fmt.name} is suspected to saturate on every "
                          f"element ({v.reason}); launching anyway",
                          SFPreflightWarning, stacklevel=_errstate._user_stacklevel())
    if v.kind == "certain":
        policy = _config.get("preflight")
        msg = (f"pre-flight: {op} on {fmt.name} would saturate on every element "
               f"({v.reason}); not launched")
        if policy == "raise":
            raise SFOverflowError(msg)
        if policy == "warn":
            warnings.warn(msg, SFOverflowWarning, stacklevel=_errstate._user_stacklevel())
    return v
