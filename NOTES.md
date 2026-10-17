# Stage 1.0 implementation notes

Decisions the spec left open or did not state, and how this code resolves
them. Each one is a single place to change if the team decides differently.

## Representation

* Host codes are **signed Python ints** (`k` in `[-M, M]`), not sign-magnitude
  bit patterns. Arithmetic is simpler and overflow-free on signed codes, and
  the two forms are the same information. `x.bits`, `sf.from_bits` and
  `SFFormat.to_bits/from_bits` convert at the device boundary. A -0 bit
  pattern coming back from the device becomes +0 (`_format.py`).

## Semantics not fixed by the spec

| Question | Choice here | Where |
|---|---|---|
| Device value for `a / 0` | `sign(a) * max`, `0` for `0 / 0` (spec section 12, proposed) | `_reference.div` |
| Device value for `sqrt(a < 0)` | `0` (proposed) | `_reference.sqrt` |
| Rounding of div and sqrt | truncate toward zero, matching mul (proposed) | `_reference.div`, `_reference.sqrt` |
| `fma` event when the product underflows **and** the sum overflows | overflow (what the final value shows) | `_reference.fma` |
| `x ** 0` | overflow (1 is not representable); `x ** k`, `k >= 1`, is the left-to-right mul chain; negative / non-integer `k` is unsupported | `_scalar.__pow__` |
| `under="raise"` | raises `SFUnderflowError`; the spec only names the warning | `_errors.py` |
| `over/divide/invalid = "warn"` | `SFOverflowWarning`, `SFZeroDivisionWarning`, `SFInvalidWarning` (spec names none) | `_errors.py` |
| Foreign operator operands (`x + "0.25"`, `None - x`) | `SFTypeError` (a `TypeError` subclass), per the catalogue. The Python protocol (`NotImplemented`) is used only for a non-builtin type that itself defines the matching reflected operator, so third-party numeric types still interoperate | `_scalar._reject` |
| Exponent / code / bits types | anything `operator.index` accepts (Python int, NumPy ints, `__index__`); `bool` rejected | `_scalar._as_index` |
| `x ** n` cost | the mul chain stops at the first 0, so cost is bounded by the steps to reach 0 (at most `2^(N-1) - 1`, for `x = max`), not by `n` | `_reference.pow_int` |
| NaN comparisons | like IEEE: `==` False, `!=` True, ordering False | `_scalar._cmp_value` |
| `hash(x)` | `hash(x.exact)`, so `sf.sf16(0.5) == 0.5` and the hashes agree; values are usable as dict keys next to floats | `_scalar.__hash__` |
| Strings | constructor only (`sf.sf16("0.1")` is the exact decimal 1/10); never an operator literal | `_reference.to_exact`, `_scalar._coerce` |
| `sf.where` | scalar `x if c else y`; both branches are evaluated by Python before the call | `_functions.where` |

## Immutability

`SFScalar.__setattr__` allows `_code` to be set once (construction) and
rejects everything else; `__delattr__` always rejects. Hash and equality can
therefore never change after a value is used as a dict key.

## Error policy mechanics

* `contextvars.ContextVar`, so `with sf.errstate` is per thread and per task.
* An `errstate` object keeps its entry tokens in a context-local stack, so the
  same object (including the one a decorator holds) can be entered from many
  threads or tasks at once. The decorator wraps `async def` functions with an
  `async` wrapper that holds the policy across `await`; generator functions are
  rejected with `TypeError` (the policy would not cover the body).
* Warnings carry a `stacklevel` that walks out of the package to the user's
  frame, so `-W error` and `warnings.catch_warnings` point at the right line.
* The test suite runs with `filterwarnings = error::sf.SFWarning`, so an
  unexpected warning fails a test.

## Still open for the compiler / hardware teams (spec section 12)

* `__sf16_sub` / `__sf16_neg` and a true `__sf16_div` are needed for the device
  path; nothing in Stage 1.0 depends on them.
* Accumulation order and accumulator width per target are needed before
  Stage 1.1 `sum` / `dot` / `@` can be bit-identical on host and GPU.
* Per-tile sticky SAT / DIV0 / INVALID flags decide whether heavy-op events
  can ever `raise` instead of `warn`.

## Review history

An independent review of the first Stage 1.0 drop (2026-10-17) found: mutable
scalars; the decorator sharing one token list across threads and not wrapping
coroutines; plain `TypeError` for foreign operands; NumPy ints rejected by
`**`, `from_code`, `from_bits`; unbounded `x ** n`; sampled rather than
exhaustive single-value tests on sf14..sf16; and the nightly oracle being too
close to the implementation. All are fixed and covered by
`tests/test_review_findings.py`; the nightly tool now adds a Fraction-oracle
boundary sweep through the public operators, and `.github/workflows/ci.yml`
automates the per-commit and nightly gates.

## Test cost

Every commit: exhaustive sf4..sf10 (sf10 is ~1M pairs per op through the
Fraction oracle, ~45 s per op) plus sampled sf11..sf16 and hypothesis.
`pytest -m "not slow"` skips sf9/sf10 for a ~35 s loop while iterating.
Nightly: `tools/exhaustive.py --bits 16` checks all 4.29e9 pairs per binary
op with the vectorised reference against an independent int64 oracle
(2 to 8 minutes per op), ties random samples back to the scalar reference,
and then sweeps every code against every boundary-class code (both operand
orders, about 7M pairs per op) through the public scalar operators and the
Fraction oracle, counting the event transitions it crosses.
