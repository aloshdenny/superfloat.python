# sf: the Python DSL for Superfloat

Python front end for the Superfloat formats (sf4 to sf16) on the Atreides GPU.
This is **Stage 1.0** of the language roadmap: scalar arithmetic, types and
casts, the error model, and the bit-exact host reference that every later
backend is checked against.

```python
import sf

a, b = sf.sf16(0.5), sf.sf16(0.25)
a + b          # sf16(0.75)
a * b          # sf16(0.125)
b / a          # sf16(0.5)
a / b          # SFOverflowError: 2.0 is outside sf16
a / sf.sf16(0) # SFZeroDivisionError, caught on host
sf.sf16(1)     # SFOverflowError: 1 is not representable
a < 1          # True: comparisons are exact

with sf.errstate(over="ignore"):      # silent saturation, same bits as the GPU
    sf.sf16(0.75) + 0.5               # sf16(0.999969482421875)
```

## Install

```
pip install -e ".[test]"
pytest                    # every commit: exhaustive sf4..sf10, sampled sf11..sf16
pytest -m "not slow"      # ~35 s while iterating (skips exhaustive sf9/sf10)
python tools/exhaustive.py --bits 16    # nightly: all 4.3e9 pairs per binary op
```

No runtime dependencies. NumPy is only needed by the tests and the nightly tool.

## What is in the package

| Module | Role |
|---|---|
| `sf/_format.py` | Format constants for each width: scale, max code, LSB, cast thresholds, sign-magnitude bit packing |
| `sf/_reference.py` | **Bit-exact host reference.** Pure functions on integer codes returning `(code, event)`. Never raises for numeric events |
| `sf/_errstate.py` | `sf.seterr` / `sf.geterr` / `sf.errstate`, context-local (threads and asyncio safe). Turns an event into raise / warn / ignore |
| `sf/_errors.py` | The exception and warning hierarchy of spec section 3.2 |
| `sf/_scalar.py` | `sf.sf4` .. `sf.sf16` classes: construction, promotion, literals, operators, comparisons, unsupported ops |
| `sf/_functions.py` | `sf.cast`, `sf.info`, `sf.fma`, `sf.div`, `sf.sqrt`, `sf.where`, `sf.from_code`, `sf.from_bits`, `sf.dtype` |
| `sf/_vectorized.py` | The same ops over NumPy int64 code arrays (nightly tool now, Stage 1.1 arrays later) |

The split that matters: **semantics live in `_reference.py` only.** The scalar
classes are a thin layer that promotes operands, calls the reference, and hands
the event to the error policy. A compiler backend or the device is correct when
it agrees with `_reference.py` on codes and on events.

## Semantics (spec sections 2 and 3)

sfN holds a signed integer code `k` in `[-M, M]`, `M = 2^(N-1) - 1`, value `k / 2^(N-1)`.
No exponent, NaN, infinity or subnormals; -0 is canonicalised to +0.

| Op | Result code | Overflow (saturates to ±max) | Underflow (becomes 0) |
|---|---|---|---|
| cast from host | `round_half_even(v * 2^(N-1))` | `|v| >= 1 - 2^-N`, ±inf, every nonzero int | `0 < |v| <= 2^-N` |
| `a + b`, `a - b` | `k_a ± k_b` | `|k_a ± k_b| > M` | never |
| `-a`, `abs(a)` | `-k_a`, `|k_a|` | never | never |
| `a * b` | `trunc(k_a k_b / 2^(N-1))` | never | `a, b != 0` and `|k_a k_b| < 2^(N-1)` |
| `a / b` | `trunc(k_a 2^(N-1) / k_b)` | `|a| >= |b|` (so `a / a` overflows) | never |
| `sf.fma(a, b, c)` | `k_c + trunc(k_a k_b / 2^(N-1))` | `|result| > M` | when the product underflows |
| `sf.sqrt(a)` | `isqrt(k_a 2^(N-1))` | never | never; `a < 0` is a domain error |

* Casts round to nearest, ties to even. Device arithmetic truncates toward zero.
* `a * b + c` gives the same bits as `sf.fma(a, b, c)` (tested on every width).
* Literals take the SF operand's type and are quantized first: `x + 1` overflows at the literal. `0` is the only representable integer.
* Comparisons compare exact values and never quantize: `x < 1` is always `True`, `sf.sf16(0.1) != 0.1`.
* `int(x)` is 0 for every value. Use `x.code` (signed), `x.bits` (sign-magnitude), `x.sign`, `x.mantissa`.
* `sfN op sfM` gives the wider type; narrowing needs `sf.cast`. Mixing with a narrower target raises `SFTypeError`.
* `//`, `%`, bitwise ops, `round()`, non-integer or negative `**` raise `SFUnsupportedError`.
* `x ** k` for integer `k >= 1` is the left-to-right chain of multiplies.

## Error policy

```python
sf.seterr(over="warn", under="ignore", divide="raise", invalid="raise")   # returns the old policy
with sf.errstate(all="ignore"): ...
@sf.errstate(over="warn")
def f(): ...
```

Defaults: `over=raise, under=warn, divide=raise, invalid=raise`.
The returned value is always the hardware value; the policy only decides whether
the host also raises or warns. Errors are raised at, and warnings point to, the
user's own line.

| Event | raise | warn |
|---|---|---|
| overflow | `SFOverflowError` (OverflowError) | `SFOverflowWarning` |
| underflow | `SFUnderflowError` | `SFUnderflowWarning` |
| divide | `SFZeroDivisionError` (ZeroDivisionError) | `SFZeroDivisionWarning` |
| invalid | `SFInvalidError` (ValueError) | `SFInvalidWarning` |

Static errors are not subject to the policy: `SFTypeError`, `SFUnsupportedError`,
`SFShapeError`, `SFLaunchError`, `SFCompileError`, `SFDeviceError`.
Everything inherits from `sf.SFError` or `sf.SFWarning`.

## Tests (spec section 10)

`tests/oracle.py` is an independent oracle written on `fractions.Fraction`
values, straight from the formulas in section 3.1. It never sees integer
codes. The reference works on codes. Tests assert that the two agree on the
code and on the event, for every pair (sf4 to sf10) or a seeded sample plus
boundary corners (sf11 to sf16), plus hypothesis property tests.

| File | Milestone |
|---|---|
| `test_1_0_1_types_casts.py` | 1.0.1 types and casts; golden file of all codes (`tests/golden/`) |
| `test_1_0_2_add_sub.py` | 1.0.2 add, sub, neg, abs; chains; saturation is not undone |
| `test_1_0_3_mul_fma.py` | 1.0.3 mul, fma; precedence; distributivity recorded, not asserted |
| `test_1_0_4_div.py` | 1.0.4 div (and sqrt); `|(a/b) b - a| <= 1 LSB` |
| `test_1_0_5_comparisons.py` | 1.0.5 comparisons, bool, min/max, where, guarded division |
| `test_bodmas.py` | Random expression trees over the whole 1.0 grammar; first raised error must match |
| `test_properties.py` | Commutativity, identities, monotonic saturation; associativity recorded only |
| `test_errstate.py` | Policy, catalogue types, user-line attribution, context locality |
| `test_spec_examples.py` | Every code example in the spec, as written |

Results are compared as integer codes, never with float tolerances.
`SF_TEST_SEED` and `SF_TEST_RANDOM_PAIRS` control the sampled widths.
Regenerate the golden files with `python tools/make_golden.py` only when the
format definition itself changes.

## Beyond the spec (decisions taken here; see NOTES.md)

* Divide by zero and sqrt of a negative return the spec's *proposed* device
  values (sign(a) × max, 0 for 0/0, 0 for sqrt of a negative) when the policy is
  not `raise`. Change `_reference.div` / `_reference.sqrt` if hardware decides otherwise.
* `sf.sqrt` ships now (it is the invalid-event example in section 3.2); the
  rest of 1.1.4 waits for arrays.
* Strings are accepted by the constructor only (`sf.sf16("0.1")` quantizes the
  exact decimal 1/10), never as operator literals.

## Roadmap

Stage 1.1 (arrays, dot, matmul, built-in GPU kernels) builds on
`_vectorized.py`; Stage 1.2 diagnostics; Stage 2 PyTorch; Stage 3 kernels.
