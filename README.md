# sf: the Python DSL for Superfloat

Python front end for the Superfloat formats (sf4 to sf16) on the Atreides GPU.
This covers **Stage 1.0** (scalar arithmetic, types and casts, the error model,
the bit-exact host reference), **Stage 1.1** (NumPy-style arrays, reductions,
dot, matmul, elementary functions, and the device path with a mock Atreides
target behind a device interface) and **Stage 2** (PyTorch: fake quantization,
model conversion with a scale policy, and a `torch.compile` backend that
offloads SF ops to the target).

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

```python
x = sf.array([0.10, -0.25, 0.50], dtype=sf.sf16)                 # host array: light
h = sf.asarray(np_inputs, dtype=sf.sf16, device="atreides")
w = sf.asarray(np_weights, dtype=sf.sf16, device="atreides")
y = h @ w                                                         # heavy: built-in kernel
y.status            # SFStatus(saturated<=12, div0=0, invalid=0, exact=False)
y.numpy()           # read back to host
sf.div(a, b, where=b != 0)                                        # never raises
```

```python
import sf.torch as sft
model = sft.convert(model, dtype=sf.sf16, scale="per_tensor")   # host emulation, chip-exact
fast = torch.compile(model, backend="superfloat")                # offload to Atreides
with torch.no_grad():
    out = fast(x)
sft.last_status(model)                                           # post-op events per layer
```

## Install

```
pip install -e ".[test]"
pip install -e ".[torch]"   # adds PyTorch for sf.torch (Stage 2)
pytest                    # every commit: exhaustive sf4..sf10, sampled sf11..sf16
pytest -m "not slow"      # ~40 s while iterating (skips exhaustive sf9/sf10)
python tools/exhaustive.py --bits 16    # nightly: all 4.3e9 pairs per binary op + Fraction boundary sweep
```

`.github/workflows/ci.yml` runs the per-commit gate on every push (Python 3.10,
3.12, 3.13) and the nightly gate on a schedule.

```
```

NumPy is the only runtime dependency (arrays are int64 code arrays).

## What is in the package

| Module | Role |
|---|---|
| `sf/_format.py` | Format constants for each width: scale, max code, LSB, cast thresholds, sign-magnitude bit packing |
| `sf/_reference.py` | **Bit-exact host reference.** Pure functions on integer codes returning `(code, event)`. Never raises for numeric events |
| `sf/_errstate.py` | `sf.seterr` / `sf.geterr` / `sf.errstate`, context-local (threads and asyncio safe). Turns an event into raise / warn / ignore |
| `sf/_errors.py` | The exception and warning hierarchy of spec section 3.2 |
| `sf/_scalar.py` | `sf.sf4` .. `sf.sf16` classes: construction, promotion, literals, operators, comparisons, unsupported ops |
| `sf/_functions.py` | `sf.cast`, `sf.info`, `sf.fma`, `sf.div`, `sf.sqrt`, `sf.where`, `sf.from_code`, `sf.from_bits`, `sf.dtype` |
| `sf/_vectorized.py` | The same ops over NumPy int64 code arrays: host arrays, the mock target and the nightly tool all run this |
| `sf/_array.py` | `SFArray`: construction, indexing, broadcasting, elementwise ops, reductions, `dot`, `matmul`, `addmm`, and the light/heavy dispatch |
| `sf/_linalg.py` | Reference accumulation: `sum`, `dot`, `@` as sequential saturating fma chains; the spec 3.4 matmul bound |
| `sf/_device.py` | `Device` interface, `MockAtreides` (sf16 only, no flags, like v1), `register_device` for the simulator and hardware |
| `sf/_preflight.py` | The gate every heavy op passes before launch: known answers, range bound, golden-vector probe |
| `sf/_status.py` | `SFStatus(saturated, div0, invalid, exact)` |
| `sf/_config.py` | `sf.config(preflight=, suspect=, probe=, probe_size=, status=, offload_threshold=)` |
| `sf/torch/_core.py` | Torch-free chip-exact SF linear / matmul with the scale policy; one function for eager, autocast and the compile backend, parameterised by a runner (host reference or device) |
| `sf/torch/_ops.py` | `sf::quantize`, `sf::linear`, `sf::matmul` custom ops with straight-through gradients; one opaque node each in a compiled graph |
| `sf/torch/_modules.py` | `SFLinear`, `SFConv2d` (unfold + SF linear), `sft.convert`, the conversion report |
| `sf/torch/_autocast.py` | `sft.autocast`: `F.linear`, `F.conv2d`, `matmul`/`@`/`mm`/`bmm` routed to the SF ops inside a block |
| `sf/torch/_backend.py` | `torch.compile(..., backend="superfloat")`: SF nodes retargeted to the device runner, CPU fallback for the rest, per-graph coverage report |
| `sf/torch/_status.py` | `sft.last_status()`: per-layer saturation, underflow, headroom and scale |

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

## Arrays and the device path (Stage 1.1)

An SF array is an int64 NumPy array of codes plus a dtype and a placement.
Every scalar rule applies element by element with NumPy broadcasting. Events
on host arrays are counted (`overflow in 12 of 4096 elements`) and go through
the same policy, in catalogue order: invalid, divide, overflow, underflow.

* `sum`, `dot` and `@` are chains of fma in the target's accumulation order
  (sequential from index 0, saturating at each step, accumulator at data
  width). `dot` equals the scalar fma chain bit for bit, and `@` equals
  per-output `dot`. `addmm(a, b, d)` is the `@` chain, then one saturating add.
* `device="atreides"` makes an array heavy. Any arithmetic op with a heavy
  operand (including masked `sf.div`) is a heavy op: it is checked on the host,
  run on the target, and reports `result.status`. Selection and data-movement
  ops (`maximum`, `minimum`, `clip`, `where`, `cast`, indexing) run on the host
  and carry no status. Only sf16 is accepted on Atreides v1.
* **Pre-flight (decide on the host, launch only the undecidable).** Before a
  launch: shape, dtype and limit checks; `div0` / `invalid` read off the inputs
  (exact, raised before launch under the policy); the range bound (safe means
  no scan and `status.exact=True`). When the host can *prove* the op saturates
  on every element, `sf.config(preflight="raise")` (default) refuses it before
  the chip sees it. For the undecidable band a golden-vector probe samples the
  real inputs; a probe hit is a *suspicion*, never a proof: by default it warns
  (`SFPreflightWarning`) and launches; `sf.config(suspect="raise")` refuses,
  `"verify"` replays the whole op on the host and refuses only on a real proof.
* **Status honesty.** Atreides v1 has no saturation flag. For an elementwise
  op, `status.saturated` is an upper bound from scanning the output for ±max
  (`exact=False`); for a reduction (`sum`, `dot`, `@`) a partial sum can
  saturate and come back into range, so `saturated` is `None` (unknown) and
  `status.outputs_at_max` holds the raw scan. `sf.config(status="replay")`
  recomputes the events on the host for an exact count at the cost of running
  the op twice. A target whose `Device.run` returns flags (`sticky_flags=True`,
  see `MockAtreidesFlags`) gets exact counts, and `over="raise"` is honoured.
  `status.ok` is `True`, `False`, or `None` when unknown.
* `sf.config(offload_threshold=n)` moves host ops larger than `n` elements to
  the default device.

## PyTorch (Stage 2)

Integrate, do not clone: PyTorch keeps autograd, modules, optimisers, data
loading and serialisation; SF owns numerics and offload.

* **2.0 `sft.quantize(t, dtype, scale=None|"dynamic"|float)`**: values snapped
  to SF codes, float storage (float32, which holds every sf4..sf16 value
  exactly; bfloat16 and float16 do not), the same codes and the same events as
  `sf.asarray` under every policy, clipped straight-through gradient.
* **2.1 `sft.convert(model, dtype, scale, headroom)`** swaps `nn.Linear` and
  `nn.Conv2d` (groups=1, zero padding) for `SFLinear` / `SFConv2d`; everything
  else stays float. `sft.autocast(dtype)` does the same for a block of code
  through a `TorchFunctionMode`. Both run the **scale policy**, because real
  activations exceed ±1: weights get a per-tensor or per-channel scale so the
  largest weight maps to the largest code; activations get a dynamic per-call
  scale; the float scales stay on the host. With `headroom="bound"` (default)
  the activation scale is raised until the L1 bound (`_linalg.matmul_bound`)
  proves that no partial sum of the accumulation chain and no bias add can
  saturate, so `saturated` is an exact 0 on every layer, on the host and on the
  device, and the bits given up are reported as `headroom_bits`.
  `headroom="none"` uses the raw scale and lets the chain saturate (counted).
  The chain itself is the Stage 1.1 reference: per-product truncation,
  per-step saturation, sequential order, then one saturating bias add.
  Training works through the straight-through gradient (quantization-aware
  fine-tuning recovers most of sf8's loss in the tests).
* **Status.** `sft.last_status(model)` is the most recent call of each of the
  model's layers from a process-wide registry keyed by layer name;
  `with sft.capture_status() as st:` collects the statuses of one request for
  the current thread or task only, for concurrent serving. NaN and ±inf in an
  input, weight or bias go through `sf.errstate` like any Stage 1 cast (NaN is
  `invalid`, inf is `overflow`), never silently through the arithmetic.
  `sft.convert` keeps the source layers' own parameters (identity,
  `requires_grad`, train/eval state), so an optimizer built before an in-place
  conversion keeps working, and shared layers are converted once.
* **2.2 `torch.compile(model, backend="superfloat")`**: every `sf::linear` /
  `sf::matmul` node the target supports (sf16 on Atreides v1) runs through the
  Stage 1.1 heavy-op path (pre-flight gate, launch, `result.status`); other
  widths are host-emulated; everything else stays on the CPU. Compiled output
  equals emulated output bit for bit by construction (same core function,
  different runner). `sft.compile_report()` lists what went where and flags
  float linear algebra that was never converted. The backend is inference-only;
  train with the eager emulation.
* **2.3** (Atreides as a PyTorch `PrivateUse1` device) is deferred, as the spec
  recommends, until 2.2 coverage is high.

Measured on scikit-learn digits (tests): MLP float 0.947, sf16 0.947, sf12
0.950, sf8 0.924; CNN float 0.980, sf16 0.980, sf12 0.982. Without headroom
sf16 drops to 0.18 because the accumulation saturates, which is why `bound` is
the default.

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
| `test_review_findings.py` | Regressions from the independent Stage 1.0 review |
| `test_1_1_1_arrays.py` | 1.1.1 every Stage 1.0 op re-run on shapes `()`, `(n,)`, `(m, n)`; broadcasting, literals, promotion, casts |
| `test_1_1_2_reductions.py` | 1.1.2 `sum` is the sequential chain; `dot` equals the scalar fma chain; max/min/argmax |
| `test_1_1_3_matmul_device.py` | 1.1.3 `@` equals per-output `dot`; shape errors before launch; host and target bit-identical; status; pre-flight |
| `test_1_1_4_functions.py` | 1.1.4 sqrt, pow, clip, maximum, minimum, masked div, where |
| `test_review_findings_1_1.py` | Regressions from the independent Stage 1.1 review |
| `test_2_0_quantize.py` | 2.0 same codes and events as Stage 1 under every policy; STE; dtypes |
| `test_2_1_convert.py` | 2.1 SF linear recomputed through the Stage 1 public API; adversarial no-saturation cases; conv as unfold + linear; convert report; autocast; accuracy gate on digits; QAT recovery |
| `test_2_2_compile.py` | 2.2 compiled == emulated bit for bit on MLP, CNN and attention; coverage report; unsupported widths; inference-only; pre-flight on the compiled path |
| `test_review_findings_2.py` | Regressions from the independent Stage 2 review |

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
  exact decimal 1/10), never as operator literals. A foreign operand such as
  `x + "0.25"` raises `SFTypeError`.
* Values are immutable; `x ** n` stops as soon as the chain reaches 0, so it is
  cheap for any `n`.

## Status of the Stage 1.1 gate

Stage 1.1 is a **host and mock implementation**. `MockAtreides` runs the same
vectorised reference as the host, so "host == mock" proves the routing,
placement, pre-flight and status machinery, not hardware correctness. The
spec's gate ("host and GPU bit-identical on every op") stays open until a
simulator or silicon is registered with `sf.register_device`, tested against
the Fraction oracle in `tests/oracle.py`, and its event reporting (flags or
not, accumulation order, accumulator width, `mat_fma` meaning) is confirmed.
Stage 2 runs on the same mock, so the same caveat applies to the
`superfloat` backend. Stage 1.2 (diagnostics) and Stage 2.3 are deferred;
Stage 3 kernels next.
