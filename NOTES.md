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

## Stage 1.1 decisions

| Question | Choice here | Where |
|---|---|---|
| Accumulation order and width | sequential from index 0, saturate per step, accumulator at data width (v1); attributes of the target, read by the reference | `_linalg.py`, `_device.Device` |
| `sum(axis=None)` order | C order over the flattened array, sequential | `_array.sum_` |
| `addmm(a, b, d)` | the `@` chain first, then one saturating add of `d` (not `d` as the initial accumulator; the two differ under saturation). Confirm against `__sf16_mat_fma` | `_array.addmm` |
| Event order on arrays | catalogue order: invalid, divide, overflow, underflow; the first in that order with a `raise` policy is raised. `dot` (single output) reports the first event *in time*, like the scalar chain | `_array._handle_events`, `_array.dot` |
| `a * b + c` vs `sf.fma` on arrays | same bits always; the fused form may report an overflow before an underflow elsewhere in the array, the two-step form the reverse | `_vectorized.fma` |
| `sf.ones` | 1 is not representable, so it follows the literal rule: raises by default, saturates under `over="ignore"`; the message points at `sf.full(shape, sf.info(dt).max, dt)` | `_functions.ones` |
| `sf.div(a, b, where=mask)` | elements where the mask is False are 0 and raise nothing | `_functions.div` |
| Strings in `sf.array([...])` | accepted (constructor rule), rejected as operator literals | `_array._to_codes` |
| Heavy-op underflow | not reported on v1 (no flags); only saturation, div0 and invalid (from inputs) | `_array._run_heavy` |
| `status.saturated` | exact when the bound proved safety, when the target returns flags, or under `sf.config(status="replay")`; otherwise an upper bound (scan of ±max) for elementwise ops and `None` (unknown) for reductions, with `outputs_at_max` always reported | `_array._run_heavy`, `_status.py` |
| Probe hit | `suspected`, never `certain`; `sf.config(suspect="warn"|"raise"|"verify"|"ignore")` decides. Only a host proof (range bound, or exact replay under `verify`) can refuse a launch as `certain` | `_preflight.decide` |
| `Device.run` contract | returns `(codes, flags)`; `flags` is `None` for v1 and a dict of event masks for a target with sticky flags; trusted only when `sticky_flags` is True | `_device.Device` |
| Masked `sf.div` on device arrays | heavy: masked-out elements become `0 / 1` (exact 0, no event) and the gate sees only the live elements | `_functions.div`, `_preflight._live` |
| Selection ops on device arrays | `maximum`, `minimum`, `clip`, `where`, `cast`, indexing run on the host and return device-labelled arrays with no status (data movement, not kernels) | `_functions.py` |
| Mixed placement (`host @ device`) | heavy; the host operand is sent along. Two different devices is an error | `_array._device_of` |
| Probe policy | the matmul probe samples random rows and columns plus the largest-magnitude ones; elementwise probes sample elements plus the extremes. `sf.config(probe=False)` disables it | `_preflight.probe` |
| 0-d results | host ops that produce a 0-d result return an SF scalar; heavy ops always return an array (it carries the status) | `_array._result` |

## Stage 2 decisions

| Question | Choice here | Where |
|---|---|---|
| Default scale policy (spec section 12 open question) | weights per tensor by default, per channel on request (`scale=`); activations dynamic per call; float scales on the host | `torch/_core.py` |
| Headroom | `"bound"` (default): raise the activation scale until the L1 bound proves no partial sum or bias add can saturate; `"none"`: raw scale, saturation counted. Measured: without headroom a trained MLP drops from 0.95 to 0.18 accuracy | `torch/_core._activation` |
| Matmul bound | tightened from the spec's `K max|A| max|B|` to `min(max_i ||a_i||_1 max|b|, max|a| max_j ||b_j||_1)`, never looser, exact integer arithmetic; used by both pre-flight and the headroom search, so what the host proves safe the device pre-flight proves safe too | `_linalg.matmul_bound` |
| Bias | added after the chain as one saturating add (`addmm` semantics of 1.1), quantized in the accumulator scale `a s_w`; a bias that does not fit is counted as `bias_saturated` | `torch/_core.linear` |
| Gradients | straight-through: `quantize` passes the gradient where the input was in range, 0 where it saturated; `linear` / `matmul` use the float gradient with the dequantized weight | `torch/_ops.py` |
| Output dtype | float32 for float16 / bfloat16 / float32 inputs, float64 for float64 (bfloat16 and float16 cannot hold 32767/32768) | `torch/_ops.out_dtype` |
| Conv2d | unfold on the host (data movement), then the SF linear chain; groups != 1 and string / non-zero padding are skipped and reported | `torch/_modules.SFConv2d` |
| Modules not converted | Conv1d/3d, transposed convs, MultiheadAttention, RNNs, Bilinear: skipped with a reason; custom attention uses `sft.matmul` | `torch/_modules.convert` |
| `last_status` | process-wide registry keyed by layer name; `last_status(model)` filters by that model's layers | `torch/_status.py` |
| Compile backend | retargets `sf::linear` / `sf::matmul` nodes to a `DeviceRunner` on `sf.config(default_device)`; unsupported widths stay host-emulated; inference-only (raises with grad enabled) | `torch/_backend.py` |
| `autocast` as a decorator | sync functions only; torch function modes are thread-local, so a coroutine or generator would leak the mode across awaits | `torch/_autocast.py` |
| NaN / ±inf in SF layers | routed through the SF policy before any scale is chosen, like a Stage 1 cast: NaN is `invalid` (device value 0), ±inf is `overflow`; under a non-raising policy ±inf stands in as the largest finite magnitude of the tensor with its sign, so it saturates in the scale the finite values choose | `torch/_core.sanitize` |
| Conversion and parameters | `SFLinear.from_float` / `SFConv2d` wrap the source layer's own `Parameter` objects (identity, `requires_grad` and training state preserved), so `convert(..., inplace=True)` keeps a pre-existing optimizer connected; a non-inplace `convert` deep-copies first | `torch/_modules.py` |
| Shared / aliased modules | converted once per distinct module; every alias (including inside nested containers) is replaced by the same SF object; reported under `shared` | `torch/_modules.convert` |
| Root `Conv2d` | converted like a root `Linear`; an unconvertible one is reported as skipped | `torch/_modules.convert` |
| Huge float64 values | the bias scale is derived with overflow guarded (`a_bias` from finite ratios, a bias no finite scale can encode is counted as `bias_saturated`); a result outside float64 is `inf`, as in float | `torch/_core.linear` |
| Request-scoped status | `sft.capture_status()` collects statuses for the current thread or task (context-local), alongside the process-wide `last_status()`; use it for concurrent serving where models share layer names | `torch/_status.capture_status` |
| Research `sf_quantize` | the spec says to promote it from the superfloat research code; that code is not in this repository, so `quantize` is written from the Stage 1 cast and gated on producing the same codes. Replace or cross-check when the research code is available | `torch/_core.fake_quantize` |

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
  can ever `raise` instead of `warn`. The code path exists: a `Device` with
  `sticky_flags = True` gets exact status and honours `over="raise"`.
* Confirm `__sf16_mat_fma` means `(A @ B) + D` and not `D` as the initial
  accumulator (see Stage 1.1 decisions).

## Design rule for the device path: decide on the host, launch only the undecidable

Agreed 2026-10-17. The chip is for work whose answer the host does not know.
Nothing that the host can already decide is sent to the device, and the
host never waits for the device to report an event it could have predicted.
Stage 1.0 already behaves this way for light ops; Stage 1.1 must apply it
to every heavy op *before launch*, in this order:

1. **Static checks**: shape, dtype, target support, launch limits (spec 3.2).
2. **Known answers**: literals that do not fit, `x * 0`, `a / a`, a divisor
   array containing a zero, a divide where every `|a| >= |b|`. Answered or
   rejected on the host; no launch.
3. **Range bound**: carry operand min/max through the reference rules. No
   event possible (e.g. `K * max|A| * max|B| <= max`, spec 3.4): launch and
   skip the post-op scan. Event certain for every input: reject before launch,
   naming the op and the reason.
4. **Golden-vector probe**: for the undecidable band, run the golden boundary
   vectors plus a slice of the real data through the host reference. A hit is
   a suspicion: warn by default, refuse or verify by exact replay under
   `sf.config(suspect=...)`. A sample never by itself justifies "every element".

Strictness is a policy knob alongside `errstate`, e.g.
`sf.config(preflight="raise" | "ask" | "warn" | "off")`, subtle by default.
Only step 4 costs anything, and it is host-side and small. Step 3 is also
what `sf.explain` (Stage 1.2) reports per op.

## Review history

A third independent review (Stage 2 drop, 2026-10-18) found: NaN silently
treated as 0 in SF layers and a raw `ValueError` for a NaN bias; conversion
re-creating parameters (losing `requires_grad`, training state, and the link
to a pre-existing optimizer); shared layers converted only under their first
name; a root `Conv2d` left unchanged; and an intermediate overflow in the
bias scale search for a float64 bias near the maximum. All fixed and covered
by `tests/test_review_findings_2.py`; `sft.capture_status()` addresses the
process-wide status registry the review flagged for concurrent serving.

A second independent review (Stage 1.1 drop, 2026-10-18) found: a probe
sample promoted to a "certain" refusal; the ±max scan presented as a
saturation count for reductions (false in both directions); array casts
reversing the sign of huge inputs and `from_codes` accepting out-of-range
unsigned codes; `sf.config` not re-entrant and not async-aware; masked `div`
bypassing the device path; and `offload_threshold` broken for `sum`, `dot`
and `@`. All fixed and covered by `tests/test_review_findings_1_1.py`; the
README now describes 1.1 as a host/mock implementation with the hardware gate
still open.

### First review

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
