# Stage 1.0 implementation notes

Decisions the spec left open or did not state, and how this code resolves
them. Each one is a single place to change if the team decides differently.

## Integer nonlinears, P·V over visible keys, a width per layer (0.15.0, spec 1.9.0)

* Alosh (2026-11-18): port his `sf_intops_ref.py` (exp, softmax, GELU, SiLU,
  reciprocal, rsqrt, norms) bit for bit as sf's reference, with golden
  vectors; fix the causal-masked softmax rows; take P·V's exponent over the
  visible keys only; build width per layer from the cost table, read as
  input.
* `sf._intops` (`sf.intops`): his file verbatim (his `softmax_row` renamed
  `softmax_row_v0`), compared on every input code (`tests/test_intops.py`);
  the original sits unchanged in `tests/data/sf_intops_ref_v0.py`.  Golden
  vectors: `spec/golden/intops/` from `tools/gen_golden_intops.py`.
* sf's `softmax_row` = v0 with three changes, **pending Alosh's sign-off**:
  (1) e_i flushed by the main exp get a second exp at d + 15 ln2 / 2^k and
  are summed in a third MX slot popped 15 lower; (2) q < 2^14 after
  normalising: pop once more one step higher; (3) P returned with the row's
  exponent `(t, ep)`, not shifted to scale 1.  His row: worst error 0.0037 to
  under 0.0003.  Left as defined: `mul` truncation (up to 1 ulp low per
  nonzero P, as v0) and scores above 64 saturating under k <= 6.
* Float rows on the host (`softmax_rows`): **the row max is subtracted before
  quantizing** (d = s - max, k = 4, or 5 when a visible key is more than 16
  below the max; anything 32 below is under 2^-46 and may saturate).  Alosh's
  convention quantized raw scores (k = 4, redo at 5, 6; kept as
  `softmax_rows_raw`): Qwen2.5-0.5B's attention scores reach 1728 (its sink
  token), 6% of rows saturate at 64, keys above 64 collapse, and perplexity
  doubles (17.40 to 36.9 on one window; with the max first, 17.41).  On the
  chip this needs one max over the wide Q·Kᵀ output before narrowing
  (question to Alosh).  A score at or below -10^4 is a mask (models mask
  with `finfo.min` as well as -inf).
  GELU / SiLU rows: one exponent from max |x|, clamped to 0..6 (his tables),
  so |x| above 64 saturates.
* `convert(nonlinear="integer")`: softmax in `sft.attention` (wide mode) and
  in eager attention scopes; tanh-GELU and SiLU modules swapped for
  `IntGELU` / `IntSiLU` (MoE `act_fn` too).  erf GELU has no integer form:
  stays float, reported.  Norms stay float (Alosh: up to 4% off with squares
  in sf16).  Numpy round trip: no gradient, graph break under compile.
* P·V (`out_rule="max_b"`): each output's exponent is the exact bound of the
  quantized product, (sum of the row's quantized P) x max |quantized V| over
  the keys the row uses (P != 0), times 32768/32767 for POP's rounding.  No
  P·V output can saturate, so none is redone: max |V| alone left rows that
  see one key with V just under a power of two at the top code (128 redos in
  `test_pv_is_never_redone`).
* Redos on Qwen2.5-0.5B at sf16 (one window of 256 tokens, about 300k): q_proj
  173k (3% of its outputs), scores 65k, k_proj 40k (5%), every other kind
  under 4k, P·V 0.  q_proj and k_proj carry Qwen's large biases: the first
  POP's exponent (l2 - h) ignores the bias added after it, so an output
  dominated by its bias saturates and is redone.  Taking the bias into the
  first exponent would remove most of these; left as is (the bias path is
  Alosh's), raised with him.  Measured (one window, `diag_bias.py` outside
  the repo): first exponent max(l2 - h, pow2_exp(|b| + 2^(l2 - h))) takes
  redos from 285k to 72k (q_proj 172,852 to 79, k_proj 40,155 to 0; the
  scores' 65k stay), perplexity 17.414 to 17.417 (float 17.399).  V's own per-channel quantization exponent still
  spans every key: a row is not yet fully invariant to masked keys appended
  after it (question to Alosh).  Measured: 128 keys, the last 28 with V 8x
  larger; every earlier row changes when they are appended (932 of 932, up
  to 7e-4), since V's per-channel exponent rises 3 bits for all rows.  K's
  per-channel exponent in Q·Kᵀ does the same.  Full invariance needs K and V
  quantized per block of keys (or re-quantized per prefix), not per channel
  over the whole sequence.
* Configuration chooser (`sf.torch._configs`): the 56 cases of
  `sf_golden_v4.9.2.json` (`heuristic.configuration`, the ones Alosh meant)
  reproduced exactly (`tests/test_configs.py`).  Read from the file: cycles =
  round(MACs x c), c per GEMV / GEMM and plain / packed sf8, sf10, sf12, times
  1.066 for `tbl2`; bits kept = (bits - h(K) for `l2`) - 4.2.  Inferred, holds
  on all 56: LM head and grouped conv at sf16 plain; `tbl2` measured from
  sf12 for `llm` / `any`, every width for `other`; `l2` unmeasured; under 1
  bit kept is lossy; deterministic needs no redo; speed by cycles then bits,
  accuracy by bits then cycles, top 3.  On Qwen2.5-0.5B (decode, LLM, speed):
  sf12 everywhere, LM head sf16: 88% of all-sf16 decode cycles (95% prefill),
  perplexity +0.99% (22.636 against 22.415).  Priced with the same v4.9.2
  model, the measured plans: fit A (energy, sf8 / sf16) 76% decode / 89%
  prefill at +1.78%; fit B (cycles, sf8 to sf16) 69% / 86% at +1.22%; all
  sf8 63% / 84% at +10.4%.
* Width per layer from measured accuracy (`sf.torch._widths`): `cost_table_v0.json` packaged and
  read as given; `layer_sensitivity` (one layer at a time, rest float);
  `choose_widths` (greedy on loss saved per cost, from all-narrowest; widths
  not measured count as no better than the next narrower measured width, 0
  at sf16, and are flagged: the 4x-per-bit rule was wrong on Qwen, where the
  top three layers lose only 1.1-1.3x less at sf10 than at sf8);
  `fit_widths` measures the plan on the whole model and corrects the budget,
  since the sum of single-layer losses is off by up to 1.5x on Qwen2.5-0.5B.
  `convert(widths={name: dtype})` takes the plan.  Qwen2.5-0.5B, +2% budget,
  sf8 / sf16, energy at 130 nm, LM head free: round 1 predicted +1.99%,
  measured +1.78% (65% of MACs at sf8, 64% of all-sf16 energy); round 2 at
  +2.25% predicted +2.18%, measured +2.60% (rejected).  All sf8 with the LM
  head at sf16: +10.4% at 60%.  Cycles, sf8 to sf16: round 1 predicted
  +1.89%, measured +1.29% (82% of all-sf16 cycles); round 2 at +3.11%
  predicted +2.57%, measured +1.22% (81%; 56% sf8, 37% sf10, 4% sf12, 3%
  sf16).  Measured loss is not monotone in the predicted one: the sum of
  single-layer losses is a rough guide, the whole-model check decides.  Cycles and bus words use
  the better of plain and weights-packed (sf16 packed is slower than plain).

## dot_wide with a shift per output in kernels (0.14.0, spec 1.8.0)

* Alosh (2026-11-18): the chip already takes a shift per output (POP's Rs per
  token and column); `@sf.jit` did not expose it.  Kartik follows on the
  compiler side.
* IR: `dot_wide` takes an optional third operand, an int value broadcastable
  to (m, n); the `shift` attribute is then None.  Interpreter: the host's
  `matmul_wide_codes` with the broadcast block; C: `sh` read per output and
  passed to `__sf16_dot_wide`.  A live lane outside -32..31 faults (C fault
  code 6); masked lanes use 0 and never fault.
* Int buffers: a NumPy integer array argument is a `ptr<int>` (`PtrType(0)`,
  same plumbing as sf pointers), copied in with the launch, read-only (a
  store is refused at trace time, so nothing has to be copied back), every
  value within int16 (one word on the chip) or `SFLaunchError`.
* `sf.dot` dispatches to the kernel version before checking the shift, so a
  traced shift reaches it.
* Spec 1.8.0 (a contract change: a new operand); golden files only change
  their version field.  `tests/test_kernel_dot_wide_shift.py`: every v4.9.2
  sf16 case of `dot_wide_shift.json` through a kernel on both backends, codes
  and saturation counts.

## MXFP4 (0.13.1)

* Alosh (2026-11-18): add MXFP4, the format the real gpt-oss checkpoints ship
  in; same pattern as FP8 (read exactly, then quantize to sf).  NVFP4 is
  skipped (native to Blackwell GPUs only).
* `sf.torch._mxfp4`: `dequantize_mxfp4` (E2M1 low nibble first, E8M0 bias
  127, 255 = NaN; exact in float32), `quantize_mxfp4` (the OCP encoder, for
  tests and writing checkpoints), `load_mxfp4` (every `*_blocks` /
  `*_scales` pair of a dict, .safetensors file or folder into the model's
  float parameter; 3-D pairs transposed as gpt-oss stores them: always, not
  only when the shapes need it, since a square expert matrix fits either way).
* `convert` reads an `nn.Linear` with uint8 (out, in/2) weights and a (out,
  in/32) E8M0 scale; `Mxfp4GptOssExperts` (GPU-packed) is skipped with the way
  out.
* Checked: the decoder equals `transformers`' `_convert_moe_packed_tensors`;
  a tiny gpt-oss checkpoint written in the real layout loads to the same
  values through `sft.load_mxfp4` and through `transformers`
  (`from_pretrained`, CPU, dequantize); `tests/test_mxfp4.py`.

## Every matmul on sf, or a reason (0.13.0, closing Stage 2)

Alosh's questions (2026-11-17): does `convert` find every layer at the right
width, read the bias, and cope with MoE and block-scaled models?  A probe of
14 tiny random Hugging Face models (counting the matmuls left in torch after
conversion) found three silent gaps, now closed (`sf.torch._scopes`):

* **MoE experts** (one 3-D parameter, grouped matmul): 31 to 76% of the
  matmul work stayed in float, unreported.  Now `sf_experts_forward` runs a
  Hugging Face experts module (the `use_experts_implementation` flags:
  has_gate, has_bias, is_transposed, post-expert norm, its own
  `_apply_gate`) with one SF linear per expert and projection, installed on
  the instance (parameters and state dict keys unchanged, deep copy safe).
  Under `torch.compile` it runs each expert on every token weighted by 0 or
  the routing weight: a data-dependent shape broke the graph, and a graph
  break while the attention mode (entered from a forward pre-hook) is active
  fails in dynamo.  Wide mode computes rows independently, so the bits are
  those of the sparse loop (tested).  An experts module without the flags
  falls back to a `TorchFunctionMode` scope that takes its own `F.linear`
  calls (eager only is guaranteed).
* **Routers** (a module with a 2-D `weight` that calls `F.linear`): taken
  by a scope; sf16 when the layers are narrower (like the LM head).
* **Eager attention** (`torch.matmul`, no SDPA; Gemma 2 by default): a scope
  on every module whose class name ends in `Attention` takes each matmul
  between activations; the first after a softmax is P @ V
  (`matmul_wide_pv`).  Decided by call order and shapes, not values, so
  `torch.compile` makes the same choices.  Rotary modules under it are
  suspended (angles stay float).
* **Quantized weights**: an FP8 `nn.Linear` subclass was converted as float
  (block scales ignored, output about 100 times too large); an int8 one gave
  garbage.  `read_weight`: FP8 with `weight_scale_inv` / `weight_scale` (per
  tensor or per block, block from `block_size` or the shapes) is dequantized
  to float32 and converted; any other non-float weight, packed integer
  buffers (`qweight`, int8 / uint8 / int32) and an `nn.Linear` subclass
  with its own forward are skipped with the reason.
* **Report**: `widths` per module; any module with an unconverted 2-D float
  weight is listed in `skipped` (embedding tables excluded).
* Probe after: GPT-2, Llama (SDPA and eager), Qwen2, Gemma 2, Mixtral,
  Qwen2-MoE, Qwen3-MoE, DeepSeek-V3, gpt-oss, BERT, T5, ViT: 0 matmuls left
  in float.  Mamba: its depthwise Conv1d (reported) and selective scan stay
  float.  `tests/test_stage2_coverage.py`.

## Weights prepared once (0.12.1)

* Generating text was about 20 s per token on Qwen2.5-0.5B (2 cores): every
  call re-read the float weight, checked it for NaN, quantized it per channel,
  took its norms and laid out its codes.  The arithmetic itself was a small
  share.
* `_core.PreparedWeight` / `prepare_weight`: per-channel exponents, the
  dequantized column norms (for l2), the largest column norm (for the guard),
  the weight-underflow count and the codes as the compiled loop reads them,
  uint16 magnitudes and int8 signs (3 bytes per weight).  Built in the same
  blocks of rows as `_wide_linear_blocks`.  `_heuristic.plan(nb=...)` takes
  the norms; `_linalg.matmul_wide_acc_prepared` the codes.  The C loop now
  reads B as uint16 / int8 for every call.
* `sf.torch._ops.weight_cache`: LRU, keyed by storage, offset, shape,
  strides, version counter, dtype, width and MX version, with a weak
  reference to the tensor that owns the memory (a new tensor at a freed
  address is a miss).  `SF_WEIGHT_CACHE_MB` (default 4096, 0 = off).  Host
  only; device runners and non-finite weights take the old path.
* Same bits: `tests/test_weight_cache.py` (31), and Qwen2.5-0.5B's logits
  over all 151,936 tokens identical with the cache on and off.
* Qwen2.5-0.5B decode: 20.5 s to 1.1 s per token; 18 s once to prepare;
  1.4 GB of cache.  `examples/chat_wide.py`: chat with the model, optionally
  next to float (`--compare`).

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
| Hugging Face `Conv1D` | GPT-2 and relatives build their linear layers with `transformers`' `Conv1D`, weight stored `(in, out)`. `SFHFConv1D` wraps it (own parameter objects, `weight.t()` into `sf::linear`); detected by class name plus `nf`, `nx` and the weight shape, so `transformers` is never imported. Before this, `convert` converted 1 of GPT-2 small's 49 matmul layers and did not report the other 48 | `torch/_modules.SFHFConv1D` |
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

## Stage 3 decisions

| Question | Choice here | Where |
|---|---|---|
| Kernel representation | tracing, not AST rewriting: the Python function runs once per distinct (argument kinds, constexpr values, block) with symbolic arguments and emits SF IR, typed SSA with static shapes. Python control flow over Python ints and `sf.constexpr` runs at trace time and unrolls | `kernels/_ir.py`, `kernels/_runtime.Kernel` |
| Lockstep rule | `if` / `while` / `and` / `or` / `not` on a traced value raise `SFCompileError` from `Sym.__bool__`; a traced value in `range()` raises from `Sym.__index__`; `sf.where` for values, `with sf.predicate(mask)` for statements | `kernels/_sym.py` |
| Predication semantics | a predicate masks stores and event counts, never computation (every lane computes, as on the machine). Nested predicates AND. A scalar computed under a vector predicate is one value per lane: the IR broadcasts its shape with the predicate's at trace time, so the interpreter and the C backend agree on it | `kernels/_ir.Builder.emit` |
| Literals | the Stage 1 rules: an arithmetic literal (int, float, Fraction, Decimal, NumPy scalar) is quantized at trace time and goes through the policy (`v + 1` raises before anything is launched); comparisons against any host number are exact (`v < 1` always True, `v == 0.1` never) and are lowered at trace time to one integer threshold per comparison, folded to a constant when the threshold leaves the format's range, so the backends compare integers only | `kernels/_sym._compare_literal` |
| Store races | two live lanes storing to one address: the interpreter and C keep the lane-order last write, like NumPy fancy assignment. The machine is not specified here; kernels should not rely on it. Shared-memory stores are read-all-then-write-all inside one op (lockstep) | `kernels/_interp.py`, `_cgen.py` |
| Status and events | the simulator returns an exact `SFStatus` (`exact=True`) with counts per *result that saturated*: an elementwise op counts lanes, a reduction (`sf.sum`, `sf.dot`) counts each output at most once, as the Stage 1.1 built-ins do; a hand-written loop of adds counts each add. Underflow is in `counters.underflow` only, since the spec's `SFStatus` carries none | `kernels/_interp.Counters`, `sf16_builtins.h: sf16_fold` |
| Policy after launch | invalid, div0, overflow in catalogue order through `errstate`, like the Stage 1.1 heavy ops; `raise` is honoured because the counts are exact | `kernels/_runtime._apply_policy` |
| Before launch | limits (1024 lanes, 16384-value tiles, 65535 per grid axis, 32 KiB of shared memory in device-format bytes, 2^28 elements), argument placement (one device, no host arrays, no non-contiguous views: a transpose or strided slice must be copied first), widths; any failure is `SFLaunchError` with nothing launched. The Stage 1.1 pre-flight (range bound, probe) does not run for user kernels: the kernel body is opaque to it | `kernels/_runtime.KernelDevice.check_launch` |
| Device faults | out-of-bounds load / store / shared access and integer division by zero stop the launch with `SFDeviceError`. Contract: the faulting op writes nothing; writes by earlier ops and by earlier programs stay visible; nothing after the fault runs. Both backends are tested to leave the same buffer state | both backends |
| Cycle model | `ceil(lane_ops / 256) + ceil(macs / 4096) + loads + stores + barriers`. A placeholder for autotune ordering, not a performance claim; replace the constants when the hardware team publishes them | `kernels/_interp.Counters.cycles` |
| C backend | sf16 only (the device format); every value a fixed-size `int64_t` array, every op a loop, events through the `sf16_events` block of `sf16_builtins.h`; user parameter names never appear in the C (`arg<i>`); compiled with `cc -O2 -std=c99` into a private per-user cache (`sf/_cache.py`, see below) keyed on the emitted C, the header, the compiler banner and the flags. The header is the stand-in for the custom Clang's builtins: when the real ones arrive, the emitted C is what to hand them | `kernels/_cgen.py` |
| Hardware target | `atreides-hw` registered now so code can be written against it; raises `SFDeviceError` until `HardwareAtreides.attach(driver)` is given an object with `launch(ir, grid, block, buffers, scalars) -> Counters` | `kernels/_runtime.HardwareAtreides` |
| Autotune | by simulated cycles, ties by wall time, one tuning run per distinct key; configs supply constexpr names the launch does not pass; the grid may be a function of the chosen config | `kernels/_runtime.autotune` |
| `from __future__ import annotations` | `BLOCK: sf.constexpr` arrives as a string in such modules; the kernel accepts both the object and the string | `kernels/_runtime._is_constexpr_annotation` |

## Wide mode (v4.8.1 MX engine)

* `accumulate="wide"` on `matmul`, `dot`, `addmm`, and `sf.accumulate` /
  `sf.setaccumulate` / `sf.getaccumulate` for `@`, follow Alosh's design
  (`sf wide mode (MX).md`, 2026-11-07).  The default stays `"chain"`, so no
  existing result changes.
* The model (`_linalg.matmul_wide_codes`) is written from the chip repo's
  `test/v4/mxref.py` (superfloat.gpu, branch v4.8.1-openframe) and was checked
  against `mxref.Engine` on 3,000 random and edge matmuls (107,416 outputs,
  sf4 to sf16, every shift): 0 mismatches.  `tests/data/mx_golden.json` keeps
  240 of those cases, computed by mxref itself, so CI checks the model
  without the chip repo.
* The same run showed `q115.py` (the chip's FMA / ACT reference) equals our
  sf16 add, sub, mul and fma on 2,266,050 pairs each: elementwise needs no
  change for v4.8.1.
* Width picks the block: sf9..sf16 on SF16, sf4..sf8 on SF8.
* Underflow in wide mode means a nonzero total rounded to 0.  A product below
  the SF16 block's resolution adds nothing inside the engine, so a total of
  exactly 0 raises nothing.
* Device status: the accumulator clip at +-32 is invisible, so the output scan
  is an upper bound only when the host proves no total can reach +-32
  (`matmul_wide_acc_safe`); otherwise unknown.  Underflow is always unknown
  without replay.
* The hardware version is v4.8.1 and it already has MX (Alosh, 2026-11-07),
  so every Atreides target (`atreides`, `-flags`, `-sim`, `-hw`) has
  `Device.mx = True`.  A custom target without MX refuses wide mode before
  launch and never falls back.  The DSL version goes to 0.5.0.
* Kernels (DSL 0.6.0, spec 1.1.0): `sf.dot(..., accumulate="wide", shift=)`
  and `a @ b` traced under `sf.accumulate("wide")` emit the `dot_wide` IR op.
  Both back ends run it: the interpreter through `matmul_wide_codes`, the C
  back end through `__sf16_dot_wide` in `sf16_builtins.h`; they agree bit for
  bit, events included.  The kernel cache keys on the accumulate mode, so a
  kernel traced in chain mode is never reused inside a wide block.  Spec
  golden vectors for `dot_wide` are the mxref-computed cases.
* Follow-up: `sf.torch` (shifts per token and per column, chosen
  automatically with an override, reported in `SFLayerStatus`; Alosh is
  computing golden values chip-side).

## nn.MultiheadAttention (0.11.3)

* `sft.convert` replaces `nn.MultiheadAttention` with `SFMultiheadAttention`
  (in `torch/_attention.py`), keeping the parameter objects.  Projections are
  SF linears (`{name}.q/.k/.v/.out`); in wide mode the attention is
  `sft.attention` on MX (`{name}.scores/.pv`), in chain mode SDPA in float.
* Same call and returns as PyTorch: `batch_first`, unbatched input,
  `attn_mask` (bool True = masked, as MHA; or additive; 2-D or per head),
  `key_padding_mask`, `need_weights` / `average_attn_weights`, `is_causal`,
  dropout when training, separate `kdim` / `vdim`.  `sft.attention` gained
  `need_weights=True` (returns P).
* Not converted (reported): `bias_k` / `bias_v`, `add_zero_attn`.
* Tests: 7 in `tests/test_torch_attention.py`; outputs and weights within
  2e-4 of PyTorch in wide mode.

## Big layers in blocks (0.11.2)

* Qwen2.5-0.5B's LM head (151,936 x 896) ran out of memory on a 7 GB
  machine: the weight was quantized at once, with several full-size copies.
* `_core.wide_linear` now takes a weight bigger than `WIDE_BLOCK` (8M
  elements) a block of output channels at a time.  Each channel's exponent
  depends only on its row; the guard needs the whole layer's largest column
  norm, so pass 1 finds it (`_heuristic.col_norm_max`, `acc_guard(bmax=)`)
  and pass 2 computes.  Same bits and the same status
  (`test_big_layers_in_blocks_give_the_same_bits_and_status`).
* Qwen2.5-0.5B, first 256 tokens: float 17.40, sf16 wide with attention on
  MX 17.42.  About 2 minutes per 256 tokens on 2 cores.

## A compiled MX accumulator on the host (0.11.1)

* `sf/_fastmx.py`: the MX accumulation in C (compiled once with `cc`,
  cached on disk, published atomically), rows split across threads (ctypes
  releases the GIL).  `_linalg.matmul_wide_acc` uses it when it builds
  (`FAST_MX`); otherwise, or with `SF_FAST_MX=0`, the NumPy reference.
* SF16 products in closed form: `t = (comp + a*w - C[a mod 2^tk][w mod
  2^tk]) >> (30 - F)`, where `C` is the sum of the dropped columns (a 64 x 64
  table on v4.9.2, 512 x 512 on v4.8.1).  No bit loop.  SF8: `a*w`.
  The clip once per group of 8 and the clip mask as in the reference.
* Bits: `tests/test_fastmx.py` (22) against the reference on every width,
  both chips, clipping and edge codes, batches, and the emulator's 240
  golden cases.  GPT-2 perplexity identical to every digit (60.70888535488952).
* Speed (2 cores): a 128 x 768 x 3072 SF16 layer 0.36 s against 10.6 s (29x);
  GPT-2 small about 30x faster per window.
* Where the libraries live (`sf/_cache.py`).  Both compiled paths (this one
  and the kernel C backend) load a `.so` with ctypes, which runs its code, so
  the cache must be writable by this user only.  The first version used
  `/tmp/sf-fastmx-<uid>` (and the kernel backend `/tmp/sf-kernels-c`) with
  `makedirs(exist_ok=True)`: on a shared machine another user could create
  that folder first and plant a library under the predictable name.  Now the
  cache is `$SF_CACHE_DIR/<name>`, else `$XDG_CACHE_HOME/sf/<name>`, else
  `~/.cache/sf/<name>`, created 0700; on POSIX the folder and each library
  are checked (ours, not a link, not group/other-writable) before loading.
  A failed check means the NumPy reference (`_fastmx`) or "C backend
  unavailable" (`_cgen`), never a load.  No home folder: a per-process
  `mkdtemp` folder.  Tests: `tests/test_cache.py`.
* `pyproject.toml` version was 0.4.0; now matches `sf.__version__` (tested).

## Golden vectors and docs for wide mode (spec 1.6.1)

* Items 9 and 10 of Alosh's build order.
* `tools/gen_golden_wide.py` writes `spec/golden/wide/dot_wide_shift.json` (144
  cases: sf4, 6, 8, 10, 12, 16 x full, small and sparse codes x shift per
  token, per column, a full grid and a scalar, on v4.9.2 and v4.8.1; 60
  saturate) and `spec/golden/wide/sft_wide.json` (6 wide layers with every
  `SFLayerStatus` field, including the guard raising a token and two redos;
  3 attention cases; a one-block model with a tied head, at sf16 and sf8).
  `--check` fails if the files are out of date.
* Source: sf's host model, which matches the emulator's 240 `dot_wide` and
  13 heuristic cases.  To confirm on the RTL (item 9 says "then the RTL").
* `tests/test_golden_wide.py` (160) replays them on the host, `atreides` and
  `atreides-sim`, and checks the generator writes the same bytes.
* Docs (item 10): wide-mode status on the device (`arrays/status`),
  `errors/policy`, `concepts/events` (underflow in wide mode),
  `concepts/rounding-and-truncation` (wide rounds once), `guides/kernels`,
  `reference/hardware` (both MX engines), `concepts/host-and-device`.  "v1"
  wording removed outside the decision records.
* Still open from item 9: BFCL accuracy on Qwen2.5-0.5B and Qwen3-0.6B
  (needs a GPU and Alosh's harness).

## Attention on MX (0.11.0, spec 1.6.0)

* Item 4 of Alosh's build order.  `sft.attention(q, k, v, attn_mask,
  dropout_p, is_causal, scale, enable_gqa, name=)`, in `torch/_attention.py`.
* Scores: `sf::matmul_wide(q * scale, k^T)`, one exponent per query row and
  per key, `eo = l2 - h`.  The 1/sqrt(d) scale is applied to Q in float
  before quantizing (Q's per-row power of two absorbs it).
* Softmax and masks: float on the host.  `dropout_p` applies to P in float.
* `P @ V`: new op `sf::matmul_wide_pv` (`_core.wide_matmul(out_rule="max_b")`):
  `eo` per channel = V's exponent there, `pow2_exp(max|V[:, c]|)`, via
  `_heuristic.plan(eo=...)`.  Never saturates; the l2 redo stays as a guard.
  Measured: error within 2^-15 of each channel's max.  The general rule was
  about 1.5x more precise on outlier-heavy V in a quick check: to discuss.
* GQA: key and value heads repeated to the query's (on the host).
* `sft.autocast(accumulate="wide")` catches `F.scaled_dot_product_attention`
  (chain: left in float).  `sft.convert(..., attention="auto")` installs a
  forward pre-hook / post-hook (`always_call`) on the root that enters an
  `AttentionMode` for each forward; calls named `attention.0`, `attention.1`,
  ... per forward.  `ConversionReport.attention` says where it runs.
* Attention is always sf16, also when the layers are sf8.
* Compile backend: `matmul_wide_pv` goes to the target's MX engine.
* Not yet: `nn.MultiheadAttention`, flash/varlen kernels, KV caches beyond
  what SDPA sees.
* Tests: `tests/test_torch_attention.py` (11).

## The sf8 / sf16 split (0.10.0, spec 1.5.0)

* Items 7 and 3 of Alosh's build order.
* Devices: `Device.wide_bits` (widths the MX engine takes), `widths_for(op)`
  and `can_run(op, fmt)`.  Atreides targets: `wide_bits = sf4..sf16`,
  `supported_bits = (16,)` unchanged.  `check(fmt, n, op)` uses the op:
  placement accepts either set, `matmul_wide` the wide set, every other op
  sf16.  A narrower non-wide op gets an error that names `accumulate="wide"`.
* Offload: `_should_offload(total, op, fmt)` now also checks the width.
  Before, a host sf8 op with `offload_threshold` set raised
  `SFUnsupportedError`; now it stays on the host with the same bits, as a
  host divide already did (0.6.1).  Decision to confirm with Alosh: the old
  test (`test_offload_threshold_for_reductions`) asserted the error.
* Kernels: unchanged, sf16 only (the C back end emits sf16; an sf8
  `dot_wide` in a kernel is future work).  `dev.alloc(shape, sf.sf8)` now
  works on `atreides-sim`, for wide matmuls.
* `sft.convert(..., keep_sf16="auto")`: with a dtype below sf16 the LM head
  stays sf16.  `find_lm_heads(model)`: `get_output_embeddings()`, else a
  linear tied to an `nn.Embedding`, else the last linear whose output size
  is a vocabulary.  No embedding, no head.  `keep_sf16=False` or a list of
  names.  `ConversionReport.kept_sf16` lists (name, why).  Grouped convs are
  not converted at all yet (they run in float), so nothing to keep there.
* Tests: `tests/test_sf8_split.py` (20); three older tests updated to the
  new placement and offload rules.

## sf.torch wide mode (0.9.0, spec 1.4.0)

* Items 2, 5 and 6 of Alosh's build order.  `accumulate="wide"` on
  `sft.convert`, `sft.autocast`, `sft.linear`, `sft.matmul`, `SFLinear`,
  `SFConv2d` and `SFHFConv1D`; default `"chain"`, which is unchanged bit for bit.
* The rule is `sf/_heuristic.py`, torch-free and in one place: `h_for(K)`
  (margin 2), `pow2_exp`, `quantize_pow2` (one exponent per slice, round
  half to even, clamp), `plan` (shift and the l2 redo shift) and
  `acc_guard`.  `calibration="heuristic"` is the only value.
* Scale policy in wide mode: activations one power-of-two exponent per token,
  weights one per output channel, no `headroom` (partial sums cannot clip
  at 1).  `scale=` and `headroom=` are ignored in wide mode.
* Accumulator guard: before launch, per token, Cauchy-Schwarz on the block's
  magnitudes (|S| <= |a_t| max|w_c| / 2^(30-F) + K on SF16).  A token that
  could reach the limit gets its exponent raised until it cannot, so
  `acc_clipped` is 0 on the host and on the chip.  It triggers on v4.8.1
  (+-32) and only past about 512 terms near 1 on v4.9.2.
* Redo: an output whose code reaches +-max with a nonzero total is popped
  again at l2 (the golden rule).  The host accumulates once and pops twice
  (`_linalg.matmul_wide_acc` + `mx_pop`), the device relaunches at the l2
  shifts; both give the same bits.
* Separate custom ops `sf::linear_wide(x, w, b, bits, shift, name)` and
  `sf::matmul_wide(a, b, bits, shift, name)`, so chain graphs and their
  schemas do not change.  The compile backend sends them to the target's MX
  engine, or keeps them on the host when the target has none.
* Status: `SFLayerStatus` gains `accumulate`, `shift_min`, `shift_max`, `h`,
  `redone`, `flushed`, `acc_clipped`, `guard_raised` (defaults keep chain
  statuses as before).  On a device `flushed` is `None` (the chip cannot
  report it) and `saturated` after the redo is an upper bound.
* The bias was added in float after POP on the host, as in the GPT-2
  prototype.  Decided (Alosh, 2026-11-17): from 0.12.0 it is added on the
  chip at each output's exponent (see "Bias on the chip" below).
* `_linalg.matmul_wide_acc` computes in blocks of columns (bounded memory,
  the same bits), so a 50,257-column LM head no longer runs out of memory.
* Tests: `tests/test_torch_wide.py` (43): the `h` table, the 13 golden cases
  step by step and end to end (`final_value` exact), a device redo, the
  guard on both chips, convert / autocast / Conv2d / Conv1D, gradients, the
  compile backend on `atreides` and `atreides-sim`, no-MX fallback, bad input.
* Not in this change: attention (item 4, scores and values stay float), the
  sf8 / sf16 split (item 3), device targets at sf8 (item 7), the per-matmul
  configuration (item 11).

## A shift per output (0.8.0, spec 1.3.0)

* Item 1 of Alosh's build order.  POP takes one shift per token and column
  (Rs = t << 8 | shift), and the heuristic picks one per output, so a wide
  matmul now takes `shift` as an int or an int array broadcastable to the
  result, every element from -32 to 31.
* `_linalg.check_shift` returns an int or an int64 array (rejects bool,
  floats, out of range); `_mx_pop` works element by element; `matmul_wide_codes`
  broadcasts the shift to `(..., M, N)`.
* `sf.matmul` broadcasts against the result shape the caller sees (1-D
  operands squeezed), then lays it out as the internal shape.  `sf.dot` and
  `sf.addmm` go the same way.  Device targets get the array through the same
  keyword, so the mock and `atreides-sim` match the host bit for bit.
* Kernels: the IR attribute stays one int per block.  A shift per output in a
  kernel needs it as a third operand of `dot_wide` (Kartik's side too), so it
  raises `SFCompileError` for now.
* Tests: `tests/test_shift_arrays.py`, with the 13 matmul cases of the
  heuristic section in `sf_golden_v4.9.2.json` (`pop_q` and the saturation
  masks, 13 of 13).  `final_q` there is after the heuristic's redo at `l2`;
  that is the `sft` wide-mode work, next.
* Spec 1.2.0 had no changelog row (missed in 0.7.0); added with 1.3.0.

## Wide mode on v4.9.2 (0.7.0, spec 1.2.0)

* Alosh's `sf wide mode (MX).md` (2026-11-16) makes v4.9.2 the flagship for
  all benchmarking: v4.8.1 with 32-bit MX accumulators.  SF16 block Q10.22
  (about +-512), products keep columns 6 and up, +80, cut to 22 fraction bits;
  SF8 block Q18.14 (about +-131,072), exact.  The ISA is unchanged.
* The model is now per chip.  `_linalg.MX_TARGETS` holds each chip's units
  (`W`, `F`, `tk`, `comp`, the same names as `mxref.UNITS`);
  `matmul_wide_codes`, `matmul_wide_acc_safe` and `_mx_products16` take them
  from an `mx` version.  `MX_DEFAULT = "v4.9.2"`.
* A target names its engine: `Device.mx` is a version string (`True` means the
  default, `False` no MX).  Every Atreides target is v4.9.2;
  `MockAtreides(mx="v4.8.1", name=...)` keeps the signed-off chip reachable.
  Device launches, the host replay of their status, the kernel interpreter
  and the C back end all use the target's version.  Host arrays use the
  default device's, so offloading never changes bits.
* The C back end's `__sf16_dot_wide` takes the unit's numbers as arguments,
  emitted from the target; the build cache keys on the emitted source, so a
  v4.8.1 build is never reused for v4.9.2.
* Golden vectors: `tests/data/sf_golden_v4.9.2.json` is Alosh's file as sent
  (its `heuristic` section is for the coming sft wide-mode work).
  `tests/data/mx_golden.json` is its `mx` section, 240 cases, and
  `tests/data/mx_golden_v4.8.1.json` keeps the old v4.8.1 outputs of the same
  inputs.  Both pass 240 of 240; the chips differ on 8 of them, the sums near
  +-32 and the SF16 product cut.  Spec golden vectors for `dot_wide` are now
  v4.9.2's.
* Not in this change: the heuristic, shift arrays and `sft` wide mode
  (items 1 to 11 of the doc, separate PRs).  PR 9's target guide still
  describes v4.8.1.

## Ops the chip does not have (0.6.1)

* Atreides v4.8.1 has no SF divide and no square root (ISA section 5). Before
  0.6.1 the four Atreides targets left `supported_ops` unset, so `a / b` and
  `sf.sqrt(a)` on a device array ran on the mock and looked supported.
* `MockAtreides.supported_ops` is now the v4.8.1 set: `add`, `sub`, `mul`,
  `fma`, `neg`, `abs`, `pow`, `sum`, `matmul`, `matmul_wide`. `atreides-flags`,
  `atreides-sim` and `atreides-hw` inherit it. `pow` stays because it is a
  chain of `mul`.
* Refuse, never fall back: a device `/` or `sqrt` raises `SFUnsupportedError`
  before launch. The user moves the operands with `.to("host")`. A silent host
  fallback would hide that a model cannot run on the chip.
* Known answers still come first: a divide by zero or a square root of a
  negative is reported as that, because it is read off the inputs.
* Offloading is different, because the runtime chooses it and the user did not
  ask for the device. `_should_offload(total, op)` keeps a host op on the host
  when `default_device` lacks the op (or lacks MX for a wide matmul). Same
  bits either way.
* Tests that exercise the device rules for divide and square root (exact div0
  and invalid, masked divide) run on a test-only target with every op,
  `mock-all-ops` in `tests/conftest.py`.
* Kernels too (decided by Alosh, 2026-11-17): `@sf.jit` code that uses SF `/`
  or `sf.sqrt` is refused on `atreides-sim` and `atreides-hw` with
  `SFUnsupportedError` before launch (`KernelDevice.check_launch` reads the
  traced IR against `supported_ops`). Otherwise a kernel could pass in the
  simulator and fail on the chip. Integer `//` and `%` for index arithmetic
  are separate ops and stay allowed. Their kernel semantics (events, the C
  builtins) are tested on `sim-all-ops` in `tests/conftest.py`.

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
* Stage 3: the cycle model constants (lane width 256, systolic 4096 MACs per
  cycle), what the machine does on two lanes storing to one address, and the
  driver interface for `atreides-hw` (`launch(ir, grid, block, buffers,
  scalars)` is the proposal).

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

The v1 RTL report (superfloat.gpu, 2026-10-19 and 2026-10-27) ran the suite on the
signed-off v1 chip in Verilator: 727 of 732, the four device failures being the
missing divide and square-root instructions. Framework findings fixed, with
regressions in `tests/test_review_findings_rtl.py`:
(1) a kernel driver that cannot read events back reports counts as `None` and the
status says unknown instead of "0, exact"; `exact=False` makes counts a bound;
`status="replay"` reruns on the host, takes exact counts and checks the target's
bits; (2) `SFStatus.underflow`, exact with flags or replay, 0 where impossible,
proven 0 on the host when the two smallest nonzero factors multiply to at least
the scale, otherwise unknown; the policy applies to it; (3) the C build cache
compiles to a temp name and publishes with `os.replace`; (4) `sf.torch._bits`
no longer calls `sf.info`, whose Fraction-to-float Dynamo 2.11 cannot trace (5
graphs became 1); (5) a layer's saturation is the union of the matmul's and the
bias add's per-output masks when the target gives them; two nonzero counts are
summed and marked inexact. `Device.supported_ops` lets a target without divide or
square root refuse those ops on the host before launch. Open, for discussion:
finding 6 (a v4 numeric mode: wide accumulator, one rounding at the pop,
power-of-two scales) and finding 7 (packed N-bit storage at the device boundary).

A fourth independent review (Stage 3 drop, 2026-10-18) found: exact literal
comparisons lowered as "code * denominator op numerator", which overflows
int64 for small or huge literals (`v < 0.0001` true for 0.5); writes through
a transposed or strided device view silently lost; the C backend continuing
after a device fault, so partial writes differed from the interpreter; user
parameter names interpolated into the generated C (a pointer called `o`
collided with a loop local); and the shared-memory limit enforced in elements
rather than bytes (64 KiB instead of 32 KiB). Also two risks: the C build
cache keyed on the emitted source only, and `static` shared scratch in the
generated C. All fixed and covered by `tests/test_review_findings_3.py`.
The fixes: a literal comparison is now decided at trace time with Python
integers (`k < q` becomes `k < ceil(q)` and so on, constants folded when the
threshold leaves the format's range), so no backend ever multiplies at run
time; non-contiguous pointer arguments are refused before launch with
`SFLaunchError`; a fault returns from the C kernel at once and a faulting
store writes nothing in either backend (writes by earlier ops and earlier
programs stay visible, which is now the contract); every C argument is
`arg<i>`; `Limits.max_shared` is bytes; the build key covers the emitted C,
the builtins header, the compiler banner and the flags; shared scratch is a
per-program local.

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

## Bias on the chip, and every op at sf4 to sf16 (0.12.0, spec 1.7.0)

Decided by Alosh, 2026-11-17.

* **The bias of a wide linear is added on the chip, after POP.** Before
  0.12.0 it was added in float on the host, so a layer was not SF arithmetic
  end to end. Now `_core._wide(..., bias=b)`: for each output, the bias is
  quantized to that output's exponent `fe` (round half to even of
  `b * 2^(P - fe)`, clamped to +-max) and added to the POP code with the
  saturating sfN add. On the chip that is one `ACT f=00` per output (`ACT`
  takes a bias register; ReLU comes free with f=01). Below sf16 the add is
  masked like every other op (below).
* **Redo.** An output is redone when its POP code reached +-max, when
  `q + b` did, or when its bias code was clamped. The first two show in the
  codes; the clamp is known on the host, which aligned the codes. The POP
  code must be checked as well as the sum: a POP that clipped can be pulled
  back below max by a bias of the other sign, which would hide the clip. A
  clamped bias likewise. Both cases are tests in `tests/test_bias_on_chip.py`.
* **Redo exponent.** `_heuristic.plan(..., bias_abs=|b|)` takes the redo
  exponent from `pow2_exp(|x||w| + |b|)`, so a redone output can clip in
  neither the POP nor the add. The first POP's exponent is unchanged
  (`l2 - h` from the dot product alone).
* **Status.** `bias_saturated` counts bias codes clamped at the final
  exponent; `saturated` includes them. `sft_wide.json` pins
  `bias_saturated` from 1.7.0.
* **Cost on the chip.** One extra word per output to read (the aligned bias
  code, M x N per launch, like the shift operands) and one ACT per output;
  M x N adds against M x N x K MACs.
* **Atreides targets run every op except divide and sqrt at sf4 to sf16.**
  The cores compute in sf16. Below sf16 the compiler keeps codes
  left-aligned and masks the low `16 - N` bits after every op: `DIV` (a
  logical right shift, up to 6 bits per instruction) then `ADD` to double
  back. That gives sfN's bits exactly, because sf `mul` truncates toward
  zero and sums of left-aligned codes are exact until they saturate. The
  compiler reference (`compiler_v4.9.2.md` s5) measured it on the chip with
  the sf8 and sf4 golden vectors, 0 wrong. There is no AND instruction, so
  the mask costs `ceil((16 - N) / 6) + (16 - N)` instructions: about 10x the
  time of the sf16 op at sf8, 16x at sf4. A one-instruction mask would be
  worth adding to the 2 nm ISA. User kernels (`@sf.jit`) stay sf16
  (`KernelDevice.kernel_bits`).
* **RTL check.** `superfloat.gpu/test/test_wide_prs_chip.py` part F runs
  wide linears with biases at sf4 to sf16 on the RTL: the kernel stores q,
  loads the aligned bias code, runs ACT (masked below sf16) and stores
  q + b. Parts C and E carry biases the same way.
