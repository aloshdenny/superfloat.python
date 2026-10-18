# How sf is tested

Every result is compared as integer codes, never with float tolerances. The
reference answers come from an independent oracle written with exact
fractions straight from the format definition.

| What | Checked against | Widths | Target | When | Status |
|---|---|---|---|---|---|
| Scalar ops and casts | Fraction oracle, golden files | sf4 to sf10 every pair; sf11 to sf16 sampled plus corners | host | every commit | passing |
| Scalar ops, all 4.3 billion pairs per op | Fraction oracle | sf16 and sf14 exhaustive | host | nightly, 31 min | passing |
| Error policy, expression trees, spec examples | catalogue types and first-error order | sf4 to sf16 | host | every commit | passing |
| Arrays, broadcasting, elementwise | the scalar rule, element by element | sf4 to sf16 | host | every commit | passing |
| `sum`, `dot`, `@` | sequential saturating chain of scalar ops | sf4 to sf16 | host, mock device | every commit | passing |
| Device path, pre-flight gate, status | host == mock bit for bit; nothing launched when the host can decide | sf16 | mock device | every commit | passing on mock, real chip open |
| `sft.quantize` | same codes and events as `sf.asarray` | sf4 to sf16 | host | every commit | passing |
| `SFLinear`, `SFConv2d`, `convert`, `autocast` | recomputed through the array API; accuracy on digits; QAT | sf16, sf8 | host | every commit | passing |
| `torch.compile` backend | compiled == emulated bit for bit on MLP, CNN, attention | sf16 | mock device | every commit | passing on mock, real chip open |
| Kernel runtime: copies, limits, faults | round trip bit for bit; errors before launch | sf16 | simulator | every commit | passing |
| Thread and block kernels | the Stage 1.1 built-ins, bit for bit | sf16 | simulator, both back ends | every commit | passing |
| C back end | same bits and same event counts as the interpreter, every kernel | sf16 | simulator with `cc` | every commit | passing |
| Autotune, cache, profiler | every config identical bits; fastest by cycles | sf16 | simulator | every commit | passing |
| Regressions from four independent reviews | the failing case of each finding | as found | as found | every commit | passing |
| Real hardware | the same cases through `atreides-hw` | sf16 | silicon or RTL | when a driver exists | not started |

CI (`.github/workflows/ci.yml`) runs the suite on Python 3.10, 3.12 and 3.13
on every push, and the exhaustive sf16 gate nightly. Today: 732 tests per
commit.


## Test files

| File | Covers |
|---|---|
| `test_1_0_1_types_casts.py` | types and casts; golden file of all codes (`tests/golden/`) |
| `test_1_0_2_add_sub.py` | add, sub, neg, abs; chains; saturation is not undone |
| `test_1_0_3_mul_fma.py` | mul, fma; precedence |
| `test_1_0_4_div.py` | div and sqrt |
| `test_1_0_5_comparisons.py` | comparisons, bool, min/max, where, guarded division |
| `test_bodmas.py` | random expression trees over the whole 1.0 grammar; first raised error must match |
| `test_properties.py` | commutativity, identities, monotonic saturation |
| `test_errstate.py` | policy, catalogue types, user-line attribution, context locality |
| `test_spec_examples.py` | every code example in the spec, as written |
| `test_1_1_1_arrays.py` | every 1.0 op on shapes `()`, `(n,)`, `(m, n)`; broadcasting, literals, promotion, casts |
| `test_1_1_2_reductions.py` | `sum` is the sequential chain; `dot` equals the scalar fma chain; max/min/argmax |
| `test_1_1_3_matmul_device.py` | `@` equals per-output `dot`; shape errors before launch; host and target bit-identical; status; pre-flight |
| `test_1_1_4_functions.py` | sqrt, pow, clip, maximum, minimum, masked div, where |
| `test_2_0_quantize.py` | same codes and events as Stage 1 under every policy; STE; dtypes |
| `test_2_1_convert.py` | SF linear recomputed through the Stage 1 API; no-saturation cases; conv as unfold + linear; convert report; autocast; accuracy on digits; QAT |
| `test_2_2_compile.py` | compiled == emulated bit for bit on MLP, CNN, attention; coverage report; pre-flight on the compiled path |
| `test_3_kernels.py` | runtime copies and limits; thread and block kernels match the 1.1 built-ins; predication, lockstep, faults, status; C backend == interpreter; cache, profiler, autotune |
| `test_review_findings*.py` | regressions from the four independent reviews (1.0, 1.1, 2, 3) |

`tests/oracle.py` is the independent oracle on exact fractions. `tests/golden/`
holds the golden code files; regenerate with `python tools/make_golden.py`
only when the format definition changes. `tools/exhaustive.py --bits N` is
the nightly exhaustive sweep. `SF_TEST_SEED` and `SF_TEST_RANDOM_PAIRS`
control the sampled widths.

## Running

```
pytest                       # full per-commit suite, about 8 minutes
pytest -m "not slow"         # about 40 seconds while developing
pytest tests/test_3_kernels.py tests/test_review_findings_3.py   # Stage 3 only
python tools/exhaustive.py --bits 16                              # nightly gate, about 31 minutes
```

The C kernel backend tests run only when a C compiler (`cc`) is on PATH.
