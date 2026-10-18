# sf: Superfloat in Python

`sf` is the Python library for the Superfloat number formats (sf4 to sf16)
used by the Atreides chip. You write normal Python, NumPy-style arrays,
PyTorch models or GPU kernels, and `sf` makes sure every number behaves
exactly as the chip would compute it, bit for bit, and tells you when
something overflowed, underflowed, divided by zero or went invalid.

## What is Superfloat, in one paragraph

An sf16 value is a 16-bit signed fixed-point number strictly between -1 and 1.
There is no exponent, no NaN, no infinity. Adding or subtracting can push a
value past the edge: it then sticks at the largest value (saturates).
Multiplying two small values can land on 0 (underflows). Dividing `a / b`
only works when `|a| < |b|`. The chip does exactly this, and so does `sf`.

## What you can do with it

**1. Scalars** (Stage 1.0)

```python
import sf

a, b = sf.sf16(0.5), sf.sf16(0.25)
a + b           # sf16(0.75)
a * b           # sf16(0.125)
b / a           # sf16(0.5)
a / b           # SFOverflowError: 2.0 does not fit in sf16
a < 1           # True, comparisons are exact

with sf.errstate(over="ignore"):     # or just let it saturate, like the chip
    sf.sf16(0.75) + 0.5              # sf16(0.999969482421875)
```

**2. Arrays, matmul, and the device** (Stage 1.1)

```python
x = sf.array([0.10, -0.25, 0.50], dtype=sf.sf16)       # lives on the host
h = sf.asarray(inputs,  dtype=sf.sf16, device="atreides")
w = sf.asarray(weights, dtype=sf.sf16, device="atreides")
y = h @ w                                              # runs on the device
y.status                                               # what saturated, if anything
y.numpy()                                              # back to floats
```

Before anything is sent to the device, `sf` checks on the host whether the
answer is already known (a divide by zero, a result that must saturate, a
range that cannot saturate). Anything the host can decide never reaches the
chip. This is the rule the whole device path follows.

**3. PyTorch models** (Stage 2)

```python
import sf.torch as sft

model = sft.convert(model, dtype=sf.sf16)             # Linear and Conv2d now compute in sf16
fast = torch.compile(model, backend="superfloat")      # sends the SF layers to the device
out = fast(x)
sft.last_status(model)                                 # per-layer saturation report
```

Weights and activations are scaled so the chip's -1..1 range is used well,
and by default the scale leaves enough headroom that nothing can saturate
inside a layer. Training still works (straight-through gradients).

**4. Your own GPU kernels** (Stage 3)

```python
@sf.jit                                                # Triton style
def axpy(a, x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    mask = offs < n
    x = sf.load(x_ptr + offs, mask=mask)
    y = sf.load(y_ptr + offs, mask=mask)
    sf.store(out_ptr + offs, sf.fma(a, x, y), mask=mask)

dev = sf.device("atreides-sim")
x, y, out = dev.upload(xs), dev.upload(ys), dev.alloc(n)
r = axpy[(sf.cdiv(n, 256),)](sf.sf16(0.75), x, y, out, n, BLOCK=256)
r.status, r.cycles                                     # exact events, simulated cycles
```

There is also a CUDA-style `@sf.kernel` with `thread_idx`, shared memory and
barriers. Kernels run on a simulator today (`atreides-sim`) which has two
independent back ends, a Python interpreter and generated C, that must agree
bit for bit. `atreides-hw` is reserved for the real chip and waits for its
driver.

## Install and run

```
pip install -e ".[test]"             # library + test tools
pip install -e ".[torch]"            # add PyTorch for sf.torch
pytest                               # the full per-commit suite (about 8 min)
pytest -m "not slow"                 # quick run while developing (about 40 s)
python examples/stage_3_tour.py      # see the kernels in action
```

Only NumPy is required. The C kernel back end needs a C compiler (`cc`).

## Errors: what happens when a number does not fit

| Event | Meaning | Default |
|---|---|---|
| overflow | result saturated at the largest value | raise `SFOverflowError` |
| underflow | result became 0 | warn `SFUnderflowWarning` |
| divide | divided by zero | raise `SFZeroDivisionError` |
| invalid | no valid answer (sqrt of a negative) | raise `SFInvalidError` |

Change it per event with `sf.seterr(over="warn")` or inside a block with
`with sf.errstate(all="ignore"):`. The value you get back is always the
chip's value; the policy only decides whether Python also raises or warns.

## How it is tested

Every result is compared as integer codes, never with float tolerances,
against an independent oracle written with exact fractions. 732 tests run
on every commit (Python 3.10, 3.12, 3.13) and the exhaustive sf16 sweep
runs nightly. The full matrix of what is checked, against what, at which
widths and on which target is in [docs/TESTING.md](docs/TESTING.md).

## What is done and what is not

Done: Stages 1.0, 1.1, 2 (2.0, 2.1, 2.2) and 3 (3.0 to 3.3) of the plan,
each reviewed independently with all findings fixed.

Not done:

* Running on the real chip. Everything above is proved on the host, a mock
  and a simulator. The hardware gate needs the custom compiler, a driver to
  attach, and answers to the open hardware questions (division and sqrt
  fault values, accumulation order and width, `mat_fma` meaning, readable
  flags). See `NOTES.md`.
* Stage 1.2 diagnostics (`sf.explain`) and Stage 2.3 PyTorch device tensors,
  deferred on purpose.

## Documentation

| File | What it holds |
|---|---|
| [docs/API.md](docs/API.md) | Every function, decorator and object for Stages 1 to 3, with syntax and the Python semantics of SF values. The reference to use when writing tests against the simulator. |
| [docs/TESTING.md](docs/TESTING.md) | The testing matrix, the test files, how to run each gate. |
| [docs/DETAILS.md](docs/DETAILS.md) | Module map, exact semantics tables, every config option, design notes per stage. |
| [NOTES.md](NOTES.md) | Design decisions, open hardware questions, review history. |
| `examples/` | A runnable tour per stage. |
| `website/` | The developer docs site (Mintlify), generated from the files above by `python tools/build_site.py`. Edit the markdown, not the site. |
