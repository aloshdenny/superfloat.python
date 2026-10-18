# sf API reference

Every public name in `sf`, `sf.torch` and the kernel language, with its
syntax and what it does. Written so a person or a tool can write tests
against the simulator without reading the source. Names and signatures
below are taken from the code; `pytest tests/` is the executable version.

Contents

1. Python semantics: how SF values behave
2. Stage 1.0: scalars
3. Stage 1.1: arrays, reductions, device path
4. Error policy and configuration
5. Stage 2: `sf.torch`
6. Stage 3: kernels (runtime, language, decorators, autotune)
7. Exceptions and warnings
8. Objects: `SFStatus`, `LaunchResult`, `Counters`, devices

---

## 1. Python semantics

An **sfN** value is a signed integer code `k` in `[-M, M]` with
`M = 2^(N-1) - 1`; its value is `k / 2^(N-1)`, so every value lies strictly
inside (-1, 1). sf16 is the device format; sf4 to sf15 are host-only.

| Rule | What it means in Python |
|---|---|
| Construction quantizes | `sf.sf16(0.1)` rounds to the nearest code (ties to even). A value that does not fit (`sf.sf16(1)`, any nonzero int) is an **overflow**; a tiny value that rounds to 0 is an **underflow**. Both go through the error policy. |
| Arithmetic truncates | `+ - * /` and `fma` compute exactly on codes then truncate toward zero, exactly as the chip does. `+`/`-` saturate on overflow; `*` never overflows but can underflow; `a / b` overflows whenever `|a| >= |b|`. |
| Literals are quantized first | In `x + 0.25` the literal becomes an sf value of `x`'s type before the add, through the policy. `x + 1` therefore raises `SFOverflowError` at the literal. int, float, `Fraction`, `Decimal` and NumPy scalars are accepted; strings only in the constructor. |
| Comparisons are exact | `x < 1` is always True; `sf.sf16(0.1) == 0.1` is False (0.1 is not representable). Comparisons never quantize and never raise. |
| Width promotion | `sf8 + sf16` gives sf16 (the wider). Narrowing never happens implicitly; use `sf.cast`. |
| Immutable | SF values and the codes inside arrays cannot be changed in place; every op returns a new value. |
| Unsupported | `//`, `%`, bitwise ops, `round()`, `**` with a non-integer or negative exponent raise `SFUnsupportedError`. `int(x)` is 0 for every value. |
| `bool(x)` | True unless the code is 0. |
| Precedence | Python's. `a * b + c` gives the same bits as `sf.fma(a, b, c)`. |
| Event order | When one op produces several events, they are reported in catalogue order: invalid, divide, overflow, underflow. |
| Hardware value always returned | The policy decides whether Python raises or warns; the value you get is the one the chip would produce (saturated, 0, or the proposed fault value). |

---

## 2. Stage 1.0: scalars

### Types

| Name | Description |
|---|---|
| `sf.sf4(value=0)` ... `sf.sf16(value=0)` | Scalar classes, one per width. `value` may be int, float, Fraction, Decimal, str (constructor only), NumPy scalar or another SF value. |
| `sf.dtypes` | Tuple of all thirteen classes. |
| `sf.dtype(spec)` | Resolve `"sf16"`, `16`, `sf.sf16` or an sf16 value to the class. |
| `sf.info(dt)` | `SFInfo(bits, max, min, lsb, max_code, codes, exact_max, exact_lsb)`, like `numpy.finfo`. |
| `sf.FORMATS`, `sf.SFFormat`, `sf.MIN_BITS`, `sf.MAX_BITS` | Format constants per width (scale, max code, cast thresholds). |

### Scalar attributes

| Name | Description |
|---|---|
| `x.code` | Signed integer code. |
| `x.bits` | Sign-magnitude bit pattern as the device stores it. |
| `x.sign`, `x.mantissa` | The two fields of `x.bits`. |
| `x.exact` | The value as a `Fraction`. |
| `x.dtype` | The class. |
| `float(x)` | The value as a float (exact for every width). |

### Operators

`+ - * /`, unary `-`, `abs()`, `**` (integer exponent >= 1, as a left to
right chain of multiplies), `< <= > >= == !=`, `bool()`.

### Functions

| Name | Description |
|---|---|
| `sf.cast(x, dt)` | Convert to another width. Widening is exact; narrowing re-quantizes through the policy. Scalars and arrays. |
| `sf.fma(a, b, c)` | `a * b + c` with one truncation of the product then one saturating add. Same bits as `a * b + c`. |
| `sf.div(a, b, where=None)` | `a / b`. With `where=mask` (arrays) divides only where the mask is True; other elements are 0 and raise nothing. |
| `sf.sqrt(a)` | Square root truncated toward zero. A negative input is an `invalid` event (device value 0). |
| `sf.maximum(a, b)`, `sf.minimum(a, b)` | Elementwise selection, exact, no events. |
| `sf.clip(x, lo, hi)` | Clamp to `[lo, hi]`; `lo` and `hi` are quantized like literals. |
| `sf.where(cond, x, y)` | `x if cond else y`, elementwise. At least one branch must be an SF value. |
| `sf.from_code(code, dt)` | Build a value from its signed code. Validates the range. |
| `sf.from_bits(bits, dt)` | Build a value from its sign-magnitude pattern. |

---

## 3. Stage 1.1: arrays, reductions, device path

An `SFArray` holds an int64 NumPy array of codes, a dtype and a placement
(`"host"` or a device name). Every scalar rule applies element by element
with NumPy broadcasting.

### Construction

| Name | Description |
|---|---|
| `sf.array(obj, dtype=None, device="host")` | From nested lists, NumPy arrays, SF scalars or SF arrays. `dtype=` is required unless the input already holds SF values. Quantizes through the policy; a 0-d input gives a scalar. |
| `sf.asarray(obj, dtype=None, device="host")` | Like `array` but returns the input itself when it already matches. |
| `sf.zeros(shape, dtype="sf16", device="host")` | Zeros. |
| `sf.ones(shape, ...)` | Filled with the literal 1, which does not fit: raises by default, saturates to max under `over="ignore"`. |
| `sf.full(shape, value, dtype=None, device="host")` | Filled with `value`, quantized through the policy. |
| `sf.from_codes(codes, dt, device="host")` | From an integer code array (validated in the target width). |
| `sf.from_code(code, dt)`, `sf.from_bits(bits, dt)` | Also accept arrays. |

### Array attributes and methods

| Name | Description |
|---|---|
| `a.codes`, `a.bits` | Integer codes / sign-magnitude patterns as NumPy arrays. |
| `a.dtype`, `a.shape`, `a.ndim`, `a.size`, `a.device` | As in NumPy, plus placement. |
| `a.numpy()`, `a.tolist()` | Values as floats. |
| `a.astype(dt)` | Same as `sf.cast`. |
| `a.to(device)` | Move to `"host"` or a device by name. |
| `a.copy()`, `a.reshape(...)`, `a.flatten()`, `a.T`, `a.transpose(*axes)` | Data movement; views where NumPy would give a view. |
| `a[idx]` | NumPy indexing and slicing (read only). |
| `a.sum(axis=None)`, `a.max(axis=None)`, `a.min(axis=None)`, `a.argmax(axis=None)`, `a.argmin(axis=None)` | Reductions (see below). |
| `a.any()`, `a.all()` | On the codes. |
| `a.status` | `SFStatus` of the heavy op that produced `a`, else `None`. |

### Reductions and linear algebra

| Name | Description |
|---|---|
| `sf.sum(x, axis=None)` | Saturating sum in the target's accumulation order (sequential from index 0, saturating at each step). |
| `sf.dot(a, b)` | 1-D dot as a sequential chain of `fma`. Bit-identical to the scalar chain. |
| `sf.matmul(a, b)`, `a @ b` | NumPy rules for 1-D operands and batching; each output equals `dot` of its row and column. Shape errors before any launch. |
| `sf.addmm(a, b, d)` | `a @ b` then one saturating add of `d`. |
| `sf.max(x, axis=None)`, `sf.min(x, axis=None)`, `sf.argmax`, `sf.argmin` | Exact selection, no events. |

### Device path

| Name | Description |
|---|---|
| `device="atreides"` | Placement that makes an array heavy. Any arithmetic op with a heavy operand runs on the target and returns an array with `.status`. Only sf16 is accepted on Atreides. |
| `sf.devices()` | Names of registered targets: `host`, `atreides` (mock, no flags), `atreides-flags` (mock with sticky flags), `atreides-sim` (simulator), `atreides-hw` (hardware stub). |
| `sf.device(name="atreides-sim")`, `sf.get_device(name)` | The target object. |
| `sf.register_device(device)` | Add or replace a target (the simulator or hardware when they arrive). |
| `sf.Device`, `sf.MockAtreides`, `sf.SimulatorAtreides`, `sf.HardwareAtreides` | Target classes. |

**Pre-flight** (decide on the host, launch only the undecidable). Before
every heavy op the host runs: static checks (shape, dtype, limits); known
answers (`div0` and `invalid` read off the inputs and handled before
launch); the range bound (if no element can saturate, launch with
`status.exact=True`; if every element must saturate, refuse under
`preflight="raise"`); and for the undecidable band a golden-vector probe
whose hit is a suspicion (`SFPreflightWarning` by default).

**Status honesty.** With no device flags, `status.saturated` for an
elementwise op is an upper bound from scanning for ±max (`exact=False`);
for a reduction it is `None` (unknown). With flags, or under
`sf.config(status="replay")`, counts are exact.

---

## 4. Error policy and configuration

| Name | Description |
|---|---|
| `sf.seterr(all=None, over=, under=, divide=, invalid=)` | Set the policy; each is `"raise"`, `"warn"` or `"ignore"`. Returns the previous policy. Defaults: `over=raise, under=warn, divide=raise, invalid=raise`. |
| `sf.geterr()` | Current policy as a dict. |
| `sf.errstate(all=None, **kw)` | Scoped policy: `with sf.errstate(over="ignore"):` or `@sf.errstate(over="warn")` on plain and `async` functions. Thread and asyncio safe. |
| `sf.config(**kw)` | Runtime options, as a call, `with sf.config(...)` or a decorator. |
| `sf.getconfig()` | Current options. |

`sf.config` keys:

| Key | Values | Meaning |
|---|---|---|
| `preflight` | `raise` (default), `warn`, `ignore` | What to do when the host proves a heavy op saturates everywhere. |
| `suspect` | `warn` (default), `raise`, `verify`, `ignore` | What to do on a probe hit. `verify` replays the op on the host and refuses only on proof. |
| `probe` | bool | Run the golden-vector probe. |
| `probe_size` | int, default 256 | Elements sampled by the probe. |
| `status` | `scan` (default), `replay` | How `status.saturated` is obtained without flags. |
| `offload_threshold` | int or `None` | Host ops larger than this run on `default_device`. |
| `default_device` | name, default `"atreides"` | Target for offload and for the compile backend. |
| `kernel_backend` | `interp` (default), `c` | How the simulator executes kernels. |

---

## 5. Stage 2: `sf.torch` (import as `sft`)

| Name | Description |
|---|---|
| `sft.quantize(t, dtype="sf16", scale=None)` | Fake quantization: values snapped to the SF grid, float storage, same codes and events as `sf.asarray`, straight-through gradient. `scale=None` (values must fit), `"dynamic"` (per call) or a float. |
| `sft.to_sf(t, dtype="sf16", device="host")` | Tensor to `SFArray`. |
| `sft.from_sf(a, dtype=torch.float32)` | `SFArray` or scalar to tensor (float32 holds every width exactly). |
| `sft.linear(x, weight, bias=None, dtype="sf16", scale="per_tensor", headroom="bound", name="linear")` | Functional chip-exact SF linear. |
| `sft.matmul(a, b, dtype="sf16", headroom="bound", name="matmul")` | Functional chip-exact SF matmul of two activations. |
| `sft.SFLinear(in_features, out_features, bias=True, *, dtype="sf16", scale="per_tensor", headroom="bound", name="")` | `nn.Linear` with SF arithmetic. `SFLinear.from_float(linear)` wraps an existing layer's parameters. |
| `sft.SFConv2d(conv, name, cfg)` | `nn.Conv2d` (groups=1, zero padding) as unfold + SF linear. |
| `sft.convert(model, dtype="sf16", scale="per_tensor", headroom="bound", inplace=False)` | Swap every `nn.Linear` / `nn.Conv2d` for the SF versions; everything else stays float. |
| `sft.conversion_report(model)` | `ConversionReport(converted, skipped, shared)`. |
| `sft.autocast(dtype="sf16", scale=, headroom=, prefix="autocast.")` | Context manager / decorator routing `F.linear`, `F.conv2d`, `matmul`, `@`, `mm`, `bmm` to the SF ops. |
| `torch.compile(model, backend="superfloat")` | SF nodes run through the Stage 1.1 heavy-op path on `default_device`; the rest stays on CPU. Inference only. |
| `sft.compile_report()`, `sft.compile_reports()` | `GraphReport(on_device, host_emulated, cpu, unconverted)` for the last / all compiled graphs. |
| `sft.last_status(model=None)` | Most recent `SFLayerStatus` per layer name. |
| `sft.capture_status()` | Context manager collecting the statuses of one request (thread / task local). |
| `sft.reset_status()` | Clear the registry. |
| `sft.SFConfig(bits=16, scale="per_tensor", headroom="bound")` | The scale policy record. |

Scale policy: `scale` is `"per_tensor"` or `"per_channel"` for weights;
activations get a dynamic per-call scale. `headroom="bound"` raises the
activation scale until the L1 bound proves nothing in the layer can
saturate (bits given up are reported as `headroom_bits`); `"none"` uses the
raw scale.

---

## 6. Stage 3: kernels

### Runtime

| Name | Description |
|---|---|
| `sf.device("atreides-sim")` | The simulator. `dev.alloc(shape, dtype="sf16")`, `dev.upload(host_array_or_floats)`, `dev.download(device_array)`, `dev.synchronize()`, `dev.limits`, `dev.last` (last `LaunchResult`), `dev.kernel_launches`. |
| `sf.device("atreides-hw")` | Hardware target. Raises `SFDeviceError` until `sf.HardwareAtreides.attach(driver)` is given an object with `launch(ir, grid, block, buffers, scalars) -> Counters`. |
| `sf.synchronize(device="atreides-sim")` | Wait for the device; returns its last `LaunchResult`. |
| `sf.Limits` | `max_block=1024` lanes, `max_tile=16384` values, `max_grid=65535` per axis, `max_shared=32768` bytes, `max_elements=2^28`. |
| `sf.cdiv(a, b)` | Ceiling division for grid sizes. |
| `sf.config(kernel_backend="interp" | "c")` | Interpreter (default) or generated C through `cc`. Same bits, same events. |

### Decorators and launch syntax

| Syntax | Description |
|---|---|
| `@sf.kernel` | CUDA-style thread kernel. Launch: `k[grid, block](*args, **constexpr)`, e.g. `k[(sf.cdiv(n, 256),), (256,)](a, x, y, out, n)`. Up to 3 axes each. |
| `@sf.jit` | Triton-style block kernel. Launch: `k[(grid,)](*args, BLOCK=256)`. |
| `@sf.autotune(configs=[{...}, ...], key=["n"])` on top of `@sf.jit` / `@sf.kernel` | Runs each config once per distinct value of the `key` arguments, keeps the one with the fewest simulated cycles. Config constexprs are not passed at launch. The grid may be a function of the config: `k[lambda cfg: (sf.cdiv(n, cfg["BLOCK"]),)](...)`. |
| `BLOCK: sf.constexpr` | Annotation for an argument fixed at trace time (a Python int). Kernels are traced once per distinct (argument kinds, constexpr values, block) and cached in `k.cache`; `k.tuned` holds autotune choices. |

Argument rules: device arrays become pointers; `sf.sf16(...)` scalars and
Python ints become scalar arguments; host arrays, raw floats, bools, and
non-contiguous views (a transpose or strided slice) are rejected before
launch; all arrays must be on one device; sf16 only on the simulator. A
kernel must not return a value.

### Kernel language (valid only inside a kernel)

| Name | Description |
|---|---|
| `sf.program_id(axis=0)` / `sf.block_idx(axis=0)` | Index of this program (block) in the grid. |
| `sf.thread_idx(axis=0)` | Lane index within the block, as a lane vector. |
| `sf.block_dim(axis=0)` | Lanes per block along `axis` (a Python int). |
| `sf.arange(start, end)` | `[start, end)` as an int vector; both ends Python ints or constexprs. |
| `ptr + offs`, `ptr[i]`, `ptr[i] = v` | Pointer arithmetic, load and store on pointer arguments. |
| `sf.load(ptr, mask=None, other=0)` | Gather; masked-off lanes read `other`. |
| `sf.store(ptr, value, mask=None)` | Scatter; masked-off lanes write nothing. |
| `sf.shared(shape, dtype="sf16")` | Block scratchpad; `tile[i]` and `tile[i] = v`. Zeroed per program. |
| `sf.barrier()` | Block-wide barrier. |
| `with sf.predicate(mask):` | Stores and event counts inside are masked by `mask` (and any enclosing predicate). Values are still computed on every lane. Nested predicates AND. |
| `sf.where(cond, x, y)` | Select per lane. |
| `sf.fma(a, b, c)`, `sf.sqrt(a)`, `sf.maximum(a, b)`, `sf.minimum(a, b)`, `sf.clip(x, lo, hi)`, `sf.cast(x, dtype)` | Same semantics as the host versions. |
| `sf.dot(a, b)` | 2-D tile matmul on the systolic array (sequential fma chain per output). |
| `sf.sum(x, axis=None)`, `sf.max(x, axis=None)`, `sf.min(x, axis=None)` | Tile reductions; `sum` is the sequential saturating chain. |
| `sf.zeros(shape, dtype="sf16")`, `sf.full(shape, value, dtype="sf16")` | Tile constants. |
| `v.reshape(...)`, `v[:, None]`, `v[None, :]` | Reshape and expand dims on tile values; NumPy broadcasting applies. |
| `v + w`, `v - w`, `v * w`, `v / w`, `-v`, `abs(v)`, `v < w`, `v == 0.1`, `m & n`, `m | n`, `~m`, `i // 2`, `i % 2` (ints) | Operators on traced values. Literals follow Stage 1 (quantized for arithmetic, exact for comparisons). |
| `sf.kernels.is_traced(x)` | True for a traced value. |

**Lockstep rule.** `if`, `while`, `and`, `or`, `not` on a traced value, or
a traced value in `range()`, raise `SFCompileError` at trace time. Use
`sf.where` for a value and `sf.predicate` for statements. Loops run over
Python ints or constexprs and are unrolled.

**Faults.** Out-of-bounds load / store / shared access and integer division
by zero raise `SFDeviceError`. The faulting op writes nothing; writes by
earlier ops and earlier programs stay; nothing after the fault runs.

**Result.** A launch returns `LaunchResult` with `.status` (exact
`SFStatus`), `.counters`, `.cycles`, `.grid`, `.block`, `.device`,
`.backend`, `.config`. The policy is applied to the status after the launch
in catalogue order (invalid, divide, overflow).

---

## 7. Exceptions and warnings

| Name | When |
|---|---|
| `SFOverflowError` / `SFOverflowWarning` | Result outside ±max (saturated). |
| `SFUnderflowError` / `SFUnderflowWarning` | Nonzero exact result became 0. |
| `SFZeroDivisionError` / `SFZeroDivisionWarning` | Division by an SF zero. |
| `SFInvalidError` / `SFInvalidWarning` | NaN input or sqrt of a negative. |
| `SFPreflightWarning` | Probe suspects a heavy op will saturate. |
| `SFDeviceOverflowWarning` | Upper-bound saturation count from a heavy op's scan. |
| `SFTypeError` | Bad operand, bad dtype, implicit narrowing, host array passed to a kernel. |
| `SFShapeError` | Shape mismatch, before launch. |
| `SFUnsupportedError` | Op or width not supported on the host or target. |
| `SFLaunchError` | Grid, block, tile, shared memory, buffer size or view rules violated; nothing launched. |
| `SFCompileError` | Lockstep violation, bad kernel shape, missing launch syntax, C compile failure. |
| `SFDeviceError` | Device fault, or no hardware driver attached. |
| `SFError`, `SFWarning` | Bases of all of the above. |

Static errors (`SFTypeError`, `SFShapeError`, `SFUnsupportedError`,
`SFLaunchError`, `SFCompileError`, `SFDeviceError`) ignore the policy.

---

## 8. Objects

**`SFStatus(saturated, div0, invalid, exact, outputs_at_max)`**: what the
runtime knows after a heavy op. `saturated` is an int, or `None` when
unknown; `exact` says whether the counts are exact; `outputs_at_max` is the
raw ±max scan; `status.ok` is True / False / None; `status.known` is True
when nothing is unknown.

**`LaunchResult`**: `kernel`, `grid`, `block`, `device`, `status`,
`counters`, `backend`, `config`, `cycles`.

**`Counters`**: `programs`, `ops`, `lane_ops`, `loads`, `stores`, `macs`,
`saturated`, `underflow`, `div0`, `invalid`, `barriers`; `cycles =
ceil(lane_ops / 256) + ceil(macs / 4096) + loads + stores + barriers`.

**Devices** share: `name`, `supported_bits`, `sticky_flags`,
`accumulation_order`, `accumulator_bits`, `max_elements`, `launches`,
`run(op, fmt, *operands)` for the built-ins; kernel targets add `limits`,
`check_launch`, `launch`, `alloc`, `upload`, `download`, `synchronize`,
`kernel_launches`, `last`.

**`SFLayerStatus`** (Stage 2): `name`, `op`, `placement`, `dtype`,
`outputs`, `saturated`, `exact`, `underflow`, `input_underflow`,
`weight_underflow`, `bias_saturated`, `headroom_bits`, `act_scale`.

---

## 9. Plugging in a real simulator or the hardware

This is how to run the whole test suite against something other than the
built-in simulator (an RTL simulator such as Verilator behind your own
compiler, or the chip).

### Kernels: attach a driver

```python
import sf
from sf.kernels._interp import Counters

class VerilatorDriver:
    def launch(self, ir, grid, block, buffers, scalars):
        # ir       : sf.kernels.KernelIR   (ir.dump() prints it; see below for C)
        # grid     : (gx, gy, gz)          block: (bx, by, bz)
        # buffers  : {param name: int64 NumPy array of codes}, written IN PLACE for outputs
        # scalars  : {param name: int}     (sf scalars arrive as their integer code)
        ...run it, write results into the buffers...
        return Counters(programs=..., saturated=..., underflow=..., div0=..., invalid=...)

sf.HardwareAtreides.attach(VerilatorDriver())
dev = sf.device("atreides-hw")
```

From then on `k[grid, block](...)` on arrays placed on `"atreides-hw"` goes
through the driver, and `LaunchResult.status` is built from the `Counters`
it returns. Raise `sf.SFDeviceError` from `launch` for a device fault.

To get the generated C for the compiler (sf16 only):

```python
from sf.kernels._cgen import emit_c
ir = next(iter(k.cache.values()))      # after one launch on the simulator
print(ir.dump())                       # the SF IR
print(emit_c(ir))                      # C on the __sf16_* builtins (sf16_builtins.h)
```

The emitted function is
`void sf_kernel(int64_t *arg0, int64_t arg0__size, ..., int64_t gx, gy, gz, sf_counters *C)`
with pointers first, scalars after, each pointer followed by its element
count. `src/sf/kernels/sf16_builtins.h` is the reference for what the
builtins must compute.

### Built-ins (Stage 1.1 arrays, Stage 2 models): register a device

```python
class VerilatorDevice(sf.Device):
    name = "atreides-rtl"
    supported_bits = (16,)
    sticky_flags = True            # True if you can return per-element event flags
    def run(self, op, fmt, *operands, **kw):
        # op: "add" | "sub" | "mul" | "div" | "fma" | "sqrt" | "sum" | "dot" | "matmul" | ...
        # operands: int64 code arrays; return (codes, flags) with flags a dict of bool arrays
        ...

sf.register_device(VerilatorDevice())
x = sf.asarray(values, dtype=sf.sf16, device="atreides-rtl")
```

Look at `src/sf/_device.py` (`MockAtreides`, `MockAtreidesFlags`) for the
exact `run` contract and the op names.

### Running the suite against it

The Stage 3 tests take the target from the `DEV` constant at the top of
`tests/test_3_kernels.py` and `tests/test_review_findings_3.py`; the Stage
1.1 device tests use the mock names in `tests/test_1_1_3_matmul_device.py`.
Point those at your device name and run `pytest`. Every assertion compares
integer codes and event counts, so any difference from the host reference
is a real disagreement between the hardware and the spec.
