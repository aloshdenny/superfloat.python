"""The lowering spec: one entry per SF IR op, the single source of truth.

Everything a compiler needs to lower an op lives here: operands, attributes,
result type and shape, the exact integer-code semantics, the events it can
raise, what it reads and writes in memory, how it behaves under a predicate,
how it maps to the v1 hardware, and which rewrites are legal.

`tools/gen_spec.py` turns this file into spec/ops.json, the golden vectors in
spec/golden/ and the website pages under website/compiler/. Do not edit those
by hand. tests/test_lowering_spec.py checks that every IR op the interpreter
knows has an entry here and that every golden vector still matches the
reference.

Versioning (see website/compiler/versioning.mdx):
  * adding an op, or changing what an op does (its contract), bumps the minor
    version: 1.1 -> 1.2. The op's own `version` becomes the new spec version.
  * editing an entry without changing the contract (clearer wording, a new
    edge case written down, a fixed typo) bumps the patch: 1.1.1 -> 1.1.2.
  * the major version changes only if the whole numeric model changes (for
    example a move to the v4.8.1 wide accumulator).
"""

SPEC_VERSION = "1.0.0"

CHANGELOG = [
    ("1.0.0", "2026-10-30", "First release: every SF IR op of Stage 3 and every public Python function."),
]

# --- shared text ---------------------------------------------------------------------------

TYPES = {
    "sfN": "Superfloat code, N in 4..16. Stored as a signed integer k in [-M, M], M = 2^(N-1) - 1. "
           "Value = k / 2^(N-1). Sign-magnitude on the chip; the IR carries k as a plain signed integer. "
           "There is no code -2^(N-1), no NaN, no infinity.",
    "int": "64-bit signed integer (indices, offsets, loop math). Never overflows in practice; the "
           "reference uses int64 and does not wrap-check.",
    "bool": "1-bit predicate / mask.",
    "ptr<sfN>": "Base address of a global buffer of sfN codes. Only appears as a kernel parameter; "
                "address arithmetic is `ptr + int offsets` and is folded into load/store.",
}

EVENTS = {
    "overflow": "the exact result did not fit; the result is saturated to +-M (sign of the exact result).",
    "underflow": "the exact result was nonzero but the stored result is 0.",
    "divide": "division by a zero code. Result is +-M with the sign of the dividend (0 if dividend is 0).",
    "invalid": "no real answer (sqrt of a negative). Result is 0.",
}

PREDICATION = {
    "pure": "No side effects. Under a predicate the op may run on every lane; masked lanes produce "
            "values that are never observed. Safe to speculate.",
    "events": "Value is pure, but events are only counted on live lanes (lanes where the predicate "
              "in force is true). Lowering may compute every lane but MUST mask the event flags with "
              "the predicate, or it will report events from dead lanes.",
    "memory": "Only live lanes (mask AND predicate) touch memory. Masked lanes must not read, write or fault.",
}


def _e(**kw):
    kw.setdefault("version", "1.0")
    kw.setdefault("attrs", {})
    kw.setdefault("events", [])
    kw.setdefault("memory", "none")
    kw.setdefault("faults", [])
    kw.setdefault("rewrites", [])
    kw.setdefault("notes", [])
    return kw


SF_ELEMENTWISE_SHAPE = "numpy broadcast of the operand shapes (rank <= 3); scalars are shape ()"
NO_REASSOC = ("NOT associative: (a+b)+c != a+(b+c) when an intermediate saturates. "
              "Do not reassociate, do not reorder a chain, do not tree-reduce.")

OPS = [
    # ---- kernel parameters and constants ---------------------------------------------------------
    _e(name="param", category="index", signature="%r = param {param, kind}",
       operands=[], attrs={"param": "kernel parameter name", "kind": "'ptr' | 'int' | 'sf'"},
       result="int or sfN scalar for kind int/sf; none for ptr (the buffer is bound by value id)",
       shape="()", semantics="Reads a scalar kernel argument. Pointer params produce no value; loads and stores refer to them directly.",
       predication="pure", hw_v1="kernel argument register / constant bank",
       mlir={"name": "func argument", "traits": ["Pure"]},
       python=["kernel signature"]),
    _e(name="const", category="index", signature="%r = const {value, kind}",
       operands=[], attrs={"value": "Python int / code / bool", "kind": "'int' | 'sf' | 'bool'"},
       result="int, sfN or bool, filled to the result shape", shape="from the result type",
       semantics="A literal. For sf the value is already a code (quantized at trace time with round half to even, through the policy).",
       predication="pure", hw_v1="immediate", mlir={"name": "sf.constant / arith.constant", "traits": ["Pure", "ConstantLike"]},
       python=["Python literals inside a kernel"]),
    _e(name="program_id", category="index", signature="%r:int = program_id {axis}",
       operands=[], attrs={"axis": "0, 1 or 2"}, result="int", shape="()",
       semantics="Index of this program (block) in the launch grid on that axis.",
       predication="pure", hw_v1="tile / block id register", mlir={"name": "gpu.block_id", "traits": ["Pure"]},
       python=["sf.program_id(axis)", "sf.block_idx(axis)"]),
    _e(name="thread_idx", category="index", signature="%r:int[...] = thread_idx {axis}",
       operands=[], attrs={"axis": "0, 1 or 2"}, result="int vector",
       shape="block_dim laid out on the used axes (rank = highest axis with block_dim > 1)",
       semantics="Lane index along axis. @sf.kernel only. The IR is lockstep: the whole block is one vector.",
       predication="pure", hw_v1="lane id (8 threads per tile in v1)", mlir={"name": "gpu.thread_id", "traits": ["Pure"]},
       python=["sf.thread_idx(axis)"]),
    _e(name="arange", category="index", signature="%r:int[n] = arange {start, end}",
       operands=[], attrs={"start": "constexpr int", "end": "constexpr int, end > start"},
       result="int vector", shape="(end - start,)", semantics="start, start+1, ..., end-1.",
       predication="pure", hw_v1="lane id + start", mlir={"name": "sf.arange (or tensor.generate)", "traits": ["Pure"]},
       python=["sf.arange(start, end)"]),

    # ---- shape ops ----------------------------------------------------------------------------
    _e(name="broadcast", category="shape", signature="%r = broadcast %x {shape}",
       operands=["x: any type"], attrs={"shape": "target static shape"}, result="same type as x",
       shape="attr shape (numpy broadcast rules from x.shape)", semantics="Repeat x to the target shape. No numeric change.",
       predication="pure", hw_v1="register replication / free", mlir={"name": "tensor broadcast / linalg.broadcast", "traits": ["Pure"]},
       python=["sf.full", "sf.zeros", "implicit broadcasting"]),
    _e(name="reshape", category="shape", signature="%r = reshape %x", operands=["x"], result="same type",
       shape="result shape (same element count)", semantics="Row-major reshape. No numeric change.",
       predication="pure", hw_v1="free (layout only)", mlir={"name": "tensor.reshape", "traits": ["Pure"]},
       python=["x.reshape(...)"]),
    _e(name="transpose", category="shape", signature="%r = transpose %x", operands=["x: rank 2"], result="same type",
       shape="(n, m) from (m, n)", semantics="2-D transpose.", predication="pure", hw_v1="data movement",
       mlir={"name": "linalg.transpose", "traits": ["Pure"]}, python=["x.T"]),

    # ---- integer ops --------------------------------------------------------------------------
    *[_e(name=n, category="int", signature=f"%r:int = {n} %a, %b", operands=["a: int", "b: int"], result="int",
         shape="broadcast(a, b)", semantics=s, predication="pure", hw_v1="integer ALU",
         mlir={"name": m, "traits": ["Pure"] + (["Commutative"] if c else [])}, python=[p],
         rewrites=["ordinary integer algebra is legal (no saturation)"])
      for n, s, m, c, p in [
          ("iadd", "a + b", "arith.addi", True, "int + int"),
          ("isub", "a - b", "arith.subi", False, "int - int"),
          ("imul", "a * b", "arith.muli", True, "int * int"),
      ]],
    _e(name="idiv", category="int", signature="%r:int = idiv %a, %b", operands=["a: int", "b: int"], result="int",
       shape="broadcast(a, b)", semantics="FLOOR division (Python //), not truncation: -7 // 2 = -4.",
       predication="pure", faults=["any lane with b == 0 faults the whole launch (SFDeviceError)"],
       hw_v1="integer ALU", mlir={"name": "arith.floordivsi", "traits": []}, python=["int // int", "sf.cdiv"],
       notes=["use arith.floordivsi, not arith.divsi", "the zero check covers every lane of the broadcast, live or not"]),
    _e(name="imod", category="int", signature="%r:int = imod %a, %b", operands=["a: int", "b: int"], result="int",
       shape="broadcast(a, b)", semantics="Python modulo: result has the sign of b. -7 % 2 = 1.",
       predication="pure", faults=["b == 0 in any lane faults the launch"], hw_v1="integer ALU",
       mlir={"name": "arith.remsi + sign fix (floor semantics)", "traits": []}, python=["int % int"]),
    _e(name="ineg", category="int", signature="%r:int = ineg %a", operands=["a: int"], result="int", shape="a",
       semantics="-a", predication="pure", hw_v1="integer ALU", mlir={"name": "arith.subi 0, a", "traits": ["Pure"]}, python=["-int"]),
    _e(name="iabs", category="int", signature="%r:int = iabs %a", operands=["a: int"], result="int", shape="a",
       semantics="|a|", predication="pure", hw_v1="integer ALU", mlir={"name": "math.absi", "traits": ["Pure"]}, python=["abs(int)"]),
    _e(name="icmp", category="int", signature="%r:bool = icmp %a, %b {op}", operands=["a: int", "b: int"],
       attrs={"op": "lt | le | gt | ge | eq | ne"}, result="bool", shape="broadcast(a, b)", semantics="Integer compare.",
       predication="pure", hw_v1="integer compare", mlir={"name": "arith.cmpi", "traits": ["Pure"]}, python=["offs < n etc."]),
    _e(name="bool_to_int", category="int", signature="%r:int = bool_to_int %a", operands=["a: bool"], result="int",
       shape="a", semantics="false -> 0, true -> 1", predication="pure", hw_v1="select", mlir={"name": "arith.extui", "traits": ["Pure"]},
       python=["using a mask in int math"]),

    # ---- bool ops -----------------------------------------------------------------------------
    *[_e(name=n, category="bool", signature=f"%r:bool = {n} %a" + ("" if n == "not" else ", %b"),
         operands=["a: bool"] + ([] if n == "not" else ["b: bool"]), result="bool",
         shape="a" if n == "not" else "broadcast(a, b)", semantics=s, predication="pure", hw_v1="predicate logic",
         mlir={"name": m, "traits": ["Pure"] + ([] if n == "not" else ["Commutative"])}, python=[p])
      for n, s, m, p in [
          ("and", "a AND b. Also emitted to combine nested predicates.", "arith.andi", "m1 & m2, nested `with sf.predicate`"),
          ("or", "a OR b", "arith.ori", "m1 | m2"),
          ("xor", "a XOR b", "arith.xori", "m1 ^ m2"),
          ("not", "NOT a", "arith.xori a, true", "~m"),
      ]],
    _e(name="where", category="select", signature="%r = where %c, %x, %y", operands=["c: bool", "x", "y (same type as x)"],
       result="type of x", shape="broadcast(c, x, y)", semantics="x where c else y, per lane. Exact, no events.",
       predication="pure", hw_v1="select / predicated move", mlir={"name": "arith.select", "traits": ["Pure"]},
       python=["sf.where(cond, x, y)"]),

    # ---- sf arithmetic ------------------------------------------------------------------------
    _e(name="add", category="sf", signature="%r:sfN = add %a, %b", operands=["a: sfN", "b: sfN (same N)"], result="sfN",
       shape=SF_ELEMENTWISE_SHAPE,
       semantics="s = a + b (exact integer). If |s| > M: result = sign(s)*M, overflow. Else result = s.",
       events=["overflow"], predication="events", hw_v1="ACT unit (add)",
       mlir={"name": "sf.add", "traits": ["Pure", "Commutative", "Elementwise", "SameOperandsAndResultType"]},
       python=["a + b"],
       rewrites=["commutative: a+b = b+a (legal)", NO_REASSOC, "x + 0 -> x (legal)", "never turn a+a into 2*a via mul (mul cannot represent 2)"],
       notes=["sfN < 16 on the sf16 datapath: shift left by 16-N, add, saturate at the sfN max shifted, shift back. "
              "The saturation threshold must be sfN's M, not sf16's."]),
    _e(name="sub", category="sf", signature="%r:sfN = sub %a, %b", operands=["a: sfN", "b: sfN"], result="sfN",
       shape=SF_ELEMENTWISE_SHAPE, semantics="s = a - b; saturate like add.", events=["overflow"], predication="events",
       hw_v1="ACT unit (sub)", mlir={"name": "sf.sub", "traits": ["Pure", "Elementwise", "SameOperandsAndResultType"]},
       python=["a - b"], rewrites=["a - b -> a + neg(b) is legal (neg is exact since the range is symmetric)", NO_REASSOC]),
    _e(name="mul", category="sf", signature="%r:sfN = mul %a, %b", operands=["a: sfN", "b: sfN"], result="sfN",
       shape=SF_ELEMENTWISE_SHAPE,
       semantics="p = a * b (exact, up to 2(N-1)+1 bits). result = trunc_toward_zero(p / 2^(N-1)). "
                 "underflow if p != 0 and result == 0. |result| < M always, so mul never overflows.",
       events=["underflow"], predication="events", hw_v1="FMA unit (c = 0)",
       mlir={"name": "sf.mul", "traits": ["Pure", "Commutative", "Elementwise", "SameOperandsAndResultType"]},
       python=["a * b"],
       rewrites=["commutative (legal)", "NOT associative: truncation after each product", "NOT distributive over add",
                 "x * 0 -> 0 (legal, no event)", "there is no 1.0, so there is no multiplicative identity to fold"],
       notes=["truncation is toward zero, NOT floor: an arithmetic right shift on a negative product is wrong. "
              "Shift the magnitude, then reapply the sign."]),
    _e(name="div", category="sf", signature="%r:sfN = div %a, %b", operands=["a: sfN (dividend)", "b: sfN (divisor)"],
       result="sfN", shape=SF_ELEMENTWISE_SHAPE,
       semantics="b == 0: divide event, result = sign(a)*M (0 if a == 0). "
                 "|a| >= |b|: overflow, result = sign(a)*sign(b)*M. "
                 "Else result = trunc_toward_zero(a * 2^(N-1) / b).",
       events=["divide", "overflow"], predication="events",
       hw_v1="NOT SUPPORTED on v1 (no divider). Device.supported_ops excludes it; the runtime raises SFUnsupportedError. "
             "Host pre-flight decides every b == 0 lane before launch.",
       mlir={"name": "sf.div", "traits": ["Elementwise", "SameOperandsAndResultType"]},
       python=["a / b", "sf.div(a, b, where=mask)"],
       rewrites=["no rewrites; in particular a/b is not a * (1/b) (1/b always overflows)"],
       notes=["the event check order is divide first, then overflow; a lane with b == 0 reports divide only",
              "sf.div(where=mask) lowers to a predicated div: masked lanes give 0 and no event"]),
    _e(name="fma", category="sf", signature="%r:sfN = fma %a, %b, %c", operands=["a: sfN", "b: sfN", "c: sfN"], result="sfN",
       shape="broadcast(a, b, c)",
       semantics="t = mul(a, b) (truncated, may underflow); result = add(t, c) (may saturate). "
                 "Bit-identical to add(mul(a, b), c). This is NOT a fused single-rounding fma.",
       events=["overflow", "underflow"], predication="events", hw_v1="FMA unit",
       mlir={"name": "sf.fma", "traits": ["Pure", "Elementwise", "SameOperandsAndResultType"]},
       python=["sf.fma(a, b, c)", "a * b + c"],
       rewrites=["add(mul(a,b),c) <-> fma(a,b,c) is legal both ways (same bits)", "commutative in a and b"],
       notes=["on v4.8.1 hardware (wide accumulator, single rounding) this would change: major version bump"]),
    _e(name="neg", category="sf", signature="%r:sfN = neg %a", operands=["a: sfN"], result="sfN", shape="a",
       semantics="-a. Exact (range is symmetric), no events.", predication="pure", hw_v1="integer op (sign flip)",
       mlir={"name": "sf.neg", "traits": ["Pure", "Elementwise", "Involution"]}, python=["-a"],
       rewrites=["neg(neg(x)) -> x"]),
    _e(name="abs", category="sf", signature="%r:sfN = abs %a", operands=["a: sfN"], result="sfN", shape="a",
       semantics="|a|. Exact, no events.", predication="pure", hw_v1="integer op (clear sign)",
       mlir={"name": "sf.abs", "traits": ["Pure", "Elementwise", "Idempotent"]}, python=["abs(a)"]),
    _e(name="sqrt", category="sf", signature="%r:sfN = sqrt %a", operands=["a: sfN"], result="sfN", shape="a",
       semantics="a < 0: invalid, result 0. Else result = isqrt(a * 2^(N-1)) (floor of the exact root). Never overflows.",
       events=["invalid"], predication="events",
       hw_v1="NOT SUPPORTED on v1 (no sqrt unit); runtime raises SFUnsupportedError",
       mlir={"name": "sf.sqrt", "traits": ["Elementwise", "SameOperandsAndResultType"]}, python=["sf.sqrt(a)"]),
    _e(name="widen", category="cast", signature="%r:sfM = widen %a {bits}", operands=["a: sfN, N < M"],
       attrs={"bits": "target width M"}, result="sfM", shape="a",
       semantics="result = a << (M - N). Exact, no events.", predication="pure", hw_v1="shift",
       mlir={"name": "sf.cast (widening)", "traits": ["Pure", "Elementwise"]}, python=["sf.cast(x, wider)", "x.astype(wider)"]),
    _e(name="narrow", category="cast", signature="%r:sfM = narrow %a {bits}", operands=["a: sfN, N > M"],
       attrs={"bits": "target width M"}, result="sfM", shape="a",
       semantics="q = round_half_to_even(a / 2^(N-M)) (floor divmod, then round up if remainder > half, or == half and q odd). "
                 "If |q| > M_max: overflow, clip to +-M_max. underflow if a != 0 and result == 0.",
       events=["overflow", "underflow"], predication="events", hw_v1="shift + round (ACT)",
       mlir={"name": "sf.cast (narrowing)", "traits": ["Pure", "Elementwise"]}, python=["sf.cast(x, narrower)", "x.astype(narrower)"],
       notes=["this is the ONLY op that rounds to nearest. All arithmetic truncates toward zero."]),
    _e(name="cmp", category="sf", signature="%r:bool = cmp %a, %b {op}", operands=["a: sfN", "b: sfN"],
       attrs={"op": "lt | le | gt | ge | eq | ne"}, result="bool", shape="broadcast(a, b)",
       semantics="Integer compare of the codes (exact).", predication="pure", hw_v1="integer compare",
       mlir={"name": "arith.cmpi on codes", "traits": ["Pure"]}, python=["a < b", "a == b", "..."]),
    _e(name="cmp_lit", category="sf", signature="%r:bool = cmp_lit %a {op, value}", operands=["a: sfN"],
       attrs={"op": "lt | le | gt | ge | eq | ne", "value": "int code threshold, decided at trace time"},
       result="bool", shape="a",
       semantics="Compare against a Python literal that may not be representable (e.g. x < 1). The tracer turns the "
                 "exact literal into an equivalent integer threshold, so the op is a plain integer compare.",
       predication="pure", hw_v1="integer compare with immediate", mlir={"name": "arith.cmpi with constant", "traits": ["Pure"]},
       python=["x < 0.3", "x > 1"]),
    _e(name="maximum", category="select", signature="%r = maximum %a, %b", operands=["a: sfN", "b: sfN"], result="sfN",
       shape="broadcast(a, b)", semantics="max of codes. Exact.", predication="pure", hw_v1="integer max",
       mlir={"name": "arith.maxsi", "traits": ["Pure", "Commutative"]}, python=["sf.maximum", "ReLU = maximum(x, 0)"],
       rewrites=["associative and commutative (legal to reorder)"]),
    _e(name="minimum", category="select", signature="%r = minimum %a, %b", operands=["a: sfN", "b: sfN"], result="sfN",
       shape="broadcast(a, b)", semantics="min of codes. Exact.", predication="pure", hw_v1="integer min",
       mlir={"name": "arith.minsi", "traits": ["Pure", "Commutative"]}, python=["sf.minimum", "sf.clip"],
       rewrites=["associative and commutative"]),

    # ---- reductions and contraction -----------------------------------------------------------
    _e(name="sum", category="reduction", signature="%r:sfN = sum %x {axis}", operands=["x: sfN"],
       attrs={"axis": "int or None (None = all)"}, result="sfN", shape="x.shape with axis removed (() for None)",
       semantics="acc = 0; for i = 0 .. n-1 in index order: acc = add(acc, x[i]) (saturating each step). "
                 "Overflow if any step saturated. Saturation is sticky in the sense that later steps continue from +-M.",
       events=["overflow"], predication="events", hw_v1="ACT chain, sequential",
       mlir={"name": "sf.reduce_sum", "traits": ["Pure"]}, python=["sf.sum(x, axis)", "x.sum()"],
       rewrites=[NO_REASSOC, "result DEPENDS on order: [M, 1, -1] gives M-1, [1, -1, M] gives M"]),
    _e(name="max", category="reduction", signature="%r = max %x {axis}", operands=["x: sfN"], attrs={"axis": "int or None"},
       result="sfN", shape="axis removed", semantics="max along axis. Exact.", predication="pure", hw_v1="integer max tree",
       mlir={"name": "sf.reduce_max", "traits": ["Pure"]}, python=["sf.max"], rewrites=["any order / tree is legal"]),
    _e(name="min", category="reduction", signature="%r = min %x {axis}", operands=["x: sfN"], attrs={"axis": "int or None"},
       result="sfN", shape="axis removed", semantics="min along axis. Exact.", predication="pure", hw_v1="integer min tree",
       mlir={"name": "sf.reduce_min", "traits": ["Pure"]}, python=["sf.min"], rewrites=["any order / tree is legal"]),
    _e(name="dot", category="contraction", signature="%r:sfN[m,n] = dot %a, %b", operands=["a: sfN[m, k]", "b: sfN[k, n]"],
       result="sfN", shape="(m, n)",
       semantics="For each (i, j): acc = 0; for t = 0 .. k-1: acc = fma(a[i,t], b[t,j], acc). "
                 "Every product truncated, every add saturated, at width N, in increasing t.",
       events=["overflow", "underflow"], predication="events", hw_v1="8x8 systolic array per tile (4 tiles)",
       mlir={"name": "sf.matmul (linalg.matmul with sf semantics)", "traits": ["Pure"]},
       python=["sf.dot(a, b) in a kernel", "a @ b", "sf.matmul", "sf.addmm"],
       rewrites=["k must be accumulated in increasing order. Tiling over m and n is free; tiling over k is legal ONLY "
                 "if the partial sum of tile 1 is fed as the initial acc of tile 2 (no separate partials added later)",
                 "split-k, tree reduction, and wider accumulators are NOT legal"],
       notes=["addmm(c, a, b) = acc starts at c, not 0, then the same chain"]),

    # ---- memory -------------------------------------------------------------------------------
    _e(name="load", category="memory", signature="%r:sfN = load %ptr, %offs, %mask, %other",
       operands=["ptr: ptr<sfN> param", "offs: int", "mask: bool", "other: sfN (value for masked lanes)"],
       result="sfN", shape="broadcast(offs, mask, other)",
       semantics="live = mask AND predicate. Live lanes: ptr[offs]. Others: other. Offsets are element indices, not bytes.",
       memory="READ global buffer ptr at offs on live lanes only", predication="memory",
       faults=["any live lane with offs < 0 or offs >= buffer size faults the launch. Masked lanes never fault."],
       hw_v1="global load (64K-word data window)", mlir={"name": "sf.load (masked, like triton.load)", "traits": ["MemRead"]},
       python=["sf.load(ptr + offs, mask=, other=)"]),
    _e(name="store", category="memory", signature="store %ptr, %offs, %val, %mask",
       operands=["ptr: ptr<sfN>", "offs: int", "val: sfN", "mask: bool"], result="none",
       shape="lanes = broadcast(offs, val, mask)",
       semantics="live = mask AND predicate. ptr[offs] = val on live lanes. If two live lanes hit the same offset, "
                 "the reference keeps the last lane in row-major order; kernels should not rely on this.",
       memory="WRITE global buffer ptr at offs on live lanes only", predication="memory",
       faults=["a live lane out of bounds faults the launch. Earlier writes are NOT rolled back: after a fault the buffers are undefined."],
       hw_v1="global store", mlir={"name": "sf.store", "traits": ["MemWrite"]}, python=["sf.store(ptr + offs, val, mask=)"],
       notes=["two programs writing the same address is a race; the result is undefined (the simulator runs programs in order)"]),
    _e(name="shared_load", category="memory", signature="%r:sfN = shared_load %idx, %mask {buf}",
       operands=["idx: int", "mask: bool"], attrs={"buf": "shared buffer id (declared on the kernel)"}, result="sfN",
       shape="broadcast(idx, mask)", semantics="Live lanes read shared[buf][idx]; masked lanes give 0.",
       memory="READ block-local shared buffer", predication="memory", faults=["live lane out of bounds faults"],
       hw_v1="tile-local SRAM", mlir={"name": "memref.load on workgroup memory", "traits": ["MemRead"]},
       python=["buf[idx] where buf = sf.shared(shape)"]),
    _e(name="shared_store", category="memory", signature="shared_store %idx, %val, %mask {buf}",
       operands=["idx: int", "val: sfN", "mask: bool"], attrs={"buf": "shared buffer id"}, result="none",
       shape="broadcast(idx, val, mask)", semantics="Live lanes write shared[buf][idx] = val.",
       memory="WRITE block-local shared buffer. Shared memory is zeroed at the start of every program.",
       predication="memory", faults=["live lane out of bounds faults"], hw_v1="tile-local SRAM",
       mlir={"name": "memref.store on workgroup memory", "traits": ["MemWrite"]}, python=["buf[idx] = val"]),
    _e(name="barrier", category="sync", signature="barrier", operands=[], result="none", shape="-",
       semantics="All lanes of the block reach this point before any continues. In lockstep IR this orders shared-memory "
                 "accesses; nothing may be moved across it.",
       memory="orders shared memory", predication="pure", hw_v1="tile barrier",
       mlir={"name": "gpu.barrier", "traits": ["MemRead", "MemWrite (barrier)"]}, python=["sf.barrier()"],
       notes=["must not be placed under a divergent predicate"]),
]

# --- the public Python API and how each function reaches the hardware -------------------------
# placement: device = heavy op, runs on the target and returns status; host = runs on the host always;
# trace = only valid inside @sf.jit / @sf.kernel and becomes IR ops.

PUBLIC = [
    # name, placement, lowers to, notes
    ("sf.sf4 .. sf.sf16 (scalar constructors)", "host", "const", "quantize with round half to even, through the policy"),
    ("sf.array / sf.asarray / sf.full / sf.zeros / sf.ones", "host (+ upload)", "const / broadcast in kernels", "device= uploads codes"),
    ("sf.from_code / sf.from_codes / sf.from_bits", "host", "-", "build from raw codes, no rounding"),
    ("+ - * /", "device if an operand is on a device", "add / sub / mul / div", "scalar and array"),
    ("sf.fma", "device", "fma", ""),
    ("sf.div(where=)", "device", "div under predicate", ""),
    ("sf.sqrt", "device", "sqrt", ""),
    ("sf.sum", "device", "sum", "sequential"),
    ("sf.dot / @ / sf.matmul", "device", "dot", "batched matmul = one dot per batch element"),
    ("sf.addmm(c, a, b)", "device", "dot with acc initialised to c", ""),
    ("sf.max / sf.min / sf.argmax / sf.argmin", "host", "max / min", "argmax/argmin: first index of the extreme"),
    ("sf.maximum / sf.minimum / sf.clip / sf.where", "host", "maximum / minimum / where", "exact, no events"),
    ("sf.cast / astype", "host", "widen / narrow", ""),
    ("-x, abs(x)", "host or device", "neg / abs", "exact"),
    ("comparisons", "host", "cmp / cmp_lit", "return numpy bool"),
    ("indexing, reshape, .T", "host", "reshape / transpose", ""),
    ("sf.errstate / seterr / geterr / config / getconfig", "host", "-", "policy only, never changes bits"),
    ("sf.info / sf.dtype / sf.dtypes", "host", "-", "format metadata"),
    ("sf.device / devices / get_device / register_device / synchronize", "host", "-", "runtime"),
    ("@sf.jit / @sf.kernel / sf.autotune / sf.constexpr", "trace", "builds a KernelIR", ""),
    ("sf.program_id / block_idx / thread_idx / block_dim / arange / cdiv", "trace", "program_id / thread_idx / arange / idiv", "block_dim is a trace-time int"),
    ("sf.load / sf.store", "trace", "load / store", ""),
    ("sf.shared / sf.barrier", "trace", "shared_load / shared_store / barrier", ""),
    ("sf.predicate (with block)", "trace", "and (predicate combine), then every op inside carries pred", ""),
    ("sf.torch.convert / backend='superfloat'", "host + device", "dot (Linear/Conv2d as matmul) + add", "per-layer scale chosen on host"),
]

GLOBAL_RULES = [
    "Static shapes. Every value has a static shape of rank <= 3. No dynamic shapes.",
    "Lockstep. A program (block) is one vector of lanes. Control flow is if-converted: `if` on a lane value becomes a predicate carried on each op (`pred=`).",
    "Ops with a predicate: values on dead lanes are don't-care, but events and memory effects must be masked.",
    "Events never change the bits. The policy (raise / warn / ignore) is applied by the host after the launch from the event counts.",
    "Arithmetic truncates toward zero; only narrow rounds (half to even). Add/sub/sum saturate; mul underflows; div overflows when |a| >= |b|.",
    "No reassociation of add, sum, or the k loop of dot. No distributivity. No float math anywhere: lower on integer codes.",
    "Faults (out of bounds, integer division by zero) abort the launch with SFDeviceError and leave buffers undefined.",
    "Host pre-flight: before a launch the host decides everything it can (a zero divisor, an op that must saturate everywhere, an op that cannot saturate). The compiler never needs to handle those cases specially, but must still produce correct bits for them.",
    "Width: sfN for N < 16 runs on the sf16 datapath shifted left by 16-N and must be re-truncated (low bits masked) after every op so the result equals true sfN.",
]
