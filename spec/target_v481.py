"""How each SF IR op lowers to Atreides v4.8.1, and the target facts a compiler needs.

Source: Alosh's compiler target reference ``compiler_v4.8.1.md`` (checked by
him against the RTL, branch v4.8.1-openframe).  Every statement cites its
section as "ISA sN".  Where the ISA says nothing, the entry says so.  Text
marked "(suggested)" is a lowering strategy we propose, not an ISA statement;
"(derived)" follows from the ISA's encoding but is not stated there.  Section
8 of that doc (README disagreements) is not used.

Verdicts:
  native    one instruction (plus at most a constant it needs)
  expand    a fixed sequence of instructions, given here
  limited   lowers only in some cases; the cases are given
  host      no instruction or sequence on the chip: run it on the host
  none      no lowering on v4.8.1: the compiler must reject it

Registers below: Z holds 0 (R0..R12 are 0 at block start, ISA s4.1), S holds
0x8000 (``CONST -128`` then 8 doublings with ``ADD x,x,x``, ISA s5).
"""

VERDICTS = ("native", "expand", "limited", "host", "none")

# --- machine limits (ISA s1, s4) ---------------------------------------------------------------

LIMITS = [
    ("Tiles", "4; each runs one block at a time", "ISA s1.1"),
    ("Threads per block", "8 (R15 = 0..7). The DSL allows up to 1024 lanes per block", "ISA s1.1"),
    ("Threads per launch", "1 to 255, so at most 32 blocks; the last block may be partial", "ISA s4.2"),
    ("Grid", "1-D only: R13 = block index 0..31, restarting at 0 every launch", "ISA s1.3, s4.2"),
    ("Registers", "R0..R12 read/write (13), R13..R15 read-only; 16 bits each", "ISA s1.3"),
    ("Integers", "16 bits, wrapping mod 2^16. The DSL's int is int64; wider than 16 bits is unsupported on the chip", "ISA s3.1, s5"),
    ("Data memory", "64K 16-bit words, word-addressed; 0x0000..0xFFBF is host memory", "ISA s1.5"),
    ("Scratchpad", "64 words at 0xFFC0..0xFFFF per tile; never cleared, survives blocks and launches", "ISA s1.5"),
    ("Program", "at most 512 words; PC wraps at 512; every path must end in RET", "ISA s1.4, s4.2"),
    ("Instruction cache", "32 entries shared by all tiles: keep hot loops to 32 instructions or fewer", "ISA s1.4"),
    ("Control flow", "lockstep, no divergence: only thread 0's NZP steers a branch; others silently follow", "ISA s1.2"),
    ("Flags", "only NZP, written only by CMP; no overflow or underflow flag", "ISA s1.1, s3.1"),
    ("Bounds checks", "none: an address is used as is; 0xFFC0 and up silently hits the scratchpad", "ISA s1.5"),
    ("Hazards", "none: each instruction completes before the next is fetched; no NOPs needed", "ISA s1.2, s3.3"),
]

# --- how DSL kernel concepts map onto the chip ----------------------------------------------

MAPPING = [
    ("program_id(0)", "R13 (%blockIdx). Axes 1 and 2 do not exist: flatten the grid into axis 0, and pass a block offset through memory for grids over 32 blocks", "ISA s1.3, s4.2"),
    ("thread_idx(0)", "R15 (%threadIdx). Axes 1 and 2 do not exist: flatten", "ISA s1.3"),
    ("A block of L lanes (@sf.jit vector, @sf.kernel block)", "8 hardware threads. L <= 8: one lane per thread, the rest disabled. L > 8 (suggested): each thread handles lanes j, j+8, j+16, ... in sequence (strip-mining)", "ISA s1.2, s4.2"),
    ("Kernel parameters", "No argument registers. Lower to LDR from a parameter block at a fixed data address, indexed by R13 if per block (GPT-2 puts a 4-word block table at word 0)", "ISA s4.1, s7"),
    ("Pointers and offsets", "16-bit word addresses: offsets are already element indices, one word per sf16 value. Buffers must lie in 0x0000..0xFFBF, and a range compared with CMP must not cross 0x8000 (CMP is signed)", "ISA s1.5, s4.3"),
    ("sfN values", "16-bit sign-magnitude words, sfN left-aligned; 0x8000 reads as 0 and results never produce it. No conversion on load or store", "ISA s3.2"),
    ("int values", "i16, wrapping. Keep every int in -32768..32767", "ISA s3.1"),
    ("bool values", "Cannot be stored: nothing writes a comparison to a register. A condition exists only as NZP after CMP, and only thread 0's steers a branch", "ISA s5"),
    ("Masks on load and store", "No predication. A tail mask offs < n lowers to launching exactly n threads, so the spare threads are disabled and touch no memory. Any other mask has no lowering", "ISA s1.2, s4.2"),
    ("Predicates (sf.predicate, if on lane values)", "No predication. Only SF values can be if-converted, arithmetically with ReLU (ISA s5); everything else must be uniform", "ISA s5, s7"),
    ("Unrolled loops", "The SF IR is straight-line: Python loops are unrolled when a kernel is traced. With 512 program words the compiler must re-roll loops into BR loops (uniform trip counts only)", "ISA s1.4, s4.2"),
    ("Kernel entry and exit", "Emit CMP R0, R0 first so an unconditional BRnzp is taken; end every path with RET", "ISA s3.1, s4.2"),
    ("Events (overflow, underflow, divide, invalid)", "The chip has no flags, so the compiler emits nothing for them. The host gets counts by replay (sf.config(status=\"replay\"))", "ISA s1.1"),
    ("Spills", "13 registers. Spill to the scratchpad, 8 words per thread at 0xFFC0 + 8*tid + k", "ISA s7"),
]

# --- register constraints ---------------------------------------------------------------------

REGISTERS = [
    ("ACT f=00 (SF add)", "destination must be R0, R4, R8 or R12", "the function code is the destination's low 2 bits"),
    ("ACT f=01 (ReLU)", "destination must be R1, R5 or R9", "R13 would be read-only, so its write is dropped"),
    ("ACT f=10 (leaky ReLU)", "destination must be R2, R6 or R10", ""),
    ("ACT f=11 (clipped ReLU, = ReLU in sf16)", "destination must be R3, R7 or R11", ""),
    ("FMA", "Rd is the accumulator in AND out: it overwrites the addend", "copy the addend first if it is still live"),
    ("R13..R15", "read-only; writes are silently dropped", ""),
    ("MX POP, PULL, STAT", "the only MX ops that write Rd", ""),
]

# --- cost (ISA s1.2, s1.5, s7) ---------------------------------------------------------------

COST = [
    ("Any instruction", "about 10 cycles (unpipelined: fetch 5, then 1 each for decode, request, wait, execute, update)", "derived"),
    ("FMA", "about 11 cycles (2 execute cycles)", "derived"),
    ("LDR / STR to host memory", "about 10 + 8 x 8 cycles for 8 threads on an idle bus: each thread is a separate bus transaction, no coalescing; a write is 2 beats", "measured: about 7.7 cycles per beat"),
    ("LDR / STR to the scratchpad", "about 3 cycles per thread read, 2 per write, serialised", "derived"),
    ("Instruction cache miss", "one bus read, shared with data traffic", "ISA s1.4"),
    ("MX MAC", "one instruction does 64 MACs (8 x 8)", "ISA s3.4"),
    ("SF max or min", "16 instructions, about 160 cycles", "derived"),
]

# --- per-op lowering ---------------------------------------------------------------------------
# verdict, lowering (instruction sequence), cost

OPS_V481 = {
    "param": ("expand", "LDR Rd, [param block + i]: one load per parameter (no argument registers, ISA s4.1)",
              "one external LDR"),
    "const": ("native", "int or code in -128..127: CONST Rd, imm (sign-extended). Larger: CONST then ADD x,x,x doublings "
                        "plus adds per set bit (build_large_const, ISA s7), or LDR from a constant table. An sf constant is "
                        "its 16-bit sign-magnitude word, built the same way. 0xFFC0 is one CONST 0xC0",
              "1 instruction, or about 1 + log2 + 2 per set bit"),
    "program_id": ("limited", "axis 0: read R13 directly (no instruction). Axes 1, 2: not on the chip; flatten the grid. "
                              "At most 32 blocks per launch", "free"),
    "thread_idx": ("limited", "axis 0: read R15 directly. Axes 1, 2: flatten. More than 8 lanes: strip-mine (see the mapping)",
                   "free"),
    "arange": ("expand", "lanes <= 8: ADD Rd, R15, Rstart. More lanes: the same per strip-mined chunk, plus 8*chunk",
               "1 instruction per chunk"),
    "broadcast": ("expand", "a value every thread computes the same way needs nothing. Moving one lane's value to all "
                            "threads (suggested): STR it to the scratchpad, then every thread LDRs that word", "free, or 2 scratchpad ops"),
    "reshape": ("native", "layout only: no instruction when the lane-to-thread mapping is unchanged. Otherwise go through "
                          "the scratchpad", "free"),
    "transpose": ("expand", "no shuffle instruction. Suggested: STR to the scratchpad in one layout, LDR in the other (64 words per tile)",
                  "2 scratchpad ops per element per thread"),
    "iadd": ("native", "ADD Rd, Rs, Rt (mod 2^16)", "1 instruction"),
    "isub": ("native", "SUB Rd, Rs, Rt (mod 2^16)", "1 instruction"),
    "imul": ("limited", "both operands provably < 256: MUL (8 x 8 unsigned; high bytes are ignored). Otherwise the "
                        "16 x 16 sequence lo*lo + ((hi*lo + lo*hi) << 8), about 17 instructions (ISA s5)",
             "1 or about 17 instructions"),
    "idiv": ("limited", "non-negative dividend and divisor 1, 2, 4, 8, 16, 32 or 64: DIV (a logical shift of all 16 bits, ISA s3.1). "
                        "Divisor 3, 9, 10, 11 or 12 with dividend < 256: DIV. Anything else: no lowering, and DIV silently "
                        "returns 0, so precompute on the host. Negative dividends break (derived): DIV shifts logically, idiv floors",
             "1 instruction"),
    "imod": ("limited", "non-negative x mod 2^k, k <= 6: x - ((x >> k) << k), using DIV for >> and ADD doubling for << "
                        "(ISA s5). Other divisors: no lowering", "about k + 3 instructions"),
    "ineg": ("native", "SUB Rd, Z, Rs", "1 instruction"),
    "iabs": ("none", "no instruction, no select, and no sequence in the ISA. Avoid it, or compute on the host", "-"),
    "icmp": ("limited", "only as a branch condition that is uniform across the block: CMP Ra, Rb sets NZP (signed: lt, eq, "
                        "gt; le, ge, ne by the BR mask), then BR. The result cannot be put in a register. Unsigned compare: "
                        "add 0x8000 to both operands first", "1 instruction plus the branch"),
    "bool_to_int": ("none", "nothing writes NZP to a register (ISA s5)", "-"),
    "and": ("none", "no AND instruction (ISA s5). Two uniform conditions can be combined with branch structure instead",
            "-"),
    "or": ("none", "no OR instruction (ISA s5). Uniform conditions: branch structure", "-"),
    "xor": ("none", "no XOR instruction (ISA s5)", "-"),
    "not": ("none", "no NOT instruction (ISA s5). A uniform condition: branch on the inverse NZP mask", "-"),
    "where": ("limited", "only for SF values, if-converted arithmetically with ReLU (ISA s5). A general select on a mask: "
                         "no lowering", "a few ACT instructions"),
    "add": ("native", "ACT f=00 Rd, Ra, Rb with Rd in {R0, R4, R8, R12}. Not integer ADD, which is wrong for negative "
                      "sign-magnitude values (ISA s3.2)", "1 instruction"),
    "sub": ("expand", "ADD n, b, S (flips b's sign bit), then ACT f=00 Rd, a, n (ISA s5)", "2 instructions"),
    "mul": ("native", "FMA with Rd = +0: CONST Rd, 0 (or copy Z), then FMA Rd, Ra, Rb. Truncates toward zero (ISA s5)",
            "2 instructions"),
    "div": ("host", "no SF divide or reciprocal on the chip (ISA s5)", "-"),
    "fma": ("native", "FMA Rd, Ra, Rb with Rd holding c on entry: Rd = c + a*b. Copy c first if it is still needed",
            "1 instruction"),
    "neg": ("expand", "ADD n, x, S (flip bit 15), then ACT f=00 Rd, n, Z so a 0 input never leaves 0x8000 (ISA s5)",
            "2 instructions"),
    "abs": ("expand", "abs(x) = ReLU(x) + ReLU(-x): ACT f=01 p, x, Z; ADD n, x, S; ACT f=01 q, n, Z; ACT f=00 Rd, p, q "
                      "(ISA s5; mind the ACT destination classes)", "4 instructions"),
    "sqrt": ("host", "no sqrt or rsqrt on the chip (ISA s5)", "-"),
    "widen": ("native", "(derived) free when sfN values are kept left-aligned in 16 bits, as the MX engine expects: the "
                        "word of an sfN code is already the sf16 word of the same value", "free"),
    "narrow": ("none", "no conversion or rounding instruction on FMA / ACT (ISA s5); the only hardware rounding is POP. "
                       "Run it on the host, or produce sfN directly from MX with POP at N bits", "-"),
    "cmp": ("limited", "R = ACT f=00 (a, -b) using ADD n, b, S; then CMP R, Z. Exact over the whole range, because "
                       "saturation keeps the sign (ISA s5). Like icmp, usable only as a uniform branch condition",
            "3 instructions plus the branch"),
    "cmp_lit": ("limited", "as cmp, with -threshold built as a sign-magnitude constant: ACT f=00 R, a, (-t); CMP R, Z",
                "2 instructions plus the constant and the branch"),
    "maximum": ("expand", "the exact 16-instruction sequence of ISA s5: max(a, b) = max(a+, b+) - min(a-, b-). Never "
                          "b + ReLU(a - b): the difference saturates (wrong on 118 of 1,024 pairs, measured)",
                "16 instructions, about 160 cycles"),
    "minimum": ("expand", "min(a, b) = -max(-a, -b) with the ISA s5 sequence, finished as M - P so the result is never -0",
                "about 18 instructions"),
    "sum": ("expand", "an ACT f=00 chain in index order (order matters: it saturates). Across threads (suggested): each "
                      "thread STRs its value to the scratchpad, then the chain LDRs them in order", "1 ACT per element, plus scratchpad traffic"),
    "max": ("expand", "the maximum sequence folded over the elements; any order. Across threads via the scratchpad (suggested)",
            "16 instructions per element"),
    "min": ("expand", "the minimum sequence folded over the elements; any order", "about 18 instructions per element"),
    "dot": ("expand", "(suggested) per output, an FMA chain in increasing k: FMA acc, a[k], b[k] (LDR the operands "
                      "first). Never MX: MX gives different bits (ISA s3.4 rule 5)", "per output: K FMA plus 2K loads"),
    "dot_wide": ("expand", "the MX sequence of ISA s3.4: CFG (block and N, from thread 0) and CLR; per chunk of 8 k: WL rows "
                           "7..0 then SWAP; MAC per token into its slot (up to 8 tokens per weight tile); then POP with "
                           "Rs(c) = slot << 8 | (shift & 63). Thread j supplies lane or column j. 8 output columns per "
                           "weight tile. Weights are not reset at block start: always load all 8 rows and SWAP",
                 "8 WL + 1 SWAP + 1 MAC per token per chunk of 8 k, then 1 POP per token"),
    "load": ("limited", "LDR Rd, Ra with Ra a word address below 0xFFC0. Masked lanes: only a tail mask, lowered as the "
                        "launch's thread count (see the mapping). No bounds check: an out-of-range address silently reads "
                        "other memory or the scratchpad", "about 10 + 8 x 8 cycles for 8 threads"),
    "store": ("limited", "STR Ra, Rv (address register first). Same mask and bounds rules as load. If several threads store "
                         "to one address in one instruction, the highest thread's value lands last (derived)",
              "about 10 + 8 x 16 cycles for 8 threads (2 beats each)"),
    "shared_load": ("limited", "LDR from 0xFFC0 + index. At most 64 words per tile. The scratchpad is NEVER cleared, but "
                               "DSL shared memory starts at 0: emit the zeroing stores if the kernel reads before writing",
                    "about 3 cycles per thread"),
    "shared_store": ("limited", "STR to 0xFFC0 + index; at most 64 words per tile", "about 2 cycles per thread"),
    "barrier": ("native", "nothing to emit inside a block: every instruction finishes for all 8 threads before the next, so "
                          "a store is visible to the whole block from the next instruction (ISA s1.5). There is no "
                          "barrier between blocks", "free"),
}

GAPS = [
    "Blocks bigger than 8 lanes: strip-mine (mapping table).",
    "Integers wider than 16 bits: unsupported on the chip.",
    "Divergent control flow and general masks: not lowerable; only tail masks and SF if-conversion.",
    "and, or, xor, not, bool_to_int, iabs: no lowering.",
    "General integer division and modulo: only the ISA's constant divisors.",
    "div, sqrt (and exp, log, softmax, LayerNorm and similar in models): host only.",
    "narrow: no conversion instruction. Note that the width rule (sfN on the sf16 path, re-truncated after every op) "
    "needs masking, and v4.8.1 has no AND, so sfN below 16 is best done through MX, where N is a CFG setting.",
    "Faults: the DSL faults on out-of-bounds; the chip has no bounds checks at all.",
]
