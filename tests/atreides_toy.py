"""A toy model of the 7 Atreides instructions Kartik's hand-written axpy uses,
for the driver tests only.  It is not the chip and not an ISA model: anything
outside CMP, CONST, LDR, ADD, FMA, STR and RET raises NotImplementedError.

Semantics from compiler_v4.9.2.md sections 2 and 3: lockstep blocks of 8
threads, R13 = block index (restarting at 0 every launch), R14 = 8,
R15 = thread index; unwritten data words read 0; FMA is the sign-magnitude
add(Rd, mul(Rs, Rt)) with the product truncated and the sum saturated.
"""
from __future__ import annotations

def _sm(x):
    return -(x & 0x7FFF) if x & 0x8000 else x & 0x7FFF


def _mul(a, b):                       # fma.sv: magnitude truncated toward zero
    m = ((a & 0x7FFF) * (b & 0x7FFF)) >> 15
    return 0 if m == 0 else ((a ^ b) & 0x8000) | m


def _add(a, b):                       # saturating sign-magnitude add
    s = max(-32767, min(32767, _sm(a) + _sm(b)))
    return 0x8000 | -s if s < 0 else s


def run_toy(plan, memory: dict) -> dict:
    mem = dict(memory)
    for launch in plan.launches:
        mem.update(launch.writes)
        nblocks = -(-launch.threads // 8)
        for blk in range(nblocks):                    # R13 restarts at 0 every launch
            live = range(8 if blk < nblocks - 1 else launch.threads - 8 * (nblocks - 1))
            R = [[0] * 16 for _ in range(8)]
            for t in range(8):
                R[t][13], R[t][14], R[t][15] = blk, 8, t
            pc = 0
            while True:
                w = plan.program[pc]
                op, a, b, c = w >> 12, (w >> 8) & 15, (w >> 4) & 15, w & 15
                if op == 0xF:                         # RET
                    break
                for t in live:
                    r = R[t]
                    if op == 0x2:                     # CMP: sets NZP; axpy never branches
                        pass
                    elif op == 0x9:                   # CONST: sign-extended imm8
                        imm = (b << 4) | c
                        r[a] = (imm - 256 if imm & 0x80 else imm) & 0xFFFF
                    elif op == 0x7:                   # LDR Rd, Ra (unwritten words read 0)
                        r[a] = mem.get(r[b], 0)
                    elif op == 0x3:                   # ADD mod 2^16
                        r[a] = (r[b] + r[c]) & 0xFFFF
                    elif op == 0xA:                   # FMA Rd = add(Rd, mul(Rs, Rt))
                        r[a] = _add(r[a], _mul(r[b], r[c]))
                    elif op == 0x8:                   # STR Ra, Rv
                        mem[r[a]] = r[b]
                    else:
                        raise NotImplementedError(f"toy model: opcode {op:#x} at pc {pc}")
                pc += 1
    return mem



def toy_runner(plan) -> dict:
    """The runner interface: execute a LaunchPlan, return the memory after it."""
    return run_toy(plan, plan.memory)
