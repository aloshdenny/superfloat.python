"""Chat with an LLM whose every matmul runs through sf's bit-exact model of the
Atreides MX engine (wide mode), and compare it with plain float.

    pip install transformers                     # once, besides sf[torch]
    python examples/chat_wide.py                 # sf16 wide mode
    python examples/chat_wide.py --dtype sf8     # sf8 layers (attention and LM head stay sf16)
    python examples/chat_wide.py --compare       # answer each message twice: float, then sf
    python examples/chat_wide.py --model Qwen/Qwen2.5-1.5B-Instruct

Type a message and press Enter. Commands: /reset (new conversation), /status
(what the last reply did on MX), /quit.

Decoding is greedy, so the same prompt always gives the same answer; that makes
float and sf directly comparable. This runs on the CPU: sf emulates the chip's
arithmetic bit for bit.  Qwen2.5-0.5B on 2 cores: about 18 s to prepare the
weights once, then about 1 token per second (float: about 8).
"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def load(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa").eval()
    return tok, model


def to_sf(model, dtype, keep_float):
    """The sf version of ``model``.  With ``keep_float`` the float model stays
    usable and both share the same weight tensors (sf only reads them), so
    comparing costs about one model's memory, not two."""
    import copy
    import sf.torch as sft
    if keep_float:
        shared = {id(t): t for t in list(model.parameters()) + list(model.buffers())}
        model = copy.deepcopy(model, memo=shared)              # new modules, same tensors
    m = sft.convert(model, dtype=dtype, accumulate="wide", inplace=True)
    r = sft.conversion_report(m)
    kept = ", ".join(n for n, _ in r.kept_sf16) or "none"
    print(f"  sf {dtype} wide mode: {len(r.converted)} layers on MX, kept at sf16: {kept}; attention on MX at sf16")
    return m


@torch.no_grad()
def reply(model, tok, history, max_new, label, stream=True):
    from transformers import TextStreamer
    text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
    print(f"\n{label}> ", end="", flush=True)
    t = time.time()
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, streamer=streamer,
                         pad_token_id=tok.eos_token_id)
    new = out[0, ids.input_ids.shape[1]:]
    dt = time.time() - t
    answer = tok.decode(new, skip_special_tokens=True)
    if not stream:
        print(answer)
    print(f"  [{len(new)} tokens in {dt:.0f} s, {len(new) / max(dt, 1e-9):.1f} tokens/s]")
    return answer


def status_line():
    import sf.torch as sft
    st = sft.last_status()
    if not st:
        return "  no sf status yet"
    sat = sum((s.saturated or 0) for s in st.values())
    redo = sum(s.redone for s in st.values())
    outs = sum(s.outputs for s in st.values())
    guard = sum(s.guard_raised for s in st.values())
    att = sum(1 for k in st if k.endswith(".pv"))
    lo = min(s.shift_min for s in st.values() if s.shift_min is not None)
    hi = max(s.shift_max for s in st.values() if s.shift_max is not None)
    return (f"  last step on MX: {len(st)} matmuls ({att} attention), shifts {lo}..{hi}, "
            f"{redo} of {outs} outputs redone, {sat} saturated, {guard} guard raises")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="sf16", help="sf16 or sf8 (the layers; attention and LM head stay sf16)")
    ap.add_argument("--compare", action="store_true", help="answer each message in float first, then in sf")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--ask", help="answer one message and exit (no chat loop)")
    a = ap.parse_args()

    print(f"Loading {a.model} ...")
    tok, base = load(a.model)
    sfm = to_sf(base, a.dtype, keep_float=a.compare)
    system = {"role": "system", "content": "You are a helpful assistant. Answer briefly."}
    history = [system]
    print("Ready. Type a message (/reset, /status, /quit).")

    msgs = [a.ask] if a.ask else None
    while True:
        if msgs is not None:
            if not msgs:
                break
            msg = msgs.pop(0)
            print(f"\nyou> {msg}")
        else:
            try:
                msg = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not msg:
            continue
        if msg == "/quit":
            break
        if msg == "/reset":
            history = [system]
            print("  new conversation")
            continue
        if msg == "/status":
            print(status_line())
            continue
        history.append({"role": "user", "content": msg})
        if a.compare:
            reply(base, tok, history, a.max_new_tokens, "float")
        ans = reply(sfm, tok, history, a.max_new_tokens, f"sf ({a.dtype}, on MX)")
        print(status_line())
        history.append({"role": "assistant", "content": ans})


if __name__ == "__main__":
    sys.exit(main())
