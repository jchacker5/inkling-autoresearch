"""Overthinking analysis of a looped checkpoint: when does each answer "land", and what does stopping early cost?

    uv run swarm/landing.py --name base               # landing-loop statistics + utility-exit sweep
    uv run swarm/landing.py --name base --no-exit     # landing statistics only

Inspired by TRACE (Zhang et al., "Do LLMs Really Need 10+ Thoughts for 'Find the Time 1000 Days Later'?"),
which finds that reasoning models keep thinking long after their answer is fixed ("Late Landing") and argues
for a utility-based definition of overthinking. Here a "thought" is one pass of the shared core.

1. Landing loop. For every reasoning problem (by hop count) and for text bytes, read the prediction after each
   loop r = 1..L (exact logit lens: decode loop r's state through the coda, which equals running r loops).
   The landing loop is the first r after which the argmax never changes again up to L (L = the trained loop
   count). Loops after landing are overthinking: they cannot change the output. We also count "flips":
   problems that were right at some earlier loop but wrong at L (destructive overthinking, Kaya et al. 2019).
2. Utility exit. Training-free early exit at eval (train.py EXIT_TAU): a token stops once one more loop changes
   its readout distribution by less than tau (KL). We sweep tau and report accuracy by hop, val_bpb and the
   mean loops per token, i.e. the compute/accuracy frontier.

Takes the GPU lock. Writes results/landing-<name>.json.
"""
import argparse
import fcntl
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import numpy as np  # noqa: E402

import prepare as P  # noqa: E402
import pretrain_base as PB  # noqa: E402
import train as T  # noqa: E402

HOPS = (0, 1, 2, 3, 4, 6, 8, 10)
TAUS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def encode(chunk, align=1):
    enc = [P.encode_prompt(p, align) for p, _ in chunk]
    x = np.zeros((len(enc), max(len(e) for e in enc)), dtype=np.int32)
    for r, e in enumerate(enc):
        x[r, :len(e)] = np.frombuffer(e, dtype=np.uint8)
    return mx.array(x), mx.array([len(e) - 1 for e in enc])


def landing(preds):
    """preds: (n, L) argmax after each loop. Returns the 1-based loop after which the argmax never changes."""
    n, L = preds.shape
    same = preds == preds[:, -1:]
    land = np.full(n, L)
    for r in range(L - 2, -1, -1):          # 0-based index r is loop r+1; extend while it already matches the end
        ok = same[:, r] & (land == r + 2)
        land[ok] = r + 1
    return land


def reasoning_landing(model, sets, L):
    out = {}
    for h, probs in sets.items():
        preds, gold = [], []
        for i in range(0, len(probs), 64):
            chunk = probs[i:i + 64]
            x, last = encode(chunk, getattr(model, 'align', 1))
            ro = model.loop_readouts(x, L)
            preds.append(np.stack([np.array(mx.argmax(lo[mx.arange(len(chunk)), last], axis=-1)) for lo in ro], 1))
            gold += [ord(a) for _, a in chunk]
        preds, gold = np.concatenate(preds), np.array(gold)
        correct = preds == gold[:, None]
        land = landing(preds)
        out[h] = {
            "acc_by_loop": correct.mean(0).round(4).tolist(),
            "mean_landing_loop": float(land.mean()),
            "landing_hist": np.bincount(land, minlength=L + 1)[1:].tolist(),
            "overthinking_frac": float(((L - land) / L).mean()),       # share of loops spent after landing
            "flips": float((correct[:, :-1].any(1) & ~correct[:, -1]).mean()),
            "mean_landing_loop_correct": float(land[correct[:, -1]].mean()) if correct[:, -1].any() else None,
        }
    return out


def text_landing(model, L, windows=32):
    val = P._load("val")
    stride = (len(val) - P.SEQ_LEN - 1) // windows
    rows = np.stack([val[j * stride: j * stride + P.SEQ_LEN + 1] for j in range(windows)]).astype(np.int32)
    x = mx.array(rows)
    lands, bpb = [], np.zeros(L)
    for i in range(0, windows, 8):
        xb = x[i:i + 8]
        ro = model.loop_readouts(xb[:, :-1], L)
        preds = np.stack([np.array(mx.argmax(lo, axis=-1)).reshape(-1) for lo in ro], 1)
        lands.append(landing(preds))
        for r, lo in enumerate(ro):
            bpb[r] += nn.losses.cross_entropy(lo, xb[:, 1:], reduction="sum").item()
    land = np.concatenate(lands)
    n = windows * P.SEQ_LEN
    return {"bpb_by_loop": (bpb / n / math.log(2)).round(4).tolist(), "mean_landing_loop": float(land.mean()),
            "landing_hist": np.bincount(land, minlength=L + 1)[1:].tolist(),
            "overthinking_frac": float(((L - land) / L).mean())}


def mean_loops(model, x, max_loops, tau, pos=None):
    _, expected = model.forward(x, max_loops, exit_tau=tau)
    if tau == 0:
        return float(max_loops)
    return float((expected[mx.arange(x.shape[0]), pos] if pos is not None else expected).mean().item())


def exit_sweep(model, sets, max_loops):
    val = P._load("val")
    stride = (len(val) - P.SEQ_LEN - 1) // 16
    text = mx.array(np.stack([val[j * stride: j * stride + P.SEQ_LEN] for j in range(16)]).astype(np.int32))
    rows = []
    for tau in (0.0,) + TAUS:
        T.EXIT_TAU = tau
        acc = P._reasoning_acc(model, sets, max_loops, mx)
        ans = []
        for h, probs in sets.items():               # mean loops spent at the answer position
            x, last = encode(probs[:64], getattr(model, 'align', 1))
            ans.append(mean_loops(model, x, max_loops, tau, last))
        bpb = P._val_bpb(model, max_loops, mx, nn)
        txt = mean_loops(model, text, max_loops, tau)
        rows.append({"tau": tau, "max_loops": max_loops, "acc_by_hop": acc, "val_bpb": bpb,
                     "mean_loops_answer": float(np.mean(ans)), "mean_loops_text": txt})
        print(f"tau={tau:<7g} L={max_loops} loops@answer={np.mean(ans):.2f} loops@text={txt:.2f} "
              f"bpb={bpb:.4f} " + " ".join(f"{h}:{v:.3f}" for h, v in acc.items()), flush=True)
    T.EXIT_TAU = 0.0
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--per-hop", type=int, default=300)
    ap.add_argument("--no-exit", action="store_true")
    a = ap.parse_args()
    PB.apply_overrides(a.set)
    path = os.path.join(P.CACHE_DIR, f"{a.name}.safetensors")
    lock = open(os.path.join(P.CACHE_DIR, "gpu.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    model = T.Inkling()
    model.load_weights(path)
    mx.eval(model.parameters())
    model.eval()
    L = T.LOOPS
    rng = np.random.default_rng(616161)
    sets = {h: P.eval_problems(rng, h, a.per_hop) for h in HOPS}
    res = {"name": a.name, "train_loops": L, "reasoning": reasoning_landing(model, sets, L),
           "reasoning_2x": reasoning_landing(model, sets, 2 * L), "text": text_landing(model, L)}
    for h, v in res["reasoning"].items():
        print(f"hop {h:>2}: acc by loop {v['acc_by_loop']}  landing {v['mean_landing_loop']:.2f}  "
              f"overthinking {v['overthinking_frac']:.2f}  flips {v['flips']:.3f}", flush=True)
    t = res["text"]
    print(f"text: bpb by loop {t['bpb_by_loop']}  landing {t['mean_landing_loop']:.2f}  hist {t['landing_hist']}", flush=True)
    if not a.no_exit:
        small = {h: v[:150] for h, v in sets.items()}
        res["exit"] = exit_sweep(model, small, L) + exit_sweep(model, small, 2 * L)
    fcntl.flock(lock, fcntl.LOCK_UN)
    out = ROOT / "results" / f"landing-{a.name}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
