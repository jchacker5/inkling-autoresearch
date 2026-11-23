"""
Build the shared base checkpoint once ("Inkling base"). Every autoresearch experiment warm-starts from it.

    uv run pretrain_base.py --hours 6        # overnight; safe to stop at any time (checkpoints every 15 min)

Why: in-context retrieval (the circuit the reasoning task needs) only emerges after thousands of steps.
A 5-minute from-scratch run on an M4 reaches ~250 steps and scores at chance, so architecture changes are
tested as *modifications of a trained looped model*: new modules are zero-initialised and fine-tuned for
the fixed 5-minute budget. Progress (val_bpb and reasoning accuracy by hop) is appended to base_progress.tsv,
so the run itself documents when retrieval emerges.
"""
import argparse
import json
import os
import subprocess
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import prepare as P
import train as T


def quick_eval(model):
    import mlx.nn as nn
    rng = np.random.default_rng(7)
    sets = {h: [P.make_problem(rng, h) for _ in range(100)] for h in (0, 1, 2, 4, 6, 8)}
    acc = P._reasoning_acc(model, sets, None, mx)
    keep = P.VAL_WINDOWS
    P.VAL_WINDOWS = 64
    bpb = P._val_bpb(model, None, mx, nn)
    P.VAL_WINDOWS = keep
    model.train()
    return bpb, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--every-min", type=float, default=15.0)
    a = ap.parse_args()

    mx.random.seed(T.SEED)
    model = T.Inkling()
    mx.eval(model.parameters())
    n = sum(v.size for _, v in tree_flatten(model.parameters()))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(commit=commit, hours=a.hours, lr=a.lr, params=n, d_model=T.D_MODEL, n_heads=T.N_HEADS,
                prelude=T.PRELUDE, core=T.CORE, coda=T.CODA, loops=T.LOOPS, batch=T.BATCH_SIZE)
    print("base config:", meta, flush=True)
    progress = os.path.join(P.CACHE_DIR, "base_progress.tsv")
    if not os.path.exists(progress):
        with open(progress, "w") as f:
            f.write("wall_min\tsteps\tval_bpb\t" + "\t".join(f"hop{h}" for h in (0, 1, 2, 4, 6, 8)) + "\n")
    next_ck = [a.every_min * 60]
    t_start = time.time()

    def checkpoint(step, t_train, final=False):
        if not final and t_train < next_ck[0]:
            return
        next_ck[0] += a.every_min * 60
        bpb, acc = quick_eval(model)
        tmp = P.BASE_CKPT + ".tmp.safetensors"
        model.save_weights(tmp)
        os.replace(tmp, P.BASE_CKPT)
        meta.update(steps=step, train_seconds=t_train, val_bpb_quick=bpb, reasoning_quick={str(k): v for k, v in acc.items()})
        with open(P.BASE_CKPT.replace(".safetensors", ".json"), "w") as f:
            json.dump(meta, f, indent=2)
        line = f"{(time.time()-t_start)/60:.1f}\t{step}\t{bpb:.4f}\t" + "\t".join(f"{acc[h]:.3f}" for h in (0, 1, 2, 4, 6, 8))
        with open(progress, "a") as f:
            f.write(line + "\n")
        print("CHECKPOINT " + line, flush=True)

    info = T.train(model, a.hours * 3600, a.lr, log_every=500, on_tick=checkpoint)
    checkpoint(info["steps"], info["training_seconds"], final=True)
    print("done:", info, flush=True)


if __name__ == "__main__":
    main()
