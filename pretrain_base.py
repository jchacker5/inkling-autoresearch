"""
Build the shared base checkpoint once ("Inkling base"). Every autoresearch experiment warm-starts from it.

    uv run pretrain_base.py --hours 6        # overnight; safe to stop at any time (checkpoints every 15 min)
    uv run pretrain_base.py --resume         # continue from the last checkpoint up to the same --hours total

Control models for the paper use the same script with a different name and config overrides, e.g. a
conventional (non-looped) transformer with the same unique parameters, stopped at a matched point of the
same 6-hour LR schedule:

    uv run pretrain_base.py --name control-4L --set LOOPS=1 --hours 6 --stop-hours 2

Why: in-context retrieval (the circuit the reasoning task needs) only emerges after thousands of steps.
A 5-minute from-scratch run on an M4 reaches ~250 steps and scores at chance, so architecture changes are
tested as *modifications of a trained looped model*: new modules are zero-initialised and fine-tuned for
the fixed 5-minute budget. Progress (val_bpb and reasoning accuracy by hop) is appended to
<name>_progress.tsv, so the run itself documents when retrieval emerges. Whole-hour snapshots are kept
as <name>-<k>h.safetensors for matched-time comparisons.
"""
import argparse
import ast
import json
import math
import os
import shutil
import subprocess
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

import prepare as P
import train as T

HOPS = (0, 1, 2, 4, 6, 8)


def quick_eval(model):
    import mlx.nn as nn
    rng = np.random.default_rng(7)
    sets = {h: [P.make_problem(rng, h) for _ in range(100)] for h in HOPS}
    acc = P._reasoning_acc(model, sets, None, mx)
    keep = P.VAL_WINDOWS
    P.VAL_WINDOWS = 64
    bpb = P._val_bpb(model, None, mx, nn)
    P.VAL_WINDOWS = keep
    model.train()
    return bpb, acc


def apply_overrides(pairs):
    """--set KEY=VALUE on train.py's module-level config (before the model is built)."""
    out = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        if not hasattr(T, k):
            raise SystemExit(f"unknown train.py setting {k}")
        out[k] = ast.literal_eval(v)
        setattr(T, k, out[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base", help="checkpoint name in the cache dir (default: the shared base)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override train.py settings")
    ap.add_argument("--hours", type=float, default=6.0, help="length of the LR schedule")
    ap.add_argument("--stop-hours", type=float, default=None, help="stop early, keeping the --hours schedule")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--every-min", type=float, default=15.0)
    ap.add_argument("--resume", action="store_true",
                    help="load the last checkpoint and train for the rest of --hours. Optimizer state is not "
                         "saved, so AdamW restarts (with a fresh warm-up) and the cosine schedule spans the remaining time.")
    a = ap.parse_args()

    overrides = apply_overrides(a.set)
    ckpt = os.path.join(P.CACHE_DIR, f"{a.name}.safetensors")
    meta_path = ckpt.replace(".safetensors", ".json")
    mx.random.seed(T.SEED)
    model = T.Inkling()
    mx.eval(model.parameters())
    n = sum(v.size for _, v in tree_flatten(model.parameters()))
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = dict(name=a.name, commit=commit, hours=a.hours, stop_hours=a.stop_hours, lr=a.lr, params=n,
                overrides=overrides, d_model=T.D_MODEL, n_heads=T.N_HEADS, prelude=T.PRELUDE, core=T.CORE,
                coda=T.CODA, loops=T.LOOPS, batch=T.BATCH_SIZE)
    step0, sec0 = 0, 0.0
    if a.resume:
        with open(meta_path) as f:
            prev = json.load(f)
        T.load_base(model, ckpt)
        step0, sec0 = prev["steps"], prev["train_seconds"]
        meta["resumed"] = prev.get("resumed", []) + [dict(commit=prev["commit"], steps=step0, train_seconds=sec0)]
        print(f"resuming from step {step0} ({sec0/60:.1f} min of training)", flush=True)
    total = a.hours * 3600
    stop_total = min(total, a.stop_hours * 3600) if a.stop_hours else total
    if stop_total - sec0 <= 1:
        raise SystemExit("nothing to do: the checkpoint already has the requested training")
    print("config:", meta, flush=True)

    progress = os.path.join(P.CACHE_DIR, f"{a.name}_progress.tsv")
    if not os.path.exists(progress):
        with open(progress, "w") as f:
            f.write("train_min\tsteps\tval_bpb\t" + "\t".join(f"hop{h}" for h in HOPS) + "\n")
    every = a.every_min * 60
    next_ck = [math.floor(sec0 / every + 1e-6) * every + every]      # checkpoints on absolute 15-min marks
    snap_done = [int(sec0 // 3600)]

    def checkpoint(step, t_train, final=False):
        step, t_train = step + step0, t_train + sec0
        if not final and t_train < next_ck[0]:
            return
        if final and meta.get("steps") == step:          # already checkpointed at this step
            return
        while next_ck[0] <= t_train:
            next_ck[0] += every
        bpb, acc = quick_eval(model)
        tmp = ckpt + ".tmp.safetensors"
        model.save_weights(tmp)
        os.replace(tmp, ckpt)
        meta.update(steps=step, train_seconds=t_train, val_bpb_quick=bpb, reasoning_quick={str(k): v for k, v in acc.items()})
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        hour = int((t_train + 1) // 3600)
        if hour > snap_done[0]:
            shutil.copyfile(ckpt, ckpt.replace(".safetensors", f"-{hour}h.safetensors"))
            shutil.copyfile(meta_path, meta_path.replace(".json", f"-{hour}h.json"))
            snap_done[0] = hour
        line = f"{t_train/60:.1f}\t{step}\t{bpb:.4f}\t" + "\t".join(f"{acc[h]:.3f}" for h in HOPS)
        with open(progress, "a") as f:
            f.write(line + "\n")
        print("CHECKPOINT " + line, flush=True)

    seed = T.SEED + 1000 * len(meta.get("resumed", []))   # a resumed run must not replay the same batches
    info = T.train(model, total - sec0, a.lr, seed=seed, log_every=500, on_tick=checkpoint,
                   stop_after=stop_total - sec0)
    checkpoint(info["steps"], info["training_seconds"], final=True)
    print("done:", info, flush=True)


if __name__ == "__main__":
    main()
