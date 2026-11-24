"""Full evaluation of a saved checkpoint (the base, its hourly snapshots, or a control model).

    uv run swarm/eval_ckpt.py --name base                 # the shared base, with a loop sweep
    uv run swarm/eval_ckpt.py --name control-4L-2h --set LOOPS=1 --no-sweep

Runs prepare.evaluate() (the same metrics as every experiment) and, for looped models, a test-time loop
sweep: reasoning accuracy by hop and val_bpb at each loop count. Takes the GPU lock. Writes
results/eval-<name>.json.
"""
import argparse
import fcntl
import json
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

SWEEP_LOOPS = (1, 2, 3, 4, 6, 8, 12)
SWEEP_HOPS = (1, 2, 3, 4, 6, 8, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--per-hop", type=int, default=300)
    a = ap.parse_args()
    overrides = PB.apply_overrides(a.set)
    path = os.path.join(P.CACHE_DIR, f"{a.name}.safetensors")
    lock = open(os.path.join(P.CACHE_DIR, "gpu.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    model = T.Inkling()
    model.load_weights(path)
    mx.eval(model.parameters())
    meta_path = path.replace(".safetensors", ".json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    res = P.evaluate(model, {"steps": meta.get("steps"), "training_seconds": meta.get("train_seconds")})
    res.update(name=a.name, overrides=overrides, checkpoint_meta=meta, sweep={})
    if not a.no_sweep:
        rng = np.random.default_rng(515151)
        sets = {h: P.twin_problems(rng, h, a.per_hop) for h in SWEEP_HOPS}
        for k in SWEEP_LOOPS:
            acc = P._reasoning_acc(model, sets, k, mx)
            res["sweep"][k] = {"val_bpb": P._val_bpb(model, k, mx, nn), "acc_by_hop": acc}
            print(f"loops={k:<3d} bpb={res['sweep'][k]['val_bpb']:.4f} " + " ".join(f"{h}:{v:.3f}" for h, v in acc.items()), flush=True)
    fcntl.flock(lock, fcntl.LOCK_UN)
    out = ROOT / "results" / f"eval-{a.name}.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
