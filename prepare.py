"""
Fixed data preparation and evaluation harness for inkling-autoresearch.

    uv run prepare.py          # one-time: download 2 ClimbMix shards, build byte corpora + reasoning sets

Everything in this file is FIXED. Agents edit train.py only. The harness defines:
  - the byte-level corpora (vocab 256, so cross-entropy maps directly to bits per byte),
  - the training data mixture (text + synthetic multi-hop reasoning),
  - the time budget and the parameter budget,
  - evaluate(): val_bpb, reasoning accuracy (in-distribution and out-of-distribution hops),
    test-time loop scaling, memory, latency and analytical KV size.
"""
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Fixed constants (do not modify)
# ---------------------------------------------------------------------------
SEQ_LEN = 512                     # context length in bytes
VOCAB_SIZE = 256                  # raw bytes
TIME_BUDGET = 300                 # training wall-clock seconds
MAX_PARAMS = 12_000_000           # parameter budget (unique weights), ~24 MB in bf16
MAX_PEAK_MEM_GB = 20.0            # leave headroom on a 32 GB M4
REASONING_MIX = 8                 # rows per 32 that come from the reasoning stream
TRAIN_HOPS = (1, 6)               # inclusive hop range seen in training
EVAL_HOPS_ID = (2, 4, 6)          # in-distribution evaluation hops
EVAL_HOPS_OOD = (8, 10)           # out-of-distribution (deeper) hops
EVAL_PER_HOP = 400
VAL_WINDOWS = 384                 # 384 * 512 = ~197k scored bytes

SMOKE = os.environ.get("INKLING_SMOKE") == "1"   # tiny budget for harness tests; never log as a result
if SMOKE:
    TIME_BUDGET, VAL_WINDOWS, EVAL_PER_HOP = 20, 32, 40

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "inkling-autoresearch")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
TRAIN_SHARD, VAL_SHARD = 0, 6542
TRAIN_BYTES_CAP = 256 * 2**20
VAL_BYTES_CAP = 8 * 2**20
REASON_TRAIN_BYTES = 16 * 2**20
DOC_SEP = 0                       # byte 0 separates documents

# ---------------------------------------------------------------------------
# Synthetic multi-hop reasoning task
# ---------------------------------------------------------------------------
# "Q: k=3;b=7;x=k+4;q=b*2;z=x-5;z? A:2\n" -- answer is a single digit (mod 10).
# A chain of `hops` dependent assignments is interleaved with `hops` distractors,
# so the model must look up and compose values. Deeper chains need more serial compute,
# which is exactly what recurrent depth is supposed to buy.
_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_OPS = "+-"


def make_problem(rng, hops):
    names = rng.permutation(list(_LETTERS))[: 2 * hops + 1]
    chain, distract = names[: hops + 1], names[hops + 1:]
    value = int(rng.integers(0, 10))
    chain_stmts = [f"{chain[0]}={value}"]
    for i in range(1, hops + 1):
        op, d = _OPS[int(rng.integers(0, len(_OPS)))], int(rng.integers(1, 10))
        value = {"+": value + d, "-": value - d, "*": value * d}[op] % 10
        chain_stmts.append(f"{chain[i]}={chain[i-1]}{op}{d}")
    dist_stmts = [f"{n}={int(rng.integers(0, 10))}" for n in distract]
    # interleave while preserving chain order
    slots = np.sort(rng.choice(len(chain_stmts) + len(dist_stmts), len(chain_stmts), replace=False))
    out, ci, di = [], 0, 0
    for pos in range(len(chain_stmts) + len(dist_stmts)):
        if ci < len(slots) and pos == slots[ci]:
            out.append(chain_stmts[ci]); ci += 1
        else:
            out.append(dist_stmts[di]); di += 1
    prompt = f"Q: {';'.join(out)};{chain[-1]}? A:"
    return prompt, str(value)


def _reasoning_stream(seed, nbytes):
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    while total < nbytes:
        p, a = make_problem(rng, int(rng.integers(TRAIN_HOPS[0], TRAIN_HOPS[1] + 1)))
        s = (p + a + "\n").encode()
        parts.append(s); total += len(s)
    return np.frombuffer(b"".join(parts)[:nbytes], dtype=np.uint8)


def reasoning_eval_set():
    rng = np.random.default_rng(424242)   # disjoint seed from training stream
    sets = {}
    for h in EVAL_HOPS_ID + EVAL_HOPS_OOD:
        sets[h] = [make_problem(rng, h) for _ in range(EVAL_PER_HOP)]
    return sets

# ---------------------------------------------------------------------------
# Data download / build
# ---------------------------------------------------------------------------

def _download(index, path):
    import requests
    url = f"{BASE_URL}/shard_{index:05d}.parquet"
    for attempt in range(5):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(path + ".tmp", "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            os.replace(path + ".tmp", path)
            return
        except Exception as e:  # noqa: BLE001
            print(f"download attempt {attempt+1} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {url}")


def _shard_to_bytes(path, cap):
    import pyarrow.parquet as pq
    out, total = bytearray(), 0
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for text in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
            b = text.encode("utf-8", errors="ignore").replace(b"\x00", b"")
            out += b + bytes([DOC_SEP]); total += len(b) + 1
            if total >= cap:
                return np.frombuffer(bytes(out[:cap]), dtype=np.uint8)
    return np.frombuffer(bytes(out), dtype=np.uint8)


def prepare():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name, shard, cap in (("train", TRAIN_SHARD, TRAIN_BYTES_CAP), ("val", VAL_SHARD, VAL_BYTES_CAP)):
        out = os.path.join(CACHE_DIR, f"{name}.bin")
        if os.path.exists(out):
            print(f"{out} exists"); continue
        pq_path = os.path.join(CACHE_DIR, f"shard_{shard:05d}.parquet")
        if not os.path.exists(pq_path):
            print(f"downloading shard {shard} ..."); _download(shard, pq_path)
        arr = _shard_to_bytes(pq_path, cap)
        arr.tofile(out)
        os.remove(pq_path)            # disk is precious; the .bin is all we need
        print(f"wrote {out}: {len(arr)/2**20:.1f} MiB")
    out = os.path.join(CACHE_DIR, "reason_train.bin")
    if not os.path.exists(out):
        _reasoning_stream(1234, REASON_TRAIN_BYTES).tofile(out)
        print(f"wrote {out}")
    print("ready")

# ---------------------------------------------------------------------------
# Training data loader (fixed mixture)
# ---------------------------------------------------------------------------

def _load(name):
    path = os.path.join(CACHE_DIR, f"{name}.bin")
    if not os.path.exists(path):
        sys.exit(f"missing {path}; run `uv run prepare.py` first")
    return np.memmap(path, dtype=np.uint8, mode="r")


class Batches:
    """Infinite iterator of (B, SEQ_LEN+1) uint8 windows. Rows [0, mix) are reasoning, the rest text."""

    def __init__(self, batch_size, seed=0):
        self.text, self.reason = _load("train"), _load("reason_train")
        self.B = batch_size
        self.mix = max(1, round(REASONING_MIX * batch_size / 32))
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        return self

    def __next__(self):
        L = SEQ_LEN + 1
        rs = self.rng.integers(0, len(self.reason) - L, size=self.mix)
        ts = self.rng.integers(0, len(self.text) - L, size=self.B - self.mix)
        rows = [self.reason[s:s + L] for s in rs] + [self.text[s:s + L] for s in ts]
        return np.stack(rows).astype(np.int32)

# ---------------------------------------------------------------------------
# Evaluation (fixed)
# ---------------------------------------------------------------------------

def _call(model, x, loops):
    return model(x) if loops is None else model(x, loops=loops)


def _val_bpb(model, loops, mx, nn):
    val = _load("val")
    stride = (len(val) - SEQ_LEN - 1) // VAL_WINDOWS
    total, n = 0.0, 0
    for i in range(0, VAL_WINDOWS, 16):
        rows = np.stack([val[j * stride: j * stride + SEQ_LEN + 1] for j in range(i, min(i + 16, VAL_WINDOWS))])
        x = mx.array(rows.astype(np.int32))
        logits = _call(model, x[:, :-1], loops)
        ce = nn.losses.cross_entropy(logits.astype(mx.float32), x[:, 1:], reduction="sum")
        total += ce.item(); n += x[:, 1:].size
    return total / n / math.log(2)


def _reasoning_acc(model, sets, loops, mx):
    acc = {}
    for h, probs in sets.items():
        correct = 0
        for i in range(0, len(probs), 64):
            chunk = probs[i:i + 64]
            enc = [p.encode() for p, _ in chunk]
            T = max(len(e) for e in enc)
            x = np.zeros((len(enc), T), dtype=np.int32)
            for r, e in enumerate(enc):
                x[r, :len(e)] = np.frombuffer(e, dtype=np.uint8)
            logits = _call(model, mx.array(x), loops)
            last = mx.array([len(e) - 1 for e in enc])
            pred = np.array(mx.argmax(logits[mx.arange(len(enc)), last], axis=-1))
            correct += sum(int(pred[r]) == ord(a) for r, (_, a) in enumerate(chunk))
        acc[h] = correct / len(probs)
    return acc


def _latency_ms(model, mx):
    x = mx.zeros((1, SEQ_LEN), dtype=mx.int32)
    for _ in range(3):
        mx.eval(model(x))
    ts = []
    for _ in range(10):
        t = time.perf_counter(); mx.eval(model(x)); ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def evaluate(model, train_info):
    """Score a trained model. Prints the summary block and writes runs/last.json.

    Model contract:
      model(x) -> logits (B, T, 256) for int32 x of shape (B, T), causal.
      optional: model(x, loops=k) to run k recurrent iterations (enables test-time scaling eval).
      optional: model.train_loops (int), model.kv_bytes_per_token() (int), model.mean_loops_per_token (float, set after a call).
    """
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    model.eval()
    params = sum(v.size for _, v in tree_flatten(model.parameters()))
    base_loops = getattr(model, "train_loops", None)
    sets = reasoning_eval_set()

    t0 = time.perf_counter()
    res = {"val_bpb": _val_bpb(model, None, mx, nn)}
    acc = _reasoning_acc(model, sets, None, mx)
    res["reason_id"] = float(np.mean([acc[h] for h in EVAL_HOPS_ID]))
    res["reason_ood"] = float(np.mean([acc[h] for h in EVAL_HOPS_OOD]))
    res["reason_by_hop"] = acc
    res["mean_loops_per_token"] = getattr(model, "mean_loops_per_token", base_loops)

    # Test-time scaling: same weights, more (and fewer) loops. Only for models that accept loops=.
    res["scaling"] = {}
    if base_loops:
        for k in sorted({max(1, base_loops // 2), base_loops * 2}):
            try:
                a = _reasoning_acc(model, sets, k, mx)
                res["scaling"][k] = {
                    "val_bpb": _val_bpb(model, k, mx, nn),
                    "reason_id": float(np.mean([a[h] for h in EVAL_HOPS_ID])),
                    "reason_ood": float(np.mean([a[h] for h in EVAL_HOPS_OOD])),
                }
            except TypeError:
                break
    res["eval_seconds"] = time.perf_counter() - t0
    res["latency_ms_b1"] = _latency_ms(model, mx)
    res["params"] = params
    res["weights_mb_bf16"] = params * 2 / 2**20
    kv = getattr(model, "kv_bytes_per_token", None)
    res["kv_kb_per_token"] = kv() / 1024 if callable(kv) else None
    res.update(train_info)
    res["peak_mem_gb"] = mx.get_peak_memory() / 2**30
    res["smoke"] = SMOKE

    violations = []
    if params > MAX_PARAMS:
        violations.append(f"params {params} > {MAX_PARAMS}")
    if res["peak_mem_gb"] > MAX_PEAK_MEM_GB:
        violations.append(f"peak_mem {res['peak_mem_gb']:.1f} > {MAX_PEAK_MEM_GB}")
    res["valid"] = not violations and not SMOKE
    res["violations"] = violations

    os.makedirs("runs", exist_ok=True)
    with open("runs/last.json", "w") as f:
        json.dump(res, f, indent=2, default=str)

    print("---")
    for k in ("val_bpb", "reason_id", "reason_ood", "mean_loops_per_token", "params", "weights_mb_bf16",
              "kv_kb_per_token", "latency_ms_b1", "peak_mem_gb", "training_seconds", "steps", "tokens_M", "valid"):
        v = res.get(k)
        print(f"{k + ':':22s}{v:.6f}" if isinstance(v, float) else f"{k + ':':22s}{v}")
    print("reason_by_hop:        " + " ".join(f"{h}:{a:.3f}" for h, a in acc.items()))
    for k, s in res["scaling"].items():
        print(f"loops={k:<3d} val_bpb={s['val_bpb']:.4f} reason_id={s['reason_id']:.3f} reason_ood={s['reason_ood']:.3f}")
    if violations:
        print("VIOLATIONS: " + "; ".join(violations))
    if SMOKE:
        print("SMOKE RUN: not a valid result")
    return res


if __name__ == "__main__":
    prepare()
