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
import re
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
TRAIN_HOPS = (0, 6)               # inclusive hop range seen in training
EVAL_HOPS_ID = (2, 4, 6)          # in-distribution evaluation hops
EVAL_HOPS_OOD = (8, 10)           # out-of-distribution (deeper) hops
EVAL_PER_HOP = 400
PAIR_BOUND = 0.25                 # pair accuracy reachable by structure + a coin flip, without reading the query
TASK_VARS = 24                    # variables per problem at every hop (two chains of 11 fit at hop 10)
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
BASE_NAME = os.environ.get("INKLING_BASE", "base-v2")      # base-v2 = the v1 base continued on task-v2 data
BASE_CKPT = os.path.join(CACHE_DIR, f"{BASE_NAME}.safetensors")   # shared warm start, built by pretrain_base.py
DOC_SEP = 0                       # byte 0 separates documents

# ---------------------------------------------------------------------------
# Synthetic multi-hop reasoning task
# ---------------------------------------------------------------------------
# Variable tracking over a forest of pointer chains (RULER-style VT, made dense):
#     "q=4;k=q;x=7;b=k;y=x;?b=4?y=7?q=4?k=4?x=7\n"
# One chain of exactly `hops` pointer steps (TASK_VARS variables in all) is mixed with distractor trees.
# Evaluation asks only the chain's end: "<statements>;?b=" -> "4". Each hop is one more serial lookup.
#
# Task v3 (2026-11-24). v1 put the chain's root in the first statement about half the time (copying the first
# digit scored .48-.57 at every hop, and the 6-hour base learned exactly that). v2 fixed positions but the queried
# chain was the only long single-child path, so structure alone found it (.97 at hops >= 4) and the most common
# value was the answer at hops 8/10. v3 (evaluation debater, round 2):
#   - TWO chains of exactly `hops` pointer steps (A and B) share one statement list, plus distractor trees;
#     TASK_VARS variables at every hop; root digits distinct; uniformly random topological order.
#   - Evaluation asks both chain ends as twin prompts and scores a PAIR as correct only if both twins are.
#     A rule that ignores the question gives both twins the same answer, and the answers differ, so it scores 0;
#     structure plus a coin flip scores .25. Pair accuracy above .25 needs real lookups along the asked chain.
#   - Training lines ask one chain end first (the only deep, question-dependent query before any answer is
#     visible; answers given earlier in a line make later queries 1-hop), then every variable in random order.
#     train.py weights that first answer most (FIRST_ANSWER_W).
_LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _forest(rng, hops):
    """Returns (statements, (end_A, end_B), values)."""
    names = list(rng.permutation(list(_LETTERS))[:TASK_VARS])
    A, B, distract = names[: hops + 1], names[hops + 1: 2 * hops + 2], names[2 * hops + 2:]
    cap = max(hops, 3)                                 # distractor depth limit
    parent, depth = {A[0]: None, B[0]: None}, {A[0]: 0, B[0]: 0}
    for ch in (A, B):
        for i in range(1, hops + 1):
            parent[ch[i]], depth[ch[i]] = ch[i - 1], i
    roots, built = [A[0], B[0]], []
    for v in distract:                                 # distractors never hang off the chains
        ok = [u for u in built if depth[u] < cap]
        if ok and (len(roots) >= 10 or rng.random() < 0.5):
            p = ok[int(rng.integers(len(ok)))]
            parent[v], depth[v] = p, depth[p] + 1
        else:
            parent[v], depth[v] = None, 0
            roots.append(v)
        built.append(v)
    assert len(roots) <= 10
    digit = dict(zip(roots, rng.permutation(10)[: len(roots)].tolist()))
    kids = {v: [] for v in names}
    for v, p in parent.items():
        if p is not None:
            kids[p].append(v)
    avail, stmts, val = list(roots), [], {}
    while avail:                                       # random topological order
        v = avail.pop(int(rng.integers(len(avail))))
        p = parent[v]
        val[v] = digit[v] if p is None else val[p]
        stmts.append(f"{v}={digit[v]}" if p is None else f"{v}={p}")
        avail += kids[v]
    return ";".join(stmts), (A[-1], B[-1]), val


def position_floor(probs):
    """Best single-prompt accuracy of answering with the prompt's first or last digit (no pointer following)."""
    first = np.mean([re.findall(r"\d", p)[0] == a for p, a in probs])
    last = np.mean([re.findall(r"\d", p)[-1] == a for p, a in probs])
    return float(max(first, last))


def make_pair(rng, hops):
    """Twin prompts over one statement list: [(prompt asking A's end, answer), (prompt asking B's end, answer)]."""
    stmts, ends, val = _forest(rng, hops)
    return [(f"{stmts};?{q}=", str(val[q])) for q in ends]


def make_problem(rng, hops):
    return make_pair(rng, hops)[0]


def twin_problems(rng, hops, n):
    """n prompts (n even) as n/2 adjacent twin pairs; _reasoning_acc scores them as pairs."""
    return [t for _ in range(n // 2) for t in make_pair(rng, hops)]


def _reasoning_stream(seed, nbytes):
    rng = np.random.default_rng(seed)
    parts, total = [], 0
    while total < nbytes:
        stmts, ends, val = _forest(rng, int(rng.integers(TRAIN_HOPS[0], TRAIN_HOPS[1] + 1)))
        first = ends[int(rng.integers(2))]
        rest = [v for v in rng.permutation(list(val)) if v != first]
        qs = "".join(f"?{v}={val[v]}" for v in [first] + rest)
        s = f"{stmts};{qs}\n".encode()
        parts.append(s); total += len(s)
    return np.frombuffer(b"".join(parts)[:nbytes], dtype=np.uint8)


def reasoning_eval_set():
    rng = np.random.default_rng(424242)   # disjoint seed from training stream
    return {h: twin_problems(rng, h, EVAL_PER_HOP) for h in EVAL_HOPS_ID + EVAL_HOPS_OOD}

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
    out = os.path.join(CACHE_DIR, "reason_train_v3.bin")
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
        self.text, self.reason = _load("train"), _load("reason_train_v3")
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


_FILLER = None


def encode_prompt(prompt, align=1):
    """Bytes the model sees for an eval prompt. Training lines always follow a newline, so the prompt does too.
    align=K (CHUNK models): extra left context, the tail of a real training line, so that the prompt's last
    byte starts a K-byte chunk. Otherwise the query "?v=" shares a chunk with the answer position and the
    looped core (which sees only completed chunks) never sees which variable was asked for."""
    global _FILLER
    if _FILLER is None:
        rng = np.random.default_rng(777)
        stmts, _, val = _forest(rng, 3)
        _FILLER = (stmts + ";" + "".join(f"?{v}={val[v]}" for v in sorted(val)) + "\n").encode()
    e = prompt.encode()
    pad = 1 + (-(len(e) + 1 - 1)) % align                # filler length so that (pad + len(e) - 1) % align == 0
    return _FILLER[-pad:] + e


def _reasoning_acc(model, sets, loops, mx, return_single=False):
    """Pair accuracy per hop (sets hold adjacent twin prompts, see twin_problems); chance-free floor 0, .25 = the
    structure-plus-coin-flip bound. return_single also returns per-prompt accuracy."""
    acc, single = {}, {}
    align = getattr(model, "align", 1)
    for h, probs in sets.items():
        hits = []
        for i in range(0, len(probs), 64):
            chunk = probs[i:i + 64]
            enc = [encode_prompt(p, align) for p, _ in chunk]
            T = max(len(e) for e in enc)
            x = np.zeros((len(enc), T), dtype=np.int32)
            for r, e in enumerate(enc):
                x[r, :len(e)] = np.frombuffer(e, dtype=np.uint8)
            logits = _call(model, mx.array(x), loops)
            last = mx.array([len(e) - 1 for e in enc])
            pred = np.array(mx.argmax(logits[mx.arange(len(enc)), last], axis=-1))
            hits += [int(pred[r]) == ord(a) for r, (_, a) in enumerate(chunk)]
        hits = np.array(hits)
        single[h] = float(hits.mean())
        acc[h] = float((hits[0::2] & hits[1::2]).mean())   # twins are adjacent: a pair counts if both are right
    if return_single:
        return acc, single
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
    acc, single = _reasoning_acc(model, sets, None, mx, return_single=True)
    res["reason_single_by_hop"] = single
    res["reason_id"] = float(np.mean([acc[h] for h in EVAL_HOPS_ID]))
    res["reason_ood"] = float(np.mean([acc[h] for h in EVAL_HOPS_OOD]))
    res["reason_by_hop"] = acc
    res["reason_single_floor_by_hop"] = {h: position_floor(p) for h, p in sets.items()}   # per-prompt position rule
    res["reason_floor_id"] = res["reason_floor_ood"] = PAIR_BOUND   # pair accuracy must beat this to show lookups
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
    for k in ("val_bpb", "reason_id", "reason_ood", "reason_floor_id", "reason_floor_ood", "mean_loops_per_token", "params", "weights_mb_bf16",
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
