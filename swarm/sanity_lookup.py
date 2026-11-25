"""Sanity check: can this model and optimizer learn plain key->value lookup from scratch?

    uv run swarm/sanity_lookup.py [--minutes 8] [KEY=VALUE ...]

Every row is pure lookup lines "a=5;k=2;...;?k=2?a=5..." (NV distinct letters, random digits, every key queried
once in random order), so a working retrieval circuit is the only way to lower the answer loss. Prints answer
accuracy on fresh lines every 250 steps. Queries are drawn WITH replacement: if every key is asked exactly once,
a query-blind elimination rule (most frequent digit not yet answered) scores .458 with 8 keys, and models learn
that instead of lookup (results/sanity-elim/). Small transformers learn this (MQAR) within a few thousand steps.
"""
import argparse
import os
import string
import sys
import time
from functools import partial

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402

import pretrain_base as PB  # noqa: E402
import train as T  # noqa: E402

WITH_REPLACEMENT = True   # asking every key exactly once lets "most frequent unused digit" score .458 (8 keys)
LET = np.frombuffer(string.ascii_lowercase.encode(), np.uint8)


HOPS = 0                  # >0: equal-length chains ("b=a"), see chain_line


def chain_line(rng, nv, hops):
    """nv // (hops + 1) roots with distinct digits, each heading one chain of exactly `hops` pointers ("b=a"), so
    every value is equally common: a question-blind guess scores 1 / roots at every depth. (Random forests let
    "answer the most common value" score .51 at hop 1 and .73 at hop 2.) Statements come in a random order that
    keeps parents before children. Returns (bytes, per-question hop depth, -1 for repeats)."""
    R = max(2, nv // (hops + 1))
    vs = rng.choice(LET, R * (hops + 1), replace=False).reshape(R, hops + 1)
    digits = rng.choice(10, R, replace=False)
    nxt, order = [0] * R, []
    while len(order) < R * (hops + 1):            # random interleaving of the chains, each in its own order
        r = int(rng.choice([k for k in range(R) if nxt[k] <= hops]))
        order.append((r, nxt[r]))
        nxt[r] += 1
    st = []
    for r, d in order:
        st += [vs[r, d], 61, 48 + digits[r] if d == 0 else vs[r, d - 1], 59]
    flat = [(r, d) for r in range(R) for d in range(hops + 1)]
    q = rng.integers(0, len(flat), len(flat))
    qs, ds, seen = [], [], set()
    for i in q:
        r, d = flat[i]
        qs += [63, vs[r, d], 61, 48 + digits[r]]
        ds.append(d if int(i) not in seen else -1)    # a repeat can copy its earlier answer: score first asks only
        seen.add(int(i))
    return np.array([10] + st + qs, np.uint8), ds


def line(rng, nv):
    if HOPS > 0:
        return chain_line(rng, nv, HOPS)[0]
    vs = rng.choice(LET, nv, replace=False)
    ds = rng.integers(48, 58, nv).astype(np.uint8)
    q = rng.integers(0, nv, nv) if WITH_REPLACEMENT else rng.permutation(nv)
    st = np.stack([vs, np.full(nv, 61, np.uint8), ds, np.full(nv, 59, np.uint8)], 1).reshape(-1)
    qs = np.stack([np.full(nv, 63, np.uint8), vs[q], np.full(nv, 61, np.uint8), ds[q]], 1).reshape(-1)
    return np.concatenate([[10], st, qs]).astype(np.uint8)


def rows(rng, B, L, nv):
    out = np.zeros((B, L), np.uint8)
    for b in range(B):
        buf = np.concatenate([line(rng, nv) for _ in range(L // (8 * nv) + 2)])
        s = rng.integers(0, 8 * nv)
        out[b] = buf[s:s + L]
    return out


def answer_mask(x):
    """targets that are the digit after '?<letter>=' with the line start in the window."""
    xi = x[:, :-1]
    m = np.zeros_like(xi, bool)
    m[:, 2:] = (xi[:, :-2] == 63) & (xi[:, 2:] == 61)
    seen = np.maximum.accumulate(xi == 10, axis=1)
    return m & seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--nv", type=int, default=8)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--hops", type=int, default=0, help="chain depth (0 = pure lookup)")
    ap.add_argument("set", nargs="*")
    a = ap.parse_args()
    global HOPS
    HOPS = a.hops
    PB.apply_overrides(a.set)
    mx.random.seed(0)
    model = T.Inkling()
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=a.wd, betas=[0.9, 0.95])
    rng, erng = np.random.default_rng(0), np.random.default_rng(99)

    def loss_fn(x, w):
        logits = model(x[:, :-1])
        tok = nn.losses.cross_entropy(logits, x[:, 1:])
        return (tok * w).sum() / w.sum()

    vg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, w):
        loss, g = vg(x, w)
        g, _ = optim.clip_grad_norm(g, 1.0)
        opt.update(model, g)
        return loss

    if HOPS > 0:          # eval: whole lines, answers grouped by depth
        lines = [chain_line(erng, a.nv, HOPS) for _ in range(64)]
        width = max(len(b) for b, _ in lines)
        ex = np.full((64, width), 10, np.uint8)
        for i, (b, _) in enumerate(lines):
            ex[i, :len(b)] = b
        em = answer_mask(ex)
        edepth = np.full(em.shape, -1)
        for i, (b, ds) in enumerate(lines):
            pos = np.nonzero(em[i])[0]
            edepth[i, pos[:len(ds)]] = ds[:len(pos)]
    else:
        ex = rows(erng, 64, a.len, a.nv)
        em = answer_mask(ex)
    t0, i = time.time(), 0
    while time.time() - t0 < a.minutes * 60:
        x = rows(rng, 32, a.len + 1, a.nv)
        am = answer_mask(x)
        w = np.where(am, 10.0, 1.0).astype(np.float32)
        opt.learning_rate = a.lr * min(1.0, (i + 1) / 100)
        loss = step(mx.array(x.astype(np.int32)), mx.array(w))
        mx.eval(loss, state)
        if i % 250 == 0:
            model.eval()
            p = np.array(mx.argmax(model(mx.array(ex[:, :-1].astype(np.int32))), -1))
            model.train()
            acc = (p == ex[:, 1:])[em].mean()
            by = ""
            if HOPS > 0:
                ok = p == ex[:, 1:]
                by = " " + " ".join(f"h{d}={ok[edepth == d].mean():.3f}" for d in range(HOPS + 1) if (edepth == d).any())
            print(f"step {i:5d} t {time.time()-t0:5.0f}s loss {loss.item():.3f} answer_acc {acc:.3f}{by}", flush=True)
        i += 1


if __name__ == "__main__":
    main()
