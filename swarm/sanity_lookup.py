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


def line(rng, nv):
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
    ap.add_argument("set", nargs="*")
    a = ap.parse_args()
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
            print(f"step {i:5d} t {time.time()-t0:5.0f}s loss {loss.item():.3f} answer_acc {acc:.3f}", flush=True)
        i += 1


if __name__ == "__main__":
    main()
