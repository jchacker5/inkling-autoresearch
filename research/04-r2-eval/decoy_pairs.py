"""Decoy-twin probe (proposal; eval-only, pure Python). Two bare chains of exactly h hops with distinct root
digits share one statement list; twin prompts ask each chain's end. A rule that never reaches the root cannot
tell the chains apart (nodes at equal depth on two bare paths are structurally identical), so:
  position-only or query-blind rules:  pair accuracy 0 (same answer for both twins, answers differ)
  structure + arbitrary query hash:    .25
Usage:  make_pair(rng, h) -> (stmts, (qA, aA), (qB, aB));  probe_sets(seed, hops, n) -> {h: [pairs]}"""
import numpy as np

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def make_pair(rng, hops, nvars=20):
    assert 2 * hops + 2 <= nvars
    names = list(rng.permutation(list(LETTERS))[:nvars])
    A, B, distract = names[:hops + 1], names[hops + 1:2 * hops + 2], names[2 * hops + 2:]
    cap = max(hops, 3)
    parent, depth = {A[0]: None, B[0]: None}, {A[0]: 0, B[0]: 0}
    for ch in (A, B):
        for i in range(1, hops + 1):
            parent[ch[i]], depth[ch[i]] = ch[i - 1], i
    roots, built = [A[0], B[0]], []
    for v in distract:                                   # same distractor process as prepare._forest
        ok = [u for u in built if depth[u] < cap]
        if ok and (len(roots) >= 10 or rng.random() < 0.5):
            p = ok[int(rng.integers(len(ok)))]
            parent[v], depth[v] = p, depth[p] + 1
        else:
            parent[v], depth[v] = None, 0
            roots.append(v)
        built.append(v)
    digit = dict(zip(roots, rng.permutation(10)[:len(roots)].tolist()))
    kids = {v: [] for v in parent}
    for v, p in parent.items():
        if p is not None:
            kids[p].append(v)
    avail, stmts, val = list(roots), [], {}
    while avail:                                         # uniformly random topological order
        v = avail.pop(int(rng.integers(len(avail))))
        p = parent[v]
        val[v] = digit[v] if p is None else val[p]
        stmts.append(f"{v}={digit[v]}" if p is None else f"{v}={p}")
        avail += kids[v]
    return ";".join(stmts), (A[-1], str(val[A[-1]])), (B[-1], str(val[B[-1]]))


def probe_sets(seed=515151, hops=(0, 1, 2, 4, 6, 8), n=1000):
    rng = np.random.default_rng(seed)
    return {h: [make_pair(rng, h) for _ in range(n)] for h in hops}


def prompts(pair):
    s, (qa, aa), (qb, ab) = pair
    return (f"{s};?{qa}=", aa), (f"{s};?{qb}=", ab)
