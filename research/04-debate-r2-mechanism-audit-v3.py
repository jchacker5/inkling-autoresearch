"""Debate round 2 (Debater 1, mechanism): shortcut audit of task v3 (commit 5539a61). Pure numpy, no MLX, no model.

    .venv/bin/python research/04-debate-r2-mechanism-audit-v3.py

v3: two chains A, B of exactly h hops (24 variables), twin prompts ask both ends, a pair counts only if both
are right. A query-blind rule gives both twins one answer, so it scores 0. The question here: how far do rules
that use only a CHEAP one-bit query cue, plus query-blind structure, get above PAIR_BOUND = .25?
  S1  positional binding: "is the query the last statement's variable?" (a fixed-offset compare, no retrieval);
      if yes, chase the last statement; if no, answer the other deepest chain end. Needs a chase, not a lookup.
  S2  the same bit + root order, no chasing at all: yes -> digit of the later of the two chain roots, no -> earlier.
  S3  bind the query (one lookup), compare its statement position with the other chain end's, answer the
      later/earlier chain root accordingly. One lookup, no chasing.
  S4  bind, chase k < h steps, then answer whichever chain root lies nearest before the reached statement.
Also: how often structure alone finds both chains, and what the v3 training lines pay for these rules.
"""
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import prepare as P  # noqa: E402

assert "mlx" not in sys.modules
assert P.TASK_VARS == 24 and hasattr(P, "make_pair"), "this audit is for task v3 (prepare.py at 5539a61 or later)"


class Twin:
    def __init__(self, pair):
        (pa, aa), (pb, ab) = pair
        body = pa[:-1].split(";?")[0]
        self.st = [(s[0], s[2]) for s in body.split(";")]
        self.ends = (pa[:-1].split(";?")[1], pb[:-1].split(";?")[1])
        self.ans = (aa, ab)
        self.par = dict(self.st)
        self.pos = {v: i for i, (v, _) in enumerate(self.st)}
        self.depth, self.root = {}, {}
        for v in self.par:
            u, d = v, 0
            while not self.par[u].isdigit():
                u, d = self.par[u], d + 1
            self.depth[v], self.root[v] = d, u
        self.croots = [self.root[e] for e in self.ends]
        self.last = self.st[-1][0]

    def val(self, v):
        return self.par[self.root[v]]


def pair_rules(t, h):
    out = {}
    a, b = t.ends
    ra, rb = t.croots
    # structure: are the two chain ends exactly the two deepest variables?
    dmax = sorted(t.depth.values())[-2:]
    deepest = [v for v in t.par if t.depth[v] >= dmax[0]]
    out["structure: the 2 deepest vars are exactly the 2 chain ends"] = float(set(deepest) == {a, b})
    out["last statement is a chain end"] = float(t.last in (a, b))
    # S1: positional binding + chase
    def s1(q):
        if q == t.last:
            return t.val(t.last)
        others = [v for v in deepest if v != t.last]
        return t.val(others[0]) if len(others) == 1 else None
    out["S1 pair: last-stmt bit + chase (no lookup)"] = float(all(s1(q) == ans for q, ans in zip(t.ends, t.ans)))
    # S2: last-statement bit + root order, no chase
    early, late = sorted((ra, rb), key=lambda r: t.pos[r])
    s2 = lambda q: t.par[late] if q == t.last else t.par[early]
    out["S2 pair: last-stmt bit + root order (no chase)"] = float(all(s2(q) == ans for q, ans in zip(t.ends, t.ans)))
    # S3: one lookup (bind) + end order + root order, no chase
    s3 = lambda q, o: t.par[late] if t.pos[q] > t.pos[o] else t.par[early]
    out["S3 pair: bind + end order -> root order (no chase)"] = float(s3(a, b) == t.ans[0] and s3(b, a) == t.ans[1])
    # S4: bind, chase k steps, then the chain root nearest before the reached statement
    for k in sorted({0, 1, h // 2, max(h - 1, 0)}):
        def s4(q):
            u = q
            for _ in range(k):
                if t.par[u].isdigit():
                    break
                u = t.par[u]
            before = [r for r in (ra, rb) if t.pos[r] <= t.pos[u]]
            r = max(before, key=lambda r: t.pos[r]) if before else early
            return t.par[r]
        out[f"S4 pair: bind + chase {k} + nearest chain root"] = float(all(s4(q) == ans for q, ans in zip(t.ends, t.ans)))
    return out


def train_lines(n=20000):
    """v3 training lines (seed 1234, as prepare._reasoning_stream): the first query is a chain end."""
    rng = np.random.default_rng(1234)
    rec = []
    blob = []
    for _ in range(n):
        hops = int(rng.integers(P.TRAIN_HOPS[0], P.TRAIN_HOPS[1] + 1))
        stmts, ends, val = P._forest(rng, hops)
        first = ends[int(rng.integers(2))]
        rest = [v for v in rng.permutation(list(val)) if v != first]
        blob.append(f"{stmts};" + "".join(f"?{v}={val[v]}" for v in [first] + rest) + "\n")
        st = [(s[0], s[2]) for s in stmts.split(";")]
        pos = {v: i for i, (v, _) in enumerate(st)}
        par = dict(st)

        def root(v):
            while not par[v].isdigit():
                v = par[v]
            return v
        other = ends[0] if first == ends[1] else ends[1]
        r1, r2 = root(first), root(other)
        early, late = sorted((r1, r2), key=lambda r: pos[r])
        last = st[-1][0]
        size = Counter(val.values())
        top = [d for d, c in size.items() if c == max(size.values())]
        rec.append((hops, float(first == last),
                    float((late if first == last else early) == r1),          # S2 on the first query
                    float((late if pos[first] > pos[other] else early) == r1),  # S3 on the first query
                    (1.0 / len(top)) if val[first] in top else 0.0))          # statement plurality on the first query
    data = os.path.join(P.CACHE_DIR, "reason_train_v3.bin")
    if os.path.exists(data):
        disk = np.memmap(data, dtype=np.uint8, mode="r")
        s = "".join(blob).encode()
        print(f"\n== training lines: regenerated bytes match reason_train_v3.bin[:1 MiB]: {bytes(disk[:1 << 20]) == s[:1 << 20]}")
    R = np.array(rec)
    print("first query (a chain end, weight 30x), by line hop:")
    print("  hop  P(first query is the last statement)  S2 right  S3 right  plurality right")
    for h in range(P.TRAIN_HOPS[0], P.TRAIN_HOPS[1] + 1):
        m = R[:, 0] == h
        print(f"  {h:>3}  {R[m, 1].mean():.3f}                                  {R[m, 2].mean():.3f}     {R[m, 3].mean():.3f}     {R[m, 4].mean():.3f}")
    print(f"  all  {R[:, 1].mean():.3f}                                  {R[:, 2].mean():.3f}     {R[:, 3].mean():.3f}     {R[:, 4].mean():.3f}")


def main():
    hops = (0, 1, 2, 4, 6, 8, 10)
    rng = np.random.default_rng(20261124)
    res = {h: [pair_rules(Twin(P.make_pair(rng, h)), h) for _ in range(4000)] for h in hops}
    off = P.reasoning_eval_set()
    off_res = {h: [pair_rules(Twin((s[i], s[i + 1])), h) for i in range(0, len(s), 2)] for h, s in off.items()}
    names = []
    for h in hops:
        for r in res[h]:
            names += [k for k in r if k not in names]
    print(f"== v3 pair rules, 4,000 fresh twin pairs per hop; PAIR_BOUND = {P.PAIR_BOUND}")
    print(f"{'rule':55s}" + "".join(f"  h{h:<5d}" for h in hops))
    for n in names:
        print(f"{n:55s}" + "".join(f"  {np.mean([r[n] for r in res[h] if n in r]):.3f} " for h in hops))
    print(f"\n== same rules on the official v3 set (200 pairs per hop)")
    oh = sorted(off_res)
    print(f"{'rule':55s}" + "".join(f"  h{h:<5d}" for h in oh) + "   ID/OOD")
    for n in names:
        v = {h: np.mean([r[n] for r in off_res[h] if n in r]) for h in oh}
        print(f"{n:55s}" + "".join(f"  {v[h]:.3f} " for h in oh)
              + f"   {np.mean([v[h] for h in P.EVAL_HOPS_ID]):.3f}/{np.mean([v[h] for h in P.EVAL_HOPS_OOD]):.3f}")
    q = np.random.default_rng(7)
    quick = {h: P.twin_problems(q, h, 100) for h in (0, 1, 2, 4, 6, 8)}
    print("\n== base-v2 quick-eval sets (seed 7, 50 pairs per hop, SE ~.06): S1 / S2 / S3 pair scores")
    for h, s in quick.items():
        rr = [pair_rules(Twin((s[i], s[i + 1])), h) for i in range(0, len(s), 2)]
        print(f"  h{h}: S1 {np.mean([r['S1 pair: last-stmt bit + chase (no lookup)'] for r in rr]):.2f}"
              f"  S2 {np.mean([r['S2 pair: last-stmt bit + root order (no chase)'] for r in rr]):.2f}"
              f"  S3 {np.mean([r['S3 pair: bind + end order -> root order (no chase)'] for r in rr]):.2f}")
    train_lines()


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------------------------------------
# Part 2: (a) the same no-chase rules with CHEAP structure (k-level single-child filters instead of known chains);
# (b) a fix: "crossed twins". Swap the parents of A[i] and B[i] at one hop i (two bytes change; every position,
# letter and digit stays). The crossed problem is a valid v3 problem whose two answers are swapped, so a rule
# that never reads the pointers at hop i answers both versions identically and gets at most one of them right.
def unary_roots(t, k):
    kids = {v: [c for c, p in t.st if p == v] for v in t.par}
    out = []
    for r in (v for v, p in t.st if p.isdigit()):
        u, ok = r, True
        for _ in range(k):
            if len(kids[u]) != 1:
                ok = False
                break
            u = kids[u][0]
        if ok:
            out.append(r)
    return out


def s3_cheap(t, k, rng):
    """S3 with the two chain roots guessed by a k-level single-child filter and the other end = the other twin's
    query (known structurally only as 'the other candidate end'); falls back to random candidates."""
    cands = unary_roots(t, k)
    if len(cands) < 2:
        return 0.0
    if len(cands) > 2:
        cands = list(rng.choice(cands, 2, replace=False))
    early, late = sorted(cands, key=lambda r: t.pos[r])
    a, b = t.ends
    s3 = lambda q, o: t.par[late] if t.pos[q] > t.pos[o] else t.par[early]
    return float(s3(a, b) == t.ans[0] and s3(b, a) == t.ans[1])


def crossed(t, rng):
    """Return the crossed twin (prompt texts) or None if no hop i keeps parents before children."""
    A = [t.ends[0]]
    B = [t.ends[1]]
    for ch in (A, B):
        while not t.par[ch[-1]].isdigit():
            ch.append(t.par[ch[-1]])
        ch.reverse()                                   # root first
    h = len(A) - 1
    ok = [i for i in range(1, h + 1) if t.pos[B[i - 1]] < t.pos[A[i]] and t.pos[A[i - 1]] < t.pos[B[i]]]
    if not ok:
        return None
    i = ok[int(rng.integers(len(ok)))]
    par = dict(t.par)
    par[A[i]], par[B[i]] = B[i - 1], A[i - 1]
    body = ";".join(f"{v}={par[v]}" for v, _ in t.st)
    def val(v):
        while not par[v].isdigit():
            v = par[v]
        return par[v]
    return [(f"{body};?{q}=", val(q)) for q in t.ends]


def part2():
    rng = np.random.default_rng(99)
    print("\n== part 2a: no-chase S3 with cheap structure (k-level single-child filter), and the consistent-bit bound")
    print("   (any rule that knows both chain values and gives the twins DIFFERENT answers by an uninformative bit scores .5)")
    for h in (2, 4, 6, 8, 10):
        ts = [Twin(P.make_pair(rng, h)) for _ in range(3000)]
        row = [f"k={k}: {np.mean([s3_cheap(t, k, rng) for t in ts]):.3f}" for k in (1, 2, 3) if k <= h]
        both = np.mean([len(unary_roots(t, min(h, 3))) == 2 for t in ts])
        print(f"  h{h:<3d} S3 pair " + "  ".join(row) + f"   | P(min(h,3)-level filter finds exactly the 2 roots) {both:.3f}")
    print("\n== part 2b: crossed twins (quad = both twins right on the original AND the crossed problem)")
    print("  hop  P(valid crossing)  S2 quad  S3 quad (known structure)  S1 quad (chases, should survive)")
    for h in (1, 2, 4, 6, 8, 10):
        n_ok, s1q, s2q, s3q = 0, [], [], []
        for _ in range(3000):
            t = Twin(P.make_pair(rng, h))
            c = crossed(t, rng)
            if c is None:
                continue
            n_ok += 1
            tc = Twin(c)
            r0, r1 = pair_rules(t, h), pair_rules(tc, h)
            s1q.append(r0["S1 pair: last-stmt bit + chase (no lookup)"] * r1["S1 pair: last-stmt bit + chase (no lookup)"])
            s2q.append(r0["S2 pair: last-stmt bit + root order (no chase)"] * r1["S2 pair: last-stmt bit + root order (no chase)"])
            s3q.append(r0["S3 pair: bind + end order -> root order (no chase)"] * r1["S3 pair: bind + end order -> root order (no chase)"])
        print(f"  {h:>3}  {n_ok / 3000:.3f}              {np.mean(s2q):.3f}    {np.mean(s3q):.3f}                     {np.mean(s1q):.3f}")


if __name__ == "__main__":
    part2()


# ----------------------------------------------------------------------------------------------------------
# Part 2c: the same rules if _forest sampled a UNIFORM topological order: pop an available node with probability
# proportional to its subtree size (exact for forests: e(F - v)/e(F) = |subtree(v)|/n). A 3-line change.
def forest_uniform(rng, hops):
    names = list(rng.permutation(list(P._LETTERS))[:P.TASK_VARS])
    A, B, distract = names[: hops + 1], names[hops + 1: 2 * hops + 2], names[2 * hops + 2:]
    cap = max(hops, 3)
    parent, depth = {A[0]: None, B[0]: None}, {A[0]: 0, B[0]: 0}
    for ch in (A, B):
        for i in range(1, hops + 1):
            parent[ch[i]], depth[ch[i]] = ch[i - 1], i
    roots, built = [A[0], B[0]], []
    for v in distract:
        ok = [u for u in built if depth[u] < cap]
        if ok and (len(roots) >= 10 or rng.random() < 0.5):
            p = ok[int(rng.integers(len(ok)))]
            parent[v], depth[v] = p, depth[p] + 1
        else:
            parent[v], depth[v] = None, 0
            roots.append(v)
        built.append(v)
    digit = dict(zip(roots, rng.permutation(10)[: len(roots)].tolist()))
    kids = {v: [] for v in names}
    for v, p in parent.items():
        if p is not None:
            kids[p].append(v)
    size = {}
    def sz(v):
        if v not in size:
            size[v] = 1 + sum(sz(c) for c in kids[v])
        return size[v]
    avail, stmts, val = list(roots), [], {}
    while avail:
        w = np.array([sz(v) for v in avail], dtype=float)
        v = avail.pop(int(rng.choice(len(avail), p=w / w.sum())))
        p = parent[v]
        val[v] = digit[v] if p is None else val[p]
        stmts.append(f"{v}={digit[v]}" if p is None else f"{v}={p}")
        avail += kids[v]
    s = ";".join(stmts)
    return [(f"{s};?{q}=", str(val[q])) for q in (A[-1], B[-1])]


def part2c():
    rng = np.random.default_rng(5)
    print("\n== part 2c: uniform topological order (subtree-size-weighted pops) vs v3's random-available order")
    print("  hop  S3 pair known-structure (v3 -> uniform)   S1 pair (v3 -> uniform)   last stmt is a chain end (v3 -> uniform)")
    for h in (2, 4, 6, 8, 10):
        u = [pair_rules(Twin(forest_uniform(rng, h)), h) for _ in range(3000)]
        v = [pair_rules(Twin(P.make_pair(rng, h)), h) for _ in range(3000)]
        k3 = "S3 pair: bind + end order -> root order (no chase)"
        k1 = "S1 pair: last-stmt bit + chase (no lookup)"
        kl = "last statement is a chain end"
        print(f"  {h:>3}  {np.mean([r[k3] for r in v]):.3f} -> {np.mean([r[k3] for r in u]):.3f}"
              f"                         {np.mean([r[k1] for r in v]):.3f} -> {np.mean([r[k1] for r in u]):.3f}"
              f"                {np.mean([r[kl] for r in v]):.3f} -> {np.mean([r[kl] for r in u]):.3f}")


if __name__ == "__main__":
    part2c()


# ----------------------------------------------------------------------------------------------------------
# Part 2d: ZERO-lookup consistent rule. Structure gives the two chain roots and the two chain ends' letters;
# a bit that differs between the twins (is the asked letter alphabetically smaller than the other end's?)
# assigns the earlier root to one twin and the later root to the other: right on both or wrong on both.
def part2d():
    rng = np.random.default_rng(11)
    print("\n== part 2d: zero-lookup 'opposite answers' rule (alphabet bit + structure); hop 0 uses childless roots")
    for h in (0, 1, 2, 4, 6, 8, 10):
        known, cheap = [], []
        for _ in range(3000):
            t = Twin(P.make_pair(rng, h))
            a, b = t.ends
            for roots, store in ((list(t.croots), known), (None, cheap)):
                if roots is None:
                    k = min(h, 3)
                    if h == 0:
                        kids = Counter(p for _, p in t.st if not p.isdigit())
                        cands = [v for v, p in t.st if p.isdigit() and kids[v] == 0]
                    else:
                        cands = unary_roots(t, k)
                    if len(cands) < 2:
                        store.append(0.0)
                        continue
                    roots = list(rng.choice(cands, 2, replace=False)) if len(cands) > 2 else cands
                early, late = sorted(roots, key=lambda r: t.pos[r])
                f = lambda q, o: t.par[early] if q < o else t.par[late]
                store.append(float(f(a, b) == t.ans[0] and f(b, a) == t.ans[1]))
        print(f"  h{h:<3d} known structure {np.mean(known):.3f}   cheap structure (min(h,3)-level filter) {np.mean(cheap):.3f}")


if __name__ == "__main__":
    part2d()
