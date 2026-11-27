"""Part 2: (a) query-blind 'bare path' rules with an m-lookup budget, (b) the decoy-chain fix,
(c) training-format effective depth (answers reachable via earlier answered queries), (d) floor-feature check."""
import os
import sys
from collections import Counter, defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_v2 import P, Prob, sample  # noqa: E402

assert "mlx" not in sys.modules
HOPS = (0, 1, 2, 4, 6, 8, 10)


def path_m(pr, r, m):
    node = r
    for _ in range(m):
        if len(pr.kids[node]) != 1:
            return False
        node = pr.kids[node][0]
    return True


def path_rule(pr, m):
    c = [r for r in pr.roots if path_m(pr, r, m)]
    return (1 / len(c)) if pr.chain_root in c else 0.0


print("## (a) query-blind 'bare path' rule: uniform over roots whose first m levels are single-child (m serial lookups, no question)")
print("hop  " + " ".join(f"m={m:<4d}" for m in range(0, 7)))
for h in HOPS:
    probs = sample(11000 + h, h, 10000)
    row = [np.mean([path_rule(pr, m) for pr in probs]) for m in range(0, 7)]
    print(f"h{h:<3d} " + " ".join(f"{a:.3f} " for a in row))

# the same with the official eval set
print("official set:")
off = {h: [Prob(p, a) for p, a in ps] for h, ps in P.reasoning_eval_set().items()}
for h, probs in off.items():
    row = [np.mean([path_rule(pr, m) for pr in probs]) for m in range(0, 7)]
    print(f"h{h:<3d} " + " ".join(f"{a:.3f} " for a in row))


# ---- (b) decoy-chain variant: a second bare chain of the same length, query one of the two at random
def forest_decoy(rng, hops, nvars=P.TASK_VARS):
    letters = list(rng.permutation(list(P._LETTERS))[:nvars])
    A, B = letters[: hops + 1], letters[hops + 1: 2 * hops + 2]
    distract = letters[2 * hops + 2:]
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
    kids = {v: [] for v in parent}
    for v, p in parent.items():
        if p is not None:
            kids[p].append(v)
    avail, stmts, val = list(roots), [], {}
    while avail:
        v = avail.pop(int(rng.integers(len(avail))))
        p = parent[v]
        val[v] = digit[v] if p is None else val[p]
        stmts.append(f"{v}={digit[v]}" if p is None else f"{v}={p}")
        avail += kids[v]
    q = (A if rng.random() < 0.5 else B)[-1]
    return f"{';'.join(stmts)};?{q}=", str(val[q])


print("\n## (b) decoy chain (two bare chains of h hops; ask one end at random), 20 vars where it fits, else 24")
print("hop  vars  first  last  1/R    1-child  path(m=h)  deepest  plurality")
for h in (2, 4, 6, 8, 10):
    nv = 20 if 2 * h + 2 <= 20 else 24
    rng = np.random.default_rng(13000 + h)
    probs = [Prob(*forest_decoy(rng, h, nv)) for _ in range(10000)]
    def u(c, pr):
        return (1 / len(c)) if pr.chain_root in c else 0.0
    first = np.mean([pr.digit[pr.roots[0]] == pr.ans for pr in probs])
    last = np.mean([pr.digit[pr.roots[-1]] == pr.ans for pr in probs])
    invR = np.mean([1 / len(pr.roots) for pr in probs])
    one = np.mean([u([r for r in pr.roots if len(pr.kids[r]) == 1], pr) for pr in probs])
    pth = np.mean([u([r for r in pr.roots if pr.ispath[r] and pr.maxdepth[r] == h], pr) for pr in probs])
    dmax = np.mean([u([r for r in pr.roots if pr.maxdepth[r] == max(pr.maxdepth[x] for x in pr.roots)], pr) for pr in probs])
    plu = np.mean([u([r for r in pr.roots if pr.size[r] == max(pr.size[x] for x in pr.roots)], pr) for pr in probs])
    print(f"h{h:<3d} {nv:<5d} {first:.3f}  {last:.3f} {invR:.3f}  {one:.3f}    {pth:.3f}      {dmax:.3f}    {plu:.3f}")


# ---- (c) training format: how deep must a model chase to answer each query, given earlier answers in the line?
def train_lines(seed, n):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        h = int(rng.integers(P.TRAIN_HOPS[0], P.TRAIN_HOPS[1] + 1))
        stmts, chain_end, val = P._forest(rng, h)
        order = list(rng.permutation(list(val)))
        out.append((h, stmts, order, val, chain_end))
    return out


print("\n## (c) training lines (hops 0-6, 20 queries each): true depth vs effective depth")
print("effective depth = pointer steps until a root statement OR a variable already answered earlier in the line")
true_d, eff_d = Counter(), Counter()
plural_ok, n_ans, first_q, seen_before = 0, 0, Counter(), 0
deep_true_eff = defaultdict(Counter)
for h, stmts, order, val, chain_end in train_lines(4321, 20000):
    pr = Prob(f"{stmts};?{order[0]}=", str(val[order[0]]))
    answered, prev_digits = set(), []
    for j, v in enumerate(order):
        d = pr.depth[v]
        node, e = v, 0
        while pr.parent[node] is not None and node not in answered:
            node, e = pr.parent[node], e + 1
        if node in answered and node != v:
            pass
        elif node == v and v in answered:
            e = 0
        true_d[d] += 1
        eff_d[e] += 1
        deep_true_eff[d][e] += 1
        if prev_digits:
            c = Counter(prev_digits).most_common()
            top = [k for k, cnt in c if cnt == c[0][1]]
            plural_ok += (val[v] in top) / len(top)
        if val[v] in prev_digits:
            seen_before += 1
        if j == 0:
            first_q[d] += 1
        n_ans += 1
        answered.add(v)
        prev_digits.append(val[v])
tot = sum(true_d.values())
print("depth:            " + " ".join(f"{d:>6d}" for d in range(0, 7)))
print("true depth  frac: " + " ".join(f"{true_d[d]/tot:6.3f}" for d in range(0, 7)))
print("eff. depth  frac: " + " ".join(f"{eff_d[d]/tot:6.3f}" for d in range(0, 7)))
print("first-query frac: " + " ".join(f"{first_q[d]/tot:6.4f}" for d in range(0, 7)) + "   (of all answers)")
print(f"answers whose digit already appeared as an earlier answer in the line: {seen_before/n_ans:.3f}")
print(f"query-blind 'most common earlier answer' rule accuracy (queries 2..20): {plural_ok/(n_ans - 20000):.3f}")
print("for true depth d >= 4, distribution of effective depth:")
for d in (4, 5, 6):
    c = deep_true_eff[d]
    s = sum(c.values())
    print(f"  d={d}: " + " ".join(f"e{e}:{c[e]/s:.2f}" for e in range(0, d + 1)))
L = np.mean([len(f"{s};" + "".join(f"?{v}={val[v]}" for v in o) + "\n") for _, s, o, val, _ in train_lines(99, 2000)])
print(f"mean training line length {L:.1f} bytes; answers/line 20 -> answer bytes {20/L:.3f} of reasoning bytes, "
      f"{20/L*P.REASONING_MIX/32:.4f} of all training bytes; eval-format (first query = chain end at depth h) "
      f"{1/L/20*P.REASONING_MIX/32:.5f} of all bytes")
