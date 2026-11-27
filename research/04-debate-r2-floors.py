"""Debater 3, round 2: every [local] number in 04-debate-r2-optimization.md. Pure numpy, no MLX, no model.
    cd <repo> && .venv/bin/python research/04-debate-r2-floors.py
"""
import math
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.getcwd())
import prepare as P  # noqa: E402  (numpy only at import time)

assert "mlx" not in sys.modules

# ---------------------------------------------------------------- helpers
def parse(stmts):
    par, dig = {}, {}
    for s in stmts.split(";"):
        v, x = s.split("=")
        if x.isdigit():
            par[v], dig[v] = None, int(x)
        else:
            par[v] = x
    return par, dig


def depth_of(par, v):
    d = 0
    while par[v] is not None:
        v, d = par[v], d + 1
    return d


def root_of(par, v):
    while par[v] is not None:
        v = par[v]
    return v


def values(par, dig):
    return {v: dig[root_of(par, v)] for v in par}


# ---------------------------------------------------------------- A. the training stream on disk
data = np.fromfile(os.path.join(P.CACHE_DIR, "reason_train_v2.bin"), dtype=np.uint8)
text = data.tobytes().decode()
lines = text.split("\n")[:-1]
L = np.array([len(l) + 1 for l in lines])
n_ans = [l.count("?") for l in lines]
print(f"A. stream: {len(data)/2**20:.0f} MiB, {len(lines)} lines, bytes/line mean {L.mean():.1f} "
      f"(p5 {np.percentile(L,5):.0f}, p95 {np.percentile(L,95):.0f}), answers/line {np.mean(n_ans):.2f}")
# answer byte = the digit after '?<letter>='
arr = data.astype(np.int32)
is_ans = np.zeros(len(arr), bool)
q = np.where(arr[:-3] == ord("?"))[0]
ok = (arr[q + 2] == ord("=")) & (arr[q + 3] >= ord("0")) & (arr[q + 3] <= ord("9"))
is_ans[q[ok] + 3] = True
f_ans = is_ans.mean()
print(f"   answer digits = {100*f_ans:.2f}% of reasoning bytes -> {100*f_ans*P.REASONING_MIX/32:.2f}% of all training targets")

# random 513-byte windows: which answer targets have their whole line inside the window?
rng = np.random.default_rng(0)
Lw = P.SEQ_LEN + 1
tot_ans, ok_ans, first_ok = 0, 0, 0
nw = 20000
for s in rng.integers(0, len(arr) - Lw, size=nw):
    w = arr[s:s + Lw]
    x, y = w[:-1], w[1:]
    tgt_ans = is_ans[s + 1:s + Lw]                    # target y[t] is an answer digit
    seen_nl = np.cumsum(x == 10) > 0                   # a newline at or before input position t
    tot_ans += tgt_ans.sum()
    ok_ans += (tgt_ans & seen_nl).sum()
f_ok = f_ans * ok_ans / tot_ans
print(f"   in {nw} random windows: {tot_ans/nw:.1f} answer targets/row, {100*ok_ans/tot_ans:.1f}% answerable "
      f"(line starts in window) -> answerable = {100*f_ok*P.REASONING_MIX/32:.2f}% of all targets")
a_all = f_ok * P.REASONING_MIX / 32
for W in (1, 4, 8, 10, 16, 30):
    share = W * a_all / (W * a_all + 1 - a_all)
    print(f"   W={W:>2}: answer share of the weighted loss = {100*share:.1f}%  (text rows keep "
          f"{100*(1-P.REASONING_MIX/32)/(W*a_all + 1 - a_all):.1f}% of the weight)")
for mix in (8, 12, 16, 24, 32):
    print(f"   W=1, REASONING_MIX={mix}: answer share = {100*f_ok*mix/32:.1f}%  text rows = {32-mix}/32")

# ---------------------------------------------------------------- B. training-format structure (regenerated)
rng = np.random.default_rng(1234)
dep_all, eff_all, first_dep, roots_n, best_blind, ans_rootfrac = Counter(), Counter(), Counter(), [], [], []
first_eff = Counter()
NL = 20000
for _ in range(NL):
    h = int(rng.integers(P.TRAIN_HOPS[0], P.TRAIN_HOPS[1] + 1))
    stmts, _, val = P._forest(rng, h)
    order = list(rng.permutation(list(val)))
    par, dig = parse(stmts)
    roots_n.append(len(dig))
    sizes = Counter(val.values())
    best_blind.append(max(sizes.values()) / len(val))  # best query-blind per-answer accuracy (plurality value)
    answered = set()
    for i, v in enumerate(order):
        d = depth_of(par, v)
        # effective depth: pointer steps to the nearest ancestor that is a root or already answered in this line
        u, e = v, 0
        while par[u] is not None and u not in answered:
            u, e = par[u], e + 1
        e = 0 if v in answered else e
        dep_all[d] += 1
        eff_all[min(d, e)] += 1
        if i == 0:
            first_dep[d] += 1
        answered.add(v)
tot = sum(dep_all.values())
print("\nB. training lines (regenerated, seed 1234):")
print("   #roots per line: mean %.2f, dist %s" % (np.mean(roots_n), dict(sorted(Counter(roots_n).items()))))
print("   depth of queried var (all 20 queries): " + " ".join(f"{d}:{100*c/tot:.1f}%" for d, c in sorted(dep_all.items())))
print("   effective depth given earlier answers: " + " ".join(f"{d}:{100*c/tot:.1f}%" for d, c in sorted(eff_all.items())))
print("   first query of the line (eval format): " + " ".join(f"{d}:{100*c/NL:.1f}%" for d, c in sorted(first_dep.items())))
print(f"   best query-blind answer accuracy (plurality value): {np.mean(best_blind):.3f}; E[1/#roots] {np.mean([1/r for r in roots_n]):.3f}")

# ---------------------------------------------------------------- C. floors on the eval sets
def floors(problems_struct):
    """problems_struct: list of (prompt, answer, par, dig, q). Returns dict of heuristic accuracies."""
    out = Counter()
    n = len(problems_struct)
    for prompt, ans, par, dig, qv in problems_struct:
        ds = re.findall(r"\d", prompt)
        out["first"] += ds[0] == ans
        out["last"] += ds[-1] == ans
        out["1/#roots"] += 1 / len(dig)
        val = values(par, dig)
        sizes = Counter(val.values())
        # plurality value (largest tree), ties -> expected
        m = max(sizes.values()); top = [k for k, c in sizes.items() if c == m]
        out["plurality"] += (int(ans) in top) / len(top)
        # value of the deepest variable(s) (query-blind 'end of the longest chain')
        dep = {v: depth_of(par, v) for v in par}
        dm = max(dep.values()); deep_vals = Counter(val[v] for v in par if dep[v] == dm)
        mv = max(deep_vals.values()); topd = [k for k, c in deep_vals.items() if c == mv]
        out["deepest"] += (int(ans) in topd) / len(topd)
        # last-defined variable's value (query-blind 'resolve the final statement')
        lastv = prompt.split(";")[-2].split("=")[0]
        out["last-stmt value"] += str(val[lastv]) == ans
    return {k: v / n for k, v in out.items()}


def struct_sets(seed, hops, n):
    rng = np.random.default_rng(seed)
    sets = {}
    for h in hops:
        L_ = []
        for _ in range(n):
            stmts, qv, val = P._forest(rng, h)
            par, dig = parse(stmts)
            L_.append((f"{stmts};?{qv}=", str(val[qv]), par, dig, qv))
        sets[h] = L_
    return sets


# sanity: the regenerated sets equal the official ones
off = P.reasoning_eval_set()
mine = struct_sets(424242, P.EVAL_HOPS_ID + P.EVAL_HOPS_OOD, P.EVAL_PER_HOP)
assert all([(p, a) for p, a, *_ in mine[h]] == off[h] for h in off)
rq = np.random.default_rng(7)
quick_off = {h: [P.make_problem(rq, h) for _ in range(100)] for h in (0, 1, 2, 4, 6, 8)}
quick = struct_sets(7, (0, 1, 2, 4, 6, 8), 100)
assert all([(p, a) for p, a, *_ in quick[h]] == quick_off[h] for h in quick)

print("\nC1. official eval sets (400/hop): heuristic accuracy (letter-blind / query-blind)")
print("   hop  first  last  1/#roots plurality deepest last-stmt | reason_floor")
for h, s in mine.items():
    f = floors(s)
    print(f"   {h:>3}  {f['first']:.3f}  {f['last']:.3f}  {f['1/#roots']:.3f}    {f['plurality']:.3f}    "
          f"{f['deepest']:.3f}  {f['last-stmt value']:.3f}   | {P.position_floor([(p, a) for p, a, *_ in s]):.3f}")
print("\nC2. base-v2 quick-eval sets (pretrain_base.quick_eval: seed 7, 100/hop) -- the decision-rule floors")
print("   hop  first  last  1/#roots plurality deepest last-stmt | pos_floor  SE@floor  acc_needed(+4SE)")
qf = {}
for h, s in quick.items():
    f = floors(s)
    pf = P.position_floor([(p, a) for p, a, *_ in s])
    fl = max(pf, f["plurality"], f["1/#roots"])
    qf[h] = fl
    se = math.sqrt(fl * (1 - fl) / 100)
    print(f"   {h:>3}  {f['first']:.3f}  {f['last']:.3f}  {f['1/#roots']:.3f}    {f['plurality']:.3f}    "
          f"{f['deepest']:.3f}  {f['last-stmt value']:.3f}   | {pf:.3f}      {se:.3f}     {fl + 4*se:.3f}")
F012 = np.mean([qf[h] for h in (0, 1, 2)])
se012 = math.sqrt(np.mean([qf[h] * (1 - qf[h]) for h in (0, 1, 2)]) / 300)
print(f"   hops 0-2 pooled floor {F012:.3f}, SE of the 300-problem mean at the floor {se012:.3f}")

# ---------------------------------------------------------------- D. LR schedules (legacy vs R), 300 s
def legacy(sps, budget=300, peak=1e-3, warm=100, final=0.05):
    lrs, t, step = [], 0.0, 0
    while t < budget:
        prog = t / budget
        lrs.append(peak * min(1, (step + 1) / warm) * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * prog))))
        if step >= 2:
            t += sps
        step += 1
    return np.array(lrs)


def recipe_r(sps, budget=300, peak=1e-4, warm=10, decay=0.25):
    lrs, t, step = [], 0.0, 0
    while t < budget:
        prog = t / budget
        lrs.append(peak * min(1, (step + 1) / warm) * (1.0 if prog < 1 - decay else (1 - prog) / decay))
        if step >= 2:
            t += sps
        step += 1
    return np.array(lrs)


print("\nD. 300-s warm-start schedules")
for sps in (1.18, 1.28, 1.5):
    a, b = legacy(sps), recipe_r(sps)
    print(f"   {sps:.2f} s/step: {len(a)} steps | legacy peak {a.max():.2e} at step {a.argmax()}, "
          f"{(a > 3e-4).sum()} steps > 3e-4, area {a.sum():.4f} = {a.sum()/2.5e-5:.0f} steps at base-v2's final LR | "
          f"R area {b.sum():.4f} ({b.sum()/a.sum():.2f}x legacy)")
print(f"   base-v2 schedule: peak 5e-4, final {5e-4*0.05:.1e}; v1 base final {1e-3*0.05:.1e}")

# ===================== line entropy and query-blind answer CE =====================
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.getcwd())
import prepare as P  # noqa: E402

assert "mlx" not in sys.modules
rng = np.random.default_rng(99)
N = 20000
H_lhs = sum(math.log2(27 - k) for k in range(1, 21))     # k-th LHS letter: uniform over undefined letters
H_q = sum(math.log2(21 - k) for k in range(1, 21))       # k-th query letter: uniform over unqueried vars
H_root_dig, H_parent, H_type, ce_uniform_roots, ce_blind = [], [], [], [], []
root_by_pos = np.zeros(20); cnt = np.zeros(20)
lines = []
for _ in range(N):
    h = int(rng.integers(0, 7))
    stmts, _, val = P._forest(rng, h)
    ss = stmts.split(";")
    nroots = sum(s.split("=")[1].isdigit() for s in ss)
    H_root_dig.append(sum(math.log2(11 - j) for j in range(1, nroots + 1)))
    hp = 0.0
    for k, s in enumerate(ss):
        rhs = s.split("=")[1]
        root_by_pos[k] += rhs.isdigit(); cnt[k] += 1
        if not rhs.isdigit():
            hp += math.log2(k)           # upper bound: parent uniform over the k defined vars
    H_parent.append(hp)
    sizes = Counter(val.values())
    p = np.array([sizes[val[v]] / 20 for v in val])
    ce_blind.append(-np.mean(np.log2(p)))           # query-blind Bayes: p(answer) = its tree's share
    ce_uniform_roots.append(math.log2(nroots))
pr = root_by_pos / cnt
H_type = sum(-(q * math.log2(q) + (1 - q) * math.log2(1 - q)) for q in pr if 0 < q < 1)
tot = H_lhs + H_q + np.mean(H_root_dig) + np.mean(H_parent) + H_type
print(f"LHS {H_lhs:.1f}  query letters {H_q:.1f}  root digits {np.mean(H_root_dig):.1f}  parents<= {np.mean(H_parent):.1f}"
      f"  root/child {H_type:.1f}  => <= {tot:.0f} bits per 161-byte line = {tot/161:.3f} bits/byte (answers 0)")
print(f"answer CE per digit: uniform over roots {np.mean(ce_uniform_roots):.2f} bits; query-blind Bayes (tree share) "
      f"{np.mean(ce_blind):.2f} bits; uniform over 10 digits {math.log2(10):.2f}")
for name, ce in (("no lookup, uniform roots", np.mean(ce_uniform_roots)), ("query-blind plurality-aware", np.mean(ce_blind)),
                 ("perfect lookup", 0.0)):
    r = (tot + 20 * ce) / 161
    print(f"  reasoning-row bpb floor if {name:28s}: {r:.3f}  -> batch train_bpb with text at 1.51: {0.75*1.51 + 0.25*r:.4f}")
