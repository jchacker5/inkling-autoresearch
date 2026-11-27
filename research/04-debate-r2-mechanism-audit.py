"""Debate round 2 (Debater 1, mechanism): shortcut audit of task v2. Pure Python + numpy. No MLX, no model.

    .venv/bin/python research/04-debate-r2-mechanism-audit.py

Task v2 is prepare.py at commit 4a1c4dd (HEAD moved to task v3 at 5539a61; see 04-debate-r2-mechanism-audit-v3.py).
The v2 prepare.py is loaded from git, so this script reproduces the v2 numbers at any later commit.
Uses prepare's numpy code only (evaluate() would import mlx lazily and is never called).
Parts:
  A  query-free rules on the official set (reasoning_eval_set) and on 4,000 fresh problems per hop
  B  query-dependent but shallow rules: chase k < h steps, then guess a root by position
  C  structure: is the chain end identifiable without reading the query?
  D  training stream (seed 1234, the bytes in reason_train_v2.bin): fresh-chase depth per answer
  E  encode_prompt: what the filler prefix adds for CHUNK = 1, 2, 4
  F  floors on the base-v2 quick-eval sets (pretrain_base.quick_eval: seed 7, 100/hop)
"""
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2_COMMIT = "4a1c4dd"


def _load_v2_prepare():
    import subprocess
    import types
    src = subprocess.run(["git", "show", f"{V2_COMMIT}:prepare.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("prepare_v2")
    mod.__file__ = os.path.join(ROOT, "prepare_v2_from_git.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


P = _load_v2_prepare()
assert P.TASK_VARS == 20 and not hasattr(P, "make_pair"), "expected task v2"
assert "mlx" not in sys.modules and "mlx.core" not in sys.modules, "this audit must never load MLX"


def parse(prompt):
    body, q = prompt[:-1].split(";?")
    return [(s[0], s[2]) for s in body.split(";")], q


class Prob:
    def __init__(self, prompt, ans):
        self.st, self.q = parse(prompt)
        self.ans = ans
        self.par = dict(self.st)
        self.pos = {v: i for i, (v, _) in enumerate(self.st)}
        self.roots = [v for v, r in self.st if r.isdigit()]          # in statement order
        self.kids = Counter(r for _, r in self.st if not r.isdigit())
        self.rootof, self.depth = {}, {}
        for v in self.par:
            u, d = v, 0
            while not self.par[u].isdigit():
                u, d = self.par[u], d + 1
            self.rootof[v], self.depth[v] = u, d
        self.size = Counter(self.rootof.values())
        self.tdepth = {}
        for v, rt in self.rootof.items():
            self.tdepth[rt] = max(self.tdepth.get(rt, 0), self.depth[v])
        self.croot = self.rootof[self.q]
        assert self.par[self.croot] == ans

    def dig(self, root):
        return self.par[root]

    def ancestor(self, v, k):
        for _ in range(k):
            if self.par[v].isdigit():
                break
            v = self.par[v]
        return v

    def nearest_root_before(self, v):
        """Root statement nearest before v's statement (v itself if v is a root)."""
        p = self.pos[v]
        rs = [r for r in self.roots if self.pos[r] <= p]
        return rs[-1]

    def roots_before(self, v):
        return [r for r in self.roots if self.pos[r] <= self.pos[v]]


def exp_acc(cands, gold_root):
    """Expected accuracy of picking uniformly among candidate roots (ties split evenly)."""
    return (1.0 / len(cands)) if gold_root in cands else 0.0


def rules_query_free(p):
    out = {}
    out["first digit"] = float(p.dig(p.roots[0]) == p.ans)
    out["last digit"] = float(p.dig(p.roots[-1]) == p.ans)
    out["random root (1/R)"] = 1.0 / len(p.roots)
    nonsing = [r for r in p.roots if p.kids[r] > 0]
    out["random root with >=1 child"] = exp_acc(nonsing, p.croot) if nonsing else 0.0
    one = [r for r in p.roots if p.kids[r] == 1]
    out["random root with exactly 1 child (name seen twice)"] = exp_acc(one, p.croot) if one else 0.0
    big = max(p.size.values())
    out["largest tree = plurality value"] = exp_acc([r for r in p.roots if p.size[r] == big], p.croot)
    deep = max(p.tdepth.values())
    out["deepest tree"] = exp_acc([r for r in p.roots if p.tdepth[r] == deep], p.croot)
    out["resolve the LAST statement's variable"] = float(p.rootof[p.st[-1][0]] == p.croot)
    dv = [v for v in p.par if p.depth[v] == deep]
    out["resolve a deepest variable"] = np.mean([p.rootof[v] == p.croot for v in dv])
    # deepest *non-branching* tree (a path): the chain is always a path
    paths = [r for r in p.roots if all(p.kids[v] <= 1 for v in p.par if p.rootof[v] == r)]
    pd = max(p.tdepth[r] for r in paths)
    out["deepest path-shaped tree"] = exp_acc([r for r in paths if p.tdepth[r] == pd], p.croot)
    # counting rules with no chasing at all: a root's letter occurs once (childless) or twice (one child)
    zero = [r for r in p.roots if p.kids[r] == 0]
    out["random childless root (letter occurs once)"] = exp_acc(zero, p.croot) if zero else 0.0
    # query-free local structure: roots whose first k levels are a single path (each node exactly 1 child)
    for k in (2, 3, 4):
        out[f"path filter, {k} levels (root->1 child->1 child..)"] = path_filter(p, k)
    return out


def path_filter(p, k):
    cands = []
    for r in p.roots:
        u, ok = r, True
        for _ in range(k):
            ch = [v for v, rr in p.st if rr == u]
            if len(ch) != 1:
                ok = False
                break
            u = ch[0]
        if ok:
            cands.append(r)
    return exp_acc(cands, p.croot) if cands else 0.0


def rules_shallow(p, h):
    """Chase k = 0..h steps from the query, then guess by position. k = h is exact."""
    out = {}
    for k in range(0, h + 1):
        u = p.ancestor(p.q, k)
        out[f"chase {k}, nearest root before"] = float(p.nearest_root_before(u) == p.croot)
        out[f"chase {k}, random root before"] = exp_acc(p.roots_before(u), p.croot)
    return out


def table(sets, rulefn, title, hops=None):
    hops = hops or sorted(sets)
    print(f"\n== {title}")
    res = {h: [rulefn(Prob(pr, a)) for pr, a in sets[h]] for h in hops}
    names = list(res[hops[0]][0].keys())
    print(f"{'rule':52s}" + "".join(f"  h{h:<5d}" for h in hops) + "   ID/OOD mean")
    for n in names:
        vals = {h: np.mean([r[n] for r in res[h] if n in r]) for h in hops}
        idm = np.mean([vals[h] for h in P.EVAL_HOPS_ID if h in vals]) if any(h in vals for h in P.EVAL_HOPS_ID) else float("nan")
        oodm = np.mean([vals[h] for h in P.EVAL_HOPS_OOD if h in vals]) if any(h in vals for h in P.EVAL_HOPS_OOD) else float("nan")
        print(f"{n:52s}" + "".join(f"  {vals[h]:.3f} " for h in hops) + f"   {idm:.3f}/{oodm:.3f}")
    return res


def structure(sets, hops):
    print("\n== C. structure of the eval problems (fraction of problems)")
    print(f"{'property':60s}" + "".join(f"  h{h:<5d}" for h in hops))
    rows = {
        "chain is the unique largest tree": lambda p: max(s for r, s in p.size.items() if r != p.croot) < p.size[p.croot] if len(p.size) > 1 else True,
        "query is the unique deepest variable": lambda p: sum(1 for v in p.par if p.depth[v] >= p.depth[p.q]) == 1,
        "query's statement is the last statement": lambda p: p.pos[p.q] == len(p.st) - 1,
        "chain root is the first statement": lambda p: p.pos[p.croot] == 0,
        "mean number of roots R": lambda p: len(p.roots),
        "mean statement index of the query (0..19)": lambda p: p.pos[p.q],
        "mean statement index of the chain root": lambda p: p.pos[p.croot],
        "mean gap between consecutive chain statements": lambda p: np.mean(
            [p.pos[v] - p.pos[p.par[v]] for v in p.par if p.rootof[v] == p.croot and not p.par[v].isdigit()]) if p.depth[p.q] > 0 else float("nan"),
    }
    probs = {h: [Prob(pr, a) for pr, a in sets[h]] for h in hops}
    for name, f in rows.items():
        print(f"{name:60s}" + "".join(f"  {np.nanmean([float(f(p)) for p in probs[h]]):.3f} " for h in hops))


def train_stream(nlines=20000):
    """Re-generate the start of the v2 training stream exactly (prepare._reasoning_stream, seed 1234) and
    measure, for every answer, its depth and its *fresh* depth: pointer steps from the queried variable to the
    nearest ancestor-or-self that is a root or was already answered earlier in the same line."""
    rng = np.random.default_rng(1234)
    lines, recs, stmt_plur, last_stmt = [], [], [], []
    for li in range(nlines):
        hops = int(rng.integers(P.TRAIN_HOPS[0], P.TRAIN_HOPS[1] + 1))
        stmts, chain_end, val = P._forest(rng, hops)
        order = list(rng.permutation(list(val)))
        line = f"{stmts};" + "".join(f"?{v}={val[v]}" for v in order) + "\n"
        lines.append(line)
        st = [(s[0], s[2]) for s in stmts.split(";")]
        par = dict(st)
        depth = {}
        for v, r in st:
            depth[v] = 0 if r.isdigit() else depth[r] + 1
        answered, seen_vals = set(), Counter()
        tsize = Counter(val.values())                    # statement plurality = value of the largest tree
        top_t = [d for d, c in tsize.items() if c == max(tsize.values())]
        last_val = val[st[-1][0]]
        for qi, v in enumerate(order):
            stmt_plur.append((1.0 / len(top_t)) if val[v] in top_t else 0.0)
            last_stmt.append(float(val[v] == last_val))
            u, f = v, 0
            while not par[u].isdigit() and u not in answered:
                u, f = par[u], f + 1
            plural = (max(seen_vals.values()) if seen_vals else 0)
            top = [d for d, c in seen_vals.items() if c == plural] if seen_vals else []
            recs.append((hops, qi, depth[v], f, v == chain_end,
                         (1.0 / len(top)) if str(val[v]) in top else 0.0))
            answered.add(v)
            seen_vals[str(val[v])] += 1
    blob = "".join(lines).encode()
    path = os.path.join(P.CACHE_DIR, "reason_train_v2.bin")
    if os.path.exists(path):
        disk = np.memmap(path, dtype=np.uint8, mode="r")
        n = min(len(blob), 1 << 20)
        print(f"\n== D. training stream: regenerated bytes match reason_train_v2.bin[:{n}]: {bytes(disk[:n]) == blob[:n]}")
    R = np.array(recs, dtype=np.float64)
    hops, qi, d, f, is_end, plur = R.T
    n_ans = len(R)
    nbytes = len(blob)
    print(f"lines {nlines}, mean line length {nbytes / nlines:.1f} bytes, answers {n_ans} "
          f"({n_ans / nbytes:.1%} of reasoning bytes; x 8/32 rows = {n_ans / nbytes * P.REASONING_MIX / 32:.2%} of all training targets)")
    print("depth d of the queried variable   : " + "  ".join(f"d{k} {np.mean(d == k):.3f}" for k in range(7)))
    print("fresh depth f (memo-copy allowed) : " + "  ".join(f"f{k} {np.mean(f == k):.3f}" for k in range(7)))
    print(f"P(f>=2) {np.mean(f >= 2):.4f}  P(f>=3) {np.mean(f >= 3):.4f}  P(f>=4) {np.mean(f >= 4):.4f}  P(f>=6) {np.mean(f >= 6):.5f}")
    print(f"P(d>=2) {np.mean(d >= 2):.4f}  P(d>=4) {np.mean(d >= 4):.4f}  P(d>=6) {np.mean(d >= 6):.5f}")
    fq = qi == 0
    print(f"first query of a line: {np.mean(fq):.3f} of answers; its depth: " + "  ".join(f"d{k} {np.mean(d[fq] == k):.3f}" for k in range(7)))
    ev = fq & is_end.astype(bool)
    print(f"eval format (first query AND chain end): {np.mean(ev):.4f} of answers; with depth>=2: {np.mean(ev & (d >= 2)):.4f}")
    print(f"chain end asked at any index: {np.mean(is_end):.4f} of answers; its fresh depth when asked: "
          + "  ".join(f"f{k} {np.mean(f[is_end.astype(bool)] == k):.3f}" for k in range(7)))
    later = qi > 0
    print(f"query-free 'plurality of answers so far in the line' (2nd+ query): {plur[later].mean():.3f};"
          f" on answers with d>=2: {plur[later & (d >= 2)].mean():.3f}")
    print(f"query-free priors that training pays for: statement plurality right on {np.mean(stmt_plur):.3f} of answers; "
          f"'value of the last statement's variable' right on {np.mean(last_stmt):.3f}")
    # gradient share for genuine multi-step chasing
    tot = nbytes
    print(f"targets per reasoning byte: answers with f>=2: {np.sum(f >= 2) / tot:.4%}; f>=4: {np.sum(f >= 4) / tot:.4%}")
    return R


def filler():
    print("\n== E. encode_prompt prefixes (all v2 prompts are 83 bytes)")
    sets = P.reasoning_eval_set()
    lens = {len(p) for s in sets.values() for p, _ in s}
    print(f"prompt lengths in the official set: {sorted(lens)}")
    pr, a = sets[2][0]
    for K in (1, 2, 4):
        e = P.encode_prompt(pr, K)
        pre = e[: len(e) - len(pr)]
        print(f"K={K}: prefix {pre!r}  total {len(e)} bytes; last byte index % K = {(len(e) - 1) % K}")
    fill = P._FILLER.decode()
    print(f"_FILLER = {fill!r}")
    fd = re.findall(r"\d", fill[-2:])
    if fd:
        agree = {h: np.mean([a == fd[0] for _, a in s]) for h, s in sets.items()}
        print(f"filler digit {fd[0]!r} (seen by CHUNK=2/4 only) equals the answer: " + "  ".join(f"h{h} {v:.3f}" for h, v in agree.items()))
        inprompt = {h: np.mean([fd[0] in re.findall(r"\d", p) for p, _ in s]) for h, s in sets.items()}
        print("filler digit also occurs in the prompt: " + "  ".join(f"h{h} {v:.3f}" for h, v in inprompt.items()))
    # K=4 chunk phase of statements at the eval alignment
    e = P.encode_prompt(pr, 4).decode()
    chunks = [e[i:i + 4] for i in range(0, len(e), 4)]
    print("K=4 chunks at eval alignment (first 6, last 3): " + " | ".join(repr(c) for c in chunks[:6]) + " ... " + " | ".join(repr(c) for c in chunks[-3:]))


def answer_digit_balance(sets):
    print("\n== answer-digit balance (chi-square vs uniform, 9 dof; 5% crit 16.9); 'best constant digit' = max share")
    for h, s in sets.items():
        c = Counter(a for _, a in s)
        exp = len(s) / 10
        chi = sum((c.get(str(k), 0) - exp) ** 2 / exp for k in range(10))
        print(f"h{h}: chi2 {chi:.1f}  best constant digit {c.most_common(1)[0][0]!r} scores {c.most_common(1)[0][1] / len(s):.4f}")


def main():
    official = P.reasoning_eval_set()
    print("== A. query-free rules on the OFFICIAL set (400/hop); `evaluate` reports max(first, last) as reason_floor")
    for h in sorted(official):
        print(f"   h{h}: position_floor (prepare) = {P.position_floor(official[h]):.3f}")
    table(official, rules_query_free, "A1. query-free rules, official set")
    rng = np.random.default_rng(20261124)
    big = {h: [P.make_problem(rng, h) for _ in range(4000)] for h in (0, 1, 2, 4, 6, 8, 10)}
    table(big, rules_query_free, "A2. query-free rules, 4,000 fresh problems per hop (SE <= .008)")
    for h in (2, 6, 10):
        table({h: big[h]}, lambda p, h=h: rules_shallow(p, h), f"B. chase k < h steps then guess by position, hop {h} (4,000)", hops=[h])
    structure(big, (0, 1, 2, 4, 6, 8, 10))
    answer_digit_balance(official)
    train_stream()
    filler()
    q = np.random.default_rng(7)
    quick = {h: [P.make_problem(q, h) for _ in range(100)] for h in (0, 1, 2, 4, 6, 8)}
    table(quick, rules_query_free, "F. base-v2 quick-eval sets (seed 7, 100/hop, SE ~.05): the floors for base-v2_progress.tsv")
    answer_digit_balance(quick)


if __name__ == "__main__":
    main()
