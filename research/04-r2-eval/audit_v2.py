"""Adversarial audit of task v2 (prepare._forest): pure Python/numpy, no MLX, no model.

For each hop, the expected accuracy of rules that answer WITHOUT following the query's pointer chain
(position, query-position, frequency, structure, filler), on the official reasoning_eval_set() (400/hop)
and on large fresh samples (N_BIG/hop, other seeds), plus the position-only learned oracle.
"""
import os
import re
import sys
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, ROOT)
import prepare as P  # noqa: E402

assert "mlx" not in sys.modules
N_BIG = int(os.environ.get("N_BIG", "20000"))
HOPS = (0, 1, 2, 4, 6, 8, 10)


def parse(prompt):
    body, q = prompt[:-1].split(";?")
    return [(s[0], s[2]) for s in body.split(";")], q


class Prob:
    def __init__(self, prompt, ans):
        st, q = parse(prompt)
        self.st, self.q, self.ans = st, q, int(ans)
        self.parent = {v: (None if x.isdigit() else x) for v, x in st}
        self.digit = {v: int(x) for v, x in st if x.isdigit()}
        self.slot = {v: i for i, (v, _) in enumerate(st)}
        self.kids = defaultdict(list)
        for v, p in self.parent.items():
            if p is not None:
                self.kids[p].append(v)
        self.roots = [v for v, _ in st if self.parent[v] is None]          # in prompt order
        rootof, depth = {}, {}
        for v, _ in st:                                                     # topological order
            p = self.parent[v]
            rootof[v], depth[v] = (v, 0) if p is None else (rootof[p], depth[p] + 1)
        self.rootof, self.depth = rootof, depth
        self.chain_root = rootof[q]
        self.size = defaultdict(int)
        self.maxdepth = defaultdict(int)
        self.ispath = {r: True for r in self.roots}
        for v in self.parent:
            r = rootof[v]
            self.size[r] += 1
            self.maxdepth[r] = max(self.maxdepth[r], depth[v])
            if len(self.kids[v]) > 1:
                self.ispath[r] = False


def uniform(cands):
    return {c: 1 / len(cands) for c in cands} if cands else {}


def acc_of(dist, pr):
    return dist.get(pr.ans, 0.0)


def rules(pr, h):
    """Each rule -> distribution over digits."""
    R = pr.roots
    dig = [pr.digit[r] for r in R]
    out = {}
    out["first digit"] = {dig[0]: 1.0}
    out["last digit"] = {dig[-1]: 1.0}
    out["second digit"] = {dig[1]: 1.0} if len(dig) > 1 else {dig[0]: 1.0}
    out["random root (1/R)"] = uniform(dig)
    # query-position rules: digits nearest the query variable's own statement (no pointer following)
    qs = pr.slot[pr.q]
    before = [pr.digit[r] for r in R if pr.slot[r] <= qs]
    after = [pr.digit[r] for r in R if pr.slot[r] > qs]
    out["nearest digit before q's stmt"] = {before[-1]: 1.0} if before else {dig[0]: 1.0}
    out["nearest digit after q's stmt"] = {after[0]: 1.0} if after else {dig[-1]: 1.0}
    near = min(R, key=lambda r: (abs(pr.slot[r] - qs), pr.slot[r]))
    out["nearest digit to q's stmt"] = {pr.digit[near]: 1.0}
    # structure, query-blind (needs the graph, not the question)
    big = max(pr.size[r] for r in R)
    out["largest tree (plurality value)"] = uniform([pr.digit[r] for r in R if pr.size[r] == big])
    deep = max(pr.maxdepth[r] for r in R)
    out["deepest tree"] = uniform([pr.digit[r] for r in R if pr.maxdepth[r] == deep])
    out["path tree of depth h (knows h)"] = uniform([pr.digit[r] for r in R if pr.ispath[r] and pr.maxdepth[r] == h]) or uniform(dig)
    nk = {r: len(pr.kids[r]) for r in R}
    out["roots with >=1 child"] = uniform([pr.digit[r] for r in R if nk[r] >= 1]) or uniform(dig)
    out["roots with exactly 1 child"] = uniform([pr.digit[r] for r in R if nk[r] == 1]) or uniform(dig)
    most = max(nk.values())
    out["root with most children"] = uniform([pr.digit[r] for r in R if nk[r] == most])
    # query-aware structure without chasing: q is a leaf; nothing else is free
    return out


def partial_chase(pr, k):
    """Follow k pointers from q (real lookups), then guess uniformly among roots whose tree is deep enough."""
    v, d = pr.q, 0
    for _ in range(k):
        if pr.parent[v] is None:
            break
        v = pr.parent[v]
        d += 1
    if pr.parent[v] is None:
        return {pr.digit[v]: 1.0}
    # v is at depth depth[v] >= 1 below an unknown root; the reader knows v is not a root
    return uniform([pr.digit[r] for r in pr.roots])


def sample(seed, h, n):
    rng = np.random.default_rng(seed)
    return [Prob(*P.make_problem(rng, h)) for _ in range(n)]


def table(sets_by_hop, title):
    names = None
    res = {}
    for h, probs in sets_by_hop.items():
        tot = defaultdict(float)
        for pr in probs:
            for name, dist in rules(pr, h).items():
                tot[name] += acc_of(dist, pr)
        res[h] = {k: v / len(probs) for k, v in tot.items()}
        names = names or list(res[h])
    print(f"\n## {title}")
    print(f"{'rule':34s} " + " ".join(f"h{h:<5d}" for h in sets_by_hop))
    for name in names:
        print(f"{name:34s} " + " ".join(f"{res[h][name]:.3f} " for h in sets_by_hop))
    meanR = {h: np.mean([len(pr.roots) for pr in probs]) for h, probs in sets_by_hop.items()}
    print(f"{'mean #roots R':34s} " + " ".join(f"{meanR[h]:.2f}  " for h in sets_by_hop))
    rootfirst = {h: np.mean([pr.roots[0] == pr.chain_root for pr in probs]) for h, probs in sets_by_hop.items()}
    print(f"{'P(chain root = first root)':34s} " + " ".join(f"{rootfirst[h]:.3f} " for h in sets_by_hop))
    return res


# ---- position-only learned oracle: conditional logit over candidate roots, features from the root mask only
def feats(pr):
    R = pr.roots
    nR = len(R)
    slots = [pr.slot[r] for r in R]
    rows = []
    for i, r in enumerate(R):
        s = slots[i]
        nxt = slots[i + 1] if i + 1 < nR else 20
        prv = slots[i - 1] if i > 0 else -1
        f = np.zeros(20 + 10 + 10 + 11 + 11 + 1)
        f[s] = 1
        f[20 + i] = 1
        f[30 + (nR - 1 - i)] = 1
        f[40 + min(nxt - s - 1, 10)] = 1                     # non-root run right after this root
        f[51 + min(s - prv - 1, 10)] = 1                     # non-root run right before
        f[62] = 1.0
        rows.append(f)
    return np.array(rows), R.index(pr.chain_root)


def fit_clogit(data, epochs=300, lr=0.5, l2=1e-3):
    d = data[0][0].shape[1]
    w = np.zeros(d)
    for _ in range(epochs):
        g = np.zeros(d)
        for X, y in data:
            z = X @ w
            p = np.exp(z - z.max()); p /= p.sum()
            g += X[y] - p @ X
        w += lr * (g / len(data) - l2 * w)
    return w


def clogit_acc(w, data):
    return float(np.mean([int(np.argmax(X @ w)) == y for X, y in data]))


def filler_rule(sets_by_hop):
    P._FILLER = None
    out = {}
    for K in (1, 2, 4):
        e = P.encode_prompt(sets_by_hop[2][0][0], K)
        pre = e[: len(e) - len(sets_by_hop[2][0][0].encode())]
        ds = re.findall(rb"\d", pre)
        out[K] = (pre, ds)
    return out


if __name__ == "__main__":
    official = {h: [Prob(p, a) for p, a in ps] for h, ps in P.reasoning_eval_set().items()}
    lens = {len(p) for ps in P.reasoning_eval_set().values() for p, _ in ps}
    print("official prompt lengths:", lens)
    print("evaluate()'s position floor:", {h: round(P.position_floor(ps), 3) for h, ps in P.reasoning_eval_set().items()})
    res_off = table(official, "official reasoning_eval_set (400/hop)")
    big = {h: sample(9000 + h, h, N_BIG) for h in HOPS}
    res_big = table(big, f"fresh samples ({N_BIG}/hop)")

    print("\n## partial chase: k real lookups from q, then a uniform guess among roots (fresh samples)")
    for h in HOPS:
        row = []
        for k in range(0, h + 1):
            row.append(np.mean([acc_of(partial_chase(pr, k), pr) for pr in big[h][:5000]]))
        print(f"h{h:<3d} " + " ".join(f"k{k}:{a:.3f}" for k, a in enumerate(row)))

    print("\n## position-only learned oracle (conditional logit on root-mask features; train 6k, test 6k + official)")
    for h in HOPS:
        tr = [feats(pr) for pr in sample(5000 + h, h, 6000)]
        te = [feats(pr) for pr in sample(7000 + h, h, 6000)]
        w = fit_clogit(tr)
        off = [feats(pr) for pr in official[h]] if h in official else None
        print(f"h{h:<3d} train {clogit_acc(w, tr):.3f}  test {clogit_acc(w, te):.3f}" + (f"  official {clogit_acc(w, off):.3f}" if off else ""))

    print("\n## encode_prompt filler (left context before the prompt)")
    for K, (pre, ds) in filler_rule(P.reasoning_eval_set()).items():
        print(f"align={K}: prefix {pre!r} digits {ds}")
        if ds:
            d = int(ds[0])
            for h, ps in P.reasoning_eval_set().items():
                print(f"   h{h}: P(answer == filler digit {d}) = {np.mean([int(a) == d for _, a in ps]):.3f}")
