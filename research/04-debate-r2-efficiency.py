"""Reproduces every number in 04-debate-r2-efficiency.md. Pure Python + numpy: no MLX, no forward passes.

    .venv/bin/python -I research/04-debate-r2-efficiency.py

1. Rules that never read the query, on task v2 (prepare.py at 4a1c4dd, loaded from git) and on task v3 (the
   current prepare.py): per-prompt accuracy and, for v3's twin pairs, pair accuracy.
2. Bytes, KV and block applications per model (bf16 / W4 / int8 KV / SWA-128 core); memory crossovers.
3. Loss shares of answer digits in the v2 and v3 training streams under the answer weights.
"""
import os
import re
import subprocess
import sys
import types
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import prepare as P3  # noqa: E402  (current harness = task v3; numpy only at import time)


def prepare_at(rev):
    """prepare.py as of a git revision, as a module (the repo's own code; numpy only at import time)."""
    src = subprocess.run(["git", "show", f"{rev}:prepare.py"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType(f"prepare_{rev}")
    exec(compile(src, f"prepare@{rev}", "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------- 1. query-blind rules
def parse(prompt):
    st = prompt.split(";?")[0].split(";")
    par = {s[0]: s[2] for s in st if not s[2].isdigit()}
    lit = {s[0]: s[2] for s in st if s[2].isdigit()}
    return st, par, lit


def resolve(v, par, lit):
    d = 0
    while v in par:
        v, d = par[v], d + 1
    return lit[v], d


def blind_rules(prompt, ans):
    """Expected accuracy of five rules that never read the queried letter (ties broken uniformly at random)."""
    st, par, lit = parse(prompt)
    info = {s[0]: resolve(s[0], par, lit) for s in st}
    md = max(d for _, d in info.values())
    deep = sorted({v for v, d in info.values() if d == md})
    cnt = Counter(v for v, _ in info.values())
    top = max(cnt.values())
    tops = [k for k, c in cnt.items() if c == top]
    digs = re.findall(r"\d", prompt)
    return np.array([digs[0] == ans, digs[-1] == ans, info[st[-1][0]][0] == ans,
                     (ans in tops) / len(tops), (ans in deep) / len(deep)], dtype=float)


NAMES = ("first digit", "last digit", "resolve last stmt", "largest tree", "deepest tree")


def table(title, sets, pairs=False):
    print(f"\n{title}\nhop  " + "  ".join(f"{n:>17s}" for n in NAMES) + "  max/prompt" + ("  max/pair" if pairs else ""))
    for h, probs in sets.items():
        R = np.array([blind_rules(p, a) for p, a in probs])
        line = f"{h:3d}  " + "  ".join(f"{x:17.3f}" for x in R.mean(0)) + f"  {R.mean(0).max():10.3f}"
        if pairs:   # twins are adjacent; independent random tie-breaks = "structure + coin flip"
            line += f"  {max(float((R[0::2, j] * R[1::2, j]).mean()) for j in range(R.shape[1])):8.3f}"
        print(line)


def query_is_unique_deepest(sets):
    out = {}
    for h, probs in sets.items():
        k = 0
        for p, _ in probs:
            st, par, lit = parse(p)
            q = p.split(";?")[1][:-1]
            info = {s[0]: resolve(s[0], par, lit) for s in st}
            md = max(d for _, d in info.values())
            k += info[q][1] == md and sum(d == md for _, d in info.values()) == 1
        out[h] = round(k / len(probs), 3)
    return out


# ---------------------------------------------------------------- 2. bytes
D, HID, V, MiB = 384, 1024, 256, 2**20
BLOCK = 4 * D * D + 3 * D * HID + 2 * D


def model(prelude=1, core=2, coda=1, loops=4, chunk=1):
    nb = prelude + core + coda
    p = V * D + nb * BLOCK + D + D * V
    small = nb * 2 * D + D
    if chunk > 1:
        p += 2 * chunk * D * D + D + chunk * D
        small += D + chunk * D
    return dict(params=p, small=small, pre=prelude, core=core, coda=coda, loops=loops, K=chunk)


def wbytes(m, mode):
    q = m["params"] - m["small"]
    return {"bf16": 2 * m["params"], "W4": q * 0.5 + q / 64 * 4 + 2 * m["small"]}[mode]


def kv(m, T, elem=2.0, win=None, loops=None):
    L = loops or m["loops"]
    core_pos = (min(T, win) if win else T) / m["K"]
    return (m["pre"] * T + m["core"] * L * core_pos + m["coda"] * T) * 2 * D * elem


def decode_apps(m):
    proj = (2 * D * D / BLOCK) if m["K"] > 1 else 0.0       # down + up per byte
    return m["pre"] + m["core"] * m["loops"] / m["K"] + m["coda"] + proj


def prefill_depth(m):
    return m["pre"] + m["core"] * m["loops"] + m["coda"] + (2 * 0.17 if m["K"] > 1 else 0)


# ---------------------------------------------------------------- 3. loss shares
def loss_shares(path, weights):
    s = np.asarray(np.memmap(path, dtype=np.uint8, mode="r")[: 4 * 2**20])
    ans = np.zeros(len(s), bool)
    ans[3:] = (s[3:] >= 48) & (s[3:] <= 57) & (s[2:-1] == ord("=")) & (s[:-3] == ord("?"))
    nl = s == ord("\n")
    c = np.cumsum(ans)
    first = ans & ((c - np.maximum.accumulate(np.where(nl, c, 0))) == 1)
    fa, oa, mix = first.mean(), (ans & ~first).mean(), 0.25
    for wa, wf in weights:
        tot = (1 - mix) + mix * (1 - fa - oa) + mix * oa * wa + mix * fa * wf
        print(f"  ANSWER_W={wa:g} FIRST_ANSWER_W={wf:g}: first answers {mix*fa*wf/tot:.3f}, other answers {mix*oa*wa/tot:.3f}, text {(1-mix)/tot:.3f}")


def main():
    P2 = prepare_at("4a1c4dd")
    v2 = P2.reasoning_eval_set()
    table("Task v2 (4a1c4dd), official eval set, 400 per hop:", v2)
    print("v2: P(query is the unique deepest variable):", query_is_unique_deepest(v2))
    rng = np.random.default_rng(99)
    table("Task v2, fresh problems, 2,000 per hop:", {h: [P2.make_problem(rng, h) for _ in range(2000)] for h in (0, 1, 2, 3, 4, 6, 8, 10)})
    v3 = P3.reasoning_eval_set()
    table(f"Task v3 (current prepare.py), official eval set, {P3.EVAL_PER_HOP} prompts = {P3.EVAL_PER_HOP // 2} twin pairs per hop:", v3, pairs=True)
    print("v3 prompt lengths:", sorted({len(p) for v in v3.values() for p, _ in v}), " PAIR_BOUND:", P3.PAIR_BOUND)

    M = {"looped": model(), "4L": model(loops=1), "10L": model(core=8, loops=1), "chunk2": model(chunk=2),
         "chunk4": model(chunk=4), "looped L=3": model(loops=3)}
    int8 = 1 + 4 / 64
    print("\nmodel        params    F bf16 T=100  T=512   F deployed T=512 (W4, int8 KV, SWA-128 core)  decode apps/byte  prefill depth")
    for n, m in M.items():
        print(f"{n:12s} {m['params']:8d}  {(wbytes(m,'bf16')+kv(m,100))/MiB:6.2f}  {(wbytes(m,'bf16')+kv(m,512))/MiB:6.2f}"
              f"   {(wbytes(m,'W4')+kv(m,512,int8,128))/MiB:6.2f}                                    {decode_apps(m):5.2f}             {prefill_depth(m):5.2f}")
    b, c = M["looped"], M["chunk4"]
    for name, wm, el in (("bf16 W + bf16 KV", "bf16", 2.0), ("W4 + bf16 KV", "W4", 2.0), ("W4 + int8 KV", "W4", int8)):
        print(f"chunk4 vs looped memory crossover ({name}): {(wbytes(c, wm) - wbytes(b, wm)) / (kv(b, 1, el) - kv(c, 1, el)):.0f} bytes")
    print(f"looped KV/byte at loops 4 / 8: {kv(b,1)/1024:.0f} / {kv(b,1,loops=8)/1024:.0f} KiB")
    print(f"KL-exit check (coda + head) in block applications: {(BLOCK + D * V) / BLOCK:.2f}")

    print("\nItem-noise MDE of paired delta-kappa (5 hops, pair discordance d = .2, one-sided .05, power .8):")
    for pairs in (200, 400):
        se = np.sqrt(0.2 / pairs) / 0.75 / np.sqrt(5)
        print(f"  {pairs} pairs per hop: SE {se:.4f}, MDE {2.487 * se:.3f}")

    for name, weights in (("reason_train_v2.bin", ((1, 1), (10, 10))), ("reason_train_v3.bin", ((1, 1), (3, 30)))):
        path = os.path.join(P3.CACHE_DIR, name)
        if os.path.exists(path):
            print(f"\nloss shares, {name}:")
            loss_shares(path, weights)


if __name__ == "__main__":
    main()
