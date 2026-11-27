"""Debate round 1 (Debater 1, mechanism) evidence: CPU-only, read-only probes. Never touches the GPU.

    .venv/bin/python research/02-debate-r1-probe.py base main,swap,format,train,sweep,floor
    .venv/bin/python research/02-debate-r1-probe.py base-5h swap
    .venv/bin/python research/02-debate-r1-probe.py chunk          # tiny random CHUNK models, no checkpoint

Parts:
  main    accuracy on the official sets split by whether the chain root is statement 0
  swap    query invariance: same statements, different question (hop-0 three-way; hops 2/6 chain end vs a literal)
  format  prepend "\\n" or 1-3 solved training lines
  train   teacher-forced accuracy inside 512-byte windows of a fresh training-format stream, by depth
  sweep   loop sweep of the statement-0 split
  floor   numpy-only: "copy statement 0's digit" and "random literal" on the exact official sets
  chunk   can the looped core's contribution at the answer byte depend on the query letter? (CHUNK 1/2/4)
"""
import os
import sys
import time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
import numpy as np  # noqa: E402
import mlx.core as mx  # noqa: E402

mx.set_default_device(mx.cpu)
mx.set_cache_limit = lambda *a, **k: 0          # train.py calls this at import; leave the GPU allocator alone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import prepare as P  # noqa: E402
import train as T  # noqa: E402

CK = sys.argv[1] if len(sys.argv) > 1 else "base"
PARTS = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"main", "swap", "format", "train", "sweep", "floor"}


def parse(prompt):
    body = prompt[:-1].split(";?")
    return [(s[0], s[2]) for s in body[0].split(";")], body[1]


def root_of(st, v):
    d = dict(st)
    while not d[v].isdigit():
        v = d[v]
    return v


if CK == "chunk":
    T.D_MODEL, T.N_HEADS = 64, 2
    for K in (1, 2, 4):
        T.CHUNK = K
        mx.random.seed(0)
        m = T.Inkling()
        mx.eval(m.parameters())
        row = []
        for pad in range(4):
            ps = ["\n" * pad + f"a=3;b=a;c=5;?{q}=" for q in "bc"]
            x = mx.array(np.stack([np.frombuffer(p.encode(), np.uint8) for p in ps]).astype(np.int32))
            logits, _, states, byte = m.forward(x, 4, return_states=True)
            core = (m.unchunk(states[-1], byte) - byte) if K > 1 else states[-1]
            row.append(f"pad {pad}: core diff {float(mx.abs(core[0, -1] - core[1, -1]).max()):.1e}")
        print(f"CHUNK={K}: " + " | ".join(row))
    sys.exit(0)

model = T.Inkling()
model.load_weights(os.path.join(P.CACHE_DIR, f"{CK}.safetensors"))
mx.eval(model.parameters())
model.eval()


def predict(texts, loops=4):
    out = []
    for i in range(0, len(texts), 128):
        enc = [t.encode() for t in texts[i:i + 128]]
        x = np.zeros((len(enc), max(map(len, enc))), np.int32)
        for r, e in enumerate(enc):
            x[r, :len(e)] = np.frombuffer(e, np.uint8)
        logits, _ = model.forward(mx.array(x), loops)
        out.append(np.array(mx.argmax(logits[mx.arange(len(enc)), mx.array([len(e) - 1 for e in enc])], -1)))
    return np.concatenate(out)


def split(probs, loops=4, prefix=lambda: ""):
    pred = predict([prefix() + p for p, _ in probs], loops)
    gold = np.array([ord(a) for _, a in probs])
    sq = [parse(p) for p, _ in probs]
    r0 = np.array([root_of(st, q) == st[0][0] for st, q in sq])
    lit = np.array([chr(pr) in {d for _, d in st if d.isdigit()} for pr, (st, _) in zip(pred, sq)])
    c = pred == gold
    f = pred == np.array([ord(st[0][1]) for st, _ in sq])
    return dict(acc=c.mean(), r0=c[r0].mean(), nr0=c[~r0].mean(), n0=int(r0.sum()), f=f.mean(), fnr0=f[~r0].mean(), lit=lit.mean())


t0 = time.time()
sets = P.reasoning_eval_set()                                       # official: hops 2,4,6 (ID), 8,10 (OOD), 400 each
rng = np.random.default_rng(616161)
sets.update({h: [P.make_problem(rng, h) for _ in range(400)] for h in (0, 1)})

if "floor" in PARTS:
    for h in sorted(sets):
        f = np.mean([parse(p)[0][0][1] == a for p, a in sets[h]])
        lits = np.mean([np.mean([d == a for _, d in parse(p)[0] if d.isdigit()]) for p, a in sets[h]])
        print(f"floor hop {h:>2}: copy-stmt0-digit {f:.3f}  random-literal {lits:.3f}")

if "main" in PARTS:
    print(f"== {CK}: 400 problems/hop, loops=4. root0 = the chain root is statement 0")
    print("hop   acc  acc|root0  acc|not-root0  n_root0  P(out=stmt0)  P(out=stmt0|not-root0)  P(out is a literal)")
    for h in sorted(sets):
        s = split(sets[h])
        print(f"{h:>3}  {s['acc']:.3f}    {s['r0']:.3f}        {s['nr0']:.3f}      {s['n0']:>4}     {s['f']:.3f}           {s['fnr0']:.3f}                {s['lit']:.3f}")

if "swap" in PARTS:
    rng = np.random.default_rng(99)
    sts = [parse(P.make_problem(rng, 0)[0])[0] for _ in range(700)]
    sts = [st for st in sts if len({d for _, d in st}) == 3][:400]
    preds = np.stack([predict([";".join(f"{v}={d}" for v, d in st) + f";?{st[j][0]}=" for st in sts]) for j in range(3)], 1)
    digits = np.array([[ord(d) for _, d in st] for st in sts])
    sel = np.stack([(preds == digits[:, [k]]) for k in range(3)], 2)
    print(f"\n== {CK} hop-0 query swap, 3 distinct digits (n={len(sts)}): P(same output for all 3 queries) = {np.mean((preds == preds[:, :1]).all(1)):.3f}")
    for j in range(3):
        print(f"  ask stmt {j}'s variable: acc {sel[:, j, j].mean():.3f} | output = digit of stmt 0/1/2: "
              f"{sel[:, j, 0].mean():.3f} / {sel[:, j, 1].mean():.3f} / {sel[:, j, 2].mean():.3f}")
    for h in (2, 6):
        q_end, q_lit = [], []
        for p, a in sets[h]:
            st, q = parse(p)
            lits = [(v, d) for v, d in st if d.isdigit() and d != a]
            if lits:
                v, d = lits[int(np.random.default_rng(len(p)).integers(len(lits)))]
                q_end.append((p, a))
                q_lit.append((p[: p.rindex("?")] + f"?{v}=", d))
        pe, pl = predict([p for p, _ in q_end]), predict([p for p, _ in q_lit])
        print(f"== hop-{h} query swap (n={len(q_end)}): acc asking chain end {np.mean(pe == np.array([ord(a) for _, a in q_end])):.3f}; "
              f"acc asking a depth-0 literal (other digit) {np.mean(pl == np.array([ord(a) for _, a in q_lit])):.3f}; "
              f"P(same output) {np.mean(pe == pl):.3f}")

if "format" in PARTS:
    r5 = np.random.default_rng(5)

    def solved_line():
        st, _, val = P._forest(r5, int(r5.integers(0, 7)))
        return f"{st};" + "".join(f"?{v}={val[v]}" for v in r5.permutation(list(val))) + "\n"

    print("\n== format check: acc / acc|root0 / acc|not-root0 / P(out=stmt0 | not-root0)")
    for name, pre in (("none", lambda: ""), ("newline", lambda: "\n"), ("1 solved line", solved_line),
                      ("3 solved lines", lambda: solved_line() + solved_line() + solved_line())):
        print(f"{name:>15}: " + "  ".join("h{} {acc:.2f}/{r0:.2f}/{nr0:.2f}/{fnr0:.2f}".format(h, **split(sets[h], prefix=pre))
                                          for h in (0, 2, 6, 8, 10)))

if "train" in PARTS:
    r7 = np.random.default_rng(777)
    buf, meta = "", []
    while len(buf) < 400_000:
        stmts, _, val = P._forest(r7, int(r7.integers(0, 7)))
        st = [(s[0], s[2]) for s in stmts.split(";")]
        dep = {}
        for v, rhs in st:
            dep[v] = 0 if rhs.isdigit() else dep[rhs] + 1
        start, line = len(buf), stmts + ";"
        for i, v in enumerate(r7.permutation(list(val))):
            line += f"?{v}="
            meta.append((start + len(line), dep[v], i, start, ord(st[0][1])))
            line += str(val[v])
        buf += line + "\n"
    B = np.frombuffer(buf.encode(), np.uint8)
    meta = np.array(meta)
    starts = r7.integers(0, len(B) - 513, size=128)
    X = np.stack([B[s:s + 513] for s in starts]).astype(np.int32)
    pred = np.concatenate([np.array(mx.argmax(model.forward(mx.array(X[i:i + 16, :-1]), 4)[0], -1)) for i in range(0, 128, 16)])
    rows = []
    for w, s in enumerate(starts):
        m = meta[(meta[:, 0] > s) & (meta[:, 0] <= s + 512) & (meta[:, 3] >= s)]     # whole line inside the window
        for g, d, qi, ls, f0 in m:
            rows.append((pred[w, g - 1 - s] == B[g], d, qi, pred[w, g - 1 - s] == f0, B[g] == f0))
    R = np.array(rows, dtype=np.int64)
    print(f"\n== training-format probe (teacher-forced, loops=4): {len(R)} answers")
    print("  acc by depth of the queried variable: " + "  ".join(
        f"d{d} {R[R[:, 1] == d, 0].mean():.2f}(n={int((R[:, 1] == d).sum())})" for d in range(7)))
    print("  acc by query index in line          : " + "  ".join(f"q{q} {R[R[:, 2] == q, 0].mean():.2f}" for q in (0, 1, 2, 4, 8)))
    fq = R[:, 2] == 0
    print("  first query of the line, by depth   : " + "  ".join(
        f"d{d} {R[fq & (R[:, 1] == d), 0].mean():.2f}(n={int((fq & (R[:, 1] == d)).sum())})" for d in range(4)))
    nf = R[:, 4] == 0
    print(f"  P(out = line's stmt0 digit | answer != it): {R[nf, 3].mean():.3f} (n={int(nf.sum())})")

if "sweep" in PARTS:
    print("\n== loop sweep: acc|root0 / acc|not-root0")
    for k in (1, 2, 3, 4, 6, 8):
        print(f"loops={k}: " + "  ".join("h{} {r0:.2f}/{nr0:.2f}".format(h, **split(sets[h], loops=k)) for h in (0, 2, 6, 8, 10)))
print(f"[{time.time() - t0:.0f}s on CPU]")
