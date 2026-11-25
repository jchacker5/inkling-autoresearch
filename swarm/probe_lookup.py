"""Can a checkpoint retrieve? Three probes that need no chain reasoning.

    uv run swarm/probe_lookup.py NAME [KEY=VALUE ...]     # overrides as for pretrain_base.py --set

1. induction: a random 100-letter string repeated; accuracy on the second copy.
2. pure lookup: "a=5;b=7;...;" with N roots only (distinct letters), then every variable queried once in random
   order ("?b=7?a=5..."); accuracy on the answer digits. Chance is .10.
3. training lines (reason_train_v4): answer accuracy split by the queried variable's hop depth and by whether it is
   the line's first query.
"""
import collections
import os
import string
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx.core as mx  # noqa: E402

import prepare as P  # noqa: E402
import pretrain_base as PB  # noqa: E402
import train as T  # noqa: E402


def main():
    name = sys.argv[1]
    PB.apply_overrides(sys.argv[2:])
    m = T.Inkling()
    m.load_weights(os.path.join(P.CACHE_DIR, name + ".safetensors"))
    mx.eval(m.parameters())
    m.eval()
    rng = np.random.default_rng(1)

    def argmax(rows):
        x = mx.array(np.stack([np.frombuffer(r.encode(), np.uint8) for r in rows]).astype(np.int32))
        return np.array(mx.argmax(m(x), -1))

    rows = ["\n" + (s := "".join(rng.choice(list(string.ascii_lowercase), 100))) + s for _ in range(32)]
    p = argmax(rows)
    ind = np.mean([p[i, j] == ord(rows[i][j + 1]) for i in range(32) for j in range(101, 200)])
    out = {"induction": ind}
    for nv in (4, 10, 20):
        rows, golds = [], []
        for _ in range(64):
            vs = rng.choice(list(string.ascii_lowercase), nv, replace=False)
            ds = rng.integers(0, 10, nv)
            q = rng.permutation(nv)
            rows.append("\n" + "".join(f"{v}={d};" for v, d in zip(vs, ds)) + "".join(f"?{vs[i]}={ds[i]}" for i in q))
            golds.append([(vs[i], ds[i]) for i in q])
        width = max(len(r) for r in rows)
        rows = [r.ljust(width, "\n") for r in rows]
        p = argmax(rows)
        hits = [p[i, r.index("?" + v + "=") + 2] == ord(str(d)) for i, r in enumerate(rows) for v, d in golds[i]]
        out[f"lookup{nv}"] = float(np.mean(hits))

    data = P._load("reason_train_v4")
    L = P.SEQ_LEN + 1
    X = np.stack([data[s:s + L] for s in np.random.default_rng(0).integers(0, len(data) - L, 64)]).astype(np.int32)
    pred = np.array(mx.argmax(m(mx.array(X[:, :-1])), -1))
    stats = collections.defaultdict(lambda: [0, 0])
    for r in range(64):
        row = bytes(X[r].astype(np.uint8)).decode("latin1")
        for start in [i + 1 for i, c in enumerate(row) if c == "\n"]:
            end = row.find("\n", start)
            line = row[start:end if end >= 0 else len(row)]
            if "?" not in line:
                continue
            st, qs = line.split("?", 1)
            par = dict(s.split("=") for s in st.strip(";").split(";") if s.count("=") == 1)

            def hop(v):
                k = 0
                while v in par and not par[v].isdigit():
                    v, k = par[v], k + 1
                return k
            for qi, pos in enumerate(start + len(st) + i for i, c in enumerate("?" + qs) if c == "?"):
                if pos + 3 > L - 1 or not row[pos + 3].isdigit():
                    break
                key = ("first" if qi == 0 else "later") + str(min(hop(row[pos + 1]), 4))
                stats[key][0] += pred[r, pos + 2] == ord(row[pos + 3])
                stats[key][1] += 1
    for k in sorted(stats):
        out["train_" + k] = stats[k][0] / stats[k][1]
    print(name, " ".join(f"{k}={v:.3f}" for k, v in out.items()))


if __name__ == "__main__":
    main()
