import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_pairs import make_pair, prompts
from audit_v2 import Prob
from pair_floor import chain_roots, nearest_before
print("pair accuracy of 'k real lookups from q, then the nearest preceding bare-chain root' (decoy twins)")
for h in (2, 4, 6, 8):
    rng = np.random.default_rng(717171 + h)
    res = {k: [] for k in range(0, h)}
    for _ in range(6000):
        pa, pb = prompts(make_pair(rng, h))
        A, B = Prob(*pa), Prob(*pb)
        cands = chain_roots(A, h)
        for k in range(0, h):
            def go(P_):
                v = P_.q
                for _ in range(k):
                    v = P_.parent[v]
                return nearest_before(P_, v, cands)
            res[k].append(float(A.digit[go(A)] == A.ans and B.digit[go(B)] == B.ans))
    print(f"h{h}: " + " ".join(f"k={k}:{np.mean(v):.3f}" for k, v in res.items()) + f"  k={h}:1.000")
