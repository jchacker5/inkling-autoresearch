import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoy_pairs import make_pair, prompts
from audit_v2 import Prob

def chain_roots(pr, h):
    return [r for r in pr.roots if pr.ispath[r] and pr.maxdepth[r] == h]

def nearest_before(pr, node, cands):
    s = pr.slot[node]
    before = [r for r in cands if pr.slot[r] <= s]
    return max(before, key=lambda r: pr.slot[r]) if before else min(cands, key=lambda r: pr.slot[r])

if __name__ == "__main__":
    print("hop  1/R^2   pos/blind  struct+hash  struct+q-position  struct+chase(k=h-1)+position  single acc: struct+q-pos | v2 struct (no decoy)")
    for h in (1, 2, 4, 6, 8):
        rng = np.random.default_rng(616161 + h)
        r2, hashp, qpos, chase, single = [], [], [], [], []
        for _ in range(10000):
            pa, pb = prompts(make_pair(rng, h))
            A, B = Prob(*pa), Prob(*pb)
            r2.append(1 / len(A.roots) ** 2)
            cands = chain_roots(A, h)
            hashp.append(0.25 if len(cands) == 2 else (1 / len(cands)) ** 2)
            ga = nearest_before(A, A.q, cands); gb = nearest_before(B, B.q, cands)
            qpos.append(float(A.digit[ga] == A.ans and B.digit[gb] == B.ans))
            single.append(float(A.digit[ga] == A.ans))
            def chased(P_):
                v = P_.q
                for _ in range(max(h - 1, 0)):
                    v = P_.parent[v]
                return v
            ca = nearest_before(A, chased(A), cands); cb = nearest_before(B, chased(B), cands)
            chase.append(float(A.digit[ca] == A.ans and B.digit[cb] == B.ans))
        print(f"h{h:<3d} {np.mean(r2):.3f}   0          {np.mean(hashp):.3f}        {np.mean(qpos):.3f}              {np.mean(chase):.3f}                          {np.mean(single):.3f}")
