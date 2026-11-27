import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NSIM", "20000")
import sim_keep as S
for nl in (3, 4):
    for a_s, n_c in ((0.10, 6), (0.10, 8), (0.20, 6)):
        f, k, _, runs = S.new_rule(None, alpha_screen=a_s, n_c=n_c, n_lanes=nl)
        pw = [S.new_rule(dl, alpha_screen=a_s, n_c=n_c, n_lanes=nl)[2] for dl in (1, 1.5, 2, 2.5, 3)]
        print(f"lanes {nl} alpha {a_s:.2f} n_c {n_c}: FWER {f:.3f} keeps|null {k:.1f} runs|null {runs:.0f} "
              f"power d=1/1.5/2/2.5/3: " + "/".join(f"{p:.2f}" for p in pw))
