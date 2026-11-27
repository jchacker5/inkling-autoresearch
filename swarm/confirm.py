"""Confirmation statistics (program.md, "Promotion"): each lane's best screening candidate is re-trained with
fresh seeds and scored on the HOLDOUT eval set (prepare.HOLDOUT_EVAL_SEED), as are fresh baseline seeds.
Each lane gets a one-sided Welch t-test on its primary metric against the baseline seeds, Holm-corrected
across lanes at family-wise alpha .05. A promoted lane must also pass the val_bpb gate (mean within +.02).

    uv run swarm/confirm.py          # (re)write results/confirm.tsv from the ledger

Pure Python (no scipy), so it runs anywhere the ledger is.
"""
import csv
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import BPB_GATE, PRIMARIES  # noqa: E402

ALPHA = 0.05
CONFIRM_SEEDS = tuple(range(100, 106))   # fresh: screening uses seed 0 (lanes) and 0..5 (baseline)
BASELINE_LANE = "baseline-holdout"
FIELDS = ["lane", "commit", "primary", "n_cand", "mean_cand", "sd_cand", "n_base", "mean_base", "sd_base", "t", "df",
          "p_one_sided", "holm_alpha", "bpb_cand", "bpb_base", "bpb_gate_ok", "promoted"]


def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (Numerical Recipes 6.4)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c; c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def t_sf(t, df):
    """P(T > t) for Student's t with df degrees of freedom."""
    tail = 0.5 * betainc(df / 2, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def welch(cand, base, sign=1.0):
    """One-sided Welch test of H1: sign * (mean(cand) - mean(base)) > 0. Returns (t, df, p)."""
    n1, n2 = len(cand), len(base)
    if n1 < 2 or n2 < 2:
        return None, None, None
    v1, v2 = st.variance(cand) / n1, st.variance(base) / n2
    diff = sign * (st.mean(cand) - st.mean(base))
    if v1 + v2 == 0:
        return (math.inf if diff > 0 else -math.inf), float(n1 + n2 - 2), (0.0 if diff > 0 else 1.0)
    t = diff / math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return t, df, t_sf(t, df)


def holm(pvals, alpha=ALPHA):
    """Holm step-down. pvals: {name: p or None}. Returns {name: (per-test alpha or None, rejected)}."""
    ranked = sorted((p, k) for k, p in pvals.items() if p is not None)
    out = {k: (None, False) for k, p in pvals.items() if p is None}
    m, stopped = len(ranked), False
    for i, (p, k) in enumerate(ranked):
        a_i = alpha / (m - i)
        rej = not stopped and p <= a_i
        stopped = stopped or not rej
        out[k] = (a_i, rej)
    return out


def best_candidate(rows, lane):
    """The lane's screening candidate with the largest t on its primary metric, or None."""
    c = [r for r in rows if r["lane"] == lane and r["verdict"] == "candidate" and r["valid"] == "True"]
    return max(c, key=lambda r: float(r.get("t_primary") or "-inf")) if c else None


def confirm_table(rows, lanes, eval_seed, alpha=ALPHA):
    """One row per lane that has a best candidate (see FIELDS). Holdout rows: lane '<lane>-confirm' and
    BASELINE_LANE, valid, with the given eval_seed."""
    expected = {str(seed) for seed in CONFIRM_SEEDS}

    def holdout(lane, finalist, commit=None):
        if not finalist.get("harness") or not finalist.get("base"):
            return [], False
        rows_by_seed = {}
        duplicates = set()
        for r in rows:
            seed = r.get("seed")
            if (r["lane"] != lane or r["valid"] != "True" or r.get("eval_seed") != str(eval_seed)
                    or seed not in expected or r.get("harness") != finalist["harness"]
                    or r.get("base") != finalist["base"] or (commit is not None and r["commit"] != commit)):
                continue
            if seed in rows_by_seed:
                duplicates.add(seed)
            rows_by_seed[seed] = r
        selected = [rows_by_seed[seed] for seed in sorted(rows_by_seed)]
        complete = (set(rows_by_seed) == expected and not duplicates
                    and len({r["commit"] for r in selected}) == 1)
        return selected, complete

    out, pvals = [], {}
    for lane in lanes:
        best = best_candidate(rows, lane)
        if best is None:
            continue
        primary = best.get("primary") or "kappa_id"
        base, base_complete = holdout(BASELINE_LANE, best)
        cand, cand_complete = holdout(f"{lane}-confirm", best, best["commit"])
        xc, xb = [float(r[primary]) for r in cand], [float(r[primary]) for r in base]
        t, df, p = welch(xc, xb, PRIMARIES[primary]) if base_complete and cand_complete else (None, None, None)
        bc = [float(r["val_bpb"]) for r in cand]
        bb = [float(r["val_bpb"]) for r in base]
        gate = bool(bc and bb) and st.mean(bc) <= st.mean(bb) + BPB_GATE
        pvals[lane] = 1.0 if p is None else p
        out.append(dict(lane=lane, commit=best["commit"], primary=primary, n_cand=len(xc),
                        mean_cand=st.mean(xc) if xc else None, sd_cand=st.stdev(xc) if len(xc) > 1 else None,
                        n_base=len(xb), mean_base=st.mean(xb) if xb else None,
                        sd_base=st.stdev(xb) if len(xb) > 1 else None, t=t, df=df, p_one_sided=p,
                        bpb_cand=st.mean(bc) if bc else None, bpb_base=st.mean(bb) if bb else None, bpb_gate_ok=gate))
    h = holm(pvals, alpha)
    for r in out:
        a_i, rej = h[r["lane"]]
        r["holm_alpha"] = a_i
        r["promoted"] = bool(r["p_one_sided"] is not None and rej and r["bpb_gate_ok"])
    return out


def write_tsv(table, path):
    def cell(v):
        return f"{v:.6g}" if isinstance(v, float) else ("" if v is None else str(v))
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(FIELDS)
        for r in table:
            w.writerow([cell(r.get(k)) for k in FIELDS])


def main():
    from run import ledger_path, read_ledger
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from prepare import HOLDOUT_EVAL_SEED
    rows = read_ledger(ledger_path())
    lanes = sys.argv[1:] or sorted({r["lane"] for r in rows if r["verdict"] == "candidate"})
    table = confirm_table(rows, lanes, HOLDOUT_EVAL_SEED)
    out = ledger_path().parent / "confirm.tsv"
    write_tsv(table, out)
    print(out.read_text())


if __name__ == "__main__":
    main()
