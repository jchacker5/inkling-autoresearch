"""Leaderboard over the shared ledger.   uv run swarm/status.py [--all] [--noise]

--noise writes results/noise.md: baseline seed mean/std and the screening bars they imply (program.md)."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import BPB_GATE, CAND_T, MIN_NOISE_SEEDS, ledger_path, noise_summary, read_ledger  # noqa: E402


def fmt(x):
    return "nan" if x is None else f"{x:.4f}"


def bar(x):
    return f"pending (need >= {MIN_NOISE_SEEDS} seeds)" if x is None else f"{x:.4f}"


rows = read_ledger(ledger_path())
if not rows:
    sys.exit("ledger empty: run the baseline first (uv run swarm/run.py --lane baseline --desc baseline)")

if "--noise" in sys.argv:
    ns = noise_summary(rows)
    b = [r for r in rows if r["lane"] == "baseline" and r["valid"] == "True"]
    lines = ["# Seed noise of the baseline", "",
             f"{ns['n']} valid baseline runs (base + 5 minutes of unchanged training, differing only in seed: data order "
             "and loop sampling), screening eval set.", "", "| metric | mean | std | n |", "|---|---|---|---|"]
    for k in ("val_bpb", "kappa_id", "kappa_ood", "reason_id", "steps"):
        m, sd, n = ns[k]
        lines.append(f"| {k} | {fmt(m)} | {fmt(sd)} | {n} |")
    lines += ["", f"Screening rule used by swarm/run.py (candidate iff t >= {CAND_T} on the lane's primary metric):", "",
              f"- kappa_id-primary lanes: candidate iff kappa_id >= {bar(ns['kappa_id_bar'])}",
              f"- val_bpb-primary lanes: candidate iff val_bpb <= {bar(ns['val_bpb_bar'])}",
              f"- gate for every run: val_bpb <= {bar(ns['val_bpb_gate'])} (baseline mean + {BPB_GATE})",
              "", "Power state per run: " + ", ".join(sorted({r.get('power', '?') for r in b}))]
    out = ledger_path().parent / "noise.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0)

cols = ["lane", "commit", "val_bpb", "kappa_id", "kappa_ood", "t_primary", "mean_loops", "kv_kb", "params", "verdict", "desc"]
show = rows if "--all" in sys.argv else rows[-25:]
w = {c: max(len(c), *(len(str(r.get(c) or "")[:48]) for r in show)) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in show:
    print("  ".join(str(r.get(c) or "")[:48].ljust(w[c]) for c in cols))

print("\nbest valid per lane (by val_bpb; by kappa_id):")
by = defaultdict(list)
for r in rows:
    if r["valid"] == "True":
        by[r["lane"]].append(r)
for lane, rs in sorted(by.items()):
    b = min(rs, key=lambda r: float(r["val_bpb"]))
    o = max(rs, key=lambda r: float(r.get("kappa_id") or "-inf"))
    n_c = sum(r["verdict"] == "candidate" for r in rs)
    print(f"  {lane:20s} bpb {b['val_bpb']} ({b['desc'][:40]}) | best kappa_id {o.get('kappa_id')} ({o['desc'][:40]}) "
          f"| runs {len(rs)} | candidates {n_c}")
