"""Leaderboard over the shared ledger.   uv run swarm/status.py [--all] [--noise]

--noise writes results/noise.md: baseline seed mean/std and the keep/discard thresholds they imply."""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import baseline_rows, ledger_path, read_ledger, thresholds  # noqa: E402

rows = read_ledger(ledger_path())
if not rows:
    sys.exit("ledger empty: run the baseline first (uv run swarm/run.py --lane baseline --desc baseline)")

if "--noise" in sys.argv:
    import statistics as st
    b = baseline_rows(rows)
    lines = ["# Seed noise of the baseline", "",
             f"{len(b)} runs of base + 5 minutes of unchanged training, differing only in seed "
             "(data order and loop sampling).", "", "| metric | mean | std | min | max |", "|---|---|---|---|---|"]
    for k in ("val_bpb", "reason_id", "reason_ood", "ood_at_2x", "steps"):
        v = [float(r[k]) for r in b]
        sd = st.stdev(v) if len(v) > 1 else float("nan")
        lines.append(f"| {k} | {st.mean(v):.4f} | {sd:.4f} | {min(v):.4f} | {max(v):.4f} |")
    th = thresholds(rows)
    lines += ["", "Keep/discard thresholds used by swarm/run.py (max of the program.md default and the measured noise):", "",
              f"- val_bpb gain >= {th[0]:.4f} (slack {th[1]:.4f})", f"- reasoning gain >= {th[2]:.3f} (slack {th[3]:.3f})",
              "", "Power state per run: " + ", ".join(sorted({r.get('power', '?') for r in b}))]
    out = ledger_path().parent / "noise.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(0)

cols = ["lane", "commit", "val_bpb", "reason_id", "reason_ood", "ood_at_2x", "mean_loops", "kv_kb", "params", "verdict", "desc"]
show = rows if "--all" in sys.argv else rows[-25:]
w = {c: max(len(c), *(len(str(r[c])[:48]) for r in show)) for c in cols}
print("  ".join(c.ljust(w[c]) for c in cols))
for r in show:
    print("  ".join(str(r[c])[:48].ljust(w[c]) for c in cols))

print("\nbest valid per lane (by val_bpb, then reason_ood):")
by = defaultdict(list)
for r in rows:
    if r["valid"] == "True":
        by[r["lane"]].append(r)
for lane, rs in sorted(by.items()):
    b = min(rs, key=lambda r: (float(r["val_bpb"]), -float(r["reason_ood"])))
    o = max(rs, key=lambda r: float(r["reason_ood"]))
    print(f"  {lane:20s} bpb {b['val_bpb']} ({b['desc'][:40]}) | best ood {o['reason_ood']} ({o['desc'][:40]}) | runs {len(rs)}")
