"""Leaderboard over the shared ledger.   uv run swarm/status.py [--all]"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run import ledger_path, read_ledger  # noqa: E402

rows = read_ledger(ledger_path())
if not rows:
    sys.exit("ledger empty: run the baseline first (uv run swarm/run.py --lane baseline --desc baseline)")

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
