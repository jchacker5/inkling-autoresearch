"""Run one experiment through the shared GPU lock and append it to the shared ledger.

    uv run swarm/run.py --lane depth-attn --desc "K=3 depth attention" --hypothesis "..."

Many agents may call this at once from different worktrees: training is serialized by an flock on
~/.cache/inkling-autoresearch/gpu.lock so the fixed wall-clock budget stays fair. Thinking and editing
happen in parallel, training never does.
"""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CACHE = Path.home() / ".cache" / "inkling-autoresearch"
FIELDS = ["time", "lane", "commit", "dirty", "val_bpb", "reason_id", "reason_ood", "ood_at_2x", "bpb_at_2x",
          "mean_loops", "params", "kv_kb", "latency_ms", "peak_gb", "steps", "valid", "verdict", "desc", "hypothesis",
          "seed", "power"]

# Default thresholds from program.md. Once the ledger holds >= 3 baseline seeds, each threshold becomes
# max(default, 2 x the measured seed std) for gains and max(default, 1 x std) for slacks (see thresholds()).
BPB_GAIN, BPB_SLACK, ACC_GAIN, ACC_SLACK = 0.004, 0.004, 0.03, 0.02
METRICS = ("val_bpb", "reason_id", "reason_ood", "mean_loops", "kv_kb", "params")


def power_state():
    """'AC' or 'battery', plus '+lowpower' when Low Power Mode is on. Wall-clock budgets depend on it."""
    batt = sh("pmset", "-g", "batt")
    low = "lowpowermode 1" in " ".join(sh("pmset", "-g").split())
    return ("AC" if "AC Power" in batt else "battery") + ("+lowpower" if low else "")


def baseline_rows(rows):
    return [r for r in rows if r["lane"] == "baseline" and r["valid"] == "True"]


def baseline_mean(rows):
    b = baseline_rows(rows)
    if not b:
        return None
    m = {k: str(sum(float(r[k] or 0) for r in b) / len(b)) for k in METRICS}
    m["params"] = b[-1]["params"]
    m.update(commit=b[-1]["commit"], desc=f"baseline mean of {len(b)} seed(s)")
    return m


def thresholds(rows):
    import statistics as st
    b = baseline_rows(rows)
    if len(b) < 3:
        return BPB_GAIN, BPB_SLACK, ACC_GAIN, ACC_SLACK
    sd = {k: st.stdev(float(r[k]) for r in b) for k in ("val_bpb", "reason_id", "reason_ood")}
    sd_acc = max(sd["reason_id"], sd["reason_ood"])
    return (max(BPB_GAIN, 2 * sd["val_bpb"]), max(BPB_SLACK, sd["val_bpb"]),
            max(ACC_GAIN, 2 * sd_acc), max(ACC_SLACK, sd_acc))


def sh(*a, cwd=None):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True).stdout.strip()


def repo_root():
    return Path(sh("git", "rev-parse", "--show-toplevel"))


def ledger_path():
    common = Path(sh("git", "rev-parse", "--path-format=absolute", "--git-common-dir"))
    p = Path(os.environ.get("INKLING_LEDGER", common.parent / "results" / "ledger.tsv"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_ledger(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def best_for(rows, lane):
    """The reference a new run must beat: the lane's last kept run, else the mean of the baseline seeds."""
    kept = [r for r in rows if r["valid"] == "True" and r["verdict"] == "keep" and r["lane"] == lane]
    return kept[-1] if kept else baseline_mean(rows)


def verdict(res, best, th=(BPB_GAIN, BPB_SLACK, ACC_GAIN, ACC_SLACK)):
    BPB_GAIN, BPB_SLACK, ACC_GAIN, ACC_SLACK = th
    if not res.get("valid"):
        return "discard", "invalid: " + ("; ".join(res.get("violations", [])) or "smoke/crash")
    if best is None:
        return "baseline", "no prior valid run in this lane or baseline"
    d_bpb = res["val_bpb"] - float(best["val_bpb"])
    d_ood = res["reason_ood"] - float(best["reason_ood"])
    d_id = res["reason_id"] - float(best["reason_id"])
    why = f"d_bpb {d_bpb:+.4f}, d_reason_id {d_id:+.3f}, d_reason_ood {d_ood:+.3f} vs {best['commit']} ({best['desc']})"
    if d_bpb <= -BPB_GAIN and d_ood >= -ACC_SLACK:
        return "keep", why
    if max(d_ood, d_id) >= ACC_GAIN and d_bpb <= BPB_SLACK:
        return "keep", why
    cheaper = (float(res.get("mean_loops_per_token") or 0) < float(best["mean_loops"] or 0) - 0.25
               or (res.get("kv_kb_per_token") or 0) < float(best["kv_kb"] or 0) * 0.8
               or res["params"] < int(best["params"]) * 0.9)
    if cheaper and abs(d_bpb) < BPB_SLACK and d_ood > -ACC_SLACK and d_id > -ACC_SLACK:
        return "keep", "cheaper at equal quality; " + why
    return "discard", why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=None, help="sets INKLING_SEED for train.py (default: train.py's SEED)")
    a = ap.parse_args()
    if a.seed is not None:
        os.environ["INKLING_SEED"] = str(a.seed)

    root = repo_root()
    os.chdir(root)
    commit = sh("git", "rev-parse", "--short", "HEAD")
    diff = sh("git", "diff", "HEAD", "--", "train.py")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / "runs" / f"{stamp}-{a.lane}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train.diff").write_text(diff)
    (run_dir / "train.py").write_text((root / "train.py").read_text())

    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CACHE / "gpu.lock", "w") as lock:
        print(f"[{a.lane}] waiting for GPU lock ...", flush=True)
        t_wait = time.time()
        fcntl.flock(lock, fcntl.LOCK_EX)
        print(f"[{a.lane}] got lock after {time.time()-t_wait:.0f}s; training {a.desc!r}", flush=True)
        lock.write(f"{os.getpid()} {a.lane} {a.desc}\n"); lock.flush()
        power = power_state()
        log = run_dir / "run.log"
        try:
            with open(log, "w") as f:
                rc = subprocess.run([sys.executable, "train.py"], stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        if (root / "runs" / "last.json").exists() and rc == 0:
            res = json.loads((root / "runs" / "last.json").read_text())
            (root / "runs" / "last.json").replace(run_dir / "result.json")
        else:
            res = {"valid": False, "violations": [f"exit {rc}"], "val_bpb": float("nan"), "reason_id": 0, "reason_ood": 0, "params": 0}
        fcntl.flock(lock, fcntl.LOCK_UN)

    ledger = ledger_path()
    with open(CACHE / "ledger.lock", "w") as llock:
        fcntl.flock(llock, fcntl.LOCK_EX)
        rows = read_ledger(ledger)
        th = thresholds(rows)
        if a.lane == "baseline":
            v, why = ("baseline", "baseline seed") if res.get("valid") else verdict(res, None, th)
        else:
            v, why = verdict(res, best_for(rows, a.lane), th)
            if v == "baseline":
                v = "keep"
        why += f" | thresholds bpb_gain {th[0]:.4f} acc_gain {th[2]:.3f} | power {power}"
        sc = res.get("scaling", {})
        big = sc[max(sc, key=int)] if sc else {}
        row = dict(time=stamp, lane=a.lane, commit=commit, dirty=str(bool(diff)), val_bpb=f"{res['val_bpb']:.5f}",
                   reason_id=f"{res['reason_id']:.4f}", reason_ood=f"{res['reason_ood']:.4f}",
                   ood_at_2x=f"{big.get('reason_ood', float('nan')):.4f}", bpb_at_2x=f"{big.get('val_bpb', float('nan')):.5f}",
                   mean_loops=f"{res.get('mean_loops_per_token') or 0:.3f}", params=res.get("params", 0),
                   kv_kb=f"{res.get('kv_kb_per_token') or 0:.2f}", latency_ms=f"{res.get('latency_ms_b1', 0):.2f}",
                   peak_gb=f"{res.get('peak_mem_gb', 0):.2f}", steps=res.get("steps", 0), valid=str(res.get("valid")),
                   verdict=v, desc=a.desc, hypothesis=a.hypothesis,
                   seed=os.environ.get("INKLING_SEED", "0"), power=power)
        new = not ledger.exists()
        with open(ledger, "a", newline="") as f:
            w = csv.DictWriter(f, FIELDS, delimiter="\t")
            if new:
                w.writeheader()
            w.writerow(row)
        fcntl.flock(llock, fcntl.LOCK_UN)

    tail = (run_dir / "run.log").read_text().splitlines()
    print("\n".join(tail[tail.index("---"):] if "---" in tail else tail[-30:]))
    print(f"\nrun dir: {run_dir}")
    print(f"VERDICT: {v} ({why})")


if __name__ == "__main__":
    main()
