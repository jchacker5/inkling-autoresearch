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
import shutil
import subprocess
import sys
import time
from pathlib import Path

CACHE = Path.home() / ".cache" / "inkling-autoresearch"
# Append-only: new columns go at the end so old ledgers stay readable.
FIELDS = ["time", "lane", "commit", "dirty", "val_bpb", "reason_id", "reason_ood", "ood_at_2x", "bpb_at_2x",
          "mean_loops", "params", "kv_kb", "latency_ms", "peak_gb", "steps", "valid", "verdict", "desc", "hypothesis",
          "seed", "power",
          "harness", "base", "eval_seed", "recipe", "primary", "kappa_id", "kappa_ood", "bytes_f", "latency_t100_ms",
          "train_peak_gb", "eval_peak_gb", "t_primary"]

# Keep / promotion rule (program.md; research/04-debate-r2-evaluation.md D4). Every run is screened against the
# POOLED baseline lane (all valid baseline rows with the same harness, base and eval set), never a lane's last keep.
#   gate:      val_bpb <= baseline mean + BPB_GATE (and, for val_bpb-primary lanes, kappa_id >= mean - max(.02, sd))
#   candidate: t = sign * (x - baseline mean) / baseline sd >= CAND_T on the lane's declared primary metric
#   pending-noise: fewer than MIN_NOISE_SEEDS baseline rows, so the noise is unknown
# A candidate is only a nomination: swarm/campaign.py confirms each lane's best candidate with fresh seeds on the
# holdout eval set (Welch t-test vs baseline seeds, Holm across lanes) and writes results/confirm.tsv.
PRIMARIES = {"kappa_id": 1.0, "val_bpb": -1.0}      # +1: higher is better
CAND_T = 0.92                                        # one-sided t(0.80, 5 df) for 6 baseline seeds
BPB_GATE = 0.02
ACC_SLACK = 0.02
MIN_NOISE_SEEDS = 3
SCREEN_EVAL_SEED = "424242"                          # prepare.EVAL_SEED's default; other seeds are holdout runs
METRICS = ("val_bpb", "reason_id", "reason_ood", "kappa_id", "kappa_ood", "mean_loops", "kv_kb", "params")


def power_state():
    """'AC' or 'battery', plus '+lowpower' when Low Power Mode is on. Wall-clock budgets depend on it.
    Off macOS (e.g. a CUDA box) there is no power state; the GPU name is recorded instead."""
    if not shutil.which("pmset"):
        gpu = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader") if shutil.which("nvidia-smi") else ""
        return os.environ.get("INKLING_DEVICE") or (gpu.splitlines()[0].strip() if gpu else "unknown")
    batt = sh("pmset", "-g", "batt")
    low = "lowpowermode 1" in " ".join(sh("pmset", "-g").split())
    return ("AC" if "AC Power" in batt else "battery") + ("+lowpower" if low else "")


def _matches(r, key, want):
    return want is None or not r.get(key) or r[key] == str(want)


def baseline_rows(rows, harness=None, base=None, eval_seed=None):
    """Valid baseline-lane rows; with harness/base/eval_seed given, only rows produced under the same ones."""
    return [r for r in rows if r["lane"] == "baseline" and r["valid"] == "True"
            and _matches(r, "harness", harness) and _matches(r, "base", base) and _matches(r, "eval_seed", eval_seed)]


def stats(vals):
    """(mean, sample sd or None, n)."""
    import statistics as st
    v = [float(x) for x in vals]
    if not v:
        return None, None, 0
    return st.mean(v), (st.stdev(v) if len(v) > 1 else None), len(v)


def verdict(res, base, primary="kappa_id"):
    """Screen one result against pooled baseline rows. Returns (verdict, reason, t or None)."""
    if not res.get("valid"):
        return "discard", "invalid: " + ("; ".join(res.get("violations", [])) or "smoke/crash"), None
    if primary not in PRIMARIES:
        raise ValueError(f"primary must be one of {list(PRIMARIES)}")
    if not base:
        return "pending-noise", "no valid baseline rows yet", None
    m_bpb, _, n = stats(r["val_bpb"] for r in base)
    m_k, sd_k, _ = stats(r["kappa_id"] for r in base)
    d_bpb = res["val_bpb"] - m_bpb
    d_k = res["kappa_id"] - m_k
    why = f"d_bpb {d_bpb:+.4f}, d_kappa_id {d_k:+.3f} vs pooled baseline (n={n})"
    if d_bpb > BPB_GATE:
        return "discard", f"gate: val_bpb more than {BPB_GATE} above the baseline mean; " + why, None
    if primary == "val_bpb" and d_k < -max(ACC_SLACK, sd_k or 0):
        return "discard", "gate: kappa_id below the baseline mean by more than max(.02, sd); " + why, None
    if n < MIN_NOISE_SEEDS:
        return "pending-noise", f"{n} < {MIN_NOISE_SEEDS} baseline seeds; " + why, None
    m, sd, _ = stats(r[primary] for r in base)
    t = PRIMARIES[primary] * (res[primary] - m) / max(sd, 1e-12)
    why = f"t({primary}) {t:+.2f} (need >= {CAND_T}); " + why
    return ("candidate" if t >= CAND_T else "discard"), why, t


def screen(res, rows, lane, primary="kappa_id"):
    """Verdict for a finished run given the ledger so far. Returns (verdict, reason, t or None, eval_seed)."""
    eval_seed = str(res.get("eval_seed", os.environ.get("INKLING_EVAL_SEED", SCREEN_EVAL_SEED)))
    if not res.get("valid"):
        return (*verdict(res, None)[:2], None, eval_seed)
    if eval_seed != SCREEN_EVAL_SEED:
        return "holdout", f"scored on holdout eval set {eval_seed}; see results/confirm.tsv", None, eval_seed
    if lane == "baseline":
        return "baseline", "baseline seed", None, eval_seed
    base = baseline_rows(rows, res.get("harness"), res.get("base"), eval_seed)
    return (*verdict(res, base, primary), eval_seed)


def ledger_row(res, v, t, eval_seed, **meta):
    """One ledger row (FIELDS) from a result dict; meta holds time, lane, commit, dirty, desc, hypothesis, seed,
    power and primary."""
    sc = res.get("scaling", {})
    big = sc[max(sc, key=int)] if sc else {}

    def fm(k, fmt):
        return format(res[k], fmt) if res.get(k) is not None else ""

    row = dict(val_bpb=f"{res['val_bpb']:.5f}", reason_id=f"{res['reason_id']:.4f}", reason_ood=f"{res['reason_ood']:.4f}",
               ood_at_2x=f"{big.get('reason_ood', float('nan')):.4f}", bpb_at_2x=f"{big.get('val_bpb', float('nan')):.5f}",
               mean_loops=f"{res.get('mean_loops_per_token') or 0:.3f}", params=res.get("params", 0),
               kv_kb=f"{res.get('kv_kb_per_token') or 0:.2f}", latency_ms=f"{res.get('latency_ms_b1', 0):.2f}",
               peak_gb=f"{res.get('peak_mem_gb', 0):.2f}", steps=res.get("steps", 0), valid=str(res.get("valid")),
               verdict=v, harness=res.get("harness", ""), base=res.get("base", ""), eval_seed=eval_seed,
               recipe=res.get("recipe", ""), kappa_id=fm("kappa_id", ".4f"), kappa_ood=fm("kappa_ood", ".4f"),
               bytes_f=res.get("bytes_f", ""), latency_t100_ms=fm("latency_ms_b1_t100", ".2f"),
               train_peak_gb=fm("train_peak_mem_gb", ".2f"), eval_peak_gb=fm("eval_peak_mem_gb", ".2f"),
               t_primary="" if t is None else f"{t:.3f}")
    row.update(meta)
    assert set(row) == set(FIELDS), set(row) ^ set(FIELDS)
    return row


def noise_summary(rows):
    """Baseline mean/sd per metric and the candidate bars they imply (for status.py --noise)."""
    b = baseline_rows(rows, eval_seed=SCREEN_EVAL_SEED)
    out = {k: stats(r[k] for r in b if r.get(k) not in (None, ""))
           for k in ("val_bpb", "kappa_id", "kappa_ood", "reason_id", "steps")}
    m, sd, n = out["kappa_id"]
    out["kappa_id_bar"] = (m + CAND_T * sd) if (sd is not None and n >= MIN_NOISE_SEEDS) else None
    m, sd, n = out["val_bpb"]
    out["val_bpb_bar"] = (m - CAND_T * sd) if (sd is not None and n >= MIN_NOISE_SEEDS) else None
    out["val_bpb_gate"] = m + BPB_GATE if m is not None else None
    out["n"] = len(b)
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True)
    ap.add_argument("--desc", required=True)
    ap.add_argument("--hypothesis", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--seed", type=int, default=None, help="sets INKLING_SEED for train.py (default: train.py's SEED)")
    ap.add_argument("--primary", default=os.environ.get("INKLING_PRIMARY", "kappa_id"), choices=list(PRIMARIES),
                    help="the lane's declared primary metric (program.md lanes)")
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
        v, why, t, eval_seed = screen(res, rows, a.lane, a.primary)
        why += f" | power {power}"
        row = ledger_row(res, v, t, eval_seed, time=stamp, lane=a.lane, commit=commit, dirty=str(bool(diff)),
                         desc=a.desc, hypothesis=a.hypothesis, seed=os.environ.get("INKLING_SEED", "0"),
                         power=power, primary=a.primary)
        new = not ledger.exists()
        with open(ledger, "a", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS, delimiter="\t")
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
