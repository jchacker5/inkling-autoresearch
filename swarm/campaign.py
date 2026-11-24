"""The full research campaign, end to end. Idempotent: every finished step leaves a marker, so it is safe
to re-run after a crash, a reboot or a kill, and it picks up where it stopped.

    uv run swarm/campaign.py --tag nov23 --lanes recipe depth-attn halting kv-reuse

1. finish the shared base (pretrain_base.py --resume until it has its full --hours)
2. baseline: 6 seeds (0..5) of "base + 5 minutes of unchanged training" -> noise report (results/noise.md);
   swarm/run.py screens every run against these pooled seeds
3. landing analysis of the base; controls and chunked-core models (6-hour schedule stopped at 2 hours); evals
4. the agent swarm (swarm/launch.sh), one lane per agent
5. confirmation: each lane's best candidate and the baseline, 6 fresh seeds each on the holdout eval set;
   one-sided Welch per lane, Holm across lanes -> results/confirm.tsv (swarm/confirm.py)

GPU work waits for AC power without Low Power Mode (every budget here is wall clock) and for 3 GB of
free disk (macOS swaps to it; a full disk gets the training process killed).
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "inkling-autoresearch"
STATE = CACHE / "campaign"
UV = os.environ.get("UV", "uv")
BASELINE_SEEDS = tuple(range(6))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "swarm"))
from confirm import BASELINE_LANE, CONFIRM_SEEDS, best_candidate, confirm_table, write_tsv  # noqa: E402
from prepare import HOLDOUT_EVAL_SEED  # noqa: E402


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "results" / "campaign.log", "a") as f:
        f.write(line + "\n")


def done(step):
    return (STATE / f"{step}.done").exists()


def mark(step, info=""):
    (STATE / f"{step}.done").write_text(info)
    log(f"done: {step} {info}")


def run(*cmd, cwd=ROOT, env=None, check=True):
    log("$ " + " ".join(cmd) + (f"   (in {cwd})" if cwd != ROOT else ""))
    rc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}).returncode
    if check and rc != 0:
        log(f"command failed with exit {rc}")
    return rc


MIN_FREE_GB = 3.0   # macOS swaps to this disk; with it full, the kernel kills the largest process (training)


def resources():
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    low = "lowpowermode 1" in " ".join(subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout.split())
    free = shutil.disk_usage(str(Path.home())).free / 2**30
    problems = []
    if "AC Power" not in batt:
        problems.append("on battery")
    if low:
        problems.append("Low Power Mode on")
    if free < MIN_FREE_GB:
        problems.append(f"only {free:.1f} GB free disk (need {MIN_FREE_GB})")
    return problems


def wait_for_power():
    """Block until the machine can train fairly and safely: AC power, no Low Power Mode, enough free disk."""
    last = None
    while True:
        problems = resources()
        if not problems:
            if last:
                log("resources OK; continuing")
            return
        if problems != last:
            log("waiting: " + "; ".join(problems))
            last = problems
        time.sleep(60)


def meta(name):
    p = CACHE / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def train_ckpt(name, hours, stop_hours=None, overrides=(), extra=()):
    target = (stop_hours or hours) * 3600 - 5
    attempts = 0
    while True:
        m = meta(name)
        if m and m["train_seconds"] >= target:
            return True
        if attempts >= 6:
            log(f"giving up on {name} after {attempts} attempts")
            return False
        wait_for_power()
        cmd = [UV, "run", "pretrain_base.py", "--name", name, "--hours", str(hours)]
        if stop_hours:
            cmd += ["--stop-hours", str(stop_hours)]
        if overrides:
            cmd += ["--set", *overrides]
        cmd += list(extra)
        if m:
            cmd.append("--resume")
        with open(ROOT / "results" / f"{name}_pretrain.log", "a") as f:
            f.write(f"=== {time.ctime()} {' '.join(cmd)} ===\n")
        log("$ " + " ".join(cmd))
        with open(ROOT / "results" / f"{name}_pretrain.log", "a") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        attempts += 1


def ledger():
    p = ROOT / "results" / "ledger.tsv"
    if not p.exists():
        return []
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lanes", nargs="+", default=["recipe", "depth-attn", "halting", "kv-reuse"])
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--experiments", type=int, default=5)
    ap.add_argument("--base-hours", type=float, default=6.0)
    ap.add_argument("--control-hours", type=float, default=2.0)
    ap.add_argument("--v2-hours", type=float, default=2.0)
    ap.add_argument("--v2-set", nargs="*", default=[], metavar="KEY=VALUE", help="overrides for base-v2 (escalation)")
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    log(f"campaign start: {vars(a)}")
    run(UV, "run", "prepare.py")          # idempotent; builds any missing data file (e.g. reason_train_v2.bin)

    # 1. base (task v1, 6 h; kept for its text ability) and base-v2 = base continued on task-v2 data.
    #    v1 had a first-digit shortcut that the base learned (README); every warm start uses base-v2.
    if not done("base"):
        if not train_ckpt("base", a.base_hours):
            sys.exit(1)
        mark("base", json.dumps({k: meta("base")[k] for k in ("steps", "train_seconds")}))
    if not done("base-v2"):
        if not train_ckpt("base-v2", a.v2_hours, overrides=a.v2_set, extra=["--init", "base", "--lr", "5e-4"]):
            sys.exit(1)
        mark("base-v2", json.dumps({k: meta("base-v2")[k] for k in ("steps", "train_seconds")}))

    # 2. baseline seeds: the pooled screening reference for every verdict (program.md), then the noise report
    for s in BASELINE_SEEDS:
        if not done(f"baseline-{s}"):
            wait_for_power()
            run(UV, "run", "swarm/run.py", "--lane", "baseline", "--seed", str(s), "--desc", f"baseline seed {s}",
                "--hypothesis", "noise floor: base + 5 min of unchanged training")
            mark(f"baseline-{s}")
    if not done("noise"):
        run(UV, "run", "swarm/status.py", "--noise")
        mark("noise")

    # 3. overthinking analysis of the base (eval only, ~15 min)
    if not done("landing-base"):
        wait_for_power()
        if run(UV, "run", "swarm/landing.py", "--name", "base-v2") == 0:
            mark("landing-base")

    # 4. controls and chunked-core models on the same 6-hour schedule, stopped at --control-hours.
    #    These need no agents, so they run before the swarm; the swarm's lanes can then use the
    #    research/debate results (results/next_lanes.txt).
    # All from scratch on task-v2 data, same 6-hour schedule stopped at 2 h. "looped" is the reference.
    # Round-2 debate: control-10L dropped (17.9M params, over the 12M cap); chunk2 dropped (control-4L beats it on
    # every cost); "looped-unweighted" tests whether the answer weighting is what creates lookup.
    controls = {"looped": [], "control-4L": ["LOOPS=1"], "chunk4": ["CHUNK=4"],
                "looped-unweighted": ["ANSWER_W=1.0", "FIRST_ANSWER_W=1.0"]}
    for name, ov in controls.items():
        if not done(name):
            if train_ckpt(name, a.base_hours, a.control_hours, ov):
                mark(name)

    # 5. evaluations (looped models get the test-time loop sweep and the landing analysis)
    h = int(a.control_hours)
    evals = [("base-v2", [], True)]
    evals += [(f"{name}-{h}h", ov, "LOOPS=1" not in ov) for name, ov in controls.items()]
    for name, ov, looped in evals:
        step = f"eval-{name}"
        if done(step) or not (CACHE / f"{name}.safetensors").exists():
            continue
        wait_for_power()
        cmd = [UV, "run", "swarm/eval_ckpt.py", "--name", name] + (["--set", *ov] if ov else []) + ([] if looped else ["--no-sweep"])
        if run(*cmd) == 0:
            mark(step)
        if looped and name.startswith("chunk") and not done(f"landing-{name}"):
            if run(UV, "run", "swarm/landing.py", "--name", name, "--set", *ov, "EXIT_METRIC='state'") == 0:
                mark(f"landing-{name}")

    # 6. swarm. Lanes come from results/next_lanes.txt when it exists (written after the debate stage).
    lanes = a.lanes
    nl = ROOT / "results" / "next_lanes.txt"
    if nl.exists() and nl.read_text().split():
        lanes = nl.read_text().split()
    if not done("swarm"):
        if subprocess.run(["pgrep", "-f", "swarm/launch.sh"], capture_output=True).returncode == 0:
            log("swarm already running (campaign restarted); waiting for it")
            while subprocess.run(["pgrep", "-f", "swarm/launch.sh"], capture_output=True).returncode == 0:
                time.sleep(60)
        else:
            wait_for_power()
            log(f"swarm lanes: {lanes}")
            run("bash", "swarm/launch.sh", a.tag, *lanes,
                env={"SESSIONS": str(a.sessions), "EXPERIMENTS": str(a.experiments)}, check=False)
        mark("swarm")

    # 7. confirmation (program.md "Promotion"): each lane's best screening candidate gets fresh seeds scored on the
    #    HOLDOUT eval set, interleaved with the same number of fresh baseline seeds on the same set; then a one-sided
    #    Welch test per lane, Holm across lanes at alpha .05 -> results/confirm.tsv.
    holdout = {"INKLING_EVAL_SEED": str(HOLDOUT_EVAL_SEED)}
    finalists = {}
    for lane in lanes:
        best = best_candidate(ledger(), lane)
        if best is not None and (ROOT / "lanes" / lane).exists():
            finalists[lane] = best
    if finalists:
        log("confirmation finalists: " + ", ".join(f"{k}@{v['commit']}" for k, v in finalists.items()))
    for s in CONFIRM_SEEDS if finalists else ():
        step = f"confirm-baseline-{s}"
        if not done(step):
            wait_for_power()
            run(UV, "run", "swarm/run.py", "--lane", BASELINE_LANE, "--seed", str(s),
                "--desc", f"holdout baseline seed {s}", "--hypothesis", "confirmation reference", env=holdout)
            mark(step)
        for lane, best in finalists.items():
            step = f"confirm-{lane}-{best['commit']}-{s}"
            if done(step):
                continue
            wt = ROOT / "lanes" / lane
            # drop any unfinished experiment, then check out the candidate's commit
            run("git", "checkout", "-q", "--", "train.py", cwd=wt, check=False)
            run("git", "checkout", "-q", best["commit"], cwd=wt, check=False)
            wait_for_power()
            run(UV, "run", "swarm/run.py", "--lane", f"{lane}-confirm", "--seed", str(s),
                "--primary", best.get("primary") or "kappa_id",
                "--desc", f"confirm {lane} best {best['commit']} seed {s}", "--hypothesis", best["desc"], cwd=wt,
                env={"UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv"), **holdout})
            mark(step)
    if finalists:
        table = confirm_table(ledger(), list(finalists), HOLDOUT_EVAL_SEED)
        write_tsv(table, ROOT / "results" / "confirm.tsv")
        for r in table:
            log(f"confirm {r['lane']}: p={r['p_one_sided']} holm_alpha={r['holm_alpha']} promoted={r['promoted']}")

    (STATE / "ALL.done").write_text(time.ctime())
    log("campaign complete")


if __name__ == "__main__":
    main()
