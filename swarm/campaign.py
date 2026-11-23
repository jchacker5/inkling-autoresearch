"""The full research campaign, end to end. Idempotent: every finished step leaves a marker, so it is safe
to re-run after a crash, a reboot or a kill, and it picks up where it stopped.

    uv run swarm/campaign.py --tag nov23 --lanes recipe depth-attn halting kv-reuse

1. finish the shared base (pretrain_base.py --resume until it has its full --hours)
2. baseline: 3 seeds of "base + 5 minutes of unchanged training" -> noise -> keep/discard thresholds
3. the agent swarm (swarm/launch.sh), one lane per agent
4. confirmation: re-run every lane's best kept commit with seeds 1 and 2
5. controls: conventional transformers trained on the same 6-hour schedule, stopped at 2 hours
   (same unique params: 4 layers, LOOPS=1; same depth: 10 layers, LOOPS=1, ~2.4x params)
6. full evaluation of the base (2 h snapshot and final, with a test-time loop sweep) and the controls

GPU work waits for AC power without Low Power Mode, because every budget here is wall clock.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path.home() / ".cache" / "inkling-autoresearch"
STATE = CACHE / "campaign"
UV = os.environ.get("UV", "uv")


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


def wait_for_power():
    warned = False
    while True:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        low = "lowpowermode 1" in " ".join(subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout.split())
        if "AC Power" in batt and not low:
            if warned:
                log("power OK (AC, no Low Power Mode); continuing")
            return
        if not warned:
            log("waiting for AC power without Low Power Mode (wall-clock budgets depend on it)")
            warned = True
        time.sleep(60)


def meta(name):
    p = CACHE / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def train_ckpt(name, hours, stop_hours=None, overrides=()):
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
    a = ap.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    log(f"campaign start: {vars(a)}")

    # 1. base
    if not done("base"):
        if not train_ckpt("base", a.base_hours):
            sys.exit(1)
        mark("base", json.dumps({k: meta("base")[k] for k in ("steps", "train_seconds")}))

    # 2. baseline seeds
    for s in (0, 1, 2):
        if not done(f"baseline-{s}"):
            wait_for_power()
            run(UV, "run", "swarm/run.py", "--lane", "baseline", "--seed", str(s), "--desc", f"baseline seed {s}",
                "--hypothesis", "noise floor: base + 5 min of unchanged training")
            mark(f"baseline-{s}")
    if not done("noise"):
        run(UV, "run", "swarm/status.py", "--noise")
        mark("noise")

    # 3. swarm
    if not done("swarm"):
        if subprocess.run(["pgrep", "-f", "swarm/launch.sh"], capture_output=True).returncode == 0:
            log("swarm already running (campaign restarted); waiting for it")
            while subprocess.run(["pgrep", "-f", "swarm/launch.sh"], capture_output=True).returncode == 0:
                time.sleep(60)
        else:
            wait_for_power()
            run("bash", "swarm/launch.sh", a.tag, *a.lanes,
                env={"SESSIONS": str(a.sessions), "EXPERIMENTS": str(a.experiments)}, check=False)
        mark("swarm")

    # 4. confirm each lane's best kept commit with seeds 1 and 2
    for lane in a.lanes:
        wt = ROOT / "lanes" / lane
        kept = [r for r in ledger() if r["lane"] == lane and r["verdict"] == "keep" and r["valid"] == "True"]
        if not kept or not wt.exists():
            continue
        best_commit = kept[-1]["commit"]
        for s in (1, 2):
            step = f"confirm-{lane}-{s}"
            if done(step):
                continue
            # the lane's branch head is its last kept commit; drop any unfinished experiment first
            run("git", "checkout", "-q", "--", "train.py", cwd=wt, check=False)
            head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
            if head != best_commit:
                run("git", "checkout", "-q", best_commit, cwd=wt, check=False)
            wait_for_power()
            run(UV, "run", "swarm/run.py", "--lane", f"{lane}-confirm", "--seed", str(s),
                "--desc", f"confirm {lane} best {best_commit} seed {s}", "--hypothesis", kept[-1]["desc"], cwd=wt,
                env={"UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv")})
            mark(step)

    # 5. controls on the same 6-hour schedule, stopped at --control-hours
    controls = {"control-4L": ["LOOPS=1"], "control-10L": ["LOOPS=1", "CORE=8"]}
    for name, ov in controls.items():
        if not done(name):
            if train_ckpt(name, a.base_hours, a.control_hours, ov):
                mark(name)

    # 6. evaluations
    h = int(a.control_hours)
    evals = [(f"base-{h}h", [], True), ("base", [], True)]
    evals += [(f"{name}-{h}h", ov, False) for name, ov in controls.items()]
    for name, ov, sweep in evals:
        step = f"eval-{name}"
        if done(step) or not (CACHE / f"{name}.safetensors").exists():
            continue
        wait_for_power()
        cmd = [UV, "run", "swarm/eval_ckpt.py", "--name", name] + (["--set", *ov] if ov else []) + ([] if sweep else ["--no-sweep"])
        if run(*cmd) == 0:
            mark(step)

    (STATE / "ALL.done").write_text(time.ctime())
    log("campaign complete")


if __name__ == "__main__":
    main()
