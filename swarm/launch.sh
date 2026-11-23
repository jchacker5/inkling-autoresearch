#!/usr/bin/env bash
# Launch an autoresearch swarm: one headless Claude Code agent per lane, each in its own git worktree,
# all sharing one venv, one GPU lock and one ledger.
#
#   swarm/launch.sh <tag> [lane ...]          e.g.  swarm/launch.sh nov23 recipe depth-attn halting
#
# Env: SESSIONS (default 4)            fresh-context agent sessions per lane, run back to back
#      EXPERIMENTS (default 6)         experiments per session before the agent writes notes and exits
#      MODEL (default: CLI default)    e.g. claude-opus-5-5
#      DRY_RUN=1                       print what would run, create nothing
#
# SECURITY: agents run with --dangerously-skip-permissions (no prompts). Run the swarm on a machine or
# user account you are willing to hand to an autonomous agent.
#
# Training is serialized by swarm/run.py, so N lanes give N-way parallel *thinking*, not N-way training.
# Roughly 10-11 experiments per hour in total, however many lanes run. 3 lanes is a good default.
set -euo pipefail
TAG=${1:?usage: swarm/launch.sh <tag> [lane ...]}; shift || true
if [ $# -gt 0 ]; then LANES=("$@"); else LANES=(recipe depth-attn halting); fi
SESSIONS=${SESSIONS:-4}; EXPERIMENTS=${EXPERIMENTS:-6}
ROOT=$(git rev-parse --show-toplevel); cd "$ROOT"
export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"   # one shared venv; no per-worktree installs
mkdir -p lanes results
BASE="$HOME/.cache/inkling-autoresearch/base.safetensors"
if [ ! -f "$BASE" ] || pgrep -f pretrain_base.py >/dev/null; then
  echo "base checkpoint missing or still training ($BASE). Finish 'uv run pretrain_base.py' first."; [ "${DRY_RUN:-}" = 1 ] || exit 1
fi

for lane in "${LANES[@]}"; do
  wt="$ROOT/lanes/$lane"; br="ar/$TAG/$lane"
  prompt="You are the '$lane' lane agent in an inkling-autoresearch swarm (tag $TAG).
Your working directory is a git worktree on branch $br. Read program.md fully, then results/notes-$lane.md if it exists,
then run 'uv run swarm/status.py' to see every lane's results so far.
Work ONLY on the '$lane' lane described in program.md. Run up to $EXPERIMENTS experiments with
'uv run swarm/run.py --lane $lane ...', following the loop and keep/discard rules exactly.
Other agents share the GPU; run.py waits for the lock, so do not kill other processes or touch other worktrees.
Before you exit, append what you learned and your next 3 ideas to results/notes-$lane.md (in the MAIN repo at $ROOT/results/)."
  if [ "${DRY_RUN:-}" = 1 ]; then
    echo "[dry-run] git worktree add -b $br $wt main"; echo "[dry-run] $SESSIONS x claude -p <prompt for $lane>"; continue
  fi
  [ -d "$wt" ] || git worktree add -q -b "$br" "$wt" main
  (
    cd "$wt"
    for s in $(seq 1 "$SESSIONS"); do
      echo "=== $(date) session $s/$SESSIONS ===" >> "$ROOT/results/agent-$lane.log"
      command claude -p "$prompt" --dangerously-skip-permissions ${MODEL:+--model "$MODEL"} \
        >> "$ROOT/results/agent-$lane.log" 2>&1 || echo "session $s exited $?" >> "$ROOT/results/agent-$lane.log"
    done
  ) &
  echo "launched $lane (pid $!) -> results/agent-$lane.log"
  sleep 5
done
echo "watch: uv run swarm/status.py   |   tail -f results/agent-*.log"
echo "stop:  pkill -f swarm/launch.sh; pkill -f 'inkling-autoresearch swarm'   (launcher loops first, then agents)"
wait
