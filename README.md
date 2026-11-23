# inkling-autoresearch

**Can a compact recurrent-depth language model learn to allocate its own computation and get better
reasoning per byte of memory than a conventional transformer?**

This repo adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) (a fixed harness, one
editable file, a fixed 5-minute budget, keep/discard by metric, loop forever) to looped transformers on
Apple Silicon with MLX. It adds a swarm layer, so several agents can explore different hypotheses at
once while sharing one GPU fairly.

```
prepare.py        FIXED  data (ClimbMix bytes + synthetic multi-hop reasoning), budgets, evaluate()
pretrain_base.py  FIXED  builds the shared base checkpoint once (hours); experiments warm-start from it
train.py          AGENT  looped transformer + experiment switches + training loop
program.md        AGENT  research program: metrics, keep/discard rule, lanes, the loop
swarm/run.py             one experiment: GPU flock -> train -> evaluate -> ledger -> verdict
swarm/status.py          leaderboard over results/ledger.tsv
swarm/launch.sh          one headless Claude Code agent per lane, each in its own worktree
```

## Quick start (M-series Mac)

```sh
uv sync
uv run prepare.py                                   # ~270 MB in ~/.cache/inkling-autoresearch
uv run pretrain_base.py --hours 6                   # once: the shared base checkpoint (see below)
uv run swarm/run.py --lane baseline --desc baseline # 5 min continued training + ~1 min eval
uv run swarm/status.py
```

Single agent, Karpathy style: open Claude Code here and say *"read program.md and start the loop on
lane recipe"*. Swarm: `swarm/launch.sh nov23 recipe depth-attn halting` (see the header of
`swarm/launch.sh` for knobs, and use `DRY_RUN=1` first).

## Why there is a base checkpoint

Calibration on the M4 (`results/calibration.md`) showed that 5 minutes from scratch is about 250 steps,
far too few for in-context retrieval to emerge, so reasoning scores sit at chance and every
architecture comparison would be noise. Following the plan of testing modifications on a trained looped
model rather than training new ones, `pretrain_base.py` trains the looped backbone once, for hours.
Each experiment then adds zero-initialised modules and trains for the fixed 5 minutes from that
checkpoint. `base_progress.tsv` records when retrieval emerges during base training.

## What the harness measures

- `val_bpb`: held-out web text, the knowledge proxy.
- `reason_id` / `reason_ood`: exact match on multi-hop variable tracking (pointer chains with distractors) at 2/4/6 hops (seen in training)
  and 8/10 hops (never seen). Deeper chains need more serial computation, which is what loops should buy.
- Test-time scaling: the same weights at loops/2 and 2x loops.
- Costs: unique params (budget 12M, so "per gigabyte" is enforced), analytical KV KB/token, batch-1
  latency, peak MLX memory, and realised mean loops per token under halting.

## Model

`prelude -> [shared core blocks] x LOOPS -> coda`, with RoPE, RMSNorm and SwiGLU. Switches:
`DEPTH_ATTN` (attend over earlier loop states), `HALTING` (per-token ACT-style halting with a ponder cost),
`HISTORY` / `LOOP_EMBED` (history-state injection and loop-index conditioning), `KV_SHARE` (later loops
reuse loop-1 K/V), `RANDOM_LOOPS` / `BPTT_LOOPS` (Huginn-style training for test-time scaling).
These are first-pass implementations written from one-paragraph descriptions of the cited methods,
not reproductions of the papers.

## Why the swarm serializes training

The budget is wall-clock, so two runs sharing the GPU would each get half the steps and every
comparison would be corrupted. Agents think, read and edit in parallel. `swarm/run.py` holds an
exclusive flock while training. Throughput is about 10 experiments per hour total; extra lanes add
diversity of ideas, not speed.
