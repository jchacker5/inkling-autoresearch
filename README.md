# inkling-autoresearch

[![test](https://github.com/jchacker5/inkling-autoresearch/actions/workflows/test.yml/badge.svg)](https://github.com/jchacker5/inkling-autoresearch/actions/workflows/test.yml)

**Can a compact recurrent-depth language model learn to allocate its own computation and get better
reasoning per byte of memory than a conventional transformer?**

This repo adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to that question:
a fixed harness, one editable file, a fixed 5-minute budget, keep/discard by metric, and a loop that
never stops. It targets looped transformers on Apple Silicon with [MLX](https://github.com/ml-explore/mlx),
and adds a swarm layer so several AI agents can work different hypotheses at once while sharing one GPU
fairly. "Inkling" is the working name of the model family.

## Status (2026-11-23)

- The harness, model switches, swarm runner and tests are in place.
- The shared base checkpoint is still training (6 hours planned on an M4, 32 GB).
- **There are no architecture results yet.** Nothing here shows that any modification beats the baseline.
- The paper is being written in [`paper/`](paper/): the setup and calibration sections are drafted, and the
  results sections will be filled from the overnight campaign (`swarm/campaign.py`).

**The first two reasoning tasks had shortcuts, and the first 6-hour base learned one.** In v1 the chain's
start was the first statement about half the time, so copying the prompt's first digit scored .48-.57 at every
hop; the base's .44-.54 was that rule. In v2 the queried chain was the only long single-child path, so structure
alone found it. Task v3 (`prepare.py`) puts two chains of equal length in each problem and asks both ends as
twin prompts. A pair counts only if both answers are right, so a model that ignores the question scores 0 and
structure plus a coin flip scores .25 (`reason_floor_id/ood`). Models trained on v1 or v2 data are not
comparable with v3 results.

## How it works

```
prepare.py        FIXED  data (web-text bytes + synthetic multi-hop reasoning), budgets, evaluate()
pretrain_base.py  FIXED  builds the shared base checkpoint once (hours); experiments warm-start from it
train.py          AGENT  looped transformer + experiment switches + training loop
program.md        AGENT  research program: metrics, keep/discard rule, lanes, the loop
swarm/run.py             one experiment: GPU lock -> train -> evaluate -> ledger -> verdict
swarm/status.py          leaderboard over results/ledger.tsv
swarm/launch.sh          one headless Claude Code agent per lane, each in its own git worktree
tests/                   task correctness, causality and init checks for every switch (CPU, ~2 s)
results/                 calibration evidence, noise estimates, per-lane notes
```

**Two stages.** In-context retrieval, which the reasoning task needs, only emerges after thousands of
steps. Five minutes from scratch on an M4 is about 250 steps, and reasoning scores sit at chance
(`results/calibration.md` has the measurements and the bug checks). So `pretrain_base.py` trains the
looped backbone once, for hours, and each experiment warm-starts from it and trains for the fixed
5 minutes with its modification. The baseline is "base plus 5 more minutes of unchanged training".

**Why training is serialized.** The budget is wall clock, so two runs sharing the GPU would each get
half the steps and every comparison would be corrupted. Agents think, read and edit in parallel, but
`swarm/run.py` holds an exclusive lock while training. Expect about 10 experiments per hour in total,
however many agents run.

## Quick start

Requires an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), and about 2 GB of free disk.

```sh
uv sync
uv run pytest -q                                    # ~2 s, CPU only
uv run prepare.py                                   # downloads ~270 MB to ~/.cache/inkling-autoresearch
uv run pretrain_base.py --hours 6                   # once; checkpoints every 15 min, --resume after a stop
uv run swarm/run.py --lane baseline --desc baseline # 5 min of training + ~1 min of evaluation
uv run swarm/status.py
```

`INKLING_SMOKE=1 uv run train.py` runs a 20-second smoke test of the full pipeline. Smoke results are
marked invalid and never count.

### Running the research loop

- **One agent, Karpathy style:** open Claude Code (or another coding agent) in the repo and say
  *"read program.md and start the loop on lane recipe"*.
- **A swarm:** `swarm/launch.sh <tag> recipe depth-attn halting`. Try `DRY_RUN=1` first; the header of
  `swarm/launch.sh` lists the knobs.

> **Security:** the swarm runs `claude -p --dangerously-skip-permissions`, so agents execute commands
> without asking. Run it only on a machine or user account you are willing to hand to an autonomous
> agent. It also uses your Claude plan or API credits for as long as it runs.

## What is measured

| metric | meaning |
|---|---|
| `val_bpb` | bits per byte on held-out web text: the knowledge and fluency proxy |
| `reason_id` | exact match on multi-hop variable tracking at 2/4/6 hops (depths seen in training) |
| `reason_ood` | the same at 8/10 hops (deeper than anything in training) |
| `scaling` | the same weights at half and double the loop count (test-time compute) |
| `mean_loops_per_token` | realised compute under adaptive halting |
| `latency_ms_b1` | batch-1 forward pass over 512 bytes (prefill, a stand-in for time to first token) |
| `kv_kb_per_token`, `peak_mem_gb`, `weights_mb_bf16`, `params` | memory costs; unique params are capped at 12M |

The task looks like `q=4;k=q;x=7;b=k;y=x;?b=` → `4`: one pointer chain of exactly *n* hops, mixed with
distractor variables. Each hop is one more serial lookup, so deeper chains need more computation, which
is what loops should buy.

Not measured yet: energy, and real latency savings from halting (halted tokens still pass through the
dense kernels, so mean loops is an analytical compute measure for now).

## Model and experiment switches

`prelude → [shared core blocks] × LOOPS → coda`, with RoPE, RMSNorm and SwiGLU, 7.3M parameters by default.

| switch | idea | starts as a no-op? |
|---|---|---|
| `DEPTH_ATTN=K` | each loop attends per token over the last K loop states (ARLoop-style) | yes |
| `HALTING` | per-token ACT-style halting with a ponder cost (Mixture-of-Recursions-style) | no, changes the computation |
| `HISTORY`, `LOOP_EMBED` | inject a mean or gated summary of earlier loop states; loop-index conditioning | yes |
| `KV_SHARE` | later loops reuse loop 1's keys and values (KV cache 15 → 6 KB/token) | no, changes the computation |
| `EXIT_TAU`, `EXIT_METRIC` | training-free early exit at eval: a token stops looping once one more loop changes its prediction by less than tau (KL or state change) | yes at tau=0 (off) |
| `LOOP_CONSIST` | self-distillation: a random earlier loop learns to predict the final loop's output, so early exits lose less | yes at 0 (off); adds no parameters |
| `CHUNK` | run the looped core over K-byte chunks (hourglass, after CALM arXiv 2510.27688): core compute and core KV cache drop by K; the byte-level coda keeps bits-per-byte exact | no, new architecture; needs its own base |
| `RANDOM_LOOPS`, `BPTT_LOOPS` | Huginn-style training for test-time scaling; truncated backprop through loops | n/a, training only |

These are first-pass implementations written from short descriptions of the methods, not reproductions
of any paper. `tests/test_model.py` checks that every switch trains, stays causal, and (where claimed)
leaves a warm-started model unchanged at init.

## Open problems (good places to contribute)

1. **A non-looped control** with the same unique parameter count and its own base checkpoint. Needed
   before any "loops help" claim.
2. **Real token skipping for halting**, so halting saves wall-clock time and not just analytical FLOPs.
3. **Energy measurement** on Apple Silicon (for example by sampling `powermetrics`) for the
   per-joule half of "intelligence per gigabyte".
4. **Parallel recurrence (LoopFold-style)**, phase 2 in `program.md`: predict later loop states, run
   them in parallel and correct them, measured in real MLX wall clock.
5. **Hybrid recurrent cores** (for example mLSTM or DeltaNet layers inside the loop).
6. **Seed noise**: `results/noise.md` with baseline seed variance, so the keep/discard thresholds are
   calibrated rather than guessed.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to propose changes and how to report results.

## Acknowledgements

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the research-loop design. This
  repo is a separate implementation, not a fork.
- Training text is downloaded at run time from the
  [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)
  dataset on Hugging Face, which is derived from NVIDIA's ClimbMix. No data is redistributed in this
  repo; the dataset's own terms apply.

## License

[MIT](LICENSE)
