# inkling-autoresearch

[![test](https://github.com/jchacker5/inkling-autoresearch/actions/workflows/test.yml/badge.svg)](https://github.com/jchacker5/inkling-autoresearch/actions/workflows/test.yml)

**Can a compact recurrent-depth language model learn to allocate its own computation and get better
reasoning per byte of memory than a conventional transformer?**

This repo adapts [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) to that question:
a fixed harness, one editable file, a fixed 5-minute budget, keep/discard by metric, and a loop that
never stops. It targets looped transformers on Apple Silicon with [MLX](https://github.com/ml-explore/mlx),
and adds a swarm layer so several AI agents can work different hypotheses at once while sharing one GPU
fairly. "Inkling" is the working name of the model family.

> "Inkling" is this repo's working name for a small model family. It has no connection to Thinking Machines Lab
> or its models.

## Status (2026-11-27)

The harness, model switches, swarm runner and tests (106, CPU only) are in place, and the first full campaign
has run on one M4 with 32 GB. The results so far are mixed, and the main one is negative for looping.

**1. At matched parameters and wall clock, the non-looped model wins—at matched steps, the looped model wins.** Each row below trains 7.3M unique parameters from scratch on task-v4 data (`swarm/campaign.py`, `results/eval-*.json`):

| model | steps | `kappa_id` | `kappa_ood` | `val_bpb` |
|---|---|---|---|---|
| non-looped, 4 layers (`LOOPS=1`, 2 h) | 11,300 | .728 | .213 | 1.642 |
| chunked looped core (`CHUNK=4`, 2 h) | 11,539 | .31 | -.05 | 1.723 |
| looped, 4 loops (2 h) | 4,765 | 0 (floor) | 0 | 1.713 |
| looped, 4 loops, no answer weighting (2 h) | 4,546 | 0 | 0 | 1.712 |
| **looped, 4 loops, stretched LR (4.75 h)** | **9,815** | **.913** | **.283** | **1.635** |

A looped step costs about 2.5 times a non-looped one, so in 2 hours the looped model saw 2.4 times fewer tokens and had not yet reached the multi-hop phase transition. When allowed to run on the same per-step LR schedule (`looped-matched`, `results/eval-looped-matched.json`), it beat `control-4L` in **1,485 fewer steps** on `val_bpb` (`1.635` vs `1.642`), `kappa_id` (`.913` vs `.728`; hops 2/4/6 = `.970/.960/.825` vs `.960/.770/.505`), and `kappa_ood` (`.283` vs `.213`). At 8 test-time loops (`2x` training loops), OOD `reason_ood` jumped further from `.328` to **`.505`** (`kappa_ood = .472`), while `val_bpb` rose to `1.709`.

**2. Multi-hop composition appears suddenly, and only after long training.** In a from-scratch chain task with
equal-length chains (`swarm/sanity_lookup.py --hops 1 --nv 8`, so a model that ignores the question scores .25):

| step | direct lookup (hop 0) | one-step chain (hop 1) |
|---|---|---|
| 1,500 | 1.00 | ~.30 |
| 3,750-6,000 | 1.00 | .51-.59 |
| 6,250-9,750 | 1.00 | .75-.90 |

With 1 loop instead of 4, direct lookup never formed in 7,500 steps when the context also held pointer
statements (`results/multihop-*.txt`).

**3. The 5-minute warm-start swarm found only small text gains.** 4 lanes ran on the `base-v4` checkpoint, and
each lane's best run was rerun with 6 fresh training seeds (Welch test with Holm correction, `results/confirm.tsv`).
The reasoning examples changed, but the text score reused the same validation shard and windows used during screening.
Two text candidates passed the historical gate on that shared validation corpus: the recipe lane (decay over the whole budget plus WiSE-FT weight interpolation,
1.4698 to 1.4666) and the depth lane (1.4671). The binding and exit lanes changed nothing. Reasoning could not be
screened because every `base-v4` baseline seed scores 0 pairs. The text changes are exploratory until reproduced on an
untouched text corpus. The checked-in summary alone cannot certify the six original seed identities; keep the per-seed ledger.

**What changed in the task along the way.** The first four reasoning tasks each had a shortcut that a model
can use without reading the question. In v1, copying the prompt's first digit worked. In v2, structure found
the queried chain. In v3, each variable was asked about once, so elimination worked. In random chain forests,
the most common value worked. Task v4 (`prepare.py`) asks twin questions about equal-length chains, scores a pair
only if both answers are right, and draws later training questions with replacement. `tests/test_task.py` checks the
question-blind floors and the elimination shortcut. Models trained on earlier task versions are not comparable with v4 results.

The paper draft is in [`paper/`](paper/).

## How it works

```
prepare.py        FIXED  data (web-text bytes + synthetic multi-hop reasoning), budgets, evaluate()
pretrain_base.py  FIXED  builds the shared base checkpoint once (hours); experiments warm-start from it
train.py          AGENT  looped transformer + experiment switches + training loop
program.md        AGENT  research program: metrics, keep/discard rule, lanes, the loop
swarm/run.py             one experiment: GPU lock -> train -> evaluate -> ledger -> verdict
swarm/status.py          leaderboard over results/ledger.tsv
swarm/campaign.py        the whole campaign: base, 6 baseline seeds, controls, swarm, holdout confirmation
swarm/confirm.py         confirmation statistics (Welch + Holm) -> results/confirm.tsv
swarm/launch.sh          one headless Claude Code agent per lane, each in its own git worktree
tests/                   task correctness, causality and init checks for every switch (CPU, ~2 s)
results/                 calibration evidence, noise estimates, per-lane notes
```

**Two stages.** In-context retrieval, which the reasoning task needs, only emerges after thousands of
steps. Five minutes from scratch on an M4 is about 250 steps, and reasoning scores sit at chance
(`results/calibration.md` has the measurements and the bug checks). So `pretrain_base.py` trains the
looped backbone once, for hours, and each experiment warm-starts from it and trains for the fixed
5 minutes with its modification. The baseline is "base plus 5 more minutes of unchanged training".
Warm starts use recipe R: LR 1e-4 for tensors from the base, 1e-3 for new matrices and 1e-2 for new gates,
then a short warm-up, a constant phase and a linear decay to 0.

**Screening, then promotion.** Each run is scored on `kappa_id`, pair accuracy rescaled so the
query-blind bound is 0 (or on `val_bpb`, if its lane declares that). It is compared with the pooled
baseline seeds: a run is a *candidate* if it beats their mean by 0.92 baseline SDs while `val_bpb`
stays within +.02. Candidates are only nominations. Promotion needs 6 fresh seeds on a holdout eval set,
a Welch t-test against 6 fresh baseline seeds and a Holm correction across lanes (`program.md`).

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

### Reproducing the Status tables

```sh
uv run pretrain_base.py --name looped --hours 6 --stop-hours 2               # each control: 6 h schedule, stopped at 2 h
uv run pretrain_base.py --name control-4L --hours 6 --stop-hours 2 --set LOOPS=1
uv run pretrain_base.py --name chunk4 --hours 6 --stop-hours 2 --set CHUNK=4
uv run swarm/eval_ckpt.py --name looped-2h                                   # writes results/eval-<name>.json
uv run swarm/eval_ckpt.py --name control-4L-2h --set LOOPS=1 --no-sweep
uv run swarm/sanity_lookup.py --minutes 150 --hops 1 --nv 8                  # the chain table (4 loops)
uv run swarm/campaign.py                                                     # everything, including the swarm
```

Seeds are fixed in `train.py` and `prepare.py`, but wall-clock budgets make step counts depend on the machine.
Checkpoints are not in the repo; rebuilding the two controls takes about 4 hours on an M4.

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
| `CANON=K` | Canon layers (Allen-Zhu, Physics of LMs 4.1): a causal depthwise conv over the last K positions on each block's attention and MLP inputs. On by default (K=4); it makes in-context lookup form within ~1,000 steps | yes (zero-init) |
| `SLEEP=W`, `DREAM` | after Lee et al., in-context adaptation via sleep: a one-pass wake stream reads the looped sleep stream's keys and values for earlier W-byte windows; `DREAM` distils sleep logits into wake logits | no, changes the computation; not yet evaluated |
| `LOOP_CHOICES` | sample the training loop count per step from a tuple, e.g. `(4,)*7+(8,)`; trades default-loop bpb for 8-loop bpb | n/a, training only (off by default) |
| `R_INTERP` | warm starts only: after training, interpolate base tensors back toward the base (WiSE-FT); 0.5 by default after holdout confirmation | n/a, warm starts only |

These are first-pass implementations written from short descriptions of the methods, not reproductions
of any paper. `tests/test_model.py` checks that every switch trains, stays causal, and (where claimed)
leaves a warm-started model unchanged at init.

## Open problems (good places to contribute)

Each has a GitHub issue with details.

1. **Looping at matched steps ([#5](https://github.com/jchacker5/inkling-autoresearch/issues/5)).** `looped-matched` (9,815 steps) beat `control-4L` (11,300 steps) on `kappa_id` (`.913` vs `.728`), `kappa_ood` (`.283` vs `.213`), and `val_bpb` (`1.635` vs `1.642`), and reached `reason_ood = .505` at 8 test-time loops. Remaining: exact 11,300-step check or closing the throughput gap.
2. **A multi-hop curriculum.** Composition emerges suddenly and late (Status 2). Would training on one-step chains
   first, then deeper ones, bring it forward?
3. **Sleep and dream on task v4.** The switches are implemented and tested but have not been measured.
4. **Throughput.** Training runs at about 12k bytes/s with the GPU about 15% busy. bf16 parameters and larger batches
   would raise how much every experiment can learn.
5. **Why 1 loop fails lookup in the synthetic chain task but learns it on real data** (`control-4L` reaches
   .96/.77/.51 pair accuracy at 2/4/6 hops).
6. **Real token skipping for halting**, so halting saves wall-clock time and not just analytical FLOPs.
7. **Energy measurement** on Apple Silicon (for example by sampling `powermetrics`).

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
