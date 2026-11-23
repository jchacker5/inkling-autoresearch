# inkling-autoresearch

Autonomous research on **intelligence per gigabyte**: can a compact recurrent-depth (looped) language model
learn to allocate its own computation and get better reasoning per byte of memory than a plain transformer?

This is Karpathy's autoresearch loop (fixed harness, one editable file, fixed wall-clock budget, keep/discard
by metric) adapted to MLX on a 32 GB M4. It can run as a single agent or as a swarm of lane agents.

## Files

- `prepare.py`: FIXED. Data, the training mixture, budgets, and `evaluate()`. Never edit it.
- `train.py`: the ONLY file you edit. Model, switches, optimizer, training loop.
- `pretrain_base.py`: FIXED. Builds the shared base checkpoint once. Never re-run it during a session.
- `swarm/run.py`: run one experiment through the shared GPU lock and append it to the ledger. Always use it.
- `swarm/status.py`: leaderboard and per-lane bests.

## Two-stage design (read this first)

In-context retrieval, the circuit the reasoning task needs, only emerges after thousands of steps. A
5-minute from-scratch run on this M4 reaches about 250 steps and scores at chance (see
`results/calibration.md`). So every experiment **warm-starts from the shared base checkpoint**
(`~/.cache/inkling-autoresearch/base.safetensors`, a looped model trained once for hours) and then trains
for the fixed 5 minutes with your modification. Consequences:

- New modules must start as a no-op (zero-init output projections, gates that start closed) so the base
  is not damaged on step 1. Every switch in `train.py` already does this.
- The backbone shape (`D_MODEL`, `N_HEADS`, `PRELUDE`/`CORE`/`CODA`, MLP size) is fixed by the base.
  `train.py` refuses a mismatched load. Shape changes need `INIT_FROM_BASE = False`, which is only fair
  against a from-scratch control (recipe lane).
- The baseline is "base + 5 minutes of unchanged training", not the base itself.

## Fixed rules (from prepare.py)

- 300 s training wall clock (compilation warm-up excluded). Byte-level vocab, 512-byte context.
- Unique parameters must stay under 12M (about 24 MB of weights in bf16). That is the "per gigabyte" constraint:
  you may buy more compute with more loops, never with more weights.
- Peak MLX memory under 20 GB.
- No new packages. Do not touch `evaluate()` or the data.

## Metrics (printed after `---`, also in `runs/last.json`)

| metric | meaning | direction |
|---|---|---|
| `val_bpb` | bits per byte on held-out web text. The knowledge/fluency proxy | lower |
| `reason_id` | exact-match accuracy, multi-hop variable tracking, 2/4/6 hops (seen depths) | higher |
| `reason_ood` | same task at 8/10 hops (deeper than anything in training) | higher |
| `scaling` | the same weights evaluated at loops/2 and 2x loops | look for OOD gains at 2x |
| `mean_loops_per_token` | realised compute with halting | lower at equal quality |
| `kv_kb_per_token`, `latency_ms_b1`, `peak_mem_gb`, `weights_mb_bf16` | systems costs | lower |

The research question lives in the tension between `val_bpb` and `reason_*`. Recurrence may help reasoning
while hurting factual knowledge. Report both every time.

## Keep / discard rule

`swarm/run.py` prints a verdict against your lane's current best. Defaults, which must be re-calibrated
to the measured seed noise in `results/noise.md` once it exists:

- **keep** if `val_bpb` improves by >= 0.004 and `reason_ood` drops by no more than 0.02, OR
  `reason_ood` (or `reason_id`) improves by >= 0.03 and `val_bpb` worsens by no more than 0.004, OR
  equal quality (within noise) at lower `mean_loops_per_token`, `kv_kb_per_token` or params.
- otherwise **discard**. An invalid run (violations, crash, smoke) is always discarded.
- Simplicity criterion (Karpathy): a tiny gain bought with lots of code is not worth it. Deleting code at
  equal quality is a win.
- Claims of improvement need confirmation: before a lane's best is promoted to `main`, re-run it with
  `SEED = 1` and `SEED = 2`; promote only if the mean beats baseline by more than 2x the seed std.

## Compute control

A looped model can look smarter simply because it spent more FLOPs. The fixed wall-clock budget charges
loops honestly at training time (more loops means fewer steps). At inference, always compare at matched
`mean_loops_per_token`, and say so when you do not. Halting at eval is simulated: frozen tokens still pass
through the dense kernels, so `latency_ms_b1` does not shrink with halting. Treat mean loops as an
analytical compute measure until someone implements real token skipping.

## Research lanes

Each lane is one hypothesis family. In a swarm each agent owns one lane. Alone, work them top to bottom.

1. **depth-attn** (ARLoop-style). `DEPTH_ATTN=K`: each loop attends per-token over the last K loop states.
   Try K in {2,3,4}, `DEPTH_ATTN_DIM`, attention before vs after the core blocks, and replacing input
   injection with depth attention. Hypothesis: better `reason_ood` at equal params, small latency cost.
2. **halting** (Mixture-of-Recursions / ACT). `HALTING=True`. Sweep `PONDER_COST`, the router bias init,
   `HALT_THRESHOLD`, `LOOPS` up to 8. The matched-compute comparison is the point: adaptive at mean loops m
   vs fixed loops = round(m). Hypothesis: same `val_bpb` at fewer mean loops, and more loops on deep hops.
   Check `reason_by_hop` vs loops used.
3. **history** (history-aware recurrence). `HISTORY` in {mean, gated}, `LOOP_EMBED`. The target failure is
   overthinking: `val_bpb` getting worse at 2x loops. Hypothesis: flatter `scaling` curves with no
   knowledge loss.
4. **kv-reuse** (fixed-point / shared KV). `KV_SHARE=True` cuts KV by about 2.5x. Find the variant that keeps
   quality: share from the last loop instead of the first, share only K, or measure state convergence
   (||h_r - h_{r-1}|| / ||h_r||) and reuse once converged. Hypothesis: much smaller cache, negligible quality loss.
5. **recipe** (fair baseline). Before crediting any architecture, tune the continued-training recipe:
   LR, warmup, batch, `RANDOM_LOOPS` (Huginn-style, needed for test-time scaling), `BPTT_LOOPS`, bf16
   params via `model.set_dtype` (1.4x throughput, measured), and Muon via `mlx.optimizers.Muon`. Every
   other lane's win is measured against this lane's best, so it must not be weak. A NON-looped control
   at the same unique-param count belongs in the paper, and needs its own base (a human decision).
6. **parallel-recurrence** (LoopFold-style): phase 2, inference only. Predict later loop states, run them
   in parallel and correct them. Only start once lanes 1-4 have a confirmed model. Measure real wall clock
   with MLX, not FLOPs.

## The loop

Setup, once per session:
1. `git status` should be clean. You are on your lane branch (swarm) or `ar/<tag>` (single agent).
2. `uv run swarm/status.py` to read the ledger: baseline, noise, other lanes' results. Do not repeat
   an experiment that is already in the ledger.
3. If the base checkpoint is missing, stop and say so. Do not build it yourself.
4. If the ledger has no `baseline` lane row for this commit of `main`, run it first:
   `uv run swarm/run.py --lane baseline --desc baseline`, then again with `SEED = 1` for a noise estimate.

Then LOOP (until your session's experiment budget is used up, or forever if none was given):
1. Pick one idea. Write the hypothesis in one line before you edit anything.
2. Edit `train.py` (one idea per experiment).
3. `uv run swarm/run.py --lane <lane> --desc "<short description>" --hypothesis "<one line>"`.
   This waits for the GPU lock, trains, evaluates, appends to the ledger, saves the diff and log under
   `runs/`, and prints `VERDICT: keep|discard (reason)`.
4. keep: `git commit -am "<lane>: <desc> (val_bpb X, reason_ood Y)"`.
   discard: `git checkout -- train.py`. Never use `git reset --hard` or `git stash`.
5. Crash: read the tail of the log. Fix an obvious bug and retry once. Otherwise discard and move on.
6. Every 5 experiments, append 3 to 5 lines to `results/notes-<lane>.md`: what you learned and what is next.

Never stop to ask whether to continue. The human may be asleep. If you run out of ideas, re-read the lane
description, combine near-misses, or try the opposite of what failed.

## Science hygiene

- Negative results are results. Log them with the same care.
- Never edit the ledger by hand. Never cherry-pick seeds.
- One change per experiment, unless you are explicitly combining two confirmed wins.
- If a result looks too good, check `runs/last.json` for `valid`, check loops at eval, and re-run with another seed.
