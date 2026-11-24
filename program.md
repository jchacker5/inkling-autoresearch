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
  is not damaged on step 1. `DEPTH_ATTN`, `HISTORY` and `LOOP_EMBED` do this (checked by
  `tests/test_model.py`). `HALTING` and `KV_SHARE` change the computation itself, so they start
  slightly off the base and rely on the 5 minutes to adapt.
- Run `uv run pytest -q` after editing `train.py`. Add any new switch to `CASES` in `tests/test_model.py`.
- The backbone shape (`D_MODEL`, `N_HEADS`, `PRELUDE`/`CORE`/`CODA`, MLP size) is fixed by the base.
  `train.py` refuses a mismatched load. Shape changes need `INIT_FROM_BASE = False`, which is only fair
  against a from-scratch control (recipe lane).
- The baseline is "base + 5 minutes of unchanged training", not the base itself.
- Warm starts train with **recipe R** (`RECIPE = "R"` in `train.py`): AdamW with three parameter groups.
  Tensors loaded from the base get LR 1e-4 (with weight decay). New matrices get 1e-3 and new 1-D tensors
  (gates, scales, biases) get 1e-2, with no weight decay on new tensors. The schedule is 10 warm-up steps,
  then constant, then linear decay to 0 over the last 25% of the budget. `load_base` returns the new
  tensors and `train(recipe=..., new_keys=...)` builds the groups. `RECIPE = "legacy"` restores the old
  re-warm (one LR of 1e-3, cosine), which is a recipe experiment, not the baseline. `pretrain_base.py`
  always uses legacy.

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
| `reason_id` | pair accuracy on twin multi-hop prompts, 2/4/6 hops (seen depths); 400 pairs per hop | higher |
| `reason_ood` | same task at 8/10 hops (deeper than anything in training) | higher |
| `kappa_id`, `kappa_ood` | `(pair - 0.25) / 0.75` averaged over the same hops (`kappa_by_hop` per hop). 0 = what structure plus a coin flip gets without reading the query; 1 = perfect. **`kappa_id` is the default primary metric** | higher |
| `scaling` | the same weights at loops/2 and 2x loops, each with its own `kappa_*`, `mean_loops_per_token` and `kv_kb_per_token` | look for OOD gains at 2x |
| `mean_loops_per_token` | realised compute, token-weighted over the whole eval at the default loop count (`mean_loops_text`, `mean_loops_reason` split it) | lower at equal quality |
| `kv_kb_per_token` | KV cache per byte at the loop count evaluated | lower |
| `bytes_f`, `f_mib` | 2 x params (bf16 weights) + KV cache at a 512-byte context at the evaluated loop count | lower |
| `latency_ms_b1` | batch-1 prefill of 512 bytes with a compiled forward (`model.logits`); `latency_ms_b1_t100` for 100 bytes | lower |
| `train_peak_mem_gb`, `eval_peak_mem_gb`, `peak_mem_gb` | MLX peak in training, in evaluation (reset in between), and their max (the 20 GB check) | lower |
| `harness`, `base`, `recipe`, `eval_seed` | provenance: prepare.py sha1 (10 hex), base checkpoint name, optimizer recipe, eval set | must match to compare |

The research question lives in the tension between `val_bpb` and `reason_*`. Recurrence may help reasoning
while hurting factual knowledge. Report both every time.

## Keep / discard rule (screening) and promotion

Every lane declares **one primary metric**: `kappa_id` (the default) or `val_bpb` (lanes that target
fluency or cost). Pass it as `swarm/run.py --primary val_bpb` or set `INKLING_PRIMARY`. `swarm/run.py`
screens each run against the **pooled baseline lane**, never against your lane's last kept run. The pool is
all valid `baseline` rows with the same `harness`, `base` and eval set. The campaign runs 6 seeds (0..5).

- **gate**: `val_bpb` <= baseline mean + 0.02. For `val_bpb`-primary lanes, `kappa_id` must also be at most
  max(0.02, baseline sd) below the baseline mean. A run that fails a gate is **discard**.
- **candidate** if t = (x - baseline mean) / baseline sd >= **0.92** on the primary metric (sign flipped for
  `val_bpb`). 0.92 is the one-sided t(0.80, 5 df), which is about one baseline SD. Otherwise **discard**.
- **pending-noise** while fewer than 3 baseline seeds exist: the noise is unknown, so no verdict yet.
  Run the baseline seeds (setup step 4) before going on.
- An invalid run (violations, crash, smoke) is always discarded.
- "Cheaper" (lower `mean_loops_per_token`, `kv_kb_per_token`, `bytes_f`) is not a verdict of its own. Run it
  as a `val_bpb`-primary lane, or report it in your notes.
- Simplicity criterion (Karpathy): a tiny gain bought with lots of code is not worth it. Deleting code at
  equal quality is a win.

A candidate is only a nomination; screening at t >= 0.92 lets about 20% of null runs through by design.
**Promotion** happens in `swarm/campaign.py` after the swarm:

1. Each lane's best candidate (largest t) is re-trained with 6 fresh seeds (100..105). 6 fresh baseline
   seeds are interleaved with them.
2. All of these runs are scored on the **holdout** eval set (`INKLING_EVAL_SEED=99991`), which screening
   never sees. Their ledger verdict is `holdout`.
3. Each lane gets a one-sided Welch t-test on its primary metric against the holdout baseline seeds, with
   Holm correction across lanes at family-wise alpha 0.05. A promoted lane must also pass the `val_bpb`
   gate. The results go to `results/confirm.tsv` (`uv run swarm/confirm.py` recomputes it).

Never confirm by editing `SEED`, and never cherry-pick seeds. Every ledger row records `seed`, `power`,
`harness`, `base` and `eval_seed`. Runs on battery or Low Power Mode get fewer steps in the same wall clock
and are not comparable; the campaign waits for AC power, and so should you.

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
6. **overthinking** (utility-based exit; TRACE, Zhang et al.). Most tokens do not need every loop: run
   `uv run swarm/landing.py --name base` first. It reports, per hop and for text bytes, the loop after which
   the prediction stops changing ("landing"), the share of loops spent after landing, and "flips" (right at
   an earlier loop, wrong at the end). Then: `EXIT_TAU` (training-free; a token stops once one more loop
   changes its readout by less than tau, `EXIT_METRIC` kl or state) and `LOOP_CONSIST` (self-distil the
   final loop into a random earlier loop so early exits lose less). Report mean loops per token next to
   accuracy; a win is equal `reason_id`/`reason_ood` and `val_bpb` at fewer mean loops, or better accuracy at
   2x loops (no destructive overthinking). Frozen tokens are still computed in this implementation, so
   claim loop savings, not wall-clock savings, until a sparse kernel exists.
7. **chunked-core** (after CALM, arXiv 2510.27688). `CHUNK=K` runs the core on K-byte chunks: the byte
   prelude is pooled K-to-1, the core loops over T/K positions, and its output is shifted one chunk and
   unpooled into a byte-level coda, so bits per byte stays exact and the model stays causal. Core FLOPs and KV
   cache per byte fall by K, so the same budget buys more loops or width. It is a new architecture: the campaign
   trains `chunk2`/`chunk4` bases on the 6-hour schedule stopped at 2 h and compares them with `base-2h`.
   Lane work, once a chunk base exists: point `BASE_CKPT` at it, then test more loops, a wider core at equal
   FLOPs, and a 2-block coda.
8. **parallel-recurrence** (LoopFold-style): phase 2, inference only. Predict later loop states, run them
   in parallel and correct them. Only start once lanes 1-4 have a confirmed model. Measure real wall clock
   with MLX, not FLOPs.

## The loop

Setup, once per session:
1. `git status` should be clean. You are on your lane branch (swarm) or `ar/<tag>` (single agent).
2. `uv run swarm/status.py` to read the ledger: baseline, noise, other lanes' results. Do not repeat
   an experiment that is already in the ledger.
3. If the base checkpoint is missing, stop and say so. Do not build it yourself.
4. If the ledger has no `baseline` rows (the campaign normally runs them before any agent starts), run
   `uv run swarm/run.py --lane baseline --seed S --desc "baseline seed S"` for S = 0..5, then
   `uv run swarm/status.py --noise`.

Then LOOP (until your session's experiment budget is used up, or forever if none was given):
1. Pick one idea. Write the hypothesis in one line before you edit anything.
2. Edit `train.py` (one idea per experiment).
3. `uv run swarm/run.py --lane <lane> [--primary val_bpb] --desc "<short description>" --hypothesis "<one line>"`.
   This waits for the GPU lock, trains, evaluates, appends to the ledger, saves the diff and log under
   `runs/`, and prints `VERDICT: candidate|discard|pending-noise (reason)`.
4. candidate: `git commit -am "<lane>: <desc> (kappa_id X, val_bpb Y, t Z)"`. The campaign confirms the
   commit later.
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
