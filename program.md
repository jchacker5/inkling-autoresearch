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
   never sees for the synthetic reasoning examples. The fixed text windows do not change with this seed.
   Their ledger verdict is `holdout`; passing the text gate is not independent-corpus confirmation.
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

Each lane is one hypothesis family; in a swarm each agent owns one lane. These four came out of the round-2
research debate (research/04-debate-r2-*.md). The primary metric is `kappa_id` (pair accuracy above the .25
structure-plus-coin-flip bound). Every lane reports `kappa_ood`, `val_bpb` and `bytes_f` too.

1. **binding** (does the model read the question?). Levers: `ANSWER_W` and `FIRST_ANSWER_W`, the reasoning share
   of the batch, and the data order within a line (`prepare.py` is fixed, so propose data changes in your notes
   rather than editing it). Target: hop-0/1/2 pair accuracy well above .25. Until this lane works, no other
   lane may claim a reasoning gain.
2. **recipe** (fair warm-start baseline). Recipe R's LR grid (`R_LR`, `R_LR_NEW`, `R_LR_NEW_VEC`), warm-up and
   decay fraction, bf16 parameters (fp32 master weights only), Muon. Every other lane's win is measured against
   this lane's best, so it must not be weak.
3. **depth** (do more loops buy deeper hops?). `RANDOM_LOOPS`, `BPTT_LOOPS`, `DEPTH_ATTN`, `LOOP_EMBED`,
   `HISTORY`, and a zero-init carry gate. The claim to test: pair accuracy at hops 8-10 rises with test-time
   loops (`scaling`), and `val_bpb` does not get worse at 2x loops.
4. **exit** (overthinking; TRACE, CALM). `EXIT_TAU`/`EXIT_METRIC` (training-free) and `LOOP_CONSIST`. Run
   `swarm/landing.py` first. A win is equal `kappa` and `val_bpb` at fewer mean loops. Exited tokens are still
   computed (dense masking), so claim loops saved, not time saved.

Older switches (`HALTING`, `KV_SHARE`, `CHUNK`) remain available; `CHUNK=4` is judged by the campaign's 2-hour
from-scratch comparison against `control-4L` and `looped`, not by warm starts.

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
