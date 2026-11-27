# Decisions taken after round 1 (2026-11-24, 11:50)

Round 1 converged on one fact from three directions. Task v1 had a position shortcut: copying the first
statement's digit scored .48-.57 at every hop (debater 4). The base does not read the question: its answer is
unchanged 93-99% of the time when only the query changes (debater 1). Answer digits are only 3.1% of training
bytes (debater 3). Every v1 reasoning number measures the shortcut.

## Done (all on branch task-v2, PR #3; merged into the local main the campaign runs)
1. **Task v2** (`prepare.py`):
   - 20 variables at every hop.
   - Distinct root digits.
   - Uniformly random topological order.
   - Distractors never attach to the chain.
   - The position floor (first or last digit) is .13-.19 and is reported as `reason_floor_id` / `reason_floor_ood`.
   - Every prompt is 83 bytes.
2. **Eval prompt format** (`encode_prompt`):
   - Every prompt follows a `\n`, as training lines do.
   - `CHUNK=K` models get extra left context (the tail of a real training line) so the answer position starts a chunk. The core then sees `?v`. This fixes debater 1's finding 5.
3. **`base-v2`**:
   - The 6-hour v1 base continued for 2 h on v2 data (`--init base`, lr 5e-4, plain loss, normal mix of 8/32 reasoning rows).
   - It started at 11:42. Progress is in `~/.cache/inkling-autoresearch/base-v2_progress.tsv` (columns: hops 0,1,2,4,6,8; v2 floor about .10-.17).
   - All warm starts now use it (`INKLING_BASE`, default `base-v2`).
4. **Campaign (tag nov24)**:
   - base-v2 → 3 baseline seeds (current recipe) → landing on base-v2.
   - Then from-scratch 2-hour models on v2 data, all on the 6-hour schedule stopped at 2 h: `looped` (reference), `control-4L`, `chunk4`, `chunk2`, `control-10L`.
   - Then evals → swarm, with lanes from `results/next_lanes.txt`.
5. **The Mac watchdog-panicked at 01:58.** Swap was full while debaters ran CPU model evals beside GPU training.
   **Rule from now on: no model forward passes on CPU or GPU by agents while the campaign trains.** Pure-Python
   analysis of the task generator, code reading and arithmetic are fine.

## Open decisions for round 2 (each debater must vote with reasons)
- **D1. Answer-weighted loss** (debater 1 F2, debater 3 P): what weight? Add it to `base-v2` (restart that run now), or test it as a 5-minute warm start first? What should happen if `base-v2` shows no lookup above the floor after 1 h?
- **D2. Warm-start recipe R** (debater 3): adopt it for the swarm before the baseline seeds run, or keep the current recipe as the control?
- **D3. Score**: one formula for "intelligence per GB" (debater 2) that respects debater 4's constraints (no ratio form? the floor?). Write it exactly.
- **D4. Keep / promotion rule** for the swarm (debater 4's simulation: 23-28% false confirmations). Give exact thresholds that are detectable with 400 problems per hop, and the number of seeds.
- **D5. Controls**: keep `control-10L` (17.9M params, over the 12M cap) or drop it? Is a matched "looped" from-scratch 2-hour model the right reference for the chunk and 4L models?
- **D6. Inkling 2 Mini gate**: a zero-init warm start of `control-4L` (debater 3), its own base, or drop it?
- **D7. Swarm lanes**: exactly 3-4 lane names with a one-paragraph brief each, for `results/next_lanes.txt` and program.md. Which existing switches (`EXIT_TAU`, `LOOP_CONSIST`, `CHUNK`, `HALTING`, `DEPTH_ATTN`, `KV_SHARE`, ...) are worth any swarm time once reasoning is real?
- **D8. Harness metric bugs** (debater 2): `peak_mem` never reset, `mean_loops` from the last chunk only, latency measured uncompiled in fp32, KV counted at the trained loop count. Fix all now, or in which order?
