# 02 · Debate round 1: optimization and learning dynamics (Debater 3)

*2026-11-24. CPU-only checks on `base` (360 min) and `base-5h`. No GPU, no tracked edits. Tags as in 01.*

## 0. New evidence [local]

- **E1. Re-warm residual.** Final base 1.5113, against 1.5037 at 315 min and 1.5115 at 300 min. The two re-warm cycles (+0.081 spike) bought nothing over `base-5h`, and the 315-min weights are gone.
- **E2. 5-min schedule.** Peak LR is 6.9e-4 at step 99, and 138 of 255 steps are above 3e-4. That LR area equals about 1,750 steps at the base's final 5e-5. The interrupted baseline ran 1.5 s/step (≈200 steps/run).
- **E3. No fixed point.** ‖h_r‖ for r = 0..8 is 7, 18, 29, 42, 57, 71, 85, 99, 113: each loop adds about 15. From loop 4 on, the steps are parallel (cos ≥ .98) and the direction is frozen (cos ≥ .995). The injection e shrinks from 39% to 6% of ‖h‖. Text bpb by loop: 2.86, 1.76, 1.61, **1.58**, 1.59, 1.62, 1.65, 1.69.
- **E4. Depth binds below 4 loops; loops beyond 4 are dead.** Official sets, 128 problems per hop, exact per-loop readout:

  | hop | loop 1 | loop 4 | loop 8 | loop 10 |
  |---|---|---|---|---|
  | 2 | .45 | .45 | .43 | .48 |
  | 6 | .24 | .50 | .47 | .41 |
  | 8 | .13 | .41 | .34 | .37 |
  | 10 | .12 | .27 | .24 | .23 |

- **E5. Norm anchoring (H11)** recovers only 42% of the bpb lost at 8 loops, and gives no reasoning gain.
- **E6. Chance is a copy heuristic.** A random context digit scores .40 / .37 / .28 / .24 / .23 at hops 0 / 2 / 6 / 8 / 10.
- **E7. Answer digits are 3.1% of training bytes.** Answers in the eval's first-query format are 0.34%.
- **E8. bf16 parameters.** One AdamW step changes 14% / 27% / 75% of base weights at LR 5e-5 / 1e-4 / 3e-4. The rest round away.
- **E9. The Inkling wiring is an exact no-op from LOOPS=1.** A gated restart with gate 0 and LOOPS=4 is bitwise equal to the LOOPS=1 control. dL/dgate ≠ 0, and gate 1 reproduces our wiring.
- **Ops.** The 01:59 reboot killed the campaign at baseline seed 0, step 150: no ledger, GPU idle.

## 1. Five hypotheses

### R: the warm-start recipe every lane uses

**Mechanism.** Warm starts see the base's own data, so nothing needs re-warming. A lower peak LR preserves upstream loss ([2308.04014](https://arxiv.org/abs/2308.04014)), and re-raising the LR from its minimum raises loss ([2403.08763](https://arxiv.org/abs/2403.08763)) [snip]. The shock adds seed variance and a module × recovery interaction, in which a module that only speeds recovery "wins". It also hurts retrieval: hop 0 fell from .49 to .35 mid-cycle.

**Recipe.**
- **Optimizer.** AdamW with β (0.9, 0.95), gradient clip 1, fp32 parameters, no optimizer switch.
- **Learning rates** (via `MultiOptimizer`):
  - base tensors: **1e-4**, which is 2x the base's final LR;
  - new matrices: 1e-3;
  - new gates and scalars: 1e-2, because Adam moves at most about LR per step.
- **Weight decay.** 0.1 on base matrices, 0 on new tensors. Decay would close a gate initialized at 1.
- **Schedule.** 10-step warmup, then constant, then linear to 0 over the last 25% of wall clock.
- **Freezing.** None; a module claiming a win must also pass a frozen-backbone ablation.
- **Timing.** Exclude steps that compile a new graph from `t_train`. train.py:356 counts them after step 2, and RANDOM_LOOPS plus LOOP_CONSIST compile 22 graphs.
- **Bookkeeping.** Log step-0 metrics (they must equal the base's for no-op modules) and use a new lane, `baseline-R`. run.py:42 pools every "baseline" row.

**Test.**
- Arms:
  - the current schedule, 3 seeds (the campaign's baseline seeds);
  - R at 1e-4, 3 seeds;
  - R at 3e-4, 2 seeds.
- Positive control: an H4 readout refit under R must cut bpb at 8 loops by at least 0.02.

**Pass.** Mean `val_bpb` ≤ base + 0.002, every hop within 5 points of the base, seed SD ≤ 0.0015. Adopt the largest LR that passes. My prediction for the current schedule [est]: base + 0.005 ± 0.005, with hops swinging ≥3 points. **Cost:** about 60 GPU-min, 20 of them already scheduled.

### V: contraction through variable loops

**Mechanism.** Fixed-4 training learned a constant radial push, and eval-only tricks cannot create test-time scaling (E3–E5). Training on variable loop counts gives the model a reason to contract (TML varloop; [2610.19768](https://arxiv.org/abs/2610.19768)).

**Experiment.** A zero-init carry gate, `x = e + (1+w)⊙h` with w = 0 and LR 1e-2. Loop counts k: P(4) = .5, plus .1 each for 2, 3, 5, 6 and 8, with graphs precompiled and full BPTT. Arms: variable loops alone, and variable loops plus the gate.

**Pass.**
- ‖h₈‖/‖h₄‖ ≤ 1.3 (it is 2.0 now).
- The gap reason_ood(8) − reason_ood(4), about −5 points now, improves by ≥4 points over `baseline-R`.
- `val_bpb` at 4 loops is within 0.004 of `baseline-R`.

If it fails, fund a variable-loop base instead. **Cost:** 2 seeds, about 28 min.

### P: retrieval precision is starved of optimization

**Mechanism.** Hop 2 reaches its .45 ceiling at loop 1 (E4), so the limit is a single lookup, not depth. Two optimization causes are possible: answers carry only 3% of the gradient (E7), and weight decay on every tensor, RMSNorm gains included (train.py:327), caps the query-key logit scale [est].

**Tests.**
- (a) Eval-only: scale q in the core by β ∈ {1.25, 1.5, 2}.
- (b) A 5-minute run that weights answer bytes 4x: the byte after `?v=`, in reasoning rows only, because URLs contain `?id=`.

**Pass**, against `baseline-R`: (a) `reason_id` +3 points; (b) `reason_id` +4 points with Δbpb ≤ +0.004. If (b) passes, either every arm uses the weighting or none does. **Cost:** about 16 min.

### C: LOOP_CONSIST with a clean gradient path

**Mechanism.** train.py:293–298 distils through the *shared* coda. That rewrites the final readout (multi-exit conflict, [2407.13986](https://arxiv.org/abs/2407.13986)) and pushes the core to land early.

**Arms** (weight 0.3):
- (i) the current implementation;
- (ii) zero-init per-loop exit adapters, leaving the final path untouched;
- (iii) as (ii), plus stop-gradient into the core.

**Pass.** Some EXIT_TAU reaches ≤3 mean loops at Δbpb ≤ 0.004 and Δreason_id ≥ −1, and arms (ii) and (iii) keep the final-loop `val_bpb` within 0.002. **Cost:** 2 seeds, about 48 min.

### I: retrofit Inkling 2 Mini (see §2)

**Arms,** all from `control-4L-2h`:
- (a) LOOPS=1 continued;
- (b) gated restart, LOOPS=4 (10 layer applications, as in the base), gate LR 1e-2;
- (c) gate fixed at 1, which is not a no-op.

**Pass.** At step 0, (b) equals (a) exactly. After training, (b) beats (a) by ≥3 points on `reason_id` and ≥5 at hop 8, with Δbpb ≤ +0.004 and a mean gate ≥ 0.1. **Cost:** 2 seeds, about 40 min.

**Total for all five: about 190 GPU-min.**

## 2. Inkling 2 Mini as a warm start

**Not from our base.** Closing the gate collapses the looped base to a single-pass function its coda never saw (tiny config: max |Δlogit| = 1.83). Our base is the open-gate endpoint.

**But it is an exact zero-init warm start of `control-4L`,** with one new 384-vector (E9):

```python
CARRY, GATE_INIT = "add", 0.0    # "gated": x = e + g*h; GATE_INIT 0 = Inkling (closed), 1 = this base (H16)
# __init__: if CARRY == "gated": self.msg_gate = mx.full((D_MODEL,), GATE_INIT)
x = h if r == 0 else (h + e if CARRY == "add" else e + self.msg_gate.astype(e.dtype) * h)   # train.py:221
```

**Run it with** `BASE_CKPT=control-4L-2h`, `LOOPS=4` and `CARRY="gated"`. Zero-init `DEPTH_ATTN` can stand in for attention residuals.

**What still needs its own base** (2 h, funded only if I passes): the claim that *pretraining* with a closed gate beats our open wiring. TML's 6:12:6 split matches our 1:2:1. Restarting from e also cures E3, because the injection share stays constant and ‖h‖ stays bounded while g < 1.

## 3. Objections

**Debater 1 (mechanism)**
- Landing and truncation sweeps on a fixed-4 base mix the algorithm with radial drift. The declines after loop 4 are mostly drift, not overthinking (E3, E5).
- `EXIT_METRIC="state"` divides by ‖h‖ (train.py:197–200). The relative change falls like 1/r while the absolute step stays near 15, so it "converges" on a schedule. Use the direction's cosine instead.

**Debater 2 (systems)**
- **bf16 parameters** (program.md:101) freeze 73% of weights at a low LR (E8), and their 1.4x extra steps confound comparisons.
- **2-hour checkpoints are compared unannealed.** The controls stop at LR 7.6e-4. `base-2h` was at 9.4e-4, because its cosine restarted at 75 min (pretrain_base.py:137). Cool each one down identically before comparing or warm-starting from it.
  - run.py:92 judges chunk lanes against the 6-hour base.
  - Every reboot re-warms the controls again unless `--resume` keeps the absolute schedule (a human decision).
- **KV_SHARE** assumes a fixed point, and there is none (E3). Lanes that are not no-ops need a two-point LR grid.
- **BPTT_LOOPS** gives about 1.36x more steps (a training-amount confound), and truncation does not help retention.

**Debater 4 (statistics)**
- Seeds shrink variance, not bias. The shock and its module interactions are systematic, so 3 seeds give a tight interval around the wrong number. Fix R first, then take thresholds from `baseline-R` only.
- Steps vary from about 200 to 255 per run. Flag arms more than 5% off their baseline's step count.
- Use E6 as the null for "above chance" and for flips.

## 4. What the report gets wrong

1. **§0.2 / O1, "depth does not limit the base".** Depth binds: hop 8 rises from .13 to .41 over loops 1–4 (E4). The flat quick-eval profile is a ceiling at 4 loops, and the quick eval never ran hop 10 (.27).
2. **O2, "1/3 by guessing".** The copy baseline is .40, because digits collide. The margin over copying grows with depth (+7 points at hop 0, +17 at hop 8), so "one lookup caps every hop equally" does not fit.
3. **O3 / H3, "re-warm to 1e-3" and "0.03–0.08 above the base".** The peak is 6.9e-4. The 0.081 was a mid-cycle transient, and the residual after re-annealing was about 0.008 (E1). What matters is variance, interaction and reasoning damage; §3.6's core-frozen runs are not immune if the coda trains at that peak.
4. **H18, "q₂ and k₂ wake slowly".** Under Adam they move at full LR from step 2. The same holds for `DEPTH_ATTN`.
5. **H11 and B6.** Pre-norm does not make the blocks norm-robust, and anchoring fixes only 42% (E5). B6 extends a result about a linear input map ([2609.39892](https://arxiv.org/abs/2609.39892)) to all fine-tunes, when what binds is signal × LR × steps (test P). H5's compiles also count against the 300 s.
