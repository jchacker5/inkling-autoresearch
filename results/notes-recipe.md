# recipe lane — nov24b (session 1, 2026-11-26)

Primary: val_bpb (kappa_id is pinned at -0.0667 with sd 0 in every baseline row, so a kappa-primary t-test is meaningless: any kappa > floor is t=+inf, a tie is t=0).
Pooled baseline (base-v4, n=6): val_bpb 1.4708 +- 0.0013. Frozen base-v4 itself (results/eval-base-v4.json): 1.4675.

| run | val_bpb | t | verdict |
|---|---|---|---|
| R_LR 1e-4 (baseline) | 1.4708 | 0 | - |
| R_LR 3e-5 | 1.4689 | +1.37 | candidate, kept |
| R_LR 1e-5 | 1.4683 | +1.78 | candidate, kept |
| R_LR 1e-5 + R_WARMUP 40 | 1.4684 | +1.73 | tie with 1e-5 -> reverted (simplicity) |
| R_LR 3e-6 | 1.4679 | +2.09 | candidate, kept (branch head; best t) |

Learned:
- 5 min of recipe R *hurts* the base on val_bpb at every LR tried; bpb falls monotonically toward the frozen base as R_LR shrinks. The val_bpb-optimal "recipe" is degenerate: don't move the base. All single seeds; gaps between 3e-5/1e-5/3e-6 are under ~0.5 sd.
- Warm-up length doesn't matter at R_LR 1e-5. Fresh Adam moments aren't the source of damage at low LR. The likely cause is drift from the training mixture (16/32 reasoning rows, 10x answer weight) away from web text, not the optimizer transient.
- Caution for promotion: a lower R_LR makes the baseline "better" on bpb by training less. Base weights then barely adapt to any lane's change (binding/depth need them to move). The fair baseline for kappa-seeking lanes should keep R_LR >= 1e-5. Don't adopt 3e-6 swarm-wide without checking binding's kappa.
- tests/test_harness.py::test_lr_schedules hard-coded warm-up 10 / decay .25. It now follows R_WARMUP/R_DECAY_FRAC (commit 01c0ad1).

Next 3 ideas:
1. Split the cause: at R_LR 1e-4, train on text-only rows (reasoning share 0). If bpb returns to ~1.4675, the damage is the mixture, not the LR (coordinate with binding; it owns the reasoning share).
2. Re-run R_LR 1e-5 with 2 more seeds before the campaign. Its t (+1.78) is a near-tie with 3e-6 and it keeps the base trainable, so it's the more defensible "fair baseline".
3. Per-group LR: base attention/MLP at 1e-5 but embeddings, norms and coda at 1e-4 (or the reverse). This finds which tensors carry the damage. Muon on the core matrices once kappa leaves the floor.

## session 2 (2026-11-26), primary val_bpb, pooled baseline 1.4708 +- 0.0013 (n=6), ~200 steps/run (shared GPU)

| run | commit | val_bpb | t | verdict |
|---|---|---|---|---|
| R_LR 1e-5 + R_DECAY_FRAC 1.0 | 6d0fead | 1.46801 | +2.01 | candidate (ties 3e-6 at 3x the LR) |
| R_LR_CORE 1e-4 (core only, rest 1e-5) | 8c3d1fa | 1.46923 | +1.13 | candidate, knob kept but off |
| R_LR 1e-4 + R_DECAY_FRAC 1.0 | - | 1.47022 | +0.42 | discard |
| WiSE-FT R_INTERP 0.5 (R_LR 1e-4, full decay) | 76df3de | 1.46755 | +2.35 | candidate, **branch head, lane best** |
| WiSE-FT R_INTERP 0.75 | 4820f5d | 1.46855 | +1.62 | candidate, head restored to 0.5 |

Learned:
- Schedule shape barely matters. Full-run linear decay gains ~0.0003 at 1e-5 and ~0.0006 at 1e-4 (not significant). Damage tracks peak LR.
- The core (2 shared blocks, run 4x) carries ~0.0012 of the ~0.0028 bpb damage at 1e-4, roughly its share of the weights. It isn't unusually fragile, so lanes that need the core to adapt can give it 1e-4 for ~0.001 bpb (`R_LR_CORE`).
- WiSE-FT (train at the baseline LR 1e-4, then base tensors <- base + a*(trained - base); new tensors untouched) gives the best bpb (1.46755 at a=0.5), essentially the frozen base (1.4675). Training keeps full plasticity. The bpb ladder over a is 0 -> 1.4675, .5 -> 1.4676, .75 -> 1.4686, 1 -> 1.4702, roughly quadratic (single seeds). Caveat: this is still partly "don't move the base". Whether interpolation also shrinks a lane's kappa gain is untested, since kappa is pinned at the floor everywhere.
- Runs over 10 min of lock wait plus training get killed by the default background timeout. Launch run.py with a long timeout.
- Recommendation for the campaign's fair baseline: R_LR 1e-4 + full decay + R_INTERP 0.5 (76df3de). It beats every low-LR variant on bpb without starving other lanes of plasticity. Lanes adding new modules keep their trained values in full.

Next 3 ideas:
1. Seed-replicate 76df3de (2 seeds) and R_INTERP 0.25, to pin the interpolation curve and confirm 0.5 wasn't a lucky seed. The 0.5-vs-0.75 gap is under 1 sd.
2. Once binding moves kappa off the floor, run its best change with R_INTERP 1.0 vs 0.5. This tests whether interpolation trades away reasoning gains (WiSE-FT's known risk). Hold any swarm-wide adoption until then.
3. L2-SP / decay-toward-base during training (AdamW decay pulls to base0, not 0) at R_LR 1e-4. It's an in-training analogue of WiSE-FT that also shapes the trajectory. Compare to R_INTERP 0.5 at matched bpb.

## session 3 (2026-11-26), primary val_bpb, pooled baseline 1.4708 +- 0.0013 (n=6), ~200 steps/run

| run | commit | val_bpb | t | verdict |
|---|---|---|---|---|
| R_LR 1e-4 + R_INTERP 0.5, seed 1 (replicate 76df3de) | 91b7b8c | 1.4666 | +3.04 | candidate (seed 0 was +2.35, so it replicates) |
| R_LR 1e-4 + R_INTERP 0.25 | 924a2df | 1.4672 | +2.60 | candidate, ties 0.5 |
| L2-SP R_L2SP 100 (p -= lr*100*(p-base) each step), no interp | 0526420 | 1.4689 | +1.37 | candidate but reverted (f5d8f9f): worse than WiSE-FT, more code |
| R_LR 3e-4 + R_INTERP 0.5 | 766f956 | 1.4686 | +1.56 | candidate |
| R_LR 3e-4 + R_INTERP 0.25 | 1367995 | 1.4671 | +2.69 | candidate |

Head: 9d31dc2 = R_LR 1e-4, full decay, R_INTERP 0.5 (two seeds: 1.46755, 1.4666; mean t ~ +2.7).

Learned:
- WiSE-FT 0.5 replicates (2/2 seeds clear t 2.3). It's the most robust recipe-lane result so far.
- Interpolation curve at R_LR 1e-4 is flat for a <= 0.5 (0.25 = 1.4672, 0.5 = 1.4666/1.4676). Everything under 0.5 sits at the frozen base (1.4675) within noise. The bpb cost only appears above 0.5.
- What matters is the effective step a*R_LR, not R_LR alone. 3e-4 x 0.25 (eff 7.5e-5) = 1.4671 and 1e-4 x 0.5 (eff 5e-5) ~ 1.467, but 3e-4 x 0.5 (eff 1.5e-4) = 1.4686 and 1e-4 x 0.75 (eff 7.5e-5) = 1.4686. So it isn't purely a*R_LR either. A larger LR with heavier shrink lands in a better basin than a smaller LR with lighter shrink at the same effective step. Single seeds, gaps ~1 sd.
- L2-SP at 100 was worse than WiSE-FT (t 1.37 vs 2.35+), about the same as just dropping R_LR to 3e-5. Post-hoc averaging beats an in-training pull here. Not worth the code (reverted). Implementation note: under mx.compile the reference weights must be passed in `state` (uncaptured-input error otherwise).
- For the campaign: 3e-4 + R_INTERP 0.25 gives lanes 3x base plasticity at frozen-base bpb. If binding/depth need the base to move, it's the better fair baseline. Choose it once kappa leaves the floor.

Next 3 ideas:
1. Interp-ratio check once kappa moves: take binding's best change, run (1e-4, 0.5) vs (3e-4, 0.25) vs (1e-4, 1.0). See which preserves the kappa gain while holding bpb. WiSE-FT's risk is that it shrinks exactly the task-specific direction.
2. Per-group interpolation: interpolate only embeddings/prelude/coda (the "knowledge" tensors) and keep the core at a=1. If bpb holds, the core's adaptation is free for reasoning lanes.
3. Seed-replicate 1367995 (3e-4/0.25) once. If it's >= 1e-4/0.5 on 2 seeds, switch the head, since it buys plasticity at no cost.
