# depth lane, swarm nov24b (2026-11-26), 5 experiments, base-v4, harness 65ba393be2

Screened with `--primary val_bpb`: base-v4 scores 0 on every reasoning pair, so the baseline `kappa_id` is
-0.0667 with SD 0 (n=9), and kappa-primary screening would mark every gate-passing run "candidate".
The depth signal tracked here is val_bpb at 8 loops (`bpb_at_2x`). Pooled baseline (n=6): 4-loop 1.4708 ± 0.0014, 8-loop 1.6003 ± 0.0015.

| run | 4-loop val_bpb | 8-loop val_bpb | verdict |
|---|---|---|---|
| RANDOM_LOOPS U[1,7] | 1.4803 | 1.5570 | discard (t -6.9) |
| CARRY gate, zero-init per-channel h<-x+c(h-x) | 1.4733 | 1.5976 | discard (t -1.8) |
| LOOP_CHOICES {4,8} | 1.4879 | 1.5222 | discard (t -12.4) |
| LOOP_EMBED | 1.4809 | 1.6029 | discard (t -7.4) |
| LOOP_CHOICES {4,4,4,8} | 1.4796 | 1.5395 | discard (t -6.4) |

Learned:
- Time spent training at 8 loops trades 4-loop val_bpb for 8-loop val_bpb, roughly monotonically. {4,4,4,8} beats U[1,7] on both
  metrics, so the U[1,7] steps at short loop counts are wasted. No 5-min schedule gets 8 loops below 4 loops: overthinking stays.
- Zero-init modules still cost 4-loop bpb within 5 min: LOOP_EMBED +0.010 (a new matrix at LR 1e-3), CARRY +0.0025 (a vector at 1e-2).
  Even the unchanged baseline drifts up from the base (1.4675 -> 1.4708), so a val_bpb screen mostly measures LR re-heat.
- The lane's core claim (hop 8-10 accuracy rises with loops) cannot be tested on looped bases: base-v4 and looped-2h are at 0 pair
  accuracy, while control-4L-2h (non-looped) reaches 0.96/0.77/0.51 at hops 2/4/6.

Next 3 ideas:
1. LOOP_CHOICES {4,4,4,8} on top of the recipe lane's best (lower R_LR): check whether the 4-loop cost is re-heat times distribution shift.
2. Campaign-level: pretrain a looped base with LOOP_CHOICES or random loops for hours. Test-time depth scaling has to be in the
   base, because a 5-min warm start trades it 1:1 against default-loop bpb.
3. Add `bpb_at_2x` (or kappa_ood at 2x, once binding works) as an allowed primary for the depth lane, and report CARRY with a
   frozen backbone (only new tensors train) so zero-init additions are judged without re-heat noise.

# depth lane, swarm nov24b, session 2 (2026-11-26), 5 more experiments, base-v4, harness 65ba393be2, `--primary val_bpb`

Low-LR control is the recipe lane's R_LR 1e-5 run: 4-loop 1.4683, 8-loop 1.5941. The pooled baseline (R_LR 1e-4) is 1.4708 / 1.6003.

| run | 4-loop | 8-loop | d4 vs 1e-5 ctrl | d8 vs 1e-5 ctrl | steps | verdict |
|---|---|---|---|---|---|---|
| DEPTH_ATTN=2 (R_LR 1e-4) | 1.4733 | 1.5981 | (+0.0025 vs base) | (-0.0022 vs base) | | discard t -1.82 |
| {4,4,4,8} + R_LR 1e-5 | 1.4712 | 1.5706 | +0.0029 | -0.0235 | 165 | discard t -0.32 |
| {4,8} + R_LR 1e-5 | 1.4760 | 1.5587 | +0.0077 | -0.0354 | | discard t -3.76 |
| {4,4,4,8} + R_LR 1e-5 + BPTT_LOOPS 4 | 1.4711 | 1.5770 | +0.0028 | -0.0171 | 191 | discard t -0.21 |
| {4,4,4,6} + R_LR 1e-5 | 1.4703 | 1.5775 | +0.0020 | -0.0166 | 183 | discard t +0.36 |

Learned:
- DEPTH_ATTN=2 is null, like CARRY: zero-init depth modules cost about +0.0025 at 4 loops and buy no 8-loop gain in 5 minutes.
- Lowering R_LR 10x halves both sides of the mixed-loop trade. It does not remove the trade. At either LR every 0.001 of 4-loop bpb buys about
  0.007-0.008 of 8-loop bpb. The 4-loop cost is distribution shift in shared weights, not only re-heat.
- Truncated BPTT (last 4 loops) gives +16% steps but a smaller 8-loop gain. Gradient through all 8 loops matters more than step count.
  Confound: with BPTT_LOOPS=4, 4-loop steps also stop-grad h at r=0, so the prelude gets gradient only via the re-injected e.
- Depth extrapolation exists but is not free. Training at 6 loops improves 8-loop bpb (-0.017) at the same exchange rate as training at 8.
  So it is generic "deeper is more stable", not a cheaper route.
- Only ~165-190 steps fit in 300 s now (CANON=4 + 16/32 reasoning rows). At this budget no warm-start schedule makes 8 loops beat 4.
  The lane's real claim (hop 8-10 accuracy rises with loops) is still untestable while base-v4 is at 0 pair accuracy.

Next 3 ideas:
1. Campaign/base level: continue base-v4 for 1-2 h with LOOP_CHOICES (4,4,4,8) or (4,6). Then re-screen depth at 2x loops on that base.
   A 5-min warm start can only move along the 4-vs-8 trade line.
2. Fix BPTT_LOOPS so it only truncates when loops > BPTT_LOOPS (no stop-grad at r=0 for 4-loop steps), then retry {4,8}+BPTT 4.
   That removes the prelude confound and tests whether cheap deep steps let {4,8} reach 1.56 at 8 loops without the +0.008 at 4.
3. Let the depth lane screen on a depth metric (8-loop val_bpb, or kappa_ood at 2x once binding produces nonzero pair accuracy).
   With val_bpb at default loops as primary, every depth-scaling intervention is a discard by construction.

# depth lane, swarm nov24b, session 3 (2026-11-26), 5 experiments, base-v4, harness 65ba393be2, `--primary val_bpb`

All runs stack LOOP_CHOICES on the recipe lane's WiSE-FT candidate (R_LR 1e-4, R_DECAY_FRAC 1.0, R_INTERP, from 76df3de).
Controls are the recipe lane's runs with no mixed loops: WiSE 0.5 scored 1.4676 / 1.5949, WiSE 0.75 scored 1.4686 / 1.5949 (4-loop / 8-loop).
The test_lr_schedules failure comes from R_DECAY_FRAC 1.0 (the test hard-codes 0.25). It is not a bug.

| run | 4-loop | 8-loop | d4 vs ctrl | d8 vs ctrl | rate d8/d4 | steps | verdict |
|---|---|---|---|---|---|---|---|
| {4,4,4,8} + WiSE 0.5 | 1.4693 | 1.5665 | +0.0018 | -0.0284 | 16:1 | 168 | candidate t +1.06 (608c5c5) |
| {4,8} + WiSE 0.5 | 1.4725 | 1.5518 | +0.0049 | -0.0431 | 9:1 | 142 | discard t -1.23 |
| {4,4,4,8} + WiSE 0.75 | 1.4719 | 1.5558 | +0.0033 | -0.0391 | 12:1 | 168 | discard t -0.83 |
| (4x7,8) + WiSE 0.5 | 1.4684 | 1.5767 | +0.0008 | -0.0182 | 23:1 | 183 | candidate t +1.71 (49e8711) |
| (4x15,8) + WiSE 0.5 | 1.4679 | 1.5849 | +0.0003 | -0.0100 | 33:1 | 190 | candidate t +2.09 (HEAD) |

Learned:
- WiSE-FT bends the 4-vs-8 trade curve. Without it, every 0.001 of 4-loop bpb bought 0.007-0.008 of 8-loop bpb. With interpolation at 0.5
  the rate is 16:1 at 25% deep steps and 33:1 at 6.25%. Halving the weight delta costs the 8-loop gain less than it costs the 4-loop damage.
  So the 8-loop gain is mostly a low-curvature direction, and the 4-loop damage is curvature.
- The rate improves as the deep share shrinks, so the first few deep steps are the cheapest. A rare deep step is a near-free regulariser
  for depth extrapolation: (4x15,8) gives -0.010 at 8 loops for +0.0003 at 4.
- Caveat for promotion: the primary-metric t of these candidates is mostly the recipe lane's WiSE effect (WiSE alone has t +2.35).
  The depth contribution shows only in `scaling` (8-loop bpb), and the val_bpb screen cannot see it. The campaign should compare the depth
  candidate against WiSE-alone on 8-loop bpb, not only against the baseline on 4-loop bpb.
- No schedule makes 8 loops beat 4 loops yet (best 8-loop 1.5518 vs 1.47 at 4). The reasoning claim is still untestable at 0 pair accuracy.

Next 3 ideas:
1. Interpolate per loop count, a "depth-WiSE": average a 4-loop-only fine-tune and a {4,8} fine-tune instead of base + 0.5 * delta.
   This tests whether model soups can hold both 4-loop and 8-loop quality.
2. Sweep R_INTERP down (0.25-0.35) at 25% deep steps. Interpolation may reach the 4-loop control with more 8-loop gain than (4x15,8).
   Map the frontier as (share, alpha) rather than share alone.
3. Campaign: promote (4x7,8)+WiSE 0.5 or (4x15,8)+WiSE 0.5 with `bpb_at_2x` reported against WiSE-alone holdout seeds. If the 8-loop gain
   holds, put rare 8-loop steps in the next base pretraining run.
