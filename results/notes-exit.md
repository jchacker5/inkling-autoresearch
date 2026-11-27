# exit lane — nov24b (2026-11-26), harness 65ba393be2, base base-v4, all runs seed 0, primary val_bpb

Landing (swarm/landing.py on base-v4, results/landing-base-v4.json): reasoning is at floor (hops>=1 ~chance, kappa_id pinned at -0.0667
in every run), so this lane can only trade loops vs val_bpb. Text bpb by loop 3.15/1.73/1.49/1.46, so loop 4 still helps on average, but
mean text landing is loop 1.86. At 2x loops (8) KL exit *improves* bpb (1.597 -> 1.495 at tau 0.03): overthinking is real past the trained depth.

| exp | change | val_bpb | mean loops | t |
|---|---|---|---|---|
| 1 | EXIT_TAU=0.003 (kl, eval only) | 1.4737 | 3.50 | -2.10 |
| 2 | LOOP_CONSIST=0.1 + EXIT_TAU=0.003 | 1.4753 | 3.26 | -3.30 |
| 3 | LOOP_CONSIST=0.1, no exit | 1.4737 | 4.00 | -2.10 |
| 4 | EXIT_TAU=0.001 | 1.4731 | 3.54 | -1.69 |
| 5 | EXIT_TAU=0.01 | 1.4760 | 3.43 | -3.80 |

Learned:
- All discards under the bpb rule, but the pooled-baseline t is misleading here: the seed-0 baseline itself is 1.4735. Seed-matched,
  EXIT_TAU=1e-3 is free (-0.0004 bpb) at 11.5% fewer loops, and 3e-3 costs +0.0002 for 12.6%. 1e-2 costs +0.0025 for 14%. Knee ~1e-3.
- The 5-min fine-tune makes readouts settle much earlier than base-v4 (tau 1e-3: 3.54 loops trained vs ~3.9 on base).
- LOOP_CONSIST=0.1 costs ~+0.002 bpb at 4 loops (fewer steps: 173 vs ~198, + the extra coda pass), buys 0.24 more loops saved with exit,
  and massively improves the 2-loop readout (bpb@2 1.752 -> 1.634). Worth it only for a loops/2 deployment, not at L=4.
- Gotcha: with EXIT_TAU>0 set at module level, 17 tests fail (CHUNK>1 cases inherit it and hit the intended ValueError); not a real bug.
  `landing.py --set` needs quoted strings: --set "EXIT_METRIC='state'". Its state sweep queued > 10 min on the GPU lock and was killed.

Next 3 ideas:
1. Cheapness is not a verdict under val_bpb-primary: propose a "matched-bpb loops" verdict (or run EXIT_TAU=1e-3 at seeds 1..5) so the
   free 11.5% loop saving can be confirmed by the campaign instead of being discarded against a pool whose mean is below seed 0.
2. EXIT_TAU at 2x loops (eval L=8, tau 0.03): landing shows exit fixes overthinking there (bpb 1.597 -> 1.495); pair with depth lane's
   RANDOM_LOOPS so that 2x loops + exit beats L=4 on bpb at ~4 mean loops.
3. EXIT_METRIC='state' (no coda readout per loop, so the exit test itself is cheap) — calibrate tau first with landing.py when the GPU is idle;
   and LOOP_CONSIST at lower weight (0.02) with R_LR=1e-5 once recipe's low-LR win is confirmed.

## Session 2 (2026-11-26), harness 65ba393be2, base-v4, primary val_bpb. Seed-matched deltas use the ledger's same-harness baseline rows
(seed0 1.47347, seed1 1.46987, seed2 1.47078; status.py's "best baseline 1.46057" is from an older harness: don't compare to it).

| exp | change | seed | val_bpb | d vs seed baseline | mean loops | verdict |
|---|---|---|---|---|---|---|
| 6 | LOOP_MIX=0.2: eval output log(0.8 p_L + 0.2 p_{L-1}) | 0 | 1.47502 | +0.0016 | 4.00 | discard |
| 7 | EVAL_LOOPS=5 + EXIT_TAU=1e-3 kl | 0 | 1.48236 | +0.0089 | 4.12 | discard |
| 8 | EXIT_TAU=1e-3 kl (replicate) | 1 | 1.47020 | +0.0003 | 3.56 | discard (t +0.43) |
| 9 | EXIT_METRIC=state, tau 0.6 | 0 | 1.49924 | +0.0258 | 3.45 | discard (gate) |
| 10 | EXIT_TAU=1e-3 kl (replicate) | 2 | 1.46992 | -0.0009 | 3.55 | discard (t +0.63) |

Learned:
- **KL exit at tau 1e-3 is free, now on 3 seeds**: d_bpb -0.0004 / +0.0003 / -0.0009 (mean -0.0003) at 3.54-3.56 mean loops (-11.3%).
  It is the lane's real result but can never be a val_bpb "candidate"; it needs a matched-bpb loops verdict (or campaign it as-is).
  Caveat: baseline rows ran under heavier GPU contention (79-163 steps vs ~200 for mine), so seed matching is loose. But the 5-min
  fine-tune barely moves bpb (base 1.4675 vs pool mean 1.4676), so this hardly matters.
- The model is tuned exactly to L=4. One extra eval loop (L=5, even with KL exit) costs +0.009, and mixing in loop 3's readout costs +0.0016.
  Overthinking past the trained depth starts at loop 5, not loop 8. Loop-ensemble readouts don't help, since loop 3 is too weak.
- EXIT_METRIC='state' is a bad proxy for KL. At matched compute (3.45 vs 3.43 loops) it costs +0.026 bpb vs +0.0025 for KL tau 1e-2 (10x worse).
  The relative state change is tightly bunched per loop (loop 2->3 q10/50/90 = .48/.60/.86, loop 3->4 = .33/.38/.50 on train rows),
  so it measures "which loop" more than "which token". It also caps L=8 at 3.47 loops by itself. Its upside is prefill latency of 11.2 ms vs 12.6
  (no per-loop coda readout).
- Ops: run.py waits on the GPU lock for >10 min when 3 lanes are queued. Use a long background timeout (2 h), or the run is killed in the queue.

Next 3 ideas:
1. Campaign-promote EXIT_TAU=1e-3 under a "non-inferior at fewer loops" rule (one-sided non-inferiority on val_bpb with margin 0.002,
   plus mean_loops < 0.95*L). The 3 seeds here suggest it passes. This needs a run.py/confirm.py verdict change, not a train.py change.
2. Cheap-and-good exit signal: a learned zero-init per-token exit probe on the core state (a linear layer trained to predict KL<tau from the
   final-loop distillation target; no coda pass at eval). It should keep KL's ranking at state-metric latency.
3. If depth lane ever lands an L=8-robust model (LOOP_CHOICES / RANDOM_LOOPS), re-run KL exit at L=8 with tau 0.01-0.03. On base-v4 the exit
   cut L=8 bpb 1.60 -> 1.50, and that is where exit can beat L=4 rather than only match it.

## Session 3 (2026-11-26), harness 65ba393be2, base-v4, primary val_bpb, all seed 0, all on top of recipe's WiSE-FT 0.5
(91b7b8c: R_LR 1e-4, R_DECAY_FRAC 1.0, R_INTERP 0.5). Seed-matched reference: recipe WiSE-FT seed 0 = 1.46755 at 4.00 loops.

| exp | change (on WiSE-FT 0.5) | val_bpb | d vs WiSE seed0 | mean loops | verdict |
|---|---|---|---|---|---|
| 11 | EXIT_TAU=1e-3 kl | 1.46783 | +0.0003 | 3.544 (-11.4%) | candidate t +2.15 (d836d5d) |
| 12 | EXIT_TAU=3e-3 kl | 1.46835 | +0.0008 | 3.501 (-12.5%) | candidate t +1.77 (committed after d836d5d) |
| 13 | EXIT_METRIC=conf (1 - top1 < 0.02), exit allowed after loop 1 | 1.48754 | +0.0200 | 3.002 | discard |
| 14 | conf tau 0.02, exit only from loop 2 | 1.47648 | +0.0089 | 3.179 | discard |
| 15 | KL 1e-3 OR top-1 >= 0.995 (EXIT_CONF), loop 2+ | 1.47031 | +0.0028 | 3.245 | discard (t +0.35) |

Learned:
- KL exit at 1e-3 stays free on the WiSE-FT model (+0.0003, 4th seed-matched replicate across two training recipes). Combined with WiSE-FT it is
  finally a val_bpb *candidate* (t 2.15). The t comes from WiSE-FT, not from exit, so the campaign result for d836d5d is about "WiSE-FT at 11% fewer loops".
  Compare it with recipe's 91b7b8c holdout: if they tie, exit is confirmed free.
- The KL knee is still about 1e-3: 3e-3 buys only 1.1% more loops for +0.0005. The WiSE-FT model lands at almost the same place as the plain fine-tune (3.54 vs 3.54-3.56).
- CALM-style softmax confidence is a poor exit signal for this byte model. Loop-1 readouts are overconfident: allowing loop-1 exits doubled the cost
  (+0.020 vs +0.009). Even top-1 >= 0.995 bytes keep changing afterwards: adding them to KL costs ~0.008 bpb per loop saved, versus ~0.004/loop for raising KL tau.
  Ranking by cost per loop saved: KL (best) < KL+conf < conf < state. Distribution *stability* matters, not distribution *sharpness*.
- Ops: tests have 16-17 known failures whenever EXIT_TAU>0 at module level; with exit off, only test_lr_schedules fails (inherited from recipe's R_DECAY_FRAC=1.0).

Next 3 ideas:
1. Patience-2 KL exit at a looser tau (exit only after two consecutive KL<tau, e.g. tau 1e-2). This needs L>=5 to be useful, so pair it with depth's
   LOOP_CHOICES training: stability over 2 loops should be a stronger signal than over 1.
2. A learned exit probe (zero-init linear on the core state, trained online to predict log KL to the final loop) gives KL's ranking without the per-loop
   coda readout. It is the only route to real latency savings (state metric 11.2 ms vs KL 12.6 ms prefill).
3. Ask the campaign/confirm to add a non-inferiority verdict (val_bpb margin 0.002, mean_loops < 0.95 L) so exit can be confirmed on its own,
   rather than riding on WiSE-FT's t.
