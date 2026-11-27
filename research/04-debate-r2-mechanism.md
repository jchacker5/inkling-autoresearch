# 04 · Debate round 2 · Debater 1: mechanism skeptic

*2026-11-24, 12:30. Task v3 (5539a61) landed while I wrote this, so §2 audits both v2 and v3. No model was run. The scripts are `research/04-debate-r2-mechanism-audit.py` (v2, loaded from 4a1c4dd) and `…-audit-v3.py`; both use numpy only and never load MLX.*

## 1. Rebuttals

**Debater 4.**
- *I disagree with* r2 §1 "Fix", now `PAIR_BOUND = .25` in `evaluate()`. The claim is that structure plus any query-dependent coin scores .21–.25, so "pair_h > .25 certifies ≥ h−1 lookups". But their coin flips each twin independently.
  - A rule that gives the twins *opposite* answers is right on both or wrong on both.
  - With structure and an alphabet bit (zero lookups), such a rule scores .49–.52 at hops 4–10.
  - With one lookup and no chasing (S3 below), it scores .58–.61.
- *I accept* the protocol: paired tests, one primary metric, ≥ 6 pooled seeds, BH screening, Holm with fresh seeds.

**Debater 2.**
- *I disagree with* r1 §2's Q = (Ā−.1)/.9, which assumes a .1 null. v2's query-blind rules reach .82–1.0, and the ratio form rewards shrinking F around them.
- *I accept* r1 E4 and §4.3: every KL check costs a coda+head pass. With my F4, no exit should be claimed before a decode meter exists.

**Debater 3.**
- *I disagree with* r2 §2: "A ≥ .25 on hops 0–2 means lookup is forming". On v2 single accuracy, counting rules gave those hops a .45 ceiling. On v3 pairs, structure alone reaches .36 at hop 2. Only hop-0 pairs certify binding: their zero-lookup ceiling is .06.
- *I accept* r2 §1: my F2 as a 5-minute warm start was uninformative, so building the weighting into the base was right. I also accept R.

## 2. Audit

**v2, the shortcuts that motivated v3.** Rules that never read the query scored (ID/OOD): counting (a root letter seen exactly twice) .42/.54; a root with one child that has one child .82/.87; plurality .45/.99; resolving the *last statement* .48/.96.

The reported floor was .14/.18. Two causes: distractors never attached to the chain, so it was the only long unbranched path, and the random topological order put the query last (.93/.98 at hops 8/10). Debaters 3 and 4 reproduce these numbers.

In v2 training, once earlier answers could be copied, 87% of answers needed ≤ 1 pointer step, and plurality was right on 31%. v3's 30× first-answer weight fixes that incentive.

**v3, what runs now.** Twin pairs drive every query-blind or positional-digit rule to 0. The filler is constant (`6\n`) across twins, so it scores 0 too. What remains, as pair accuracy on 3,000–4,000 fresh pairs per hop:

| rule | query lookups | chasing | h2 | h4 | h6 | h8 | h10 |
|---|---|---|---|---|---|---|---|
| structure + alphabet bit (opposite answers) | 0 | none | .36 | .49 | .49 | .50 | .52 |
| S3: bind q; later end → later root (3-level structure) | 1 | none | .48 | .61 | .58 | .58 | .58 |
| S3 with the chains known | 1 | none | .69 | .63 | .62 | .60 | .59 |
| S1: "is q the last statement?" + chase | 0 | full | .00 | .66 | .93 | .99 | 1.0 |

On the official set, S3 with the chains known scores .66 ID and .57 OOD.

**Mechanisms behind these rules.**
- **Order leakage.** Random available-node pops make the chain that starts first finish first, so root order predicts end order (.59–.76).
- **Positional binding.** At h ≥ 6 the last statement is a chain end (.93–1.0), so S1 binds by position. It still chases: it fakes binding, not depth.
- **A training attractor.** On the 30×-weighted first answer, S3 is right .61–.70 at hops 2–6, against .5 for a coin flip and 1.0 for chasing.

**Fixes (checked in numpy).**
1. **Crossed twins.** Swap the parents of A[i] and B[i] at one hop i. Two bytes change; every position, letter and digit stays the same, and the result is a valid v3 problem with the two answers swapped.
   - Score a *quad*: both twins, on both the original and the crossed problem.
   - S2/S3 score .000 at every hop, while S1, which chases, keeps .64–1.0.
   - A valid crossing exists in 64–82% of problems at h ≥ 2; resample the rest.
2. **Uniform topological order.** Pop each available node with probability ∝ its subtree size; for forests this samples linear extensions exactly uniformly. S3 drops to .52–.60 and S1 to .36–.92. That is partial, so use it for training data, not as the eval fix.
3. **Set `PAIR_BOUND` to .5** until quads land.

**base-v2.** The plain run (W=1, v2) scored .13/.08/.13/.14/.07/.20 at 15 min, at the floor. The W=10 run was superseded before its first checkpoint. The v3 run (`FIRST_ANSWER_W=30`) started at 12:16 and has no rows yet.

## 3. Votes

**D1.** Keep the v3 run; its 30× first-answer weight is my first-query weighting.
- Judge it on hop-0 pairs. Lookup is present at ≥ .40 on two consecutive checkpoints: the ceiling is .06, and the SE is ≈ .07 at 50 pairs.
- Above hop 0, TSV pairs reach .5–.6 without chasing.
- If hop-0 pairs are ≤ .20 at both 60 and 75 min, resume with `FIRST_ANSWER_W=100`.
- Hold W fixed in every bpb comparison: the weighted mean shifts the text share.

**D2.** Use R everywhere, with baseline-R as the null. A module that only speeds re-warm recovery must not win.

**D3.** S(B) = Q subject to F ≤ B, with no ratio.
- **Q:** mean quad accuracy over hops 2–10. It is 0 for any rule that skips a hop's pointer.
- **F:** W + KV₅₁₂ + A₁, as debater 2 measures them.
- **B:** the deployed F of the reference looped model.
- **Gates:** val_bpb ≤ ref + .02 and Q ≥ .10.
- **A beats B** if F_A ≤ F_B and the paired 95% CI of ΔQ is above 0, or if F_A ≤ .9·F_B and TOST holds |ΔQ| ≤ .03 and |Δbpb| ≤ .005.

**D4.**
- **Screening:** one seed against 6 pooled baseline-R seeds. Keep if ΔQ ≥ max(3 pts, 2σ̂) with Δbpb ≤ +.004, or if Δbpb ≤ −max(.004, 2σ̂) with ΔQ ≥ −2. One seed's SE is ≈ 2.9 pts at σ = 2.5.
- **Promotion:** 6 v 6 fresh seeds, one-sided Welch with Holm. The MDE is ≈ 4 pts.
- **No per-hop claims:** the paired MDE is ≈ 6 pts at 400 problems.

**D5.**
- Drop control-10L. It is invalid and answers nothing we ask.
- Spend its 2 h on `looped --set ANSWER_W=1.0 FIRST_ANSWER_W=1.0`. That tests at base level whether the weighting creates lookup.
- `looped` is the right reference for 4L and chunk.

**D6.** Run the gated control-4L warm start only if control-4L passes hop-0 pairs. Build no separate base.

**D7.** `next_lanes.txt` lists `binding recipe` until X1 passes, then `binding recipe depth exit`.
- `binding`: first-answer weighting and uniform-order data.
- `recipe`: R's LR grid.
- `depth`: `RANDOM_LOOPS`/V plus `DEPTH_ATTN`. The claim to test is that quads at hops 8–10 rise with loops.
- `exit`: `EXIT_TAU` with a cosine metric and `LOOP_CONSIST` with zero-init adapters, scored decode-faithfully.
- No swarm time for `KV_SHARE`, `HALTING`, `CHUNK` or `HISTORY`.

**D8.** Fix in this order:
1. Quads, with `PAIR_BOUND` at .5.
2. `mean_loops` over all calls; it feeds `verdict()`.
3. A peak-memory reset.
4. KV at the evaluated loop count.
5. Compiled bf16 latency.

Land 1–3 before the warm starts, after `pytest`.

## 4. Top three experiments (GPU queue; the warm-start window is now ≈ 15:00)

**X1. Quad audit** (eval only, ≈ 10 GPU-min).
- **Checkpoints:** base-v2 (v3), base-v2-1h and v1 base.
- **Problems:** 1,000 crossed quads per hop, for h ∈ {0,1,2,4,6,8,10}, built with `crossed()` (12 lines in the audit script).
- **Pass:** hop-0 pair ≥ .30, and quad ≥ .30 at hops 2 and 4. The SE is 1.4 pts; the no-chase floor is 0.
- **Fail:** quads ≤ .05 while pairs ≥ .5. That is the positional rule: freeze the reasoning lanes.
- **Diagnostic:** pair accuracy on S3-right vs S3-wrong problems. A gap ≥ .2 means the model uses S3.

**X2. Probe staircase on base-v2** (≈ 5 GPU-min of forward passes; fit on CPU).

```python
_, _, states, _ = model.forward(x, 4, return_states=True)       # states[r]: (B, T, D)
X[r] = states[r][rows, rhs_pos]       # each statement's right-hand byte; label = its resolved value
# per (loop r, depth d): 10-way ridge-logistic in numpy, fit 4k problems, test 2k
```

- **Pass:** d\*(r), the deepest depth decoded at ≥ .9 after loop r, rises at least twice over r = 1..4, and d\*(4) ≥ 4. The answer-position probe must also read the asked value wherever the statement probe does.
- **Fail:** d\*(r) ≤ 1 at every r. Then any pair above .5 is positional; freeze `depth`.

**X3. Loop sweep on quads** (after the controls, ≈ 30 GPU-min).
- **Grid:** loops {1,2,3,4,6,8} × h {1,2,4,6,8}.
- **Models:** base-v2, looped, looped-W1, 4L, chunk2 and chunk4.
- **Pass:** all three of these hold.
  - The loop count needed for 90% of the plateau rises with h (Spearman ρ ≥ .8).
  - At h = 8, quads at 8 loops beat quads at 4 loops by ≥ 5 pts (McNemar p < .01).
  - 4L trails by ≥ 10 pts at h ≥ 4.
- **Fail:** quads are flat across loops. The paper then claims only bpb and bytes.
