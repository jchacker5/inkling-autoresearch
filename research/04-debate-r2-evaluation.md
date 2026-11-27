# 04 · Debate round 2 · Debater 4: evaluation / statistics adversary

*2026-11-24, 12:15. No model forward passes: generator audits in pure Python, keep rules by Monte Carlo. Scripts and outputs are in `research/04-r2-eval/`.*

## 1. Task v2 audit: the position shortcut is gone, a structural one replaced it

`reason_floor` is correct for position rules.
- A learned position-only rule adds only 2–6 points (.13/.14/.18/.21/.24 at hops 2/4/6/8/10).
- Digit frequency is useless: each digit occurs once.
- "Nearest digit before q's statement" scores .13–.18.
- The filler is harmless: K=2/4 see the prefix `3\n`, and the answer is 3 in 8.5–10.5% of problems.

**The leak is structure.** Distractors never attach to the chain, so the chain is the only long *bare path*. The question-blind rules below choose uniformly among roots whose first m levels each have one child (20k problems per hop; the official set agrees within 2 points):

| rule (question never read) | lookups | h2 | h4 | h6 | h8 | h10 |
|---|---|---|---|---|---|---|
| `reason_floor` | 0 | .11 | .12 | .14 | .15 | .18 |
| root with exactly 1 child | 1 | .39 | .42 | .47 | .51 | .55 |
| bare path, m=2 | 2 | .79 | .82 | .83 | .86 | .87 |
| bare path, m=3 | 3 | 0 | .97 | .97 | .97 | .97 |
| plurality value | ≈h | .02 | .48 | .85 | .98 | 1.0 |
| honest chase, k<h lookups from q | k | .11 | .12 | .13 | .15 | .18 |

Chasing earns nothing until the last hop, while structure pays from the first lookup, so depth-limited learners drift toward structure (cf. Zhang et al., IJCAI 2023; Saparov et al., ICLR 2025).

**OOD is not depth-only.** At hops 8/10:
- R falls to 7.0/6.0 (9.2 at hop 2).
- The chain holds 9–11 of the 20 variables, so the plurality value is the answer 98–100% of the time.
- 3 query-free lookups score .97.

A structure reader's profile *rises* with h. `reason_ood` supports no serial-depth claim.

**Training rarely needs depth.** Counting ancestors already answered in the line:
- 87% of answers need ≤1 lookup; 1.1% need ≥4.
- The eval format is 0.008% of bytes.
- `ANSWER_W=10` (3e4c4f6) teaches single lookups, the right first target, but not depth.

**Fix (eval-only; `decoy_pairs.py`).** Add a second bare chain of equal length (h≤8). Ask both ends as twins and score *pair accuracy*.
- Position and question-blind rules score 0.
- Structure plus any query-dependent coin scores .21–.25.
- k real lookups plus a position guess score ≈2^−(h−k): at hop 8, k = 6/7/8 give .26/.50/1.0.
- So **pair_h > .25 certifies ≥ h−1 lookups.**

**EVAL_PER_HOP=400.**
- Enough for seed-limited screening: item churn adds ≈0.9 points to reason_id.
- Not enough for per-hop curves or checkpoint pairs: the paired MDE is 6.3 points per hop (2.8 at n=2,000). There, use 2,000 per hop on a holdout seed.

**Live.** base-v2-plain at 15 min: hop0 = .13 against 1/R = .11, so no lookup. It used the old no-newline prompt and the W=10 restart uses the new one, so they are not a clean pair.

## 2. Rebuttals

- **D1 (mechanism).**
  - *Disagree:* §4's gate, "QI ≤ .50 ⇒ the report's order stands". A model that reads shallow queries and falls back on the bare-path root passes QI yet scores .97 at hops 8/10. Gate on pair_h instead.
  - *Accept:* F2 (now adopted) and F4.
- **D2 (efficiency).**
  - *Disagree:* §2's IPG = Q/F with Q = (Ā−.1)/.9 and the gate Ā ≥ .25. It is a ratio, assumes 10% chance, and includes structurally solvable hops. The 1-child rule alone gives Ā = .47.
  - *Accept:* deterministic transforms need paired tests, not seeds. With 84-byte prompts SWA-128 cannot change a reasoning answer, so test bpb only.
- **D3 (optimization).**
  - *Disagree:* §4.1, "depth binds: hop 8 .13→.41 over loops 1–4". In v1 that was the first-digit prior arriving (agreement .13→.54). In v2, loops can buy the 3-lookup structural route. Also, "seed SD ≤ .0015" from 3 seeds is unverifiable: the 95% CI is [.52, 6.3]× the estimate.
  - *Accept:* R, the ±5% step flag, and "seeds shrink variance, not bias". The plain re-warm cost +.033 bpb by 15 min.

## 3. D4 (Monte Carlo, 20k campaigns)

**Current rules** (4 lanes × 15 experiments, global null, σ_acc 1.5–4 points):
- P(≥1 lane promoted) = **.34–.38**, or .47–.48 if the discovery run is averaged in. program.md tests all three metrics.
- A true 2σ (3σ) effect is promoted .24 (.52) of the time.

**`run.py` verdict:**
- **Reference:** the pooled `baseline-<recipe>` lane (n_b=6, same harness commit), never the lane's last keep.
- **Metric:** one primary per lane, declared in `next_lanes.txt` (val_bpb or pair_id).
- **Rule:** `candidate` iff t = (x−m_b)/(s_b√(1+1/n_b)) ≥ t₀.₈₀,₅ = 0.92 (≈1 baseline SD), subject to:
  - Δbpb ≤ +.004 (bpb-primary lanes: Δreason_id ≥ −max(.02, s_b));
  - steps within ±7% of the baseline median;
  - AC power.
- **"Cheaper":** only nominates a candidate.

**`campaign.py`:**
1. Run the 6 baseline seeds just before the swarm. Add 3 A/A seeds as a fake lane; expect ≈20% candidates.
2. Each lane's finalist (its last candidate) gets 6 fresh seeds, interleaved with 3 fresh baseline seeds. Score them on a **holdout eval seed**, computed in every `evaluate()`.
3. Test with a one-sided Welch t against 9 baselines, excluding the discovery run. Apply Holm over finalists at FWER .05, plus non-inferiority on the secondary metric.
4. A stacked finalist needs a leave-one-out ablation before attribution.

| design | P(false promotion) | power at δ=1/2/3σ | GPU runs |
|---|---|---|---|
| current | .34–.38 | .07/.24/.52 | ≈67 |
| **α .20, n_c 6** | **.043** | .17/**.71**/.94 | 89 |
| α .10, n_c 6 | .045 | .14/.59/.87 | 83 |

The screen is the bottleneck: n_c=8 adds only .04. At σ ≈ 2.5 points, we detect about 5 points, not 3.

**Paired method.**
- Fixed checkpoints or eval-only transforms: exact McNemar on discordant pairs, with a paired bootstrap CI stratified by hop (bpb: resample val windows).
- Trained arms: the run is the unit (Welch). Pooling items across seeds into one McNemar compares weight files, not architectures (Dietterich 1998; Bouthillier et al. 2021).

## 4. Votes

**D1.** Keep W=10. On the new run's 60-min row:
- pass iff hop0 ≥ .30 (≥5 SE above the floor at n=100);
- flag the structural route if hop8 − hop2 ≥ .15 while hop0 < .30;
- if hop0 < .20, stop, start the controls (`looped` first), and warm-start from the first checkpoint that passes X1;
- if hop0 passes but pair₂ ≤ .25, run X2.

**D2.** Adopt R everywhere (renamed baseline lane) once step 0 reproduces the base; the old recipe measures shock recovery and is not a control.

**D3.** No ratio.
- Q = mean pair accuracy over hops 2/4/6; κ_pair = (Q−.24)/.76.
- F = W + KV₅₁₂ + A₁, measured (debater 2).
- Rank by κ_pair within ±10% F bins, ties by bpb. Plot the frontier with CIs.
- Gate: κ_pair's 95% lower bound > 0.

**D4.** As in §3.

**D5.** The matched from-scratch `looped` run is the right reference; base-2h had v1 data and 2 restarts.
- Replace control-10L (over budget) with a second `looped` seed. Its A/A gap is the yardstick for every single-run comparison.
- Give every 2-h checkpoint the same 10-min cooldown.

**D6.** A zero-init gated warm start of control-4L-2h: 2 seeds, only if it passes the pair gate. No dedicated base. The paper may say only that a gated restart, as TML's blog describes it, was reimplemented at 7M. It may not compare with or infer anything about Inkling 2.

**D7.** Three lanes, so Holm loses less:
- `signal`: answer weighting and mix; primary pair_id.
- `carry`: gated carry, DEPTH_ATTN, HISTORY; primary pair_id.
- `exit`: clean-gradient LOOP_CONSIST plus decode-faithful EXIT_TAU; primary mean loops, under bpb TOST ±.005.

Drop KV_SHARE (+.58 bpb), HALTING and CHUNK. Quantization and the KV tricks are paired evals, not lanes. Without lookup, run only bpb-primary lanes.

**D8.** Fix all four before the baseline seeds, in this order:
1. mean_loops (it feeds "cheaper");
2. KV at the evaluated loop count;
3. the peak_mem reset (it feeds validity);
4. latency (paper only).

Also add a harness-commit ledger column, the holdout set, pair_id and the structural floor.

## 5. Top three (pre-registered)

| # | slot | experiment | criterion | GPU-min |
|---|---|---|---|---|
| X1 | 14:30 | Decoy-pair audit of base-v2, base-v2-plain and the v1 base: 1,000 pairs per hop, h∈{0,1,2,4,6,8} | **Lookup:** pair₀ ≥ .30 (Wilson lower bound > .20). **Chasing at depth h:** pair_h lower bound > .25. **Structure:** ≥50% of wrong single answers at h∈{4,6} name the decoy root (chance .16/.24). No lookup ⇒ freeze the reasoning lanes. | 12 |
| X2 | 14:30, if pair₀ passes and pair₂ ≤ .25 | 5-min warm starts, 2 seeds each: W=10 vs W=10 plus ×3 on first-query answers | Δpair₂ ≥ .08 in both seeds (McNemar p < .01) and Δbpb ≤ +.005 ⇒ the controls adopt it | 30 |
| X3 | after the controls | Cooldown, then holdout eval of looped s0/s1, 4L, chunk2, chunk4: 2,000 pairs per hop + bpb | Looping helps iff Δpair_id(looped−4L) > 0, McNemar p < .01 for **both** seeds, and it exceeds 2× the s0–s1 gap. Chunk: TOST vs looped (pair ±.03, bpb ±.01). | 75 |

Then 6 baseline seeds + 3 A/A seeds (≈70 min), the swarm, and confirmation (≈27 runs, ≈200 min).
