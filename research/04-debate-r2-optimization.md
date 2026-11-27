# 04 · Debate round 2: optimization and learning dynamics (Debater 3)

*2026-11-24, 12:05. Inputs: the v2 generator, `reason_train_v2.bin` and the live `base-v2_progress.tsv`, analysed in pure numpy. No model was run. `research/04-debate-r2-floors.py` reproduces every [local] number.*

## 0. New evidence [local]

**N1. Task v2 has a query-blind shortcut at hops ≥ 4.** The chain is the deepest path. Distractors never attach to the chain, so it is also the largest tree. Several rules never read the query, yet score:

| hop | position floor | plurality value | deepest variable's value | last statement's value |
|---|---|---|---|---|
| 2 | .138 | .015 | .100 | .188 |
| 4 | .128 | .493 | .925 | .482 |
| 6 | .145 | .840 | .999 | .760 |
| 8 | .163 | .985 | 1.000 | .935 |
| 10 | .193 | 1.000 | 1.000 | .983 |

*Official sets, 400 problems per hop.*

These rules need value propagation but no lookup. A model with QI = 1 can reach `reason_ood` ≈ 1.0 and `reason_id` ≈ .70. The training data's query-blind optimum is the plurality value (.314 per answer), so plain CE rewards this circuit. Only hops 0–2 test lookup; at hops 0–1 every query-blind rule scores ≤ .13.

**N2. Answer bytes.**
- Answer digits are 12.4% of reasoning bytes and 3.11% of targets.
- 23.6% of them sit in a window's cut-off first line, whose statements are missing. They are unanswerable noise.
- Usable answers are 2.37% of targets.

**N3. Depth.**
- Training answers are 44% depth 0, 29% depth 1 and 4.6% depth ≥ 4.
- Given the line's earlier answers, 87% need at most one pointer step. Eval prompts reveal no earlier answers.

**N4. What lookup is worth.** A line holds at most 214 bits (1.33 bpb) of irreducible entropy. Answer CE is:
- 3.13 bits guessing uniformly over roots;
- 2.75 bits at the query-blind optimum;
- 0 with lookup.

So 88% of the reducible answer loss needs the query. Each answer bit moves `train_bpb` by 0.031, so perfect lookup is worth −0.097.

**N5. base-v2 at 15 min** (597 steps, 1.50 s/step; the base ran at 1.28):
- Hops 0/1/2 score .13/.08/.13, mean .113, below the .147 floor of those exact problems.
- `val_bpb` is 1.5440, +0.033 over `base`. That is the cost of re-warming to 10x the base's final LR.
- `train_bpb` at step 500 (1.568) is consistent with no lookup (predicted 1.56–1.59).

## 1. Rebuttals

**Debater 1.**
- *Disagree:* F2 as a 5-minute warm start.
  - In calibration.md, answer-only loss stayed at chance at 316–517 steps, and associative recall reached 23% only after 3,000 steps. A 5-minute null is uninformative, so F2 must be a base run.
  - Letters also don't "dominate the gradient". By step 500 they sit at their entropy (N5), so their gradient is zero-mean. Weighting competes with *text* for Adam's per-parameter step, which is why W must stay bounded.
- *Accept:* QI as the gate, and F5's linear probes over coda refits.

**Debater 2.**
- *Disagree:* §2's Q = (Ā − 0.1)/0.9 over hops 2–10. On v2, a query-blind propagator gets Ā = .82, so Q = .80 (N1). The ratio form would then pay for shrinking memory under a prior.
- *Accept:* the KL exit loses FLOPs (0.53 loop per check), and anything that changes step time changes the step count. R therefore also takes compile steps out of the budget.

**Debater 4.**
- *Disagree:* treating σ_seed ≈ 2–4 points as a constant of the harness. Most of it is the legacy re-warm (N5: +0.033 bpb from a 10x re-warm). Measure σ under R before setting thresholds.
- *Accept:* κ against the floor, paired tests, BH screening and a fresh eval seed. N1 extends their floor to query-blind propagation.

## 2. D1: the answer-weighted loss

```python
from prepare import REASONING_MIX
ANSWER_W = 10.0   # loss weight on answer digits; must equal the weight the base was trained with

def answer_weights(batch):
    """(B, T): ANSWER_W on targets answering '?<letter>=' in reasoning rows whose line starts in the window."""
    x = batch[:, :-1]
    B, T = x.shape
    q = mx.pad((x[:, :-2] == ord("?")).astype(mx.int32), [(0, 0), (2, 0)]) > 0    # x[t-2] == '?'
    seen = mx.cumsum((x == ord("\n")).astype(mx.int32), axis=1) > 0                # line's statements visible
    rows = mx.arange(B)[:, None] < max(1, round(REASONING_MIX * B / 32))           # prepare.Batches layout
    return 1.0 + (ANSWER_W - 1.0) * (q & (x == ord("=")) & seen & rows).astype(mx.float32)

# Inkling.loss(), replacing the ce line (all array ops, so it compiles):
tok = nn.losses.cross_entropy(logits, batch[:, 1:])     # (B, T)
ce = tok.mean()                                         # logged train_bpb stays unweighted
w = answer_weights(batch)
obj = (w * tok).sum() / w.sum()                         # normalised: gradient scale and clip 1.0 unchanged
return obj + ponder + LOOP_CONSIST * consist, ce
```

**What the mask covers.**
- Only digits after `?<letter>=`. Statement right-hand sides after `=` are irreducible (random digits and parents), so weighting them only multiplies noise.
- Text rows are excluded, because WordPress `?p=123` matches the pattern.

**Weight: W = 10.** That gives answers one fifth of the loss.

| W | answers' share of the loss | text weight kept |
|---|---|---|
| 1 | 2.4% | 75% |
| 4 | 8.9% | 70% |
| 8 | 16.3% | 64% |
| **10** | **19.6%** | **62%** |
| 16 | 28.0% | 55% |
| 30 | 42.2% | 44% |

The incentive ratio (lookup gain against query-blind gain, about 7:1 by N4) does not depend on W; W raises the absolute pressure.

**Not `REASONING_MIX`, instead or as well.**
- It is a fixed harness constant.
- An all-reasoning batch reaches only 9.5%.
- Mix 16 reaches 4.7% and cuts text rows by a third.

**Decision rule** (quick-eval sets, seed 7, paired across rows).
- **Statistic:** A = mean of hops 0, 1 and 2.
- **Floor:** F = .147, the per-hop maximum over the N1 rules (.11, .13 and .20). The SE is .02.
- **Ignored:** hops 4–8. Their query-blind ceilings on these sets are .47–1.0.

| run | checkpoint | condition | action |
|---|---|---|---|
| plain base-v2 | first row | A − F ≥ +.05 | keep |
| plain base-v2 | first row | otherwise | restart immediately |
| weighted run | 30 min | A ≥ .25 | lookup is forming |
| weighted run | 60 min | A < .20 | escalate: `--resume --set ANSWER_W=30.0` for the remaining hour |
| weighted run | 2 h | A < .25 | no reasoning lanes; the swarm runs bpb and efficiency lanes only |

**The 15-min row gives A − F = −.034, so restart now.** Waiting costs 15 min per row, and 17k v1 steps formed no lookup.

**Restart procedure.**
1. Stop the campaign.
2. Rename `base-v2.{safetensors,json}` and `base-v2_progress.tsv` to `base-v2-plain*`. Its 15-min row becomes the matched W = 1 arm.
3. Commit `ANSWER_W = 10.0` as train.py's default, so the base, the controls and the warm starts share one objective.
4. Relaunch. The campaign is idempotent and re-runs `--init base --lr 5e-4`.

## 3. D2: recipe R

```python
RECIPE = "R"            # warm starts only (main()); "legacy" = old cosine re-warm. pretrain_base.py is unaffected
R_LR = 1e-4             # base tensors: 4x base-v2's final LR (2.5e-5)
R_LR_NEW = 1e-3         # new matrices (absent from the base checkpoint; load_base does NEW_KEYS.update(new))
R_LR_NEW_VEC = 1e-2     # new vectors and gates; no weight decay on any new tensor
R_WARMUP, R_DECAY_FRAC = 10, 0.25   # 10 steps up, constant, then linear to 0 over the last 25% of wall clock
NEW_KEYS = set()

# train(..., recipe=False); main() passes recipe=INIT_FROM_BASE and RECIPE == "R"
groups = ((R_LR_NEW_VEC, 0.0), (R_LR_NEW, 0.0), (R_LR, WEIGHT_DECAY))
new = lambda p, a: p in NEW_KEYS
opt = optim.MultiOptimizer([optim.AdamW(lr, betas=[0.9, 0.95], weight_decay=wd) for lr, wd in groups],
                           filters=[lambda p, a: new(p, a) and a.ndim <= 1, new])
opt.init(model.trainable_parameters())          # before mx.compile captures opt.state
# per step (MultiOptimizer's own setter would give every group one LR):
s = min(1.0, (step + 1) / R_WARMUP) * min(1.0, (1 - progress) / R_DECAY_FRAC)
for o, (lr0, _) in zip(opt.optimizers, groups):
    o.learning_rate = lr0 * s
# both recipes: fresh = (k, c) not in steps_fns; count dt only if step >= 2 and not fresh
```

**Adopt R before the baseline seeds: yes.**
- The baseline seeds are the reference for every verdict. Running them under the legacy recipe and switching later wastes them.
- The legacy schedule peaks at 6.5e-4, 26x base-v2's final LR, and stays above 3e-4 for 126 of 237 steps.
- N5 shows what a 10x re-warm costs.

## 4. Votes

- **D1. Restart now with W = 10** (§2).
- **D2. Adopt R before the seeds** (§3).
- **D3. A difference, not a ratio.**
  - κ_h = (acc_h − f_h)/(1 − f_h), with f_h the best N1 rule on that set.
  - Q = mean κ_h over hops with f_h ≤ .5.
  - **S = Q − 0.05·log₂(F/F_base)**, where F = weight bytes at the evaluated precision + KV₅₁₂ at the evaluated loop count. Halving memory is worth 5 κ-points, and nothing explodes near Q = 0.
  - Gates: `val_bpb` ≤ ref + .02 and Q ≥ .10. The paper reports the frontier.
- **D4. Keep and promotion.**
  - Pooled baseline of 6 R seeds, with one pre-declared primary metric per lane.
  - Keep if ΔQ ≥ max(.05, 2.5σ̂) with Δbpb ≤ +.003, or if Δbpb ≤ −max(.003, 2.5σ̂) with ΔQ ≥ −.03.
  - The .05 comes from the paired MDE: 400 problems at each of hops 0–2 (d ≈ .19) give ≈ 4 points.
  - Promotion: 3 fresh seeds plus a fresh eval seed, one-sided Welch p < .05, Holm across lanes.
- **D5. Controls.**
  - Use the base's `ANSWER_W` and `--hours 2` (annealed). Unannealed 6-hour schedules stopped at 2 h confound architecture with LR noise.
  - "looped" is the right reference.
  - Keep 10L, last, labeled over budget.
- **D6. Zero-init gate on `control-4L-2h`** (X3). It is bitwise LOOPS=1 at step 0. Fund its own base only if X3 passes.
- **D7. Lanes.**
  - `recipe`: R_LR ∈ {5e-5, 1e-4, 2e-4}.
  - `varloop`: X2.
  - `exit`: `LOOP_CONSIST` via zero-init per-loop exit adapters, stop-gradient into the core, judged on debater 2's decode meter.
  - `depth-attn`: only if the final base has A ≥ .25.
  - Skip `KV_SHARE` (+0.58 bpb; no fixed point), `HALTING` (too slow for 5 min), `HISTORY`/`LOOP_EMBED` (subsumed by X2's gate) and `CHUNK` (needs a base).
- **D8. Fix all now, before the seeds; they are eval-only.** In this order:
  1. add the N1 floor and hops 0–1 (400 each) to `evaluate()`;
  2. `mx.reset_peak_memory()` before eval;
  3. token-weighted `mean_loops` over all calls;
  4. KV at the evaluated loop count;
  5. compiled latency with the dtype labeled.

## 5. Top 3 experiments

With the restart, the 14:30 slot moves to about 14:50.

| # | slot | switches | pre-registered pass | GPU-min |
|---|---|---|---|---|
| X1. Recipe A/A | 14:50 | R seeds 3–5 (6 with the campaign's); `RECIPE="legacy"` seeds 0–1, lane `baseline-legacy` | R: \|Δbpb\| ≤ .002 vs base, SD₆ ≤ .0015, every hop ±5 pts. Legacy predicted ≥ +.005; if not, I withdraw the shock argument | 33 |
| X2. Varloop + carry gate | after X1 | k = 4 w.p. .5, else 2/3/5/6/8 w.p. .1 each; `self.carry_w = mx.zeros((D,))`, `x = e + (1 + carry_w)*h`; ±gate, 2 seeds | ‖h₈‖/‖h₄‖ ≤ 1.3 (v1 base: 2.0); bpb(8) − bpb(4) ≤ +.01 (v1: +.11); Δbpb at 4 loops ≤ +.004 vs baseline-R; if lookup exists, κ_q(8) − κ_q(4) ≥ +.04 | 28 |
| X3. Inkling-2-Mini gate | after controls | `INKLING_BASE=control-4L-2h`, `LOOPS=4`, `CARRY="gated"` (round-1 §2 sketch), gate 0 (R vector group). Arms: LOOPS=1 continued ×2, gated ×2, gate fixed at 1 ×1 | step-0 logits bitwise equal to LOOPS=1; gated beats LOOPS=1 by Δbpb ≤ −.004 (ΔQ ≥ +.05 if lookup exists); mean gate ≥ .1 | 33 |
