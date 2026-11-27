# 04 · Debate round 2 · Debater 2: efficiency and systems

*2026-11-24, 12:20. No forward passes were run. The evidence is pure Python on `prepare.py`'s generator and byte arithmetic, and `research/04-debate-r2-efficiency.py` reproduces every number. Task v3 (5539a61) landed while I wrote this; the paper is updated to it.*

## 0. Independent confirmation of the v2 shortcut, and a check of v3

I scored rules that never read the query on the official **v2** set, before v3 existed:

| hop | harness floor | resolve last statement | root of deepest tree |
|---|---|---|---|
| 2 | .138 | .188 | .201 |
| 4 | .128 | .482 | .936 |
| 6 / 8 / 10 | .145 / .163 / .193 | .760 / .935 / .983 | .999 / 1.000 / 1.000 |

This agrees with the evaluation debater: on v2, `reason_ood` could not show lookup.

On **v3's** eval set, the same five rules score at most .50 per prompt and at most .25 per pair (.025 at hop 2, .235 at hop 4, .250 at hops 6–10). So `PAIR_BOUND` is exact at depth and conservative at shallow hops.

## 1. Rebuttals

**Debater 1.**
- *Disagree* with the CHUNK memory argument (§2). The arithmetic is right: 16.72 vs 15.35 MiB at v3's 100 bytes. The reference is wrong.
- control-4L stores chunk4's 6 KiB/byte of KV with 1.18M fewer parameters. **It is smaller at every context**: 16.9 vs 19.1 MiB at 512 bytes, and 4.9 vs 5.5 deployed.
- 4L also does less decode work (4.0 vs 4.17 block applications per byte) and has less prefill depth (4 vs 10.3). Chunking cuts FLOPs, not depth, and B=1 prefill is dispatch-bound.
- So CHUNK must beat 4L on quality. Given your F3, even a loss would not be a capacity verdict.
- *Accept* F1 (QI) and F4: exits save decode time only with exit-loop K/V reuse and a cheap state metric.

**Debater 3.**
- *Disagree* with the pass rule of I. Arm b costs 10 block applications per byte and 15 KiB of KV, against a's 4 and 6, so it wins by buying 2.5x the compute. Its opponent is looped-2h + 5 min, which sits in the same cell.
- *Accept* E8 (train in fp32, cast at eval), and accept comparing the unannealed controls only with each other.

**Debater 4.**
- *Accept* the ratio objection; I withdraw IPG = Q/F. One of my round-1 checkpoints scored 19/GB in bf16 and 73/GB deployed, a 3.8x gain from a storage format alone. Capability is concave in bytes, so a ratio always crowns the smallest model, and near the floor it is noise divided by F.
- *Disagree* with E3's fixed step count. The swarm runs on wall clock, so the 200–255-step spread is real keep-rule noise, and fixing steps shrinks σ̂. Calibrate at fixed wall clock on an idle machine, with steps as a covariate.

## 2. D3 score and D8 patch

**Each model is scored on (κ̄, F, b, cell).**
- **κ̄** = mean over h ∈ {2, 4, 6, 8, 10} of (pair_h − .25)/.75, read from v3's `reason_by_hop` and `PAIR_BOUND`.
- **F** (MiB) = 2·params + KV₅₁₂.
  - params counts every tensor, including CHUNK projections, adapters and gates.
  - KV₅₁₂ = Σ over cached layer applications of 1,536 B·min(512, window)/K. K applies to core applications only. KV is counted at κ̄'s loop count; 8 loops cost 27 KiB/byte.
  - Activations (<0.1 MiB) and the runtime are excluded.
  - F_deployed = measured nbytes after a transform that passes TOST: b within ±.005 for 8-bit and KV, ±.02 for 4-bit, and κ̄ within ±.03.
  - Also report F at T = 100.
- **b** = `val_bpb`.
- **cell** = (D, S).
  - D = decode block applications per byte: PRELUDE + CORE·L/K + CODA, +0.17 for CHUNK, +1.06 per KL-exit check.
  - S = prefill depth.
  - Two models share a cell if both D and S agree within ±10%. Measured B=1 µs/byte overrides the proxies.

**Gates.** A model that fails any gate is scored on (F, b) only.
- params ≤ 12M (10L is a reference, not a contestant).
- b ≤ b_ref + .02.
- κ̄ ≥ .10, with the 95% lower bound > 0.

**Score.**
- Within a cell and ±10% F: paired Δκ̄, plus TOST on b (±.005).
- Across cells: dominance only.
- Headline, the excess over the plain-transformer frontier in κ units:

  **E_m = κ̄_m − κ̄_4L − β̂·log₂(F_m/F_4L),  β̂ = max(0, (κ̄_10L − κ̄_4L)/log₂(F_10L/F_4L))**

  The CI comes from a joint paired bootstrap over pairs. E is not a ratio, and its exchange rate is measured.

**D8 patch, in priority order.** The ledger is empty, so no logged number changes.

| # | Change | Logged values |
|---|---|---|
| P0 | `run.py:21,173`: `harness` (prepare.py sha1) and `base` columns. `:42` pools matching rows only | none |
| P1 | `prepare.py:34`: `EVAL_PER_HOP` = 800 (400 pairs). Assuming pair discordance d ≈ .2, the item-noise MDE on κ̄ drops from .047 to .033, for a few seconds of eval | none |
| P2 | `train.py:315` keeps `last_expected` (B,T). `prepare.py:249,287` average it over text and over answers. `:335` and `run.py:109` read the averages | fixed loops unchanged |
| P3 | `train.py:196`: `kv_bytes_per_token(loops=None)`. `prepare.py:343–347` logs KV per k | 15.0 unchanged |
| P4 | `prepare.py:321`: read the training peak, then `mx.reset_peak_memory()`. `:357` reports it (20 GB gate). Add `b1_peak_mib` around `:351` | 8.1 unchanged |
| P5 | `prepare.py:299–306`: compile `model.forward(x)[0]` (`__call__` calls `.item()`). T ∈ {100, 512}, median and p90 of 30, renamed `prefill_ms_*` | none |
| P6 | `prepare.py:322,361`: reject packed uint32 parameters; add `weights_mib_stored` | none |

## 3. Votes

- **D1.** Accept v3's weighting (ANSWER_W=3, FIRST_ANSWER_W=30).
  - On `reason_train_v3.bin`, first answers carry 3.5% of the loss, other answers 8.1% and text 68%.
  - base-v2 restarted on v3 data at 12:16.
  - At 60 min, lookup means pair accuracy ≥ .10 on hops 0–2, where the floor is ≤ .03. If it is not there, let the run finish anyway, and Exp 1 decides the controls.
- **D2.** Use R at 1e-4 for the three baseline seeds, since they set every threshold. Drop the old-recipe and 3e-4 arms.
- **D3.** As in §2.
- **D4.**
  - **Screening:** compare against 6 pooled baseline seeds (3 now, 3 interleaved). Keep if Δκ̄ ≥ max(.035, 2.2σ̂) with Δb ≤ max(.004, σ̂_b), or if Δb ≤ −max(.004, 2.2σ̂_b) with Δκ̄ ≥ −.02. Here 2.2 = t₀.₉₅,₅√(7/6), and .035 is the item MDE after P1.
  - **"Cheaper":** F or D ≤ 0.8x, plus TOST.
  - **Promotion:** 5 vs 5 interleaved seeds, one-sided Welch, Holm over at most 3 finalists, fresh pairs.
  - **Eval-only transforms:** paired tests, no seeds.
- **D5.** Keep 10L if Exp 1 passes, since it fixes β̂. Drop chunk2, which 4L dominates on F, D and S. The reference is the matched looped-2h model, never base-v2.
- **D6.** After the controls, run 2 seeds against looped-2h + 5 min. No dedicated base unless it lands within ±.03 κ̄.
- **D7.** Lanes:
  - `recipe`: R, an answer-weight grid, RANDOM_LOOPS.
  - `kv`: a trained core window of 64 or 128. KV_SHARE must beat eval-only SWA-128 + int8.
  - `loops`: LOOPS = 3 or 2, plus V.
  - `exit`: strict K/V reuse, a cosine state metric and adapter LOOP_CONSIST, judged on D.

  HALTING, DEPTH_ATTN, HISTORY and LOOP_EMBED wait until κ̄ passes. The controls decide CHUNK.
- **D8.** Land everything in one commit before base-v2 ends (~14:20), or the baseline seeds must be re-run.

## 4. Top three experiments

| # | When | Run | Pass / fail | GPU-min |
|---|---|---|---|---|
| 1 | ~14:20, before seeds and controls | `eval_ckpt.py --name base-v2 --no-sweep`: the gate on the v3 base | Pass (κ̄ ≥ .10, lower bound > 0): controls as planned. Fail: run only looped and 4L (text/systems, 4 h) and give the other 6 h to continuing base-v2. There is no per-GB reasoning claim until some model reads the query | 4 |
| 2 | after controls | Patched `evaluate` on looped/4L/chunk4/10L/base-v2 in bf16, then W8/W4, int8 KV, SWA-128 | CHUNK survives only if κ̄(chunk4) − κ̄(4L) ≥ .05 (paired p < .05) and b ≤ b_4L + .005. Looping wins per GB only if E_looped > 0 with CI > 0, in bf16 and deployed | 35 |
| 3 | after controls | Compiled bf16 B=1 meter: prefill at T = 100 and 512, plus decode with one KV cache per layer application, on the controls, fixed-3 and a strict state exit | Predicted: chunk4 prefill(100) ≥ 0.9x looped, 4L ≤ 0.55x. Keep an exit only at ≤ 0.85x fixed-4 µs/byte **and** below fixed-3, with Δb ≤ +.005 | 20 |
