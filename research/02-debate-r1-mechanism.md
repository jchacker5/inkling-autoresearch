# Debate round 1: Debater 1 (mechanism skeptic)

*2026-11-24. Evidence comes from CPU-only, read-only probes of the base snapshots; `research/02-debate-r1-probe.py` reproduces every number, and the GPU was not used. Accuracies replicate `prepare._reasoning_acc` on the same problem sets.*

## 0. The finding that reorders the debate: the base never reads the question

**QI** is P(argmax unchanged when only the question changes). The swap target is a depth-0 literal with a different digit, so any real lookup must change the answer.

| Probe | 45 min | 2 h | 5 h | 6 h |
|---|---|---|---|---|
| QI, hop 0 (all 3 variables asked) | .93 | .95 | .94 | .93 |
| QI, hop 2 / hop 6 | .98/.99 | .97/.98 | .98/.96 | .975/.975 |
| `reason_id` / `reason_ood` | | | .497/.409 | .495/.340 |

- **"Copy statement 0's digit" scores `reason_id` .511 and `reason_ood` .510** on the fixed sets, at or above the 6-h base on every reasoning metric. The effective floor is about .5, not 10%.
- When the chain root is elsewhere, the 6-h base still outputs statement 0's digit .48–.57 of the time at hops 0–6. The 5-h base did so .60–.73 of the time.
- **Training format** (7,126 teacher-forced answers in 512-byte windows): accuracy by the queried variable's depth *rises*, from .48 at depth 0 to .72 at depth 6. First-query direct lookups score .37. The model emits a query-free "line value" (the plurality chain value plus a line-start prior). The eval rewards that because it always asks about the deepest chain.
- **Format**: a `"\n"` prefix lifts hop 10 from .29 to .45, while hop 0 stays at .40–.45 under any prefix. `reason_ood` is dominated by how position 0 is handled.

So `reason_*` measures a prior. No lane's reasoning delta can be interpreted until a checkpoint passes the gate in §4.

## 1. Five hypotheses to fund

**F1. The query-invariance gate.** This is the kill experiment (§4).

**F2. Binding needs gradient: an answer-weighted loss.**
- **Claim.** Answer digits are about 3% of all targets (1/8 of reasoning-row bytes × 8/32 rows). Random letters (about 4.6 bits each, per calibration.md) dominate the reasoning gradient, so a query-free prior is the cheap optimum.
- **Test.** A 5-min warm start on H3's recipe (LR 2e-4, 10 warm-up steps). Weight W∈{10, 30} on the targets that follow `?<letter>=` in rows `[0, mix)`. That is about 8 lines in `loss` (train.py:290), and the data are untouched.
- **Pass, in 3/3 seeds:**
  - QI falls by ≥.10 (snapshot-to-snapshot drift is ≤.025);
  - depth-0 first-query accuracy rises by 10 points (SE 2.3);
  - Δbpb ≤ +0.02.
- **Fail.** QI ≥.95 in all seeds means binding can't be bought in 5 minutes; spend a base on it instead.
- **Cost.** About 50 GPU-min: 2 arms × 3 seeds, sharing H3's W=1 control.

**F3. The chunked core can't see the question at the eval's alignment.**
- **Claim.**
  - The unpool shifts by one chunk (train.py:180–185), so the answer byte only receives core state from earlier chunks.
  - Eval prompts are 4n+3 bytes starting at byte 0. At K=4, `?v=` therefore sits alone in the last chunk, and only the 1-block coda can condition on the question.
- **Already checked on tiny random models.**
  - At K=4, the core's contribution at the answer byte is identical across query letters (Δ=6e-7) at left pads 0, 1 and 3, and differs only at pad 2. K=2 sees the question at pad 0.
  - Across training alignments the core is blind 3/4 of the time at K=4 and 1/2 at K=2.
  - Comparing K=4, K=2 and K=1 on the fixed eval is therefore confounded by alignment.
- **Test.** Eval-only, on the chunk bases.
  - Prefix `"\n"·p` with p∈{4..7}, paired on 2,000 problems.
  - "The core does query work" holds only if K=4 accuracy at p≡2 (mod 4) beats the other pads by ≥5 points (McNemar p<.01).
  - Also run F1 on each chunk base.
- **Cost.** About 8 GPU-min.

**F4. A decode-faithful exit.**
- **What the code does.**
  - In `recur`, frozen tokens still run both core blocks (train.py:230–231) and are only overwritten afterwards (:236).
  - Their block-2 K/V are recomputed every loop from block-1 outputs that attended to active tokens. So they keep relaying fresh information, while `expected` (:235) books them as idle.
  - A KV-cached decoder copies the exit-loop K/V and gets none of this.
- **Test.** A strict emulation, about 15 lines: a per-token mask into `Attention`, with `k,v = where(active, k_new, k_exit)`.
- **Pass (the leak is negligible).** |Δbpb| ≤0.002 at every τ in `landing.TAUS`.
- **Scope and cost.** Text only until F1 passes; about 10 GPU-min.

**F5. Linear probes, not a coda refit.**
- **Why a refit can't answer the question.** H4 refits `coda`, which is a full Block with attention (train.py:191) and can do the missing hop itself. Better early readouts would therefore not prove the loops held the answer.
- **Test.** Per-loop affine probes on the residual stream:
  - the answer position → the digit;
  - each statement position → its resolved value, by depth × loop.
- **Pass.** The answer is "present at loop r" if held-out accuracy is within 2 points of the final readout (fit on 4k problems, test on 2k).
- **What else it gives.** It separates "resolves values but can't look one up" from "resolves nothing". It also gives the iterative-vs-doubling staircase without going through the off-distribution coda.
- **Cost.** About 5 GPU-min, or CPU.

## 2. Objections to the likely champions

**D2 (CHUNK, exits, KV sharing, W4).**
- CHUNK=4's `down`/`up` add 1.18M parameters (+16%; +8% at K=2). At probe length (about 95 bytes) memory is 16.7 MiB against 15.3 MiB. It is smaller only above about 256 bytes, and by 11% at 512.
- Exit savings are optimistic (F4). The KL metric also adds a coda-plus-head pass per loop (:201).
- Quantization or KV sharing "keeping accuracy" is trivial on a query-free prior. Gate those claims on F1.

**D3 (H3, H5, H6, H4).**
- **H3 is necessary but misdescribed.** Warm-up counts steps while the cosine runs on wall clock (:349). At about 1.5 s/step the LR peaks near 6e-4 at about 45% of the budget (`runs/20261124-014816-baseline`: 6.12e-4 at step 100, 1.88e-4 at step 150).
- **H6** (`LOOP_CONSIST`, :293–298) doesn't stop the gradient into `states[r]`, so it rewrites the core. On the probe it distills a shallow prior into the early loops, which makes exits look free for the wrong reason.
- **H5**'s extrapolation claim can't be measured while QI is about .97.
- **H4** has the confound described under F5.

**D4 (evaluation).**
- Binomial SEs (±1.4–1.8) understate drift. From 5 h to 6 h, `reason_ood` moved .41→.34 while query use stayed the same. That is more than twice the .03 keep threshold, so the keep rule will select shifts in the prior.
- Use paired tests on identical problems, and report against the .51 statement-0 floor rather than 10%.
- Run F1 on the plain-4L/10L controls before claiming that looping helps reasoning.

## 3. Wrong or overstated in the report

1. **§0.2 and O2, "the binding constraint is single-lookup precision".** There is no lookup: QI is .93–.99 at every snapshot. "≈1/3 by guessing" is really .39 once coinciding digits count, and copying statement 0 gives .41 at hop 0.
2. **O2(i), "a metastable average over similar keys".** Names are sampled without replacement (prepare.py:64), so each query letter has exactly one defining statement. The outputs ignore the query; they aren't blends.
3. **O1, "hop 8 climbed to ~0.50".** That is the prior lining up with the chain value, not OOD progress. The 6-h base scores .39 at the official hop 8, against .495 for the heuristic.
4. **calibration.md, "retrieval emerges within 15 min".** Refuted: QI is already .93/.98/.99 at 45 minutes.
5. **The H1/§3.2 rule "loops=2 flat at hop 8 ⇒ doubling" misfires.**
   - A query-free prior can be flat across loop counts.
   - The sweep reads out through the coda, which the report itself calls off-distribution (W2).
   - Measured: hop 6 goes .23→.52 from 1 to 3 loops. After that, the statement-0 component decays (.55→.42 at 8 loops) while the rest rises (.46→.52). Loops reshape a prior.
6. **W3/H2, "a KL exit freezes intermediates".** Only half true in this emulation (F4).
7. **Landing and TRACE statistics on the probe** currently describe a heuristic's dynamics, not overthinking.
8. **§3.5's memory tables** leave out CHUNK's parameter overhead.

## 4. Kill experiment (run first): the query-invariance audit

**What to measure.**
- QI on the official hop-2/6 statements.
- QI at hop 0 with all three variables asked.
- Depth-0 first-query accuracy in training format.

**Where to run it.** Through the GPU eval path, on the final base, the 3 baseline seeds, the plain controls and the chunk bases. Each takes about 3 minutes, or 20 s on CPU. It is deterministic, so it needs no seeds; the SE is about 1 point at n=400.

**Pre-registered outcomes.**
- **QI ≥ .90** (the CPU run says .975):
  - freeze every lane that claims a reasoning gain: depth-attn, halting, history, probe-based overthinking, test-time scaling, H8, H12, H18, H19;
  - report `val_bpb` and memory only;
  - run F2, and if it moves QI at all, swap tonight's CHUNK=2 base for an answer-weighted base.
- **QI ≤ .50:** my CPU result is wrong, and the report's order stands.
- **In between:** run F5 first.
