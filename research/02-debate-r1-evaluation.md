# 02 · Debate round 1 · Debater 4: evaluation / statistics adversary

*2026-11-24. Read-only: numpy and CPU-only MLX evals of cached checkpoints; no GPU, no tracked files edited.*

## 0. New evidence: a ~50% letter-blind floor, and the base sits on it

`_forest` puts the chain root at the minimum of h+1 random positions, so the root is the **first statement** with probability (h+1)/N.
Copying the first statement's digit, with no letter matching, scores **.40/.46/.49/.51/.52/.52/.53** at hops 0/1/2/4/6/8/10
(20k per hop), and **.511/.510** on the official reason_id/reason_ood.

Checkpoints at 4 loops (official set plus 400 problems each at hops 0 and 1):

| ckpt | reason_id | reason_ood | P(pred = first digit), hops 0→10 | acc where answer ≠ first (ID/OOD) |
|---|---|---|---|---|
| 45 min | .381 | .174 | .52 → .15 | .33/.16 |
| 2 h | .440 | .230 | .56 → .23 | .29/.15 |
| 5 h | .497 | .409 | .63–.72 (hops 0–6); .54/.39 | .28/.30 |
| 6 h (final) | .495 | **.340** | .49–.65; .37/.32 | .39/.32 |

- **The final base is below the floor** on ID and OOD. Hop 0, a single lookup, scores .38–.41 against a .375–.40 floor.
- **Accuracy tracks first-digit agreement** (r = .87 over 35 checkpoint×hop cells), so the "OOD climb from .11 to .50" is the shortcut
  being learned. From 5 h to 6 h, OOD fell 6.9 points (paired SE 1.5) as hop-8 agreement fell from .54 to .37.
- **Loops 1→4 at hop 8 (5 h):** agreement rises .13 → .54 and accuracy .17 → .47. Most of the looping gain is the shortcut.
- **Some structure is real.** At hop 8 (5 h), where the answer ≠ first digit, answers that avoid the first digit are right 61% of the
  time (a random non-first literal: 27%). Yet accuracy on that stratum (.23–.39) is near random-literal chance (.21–.38).
- **Format (5 h):** a leading `\n` shifts hops 0/8/10 by +3.2/−4.6/+9.3 points.

**Incident.** A reboot at 01:59 killed baseline seed 0 at step 150, before it wrote a row. The ledger is empty, and the campaign is not
running. My CPU evals (01:49:56–01:52:02) probably caused that run's slowdown (13.1k → 10.4k tok/s). Run the seeds on an idle machine.

## 1. Fund these five

| # | Minimal experiment | Pre-registered criterion | GPU-min |
|---|---|---|---|
| E1 | **Shortcut-controlled probe** (analysis script; `prepare.py` untouched). Grid: h 0–12 × N ∈ {7, 15, 23}. Root position uniform. 2,000 problems per cell. Loops {1, 2, 4, 8, 12}. κ = (acc − floor)/(1 − floor), where floor is the best simulated letter-blind rule (the first-digit rule gives ≈ 1/(N−h) + .1(1 − 1/(N−h))). Counterfactual twins: flip the root digit. | **Lookup:** κ_k ≥ .10, 95% CI > 0. **Loops buy depth:** κ(8 loops) > κ(4 loops), one-sided paired McNemar p < .01, at ≥ 2 hops ≥ 7, **and** κ falls with h at fixed N. | ≈4 per ckpt, ×8 |
| E2 | **Format robustness.** Prefixes: none, `\n`, a solved line + `\n`, 200 B of text + `\n`. | An effect must keep its sign under all 4 and exceed the format spread. | ≈8 |
| E3 | **Noise floor with A/A.** 6 baseline seeds at a **fixed step count** (the idle-machine median, ≈ 240), logging steps and tok/s. Also run seeds 3–5 through `verdict()`. | All A/A runs must be discarded (else the thresholds are too lax). Report σ with χ² CIs. | ≈42 |
| E4 | **Control hygiene:** E1 + E2 on every 2 h checkpoint, reported as single training runs. Disclose the confounds below. Stratify CHUNK by (len−1) mod K. | Report effects only above the floor. | ≈10 |
| E5 | **Confirmation** of ≤ 3 finalists: 6 fresh seeds each, 6 interleaved baseline seeds, a fresh eval seed, κ. | One-sided Welch t on a pre-declared metric, Holm at α = .05. Non-inferiority on the rest. E2-consistent. | ≈170 |

Confounds for E4:
- Plain-10L has 17.9M parameters, over the 12M budget, so `evaluate()` marks it invalid.
- `chunk2`/`chunk4` add 0.59M/1.18M parameters (+8%/+16%).
- base-2h had **two optimizer restarts** (at 45 and 75 min); the controls probably had none.

## 2. Statistical protocol

**Binomial arithmetic.** Assume p ≈ .5, two-sided α = .05 and power .8 (z-sum 2.80). One score has SE 5.0/2.5/1.8/1.1 points at
n = 100/400/800/2,000.
- **Unpaired:** n per arm = 3.92/δ², giving 4,356 / 1,568 / 392 for δ = 3 / 5 / 10 points.
- **Paired:** n ≈ (1.96√d + .84√(d − δ²))²/δ², with measured discordance d ≈ .19, giving ≈ 1,650 / 590 for δ = 3 / 5.
  - reason_ood (n = 800) has a paired MDE of **4.3 points**, above the 3-point keep threshold *even with zero seed noise*.
  - Eval-only lanes: paired McNemar with ≥ 2,000 problems per hop (problems are free).
- **Seeds:** MDE ≈ 3.0σ / 2.0σ / 1.5σ at 3 / 5 / 8 seeds per arm. Item churn alone gives σ ≥ 1.1 OOD points, and the 5 h → 6 h
  swing suggests σ_seed ≈ 2–4, so 3 seeds detect only effects of 6–12 points.

**Current rules, simulated.** Monte Carlo of `run.py` plus the campaign: 60 experiments, 4 lanes, global null, σ_ood 1.5–5 points.
- **False confirmations:** E[keeps] = 2.6–3.6, 2.3–2.8 of 4 lanes keep something, and P(≥ 1 lane "confirmed") = **.23–.28**.
- **Causes:**
  - σ̂ comes from 3 seeds, so "mean > 2σ̂" fires 6.7% of the time per metric (0.7% with σ known).
  - "On some metric" adds three chances, and the keep rule takes `max(d_id, d_ood)`.
  - Excluding the discovery run barely helps (.32–.37 lanes promoted).
- **Power:** a true +3 points (σ = 2.5) is confirmed with probability .28, and +5 with .50. A true −2-point loss passes the OOD slack
  57% of the time.

**Rules.**
1. **Screening (the swarm)** compares with a pooled baseline of ≥ 6 seeds, not the lane's last kept single run (a winner's-curse
   ratchet). It uses one pre-declared primary metric and Benjamini–Hochberg at q = .2. Screening supports no claims.
2. **Promotion** is E5, with the discovery run excluded.
3. **"Cheaper at equal quality"** needs TOST equivalence (bpb ±.005, κ ±.02), not "|Δ| < 1 SD".
4. **Fresh evaluation.** Sixty adaptive looks at one eval set inflate the best run by ≈ 2.3 SD under the null. Re-evaluate finalists on
   a fresh generator seed and publish the full ledger.

**The paper may claim:** the calibration result, restated against the floor; matched-time comparisons on κ and bpb (one training run
each, eval-only CIs, E4 confounds disclosed); E5-confirmed effects, stating the number of experiments tried; mean loops as analytical
compute.

**It may not claim:**
- serial depth or OOD generalization from hops 8 and 10 as currently defined;
- "leaves chance within 15 min (52% at hop 1)", since the hop-1 floor is .46;
- "intelligence per GB" as a ratio. Use a frontier over bytes, κ and bpb, with κ ≥ .1;
- anything about Inkling 2 beyond "an Inkling-2-style gate reimplemented at 7M from a blog post". There is no shared task, the N-Hop
  data is unreleased, and the numbers are self-reported with no paper.

## 3. Objections to the other debaters

**D1 (mechanism).**
- Landing, flip and logit-lens staircases on this base will mostly time the positional prior's arrival (hop-8 agreement: .13 at loop 1, .54 at loop 4).
- "Doubling vs. iterative" is unidentifiable while κ ≈ 0, so run E1 first.
- Probes need selectivity controls (Hewitt & Liang 2019) plus causal twins or patching.
- After the H4 refit, landing describes the refit coda, not the base.

**D2 (efficiency).**
- "Accuracy holds" after W4, KV sharing or an exit proves nothing: the positional rule alone scores .51, and the little real lookup can vanish unseen.
- Test non-inferiority on κ and bpb; the per-hop paired MDE at 400 problems is ≈ 6 points.
- CHUNK needs a bytes frontier and alignment stratification.
- Latency needs idle-machine medians with CIs; today's load average was 60–86.

**D3 (optimization).**
- H3 changes the null, so re-measure σ before any lane adopts it.
- The −6.9-point OOD swing ran through the shortcut mixture, so 5-minute fine-tunes can clear the keep bar without changing lookup.
- A reasoning-weighted phase can sharpen that same prior.
- H5 needs matched mean loops **and** matched steps; under load, seed 0 managed only 150 steps in 227 s.

## 4. Wrong or overstated in 01-landscape.md

1. TL;DR 2, O1, O2 and B2 say "single-lookup precision p ≈ .5 binds". The flat profile *is* the floor, and no reliable lookup is shown.
2. "Guessing among 3 digits ≈ 1/3": it is .40, because digits coincide.
3. "Chance is 10%" (in `prepare.py`, `setup.tex` and Table 1) holds only for a model that cannot read digits.
4. §4's SEs are single-model; comparisons need a paired-difference SE plus seed variance.
5. H1's rule of "≥ 10 points at hop 0" misses hop-specific shifts of −4.6 to +9.3 points.
6. H8's pre-test (≥ 3 points at hop 0, 3 β values vs. β = 1) is within noise (paired SE ≈ 2.2): ≈ 20% risk of a false lead.
7. B3 is incomplete: OOD also changes line length (79 and 95 bytes, against ≤ 63) and the root-position prior.
8. §1.5a's "Inkling-2-Mini recipe" cannot be verified without a paper or code.
9. `paper/generated/numbers.tex` and Fig. 1 come from a 1.3 h snapshot on 100 problems per hop: stale, ±5 points.

*Reproduce:* stratify `pred == gold` by `gold == prob.split(";")[0][2]`. The scripts were lost in the reboot.
