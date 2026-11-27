# 02 · Debate round 1 · Debater 2: efficiency and systems

*My stance: count what an M4 deployment actually pays, in bytes and µs, not FLOPs or masked loops. Tags: **[local]** = CPU-only, read-only checks on `base.safetensors`. The quality figure is zero-shot KL to the fp32 base at 4 loops, measured on 256–1,024 val bytes; Δbpb ≈ KL/ln 2 is a prediction. **[est]** = my estimate.*

## 0. Measured first [local]

| Check | Result |
|---|---|
| Weight bytes, scales included | **fp32 (today) 27.76 MiB** · bf16 13.88 · W8 7.38 (7.82 if quantized before the bf16 cast) · W4 = NVFP4 3.91 · MXFP4 3.69 · NVFP4 body + MXFP8 coda/head (Inkling-2-Mini analog) 4.74 · core-8/rest-4 5.60 vs core-4/rest-8 5.69 |
| Weight-quantization KL (nats/byte) | bf16 4e-5 · W8 2e-5 · W4 0.0068 · NVFP4 0.0071 · MXFP4 0.0117 · W4 core-only 0.0040 vs W4 rest-only 0.0029 |
| Error compounding across loops | **None material.** W4 relative state error: 12.0% at loop 1, 13.1% at loop 4, 13.7% at loop 8 |
| KV savings, zero-shot KL | int8 KV 2e-5 · int4 KV 0.0057 · **SWA-128 in the core 0.0008** · `KV_SHARE` **0.388 (+0.58 bpb)** · 3 loops 0.021 (+0.026 bpb) |
| Ops in one T=1 forward (CPU graph) | base 1,292 ≈ 10L 1,289 (2.46x the params) vs 4L 527. At B=1, cost tracks **layer applications** |

## 1. Fund these five (≈80 GPU-min, mostly eval-only)

**E1. Precision ladder (eval-only).**
- *Mechanism.* Quantization error is paid per weight and barely compounds across loops. Inkling 2 Mini ships NVFP4 + MXFP8, and MLX 0.32.4 has both formats.
- *Run.* Write an untracked `swarm/quant_eval.py`. For each arm in row 1: `set_dtype(bf16)`, then `nn.quantize(mode=…, class_predicate=…)`, then `prepare.evaluate`. Record the measured bytes.
- *Pass.* 8-bit is lossless if paired Δbpb ≤ +0.002 and the one-sided 95% bound on ΔĀ is ≥ −1.5 pts. 4-bit is acceptable if Δbpb ≤ +0.02 and ΔĀ ≥ −3. Core-8/rest-4 must beat core-4/rest-8 at equal bytes by ≥ 0.001 bpb (predicted 0.0016); otherwise drop the idea that tied weights need more bits.
- *Predictions.* W8 +0.000, W4 +0.010, MXFP4 +0.017 bpb.
- *Cost.* No seeds; use a paired bootstrap. ~10 min.

**E2. KV ladder (eval-only). This is the largest win nobody has claimed.**
- *Mechanism.* The core barely uses context beyond 128 bytes (KL 0.0008). Every reasoning prompt is 4N+3 ≤ 107 bytes, so a 128-byte sliding window (SWA-128) in the core cannot change any reasoning answer. Inkling 2 Small uses 5:1 SWA (§1.1), but 01 never applies it to our cache.
- *Run.* Patch `Attention.__call__` (train.py:96) in a script: give core blocks a banded mask, and fake-quantize K/V with `mx.quantize` (group = head_dim). This matches decode, because a cache is quantized as it is written.
- *Result.* KV at 512 tokens goes 7.50 → 3.00 → **1.59 MiB**, 4.7x smaller.
- *Pass.* Δbpb ≤ +0.003, reasoning unchanged, int8 ΔĀ within ±1 pt. If it passes, it becomes the KV of record for every model, and `KV_SHARE` must beat it to keep its lane.
- *Cost.* No seeds. ~5 min.

**E3. Static depth (eval-only, then 5-min warm-start).**
- *Mechanism.* The answer is read at the last prompt position, so prefill is the whole cost. Dense prefill gets faster only with fewer layer applications for every token. Inkling 2 Mini runs a fixed 3 passes.
- *Run.* (a) An eval-only map that skips each core application (r, i), or only its FFN. (b) `swarm/run.py --lane recipe` with `LOOPS = 3` (3 seeds) and `LOOPS = 2` (1 seed).
- *Pass.* `latency_ms_b1` ≤ 0.85x (predicted 0.80x), Δbpb ≤ +0.01, ΔĀ ≥ −2 pts, all against the 3-seed baseline mean.
- *Decision.* An exit that cannot beat LOOPS=3 at ≤ 3.0 mean loops is defunded for prefill.
- *Cost.* ~30 min.

**E4. A decode meter, and exits that actually save time (eval-only).**
- *Mechanism.* At B=1, time is dispatch (row 5) plus weight traffic at 120 GB/s.
- *Why today's exit is net negative.* The emulation re-injects frozen tokens and recomputes their K/V every loop (train.py:221, 231, 236). In decode, block 1 must still run for every loop a token skips, so a skip saves ≤ 42% of that loop. Each KL check runs coda + head (train.py:201), costing 0.53 loop (+31% FLOPs at L=4). **The KL exit loses FLOPs for every exit pattern.**
- *Run.* Write `swarm/decode_bench.py`: one KV cache per layer application and an `mx.compile`d loop step, teacher-forced over 8×512 val bytes.
  - Policies: fixed-4, fixed-3, state-exit with reuse-latest-K/V, state-exit with drop, KL exit.
  - Precisions: bf16, W8, W4.
  - Report µs/byte (p50 and p90) and decode bpb, plus `latency_ms_b1` at `EXIT_TAU=1e-3` (predicted +25–35%).
- *Pass.* µs/byte ≤ 0.85x fixed-4 at Δbpb ≤ +0.005, net of host syncs, **and** better than fixed-3.
- *Cost.* 50 repeats on an idle GPU. ~15 min.

**E5. QAT-lite (5-min; run only if E1's W4 Δbpb > 0.004).**
- *Mechanism.* Tied weights receive gradient from all 4 visits, so a short fine-tune with a straight-through estimator should recover the ~0.01 bpb.
- *Run.* Add `QAT_BITS = 4` (default 0), training on `w + stop_gradient(dequantize(quantize(w,64,4)) − w)`. Use LR 2e-4 with 10 warm-up steps, then export with `nn.quantize(bits=4)`.
- *Pass.* Δbpb ≤ +0.004 and ΔĀ ≥ −1.5 pts against the bf16 baseline mean, with weights ≤ 4.0 MiB.
- *Cost.* 3 seeds. ~20 min.

## 2. The score

**Intelligence per GB, IPG = Q / F**, in units of chance-normalized accuracy per GB.

- **Q = (Ā − 0.1)/0.9.** Ā is the mean exact-match over hops {2,4,6,8,10}: 2,000 problems, SE ≈ 1.1 pt, at the deployed loop count.
- **F = W + KV₅₁₂ + A₁.**
  - **W** is the measured nbytes at the evaluated precision, including embeddings, norms, scales, `CHUNK` projections and adapters. Never params×2 (prepare.py:285).
  - **KV₅₁₂** sums, over the cached layer applications, min(512, window)/K positions × 2d × element bytes, at Q's loop count. So OOD at 2x loops pays 27 KiB/token, not the 15 reported by train.py:169.
  - **A₁** is the measured peak of one B=1 decode step. It is not `peak_mem_gb`, which is the B=32 training peak because the counter is never reset (prepare.py:289; 8.1 GB).
  - The runtime is excluded.
- **Gates.** val_bpb ≤ ref + 0.02 and Ā ≥ 0.25.
- **Fair comparison.** Compare IPG only within a latency bin (prefill and decode µs/byte, each ±10%); across bins, use the frontier. A win must exceed its paired 95% CI. As an illustration [est, Ā ≈ 0.48], the base scores ≈19/GB in bf16 and ≈73/GB deployed.

The table below is in MiB. "Deployed" means W4 weights, SWA-128 on all layers except the first and last, and int8 KV, re-validated for each model.

| model | params | block-apps/byte | F bf16 | F deployed | Q/Q_base to tie (bf16 / deployed) |
|---|---|---|---|---|---|
| base | 7.28M | 10 | 21.38 | 5.50 | 1 / 1 |
| 4L | 7.28M | 4 | 16.88 | 4.91 | 0.79 / 0.89 |
| 10L (>12M, invalid) | 17.90M | 10 | 41.64 | 11.19 | 1.95 / 2.03 |
| chunk2 | 7.87M | 6.2 | 19.51 | 5.42 | 0.91 / 0.99 |
| chunk4 | 8.46M | 4.2 | 19.14 | 5.53 | 0.90 / 1.01 |

- **Looping wins per GB only if Q_base/Q_4L exceeds 1.27 (bf16) or 1.12 (deployed).** With W4 weights but bf16 KV the bar is 1.65, so KV compression decides the paper's headline.
- **10L** is the reference at equal latency and equal KV.
- **CHUNK saves no memory once deployed (within 2%).** Judge it on latency against 4L, where it needs Q ≥ 1.13·Q_4L.

## 3. Objections

**D1 (mechanism).**
- `overthinking_frac` (landing.py:81) is not compute you can remove: statement tokens supply K/V to later tokens on every loop. An exit at the answer saves ½·1/31 to ½·1/95 of an episode.
- H4 refits the coda, which is on the loop-4 path, so "preserved by construction" is false.
- H18 adds 8% more bytes and a second SDPA. It must beat E1 or E2 per MiB.

**D3 (optimization).**
- Per-loop LoRA (H13), unmerged, adds 80 dispatches per forward (5 × 2 × 4 × 2). Merged, it needs 4 core copies, adding 10.6M params, which is over the cap.
- The gains H5 and H6 report at 8 loops cost about 2x the prefill and 1.8x the KV.
- Anything that changes step time changes the step count: bf16 (1.4x, measured), `jvp` in H17, the 8-loop readout in H4. Fix the dtype across arms.

**D4 (statistics).**
- Seeds are wasted on deterministic transforms (E1, E2, E3a). Paired tests are far stronger there.
- Rigor is moot until the instruments are right:
  - `mean_loops_per_token` comes from the last call, which is 16 hop-10 problems plus padding (prepare.py:267).
  - Latency is uncompiled fp32 prefill (prepare.py:234).
  - KV is analytic.

**Looped MoE.** At Inkling 2 Mini's 7.8:1 ratio, we would get 0.94M active params at fixed bytes, or need 57M total at fixed active params. At B=1, gathers add dispatches to save FLOPs the M4 does not charge for. Fund only static per-loop FFN masks.

## 4. Where 01 is wrong or overstated

1. **§3.5 "21.4 MiB today".** The model is fp32 (train.py:67). The real figure is 42.8 MiB with an fp32 KV cache.
2. **§2.5, H9 and failure 11, "compounding".** Not reproduced at ≤ 8 loops. W4's "+0.02–0.08" is too pessimistic; I expect ≈ +0.010.
3. **W4, "≈0.5 loop".** That cost is per check, so 1.6 loops at L=4 (+31%) and 3.7 at L=8 (+41%), against ≤ 42% saved per skip. The exit is a net loss.
4. **H2 is ranked second on a mis-measured loop count** (prepare.py:267) that does not convert to time.
5. **H14.** `KV_SHARE` starts at +0.58 bpb, while windowing is free.
6. **W3.** The danger applies only to prefill exits, which no deployment uses.
7. **§1.5(d).** Plain-10L has 17.9M params, above MAX_PARAMS, so it runs with `valid=False`.
