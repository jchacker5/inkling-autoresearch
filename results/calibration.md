# Harness calibration (2026-11-23, M4 32 GB, MLX 0.32.4)

All numbers below were measured on this machine during harness construction. They are why the harness
uses a shared base checkpoint.

## Throughput (d=384, 7.28M params, 4 loops, B=32, T=512)

| params dtype | tok/s | peak mem |
|---|---|---|
| fp32 (default) | 14.4k | 8.1 GB |
| bf16 | 20.1k | 4.1 GB |
| fp16 | 20.7k | 4.1 GB |

5 minutes in fp32 is about 255 optimizer steps (4.2M tokens).

## From-scratch 5-minute baseline (first task version, mod-10 arithmetic chains)

`val_bpb 2.293`, reasoning 14-15%. That turned out to be the digit prior, not reasoning: switching to
+/- only and 8/32 reasoning rows gave 9.7-10.3%, which is chance (10%) at every hop.

## Is the task learnable in minutes? (pure reasoning stream, no text)

| setup | steps | result |
|---|---|---|
| arithmetic chains, d=256, 4 loops, 120 s | 345 | chance |
| arithmetic chains, answer-only loss, 150 s | 509 | chance |
| pointer chains (no arithmetic), answer-only loss, 150 s | 517 | chance |
| 1-hop only, answer-only loss, fp32, 90 s | 316 | answer CE stuck at ln 10 |
| dense forest format, hops 1-6, d=384, 240 s | 414 | chance |
| dense forest format, d=256, T=256, 480 s | 1,559 | chance |
| dense forest, 0-hop only, plain 6-layer, 240 s | 1,194 | chance at 0 hops (direct lookup) |
| dense forest, hops 0-6, looped, LR 2e-3, 900 s | 2,980 | chance; loss unstable at this LR |

Per-character loss after 0-hop training: local syntax (`=`, `;`) costs about 0 bits, but query letters
cost about 4.6 bits (uniform over 26) and answers about 3.3 bits (uniform over 10). No retrieval circuit
has formed.

## Is it a bug? (multi-query associative recall, 16 pairs, 3,000 steps)

| model | LR | accuracy |
|---|---|---|
| this repo's block, 2 layers, d=128 | 1e-3 | 22.9% |
| independent minimal MLX transformer (learned positions, LayerNorm, GELU), d=128 | 1e-3 | 22.8% |
| this repo's block, d=256, LR 3e-3 | 3e-3 | 5.2% |

Fused attention matches a manual implementation (max diff 6e-7). RoPE dot products are
position-invariant. With matched settings this repo's model matches an independent reference, so there
is no bug. Retrieval just emerges slowly, and LR above about 1e-3 hurts it.

## Base pretraining progress (`pretrain_base.py`, LR 1e-3, 100 problems per hop)

| training | steps | val_bpb | hop 0 | hop 1 | hop 2 | hop 4 | hop 6 | hop 8 |
|---|---|---|---|---|---|---|---|---|
| 15 min | 763 | 2.0792 | 0.33 | 0.52 | 0.35 | 0.29 | 0.27 | 0.11 |
| 30 min | 1,522 | 1.9306 | 0.43 | 0.54 | 0.41 | 0.39 | 0.24 | 0.16 |
| 45 min | 2,277 | 1.7986 | 0.40 | 0.49 | 0.42 | 0.41 | 0.33 | 0.18 |

Retrieval emerges within the first 15 minutes (about 750 steps) once training continues past the
5-minute window. The first run was killed by a signal after the 45-minute checkpoint; it was resumed
with `--resume` (weights kept, AdamW state and data stream restarted).

## Decision

Two stages: train the looped backbone once for hours (`pretrain_base.py`, LR 1e-3, hops 0-6 curriculum
including direct lookups), then run 5-minute warm-started experiments. If base training never leaves
chance on reasoning, the next step is a dedicated reasoning-only base with a longer budget, not a
different architecture.
