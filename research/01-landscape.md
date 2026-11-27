# 01: Idea landscape for a compact looped transformer (research stage)

*Autoresearch pipeline, stage 1 (research). 2026-11-24. No code was edited.*

**Scope.** This report covers ways to improve the 7.28M-parameter byte-level looped transformer in this repo
(prelude 1 block, then a 2-block core looped 4 times, then a coda of 1 block; d=384, 6 heads, context 512). The
goal metric is "intelligence per gigabyte". There are two kinds of test: eval-only tests on the existing base
checkpoint, and fine-tunes of at most 5 minutes, warm-started from that base, where new modules start as no-ops.

**Method.** 81 web searches and about 55 primary pages were read, covering arXiv abstracts, Thinking Machines'
pages and Hugging Face. They were read through a fetch tool, and the repo's own code and logs were read
directly. Sources span 2016–2026.

**Evidence tags** (used throughout):
- **[read]**: the abstract or page at the URL was read in this pass.
- **[snip]**: only a search-engine summary or snippet was seen. The claim is probably right, but the full text was not read.
- **[local]**: a measurement from this repo or `~/.cache/inkling-autoresearch`.
- **[est]**: my own estimate or prior, not evidence.

Anything without a URL is marked **(unverified)**.

**State of the repo when this was written** (commit `4fd5aa4`). Three ideas the brief asked for are already
implemented, in commit `7705c5c`:
- (a) the landing-loop diagnostic (`swarm/landing.py`);
- (b) the training-free exit (`EXIT_TAU`, `EXIT_METRIC` kl/state);
- (c) loop-consistency self-distillation (`LOOP_CONSIST`).

The chunked core (`CHUNK`) came later, in `ee7b7f2`. Below I treat (a)–(c) as existing tools, then say how to test
them, what can go wrong, and what to refine.

---

## 0. TL;DR

1. **"Inkling 2" is real and verifiable.** It is Thinking Machines Lab's open-weight family, released in November 2026.
   **Inkling 2 Mini is a looped transformer**: 26.4B total and 3.4B active parameters, with its 12 middle layers (of 24) run 3 times.
   It is 3,600x our parameter count, and it is evaluated on agentic, coding and vision benchmarks. **Beating it "in all tasks" is not
   achievable or even measurable at 7M parameters** (§1.4). The defensible comparison is a scaled-down copy of
   *Inkling 2 Mini's recipe* trained in this harness (§1.5).
2. **The most important local finding: depth is not what limits the base on this task.** By 5–5.75 h of base training,
   accuracy is flat at 0.42–0.57 from hop 0 to hop 8, with 100 problems per hop. **Hop 0 is a direct lookup among 3
   statements, and it scores only ≈0.47.** The binding constraint for hops ≤ 8 is the precision of a single lookup, not
   serial depth. This changes which lanes are worth running (§3.1).
3. **Per-token early exit can quietly destroy multi-hop accuracy.** In this task, statement bytes are intermediates for
   other tokens. Their own next-byte prediction converges after one loop (";" after "b=k"), but their states are still
   being read through attention. A KL-on-output exit freezes them (§3.4, H2).
4. **In a fixed-4-loop model, "Late Landing" is partly an artifact of the readout.** The coda has only ever seen loop-4
   states, so the readouts at loops 1–3 are off-distribution. Refit the readout before trusting landing statistics or
   early exits (H4).
5. **Run these first, all cheap:** the landing and loop sweep with a format check (H1); exit variants that account for
   intermediates (H2); a warm-start recipe that does not re-warm the learning rate (H3). The resumed base lost 0.08 bpb
   when its LR was re-warmed [local]. Then try the core-frozen readout refit (H4) and a variable-loop fine-tune (H5).

---

## 1. Inkling 2 findings

### 1.1 What it is (verified at first-party sources)

| | **Inkling 2 Mini** | **Inkling 2 Small** |
|---|---|---|
| Maker, release | Thinking Machines Lab (TML), Nov 20, 2026 [read: [TML Mini post](https://thinkingmachines.ai/news/inkling-2-mini/)] | TML, Nov 5, 2026. Weights on HF from Nov 12 [read: [model card](https://thinkingmachines.ai/model-card/inkling-2-small/), [post](https://thinkingmachines.ai/news/inkling-2-small/)] |
| Parameters | 26.4B total / 3.4B active, and "5.4B weight activations per token to account for looping" [read: TML Mini post] | 594B total / 17B active MoE [read: model card] |
| **Looped?** | **Yes.** "24 layers, divided into four blocks of six." "The two middle blocks run three times, while the first and last blocks each run once." That gives 48 layer applications. "Inkling 2 Mini only supports a fixed number of looping passes." [read: TML Mini post] | No. The HF `Inkling2TextConfig` has no loop fields [read: [HF docs](https://huggingface.co/docs/transformers/main/en/model_doc/inkling2)] |
| Loop wiring | "Every pass starts with the output of the prelude (the first block)." Passes 2–3 "also read the middle blocks' outputs from the previous pass, which we call the message." A learned gate "initially blocks the entire message", so "each loop start[s] the same as in a non-looping model." The model "uses attention residuals" [read: TML Mini post] | 48 layers, d=4608. Hybrid 5:1 sliding-window (1024) and global attention. QK **L2 clipping** (threshold ≈22.6) instead of QK-norm. No RoPE (learned relative attention). Block attention residuals. 384 experts, top-8 [read: HF docs and model card; HF blog read in part] |
| Memory | "The total memory footprint of Inkling 2 Mini is ~20GB". NVFP4 experts, last 4 layers in MXFP8 [read: TML Mini post]. GGUF download 17.5 GB, ≥18 GB RAM [read: [LM Studio](https://lmstudio.ai/models/inkling-2-mini)] | NVFP4 needs ≥500 GB of VRAM [read: model card] |
| License, code | Apache 2.0 [read: LM Studio]. **No paper, technical report or training code was found.** | Apache 2.0. Inference code is in HF Transformers (`modeling_inkling2.py`). No training code [read: HF docs, model card] |

Name collisions to keep straight:
- This repo's "Inkling" is the user's working name for its own model family. TML's Inkling is a separate product.
- The repo's `CHUNK` switch cites **CALM = Continuous Autoregressive LMs** (Shao et al. 2025,
  [2510.27688](https://arxiv.org/abs/2510.27688) [read]). That is not **CALM = Confident Adaptive LM**
  (Schuster et al. 2022, [2207.07061](https://arxiv.org/abs/2207.07061) [read]), which is the early-exit paper used below.

### 1.2 What TML reports about looping (quotes from the [TML Mini post](https://thinkingmachines.ai/news/inkling-2-mini/) [read])

- **Middle versus last.** Looping the middle 12 layers gave larger accuracy gains than looping the last 12 "on 27 out of 28 different
  evaluations". A no-loop baseline needs "1.51–1.91×" the compute to match the middle loop's loss, against "1.24–1.36×" for the last loop.
- **Train cost versus inference cost.** "The non-looping model cost 7.5% less compute to train, but required 140% more inference compute" to
  match the looped model on 512k-token agentic traces.
- **Variable-loop research model** (`TML-d24-varloop-research`):
  - It was "trained with between one and five looping passes", mostly 3.
  - On **N-Hop Lookup** ("several lookup tables with fictional entries", where the model "must follow a chain of lookups without
    writing anything down"), it "kept improving out to ten loops—twice as many as it was trained with."
  - The no-CoT time horizon rises "from 16 seconds at one pass to 37 seconds at three" and is "roughly flat" beyond that.
- **Knowledge versus long context.** "Knowledge-intensive benchmarks don't improve with more passes", while long-context benchmarks do. Looping
  "doesn't enable its weights to store more facts".
- No KV sharing across loops, halting or early exit is mentioned.

**Direct relevance to us.**
- N-Hop Lookup is close to our variable-tracking probe, and the varloop result is the closest published evidence that
  serial lookups scale with test-time loops.
- The gate that starts closed is a way to *start a looped model as a non-looped one*. That fits the zero-init philosophy
  here (H16).

### 1.3 Reported benchmarks (TML's own numbers, not independently verified)

Selected rows from the [TML Mini post](https://thinkingmachines.ai/news/inkling-2-mini/) [read], showing Inkling 2 Mini against Qwen3.8-27B (one of TML's
four comparison columns):

| Benchmark | Inkling 2 Mini | Qwen3.8-27B |
|---|---|---|
| AA Index v5.0 | 20.1 | 20.1 |
| SWE-bench Verified (bash only) | 79.2% | 77.8% |
| BrowseComp (with compaction) | 79.0% | 61.6% |
| ARC-AGI v1 / v2 | 84.5 / 49.3% | 87.5 / 42.4% |
| HLE-Diamond | 15.2% | 15.3% |
| MMMU Pro (Standard 10) | 74.9% | 76.3% |
| GraphWalks BFS 256K | 69.7% | 50.4% |
| LongBench v2 | 66.9% | 52.3% |

TML also quotes a Stanford "intelligence per joule" test on a DGX Spark: Inkling 2 Mini 0.703, Qwen3.6-35B-A3B 0.430,
Gemma-4-26B-A4B 0.159. The method behind that test is Intelligence-per-Watt (Saad-Falcon et al.,
[2511.07885](https://arxiv.org/abs/2511.07885) [read]).

Inkling 2 Small, from its [model card](https://thinkingmachines.ai/model-card/inkling-2-small/) [read]:

| Benchmark | Inkling 2 Small |
|---|---|
| SWE-bench Pro v2 Hard | 86.3% |
| GPQA Diamond | 93.0% |
| ARC-AGI-1 / 2 | 96.5 / 81.7% |
| HLE-Diamond | 28.7% |

### 1.4 Can this project "beat Inkling 2 in all tasks"? No.

**Scale.** 26.4B / 7.28M ≈ **3,600x** the parameters, and ~20 GB against 14.6 MB of bf16 weights ≈ **1,400x** the memory [local arithmetic].

**No shared task.**
- Every Inkling 2 benchmark needs a tokenized, instruction-tuned model with tool use, vision and long context.
- Our model is a 512-byte-context byte LM with no instruction tuning, so it would score about 0 on all of them.
- The closest overlap is N-Hop Lookup, and TML publishes no data or number table for it.

**What *can* be claimed:**
1. On a public, depth-controlled reasoning probe (`prepare.py`), at 7M parameters, a given modification beats an Inkling-2-Mini-style looped
   recipe trained under the same budget (§1.5a).
2. A frontier of accuracy against memory, with external looped LMs as reference points (§1.5c).

As a ratio, "intelligence per GB" trivially favors tiny models. Our ~50% at 14 MB would "beat" a 20 GB model at 100% by about 700x.
So report a **Pareto frontier with a capability floor**, not the ratio alone [est].

### 1.5 Closest well-defined baselines we can actually run

| Baseline | What it is | Eval needed | Cost / blocker |
|---|---|---|---|
| **(a) Inkling-2-Mini recipe at 7M** | Our prelude/core/coda split already loops the middle. Set LOOPS=3. Restart each pass from the prelude output `e`, add a gated message from the previous pass with the gate starting at 0, and optionally use block attention residuals (`DEPTH_ATTN`-like) | The fixed harness: `val_bpb`, `reason_id/ood`, the `scaling` metric (accuracy at half and double the loop count), KV, latency | Needs its own base on the 6 h schedule, or a 2 h stop like the existing controls. That is a human decision |
| **(b) TML varloop protocol** | Train with 1–5 passes, mostly 3, then sweep loops 1..10 on the hop probe | `swarm/eval_ckpt.py` already sweeps loops 1–12 and hops 1–12 | A new base, or the 5-min H5 approximation |
| **(c) Public looped LMs** | Ouro-1.4B/2.6B ([2510.25741](https://arxiv.org/abs/2510.25741) [read]); Huginn-0125 3.5B ([HF](https://huggingface.co/tomg-group-umd/huginn-0125) [read]); ARLoop-1.2B, "fully open" ([2611.05234](https://arxiv.org/abs/2611.05234) [read]); Looped-MoE 1e18-FLOP checkpoints ([2605.09165](https://arxiv.org/abs/2605.09165), [HF](https://huggingface.co/ml-ryanlee/looped-16x2-d640-1e18) [snip]) | Few-shot prompts in our exact line format; greedy exact match on the first digit over the 400-per-hop eval sets; report weight and KV bytes | 1–7 GB of downloads, and the disk has ~4.8 GB free [local]. Needs user approval. Not apples-to-apples (tokenized, 200–500x larger) |
| **(d) Repo controls** | Plain-4L (equal parameters) and Plain-10L (equal depth), same schedule, stopped at 2 h | The fixed harness | Already in `swarm/campaign.py`. This is the only baseline that isolates looping |

---

## 2. Key literature (dated, with URLs)

### 2.1 Adaptive computation and halting

| Date | Work | Key finding for us |
|---|---|---|
| 2016-03 | ACT, Graves ([1603.08983](https://arxiv.org/abs/1603.08983)) | Halting unit plus ponder cost. The hand-set time penalty is one the network is "quite sensitive" to [snip] |
| 2018-07 | Universal Transformers ([1807.03819](https://arxiv.org/abs/1807.03819)) | Shared block with per-position ACT halting. Helped algorithmic tasks, slightly hurt MT [snip] |
| 2021-07 | PonderNet ([2107.05407](https://arxiv.org/abs/2107.05407)) | Geometric prior on halting. Calls ACT "notably unstable". Extrapolates on parity [snip] |
| 2024-04 | Mixture-of-Depths ([2404.02258](https://arxiv.org/abs/2404.02258)) | Routes tokens around layers. Listed in the repo's bibliography (not re-read) |
| 2025-07 | Mixture-of-Recursions ([2507.10524](https://arxiv.org/abs/2507.10524)) | Per-token recursion depth (expert-choice or token-choice). Recursive KV sharing costs ≈1/N_r of the memory but adds KV I/O [snip] |
| 2025-10 | Ouro / LoopLM ([2510.25741](https://arxiv.org/abs/2510.25741)) | Exit gates "collapse to shallow depths or overuse long loops". Fixed with entropy regularization under a uniform prior [snip] |
| 2026-02 | ANIRA ([2602.08864](https://arxiv.org/abs/2602.08864), ICML'26) | Compute that tracks difficulty can emerge without labels, but "does not guarantee algorithmic generalization". Online halting tracks execution state better than upfront decisions [snip] |
| 2026-07 | Per-token fixed-point convergence ([2607.14427](https://arxiv.org/abs/2607.14427)) | 135M model: the median token converges by loop 6 of 8. A training-free stabilization rule gives depth-8 quality at **4.94 loops (−38%)**. **A linear router trained on convergence labels gave no reduction** [read] |
| 2026-09 | Closing the Loop ([2610.00673](https://arxiv.org/abs/2610.00673)) | "Stronger exit-gate regularization". Converting a pretrained model needs only "a single learned input-mixing scalar and a smoothed exit loss" [read] |

### 2.2 Early exit and "overthinking" before LLM reasoning

| Date | Work | Key finding |
|---|---|---|
| 2016 | BranchyNet ([1709.01686](https://arxiv.org/abs/1709.01686)) | Side branches with entropy thresholds. 2–6x speedup [snip] |
| 2018-10 / ICML'19 | Shallow-Deep Networks, Kaya, Hong & Dumitras ([1810.07052](https://arxiv.org/abs/1810.07052)) | Defines **overthinking**: correct at an internal classifier, wrong at the end. Confidence exits save >50% of compute at equal accuracy. Destructive overthinking appears in up to ~50% of a CNN's errors. Disagreement among internal classifiers flags likely errors [snip] |
| 2020-04 | DeeBERT ([2004.12993](https://arxiv.org/abs/2004.12993)) | Entropy-threshold off-ramps. Up to ~40% of inference time saved [snip] |
| 2020-06 | PABEE ([2006.04152](https://arxiv.org/abs/2006.04152)) | **Patience exit**: stop after the prediction is unchanged for k layers. Beats entropy exits on GLUE and improves robustness [snip] |
| 2021-05 | Ensemble internal classifiers ([2105.13792](https://arxiv.org/abs/2105.13792)) | Vote across past exits instead of trusting one [snip] |
| 2021-10 | Learn-then-Test ([2110.01052](https://arxiv.org/abs/2110.01052)) | Picks thresholds with finite-sample risk control [snip] |
| 2022-07 | CALM, Schuster et al. ([2207.07061](https://arxiv.org/abs/2207.07061)) | Per-token exits with softmax-response, state-propagation or classifier confidence. Calibrated with risk control. Names the **missing hidden states of exited tokens** problem [snip] |
| 2023-07 | Overthinking the Truth ([2307.09476](https://arxiv.org/abs/2307.09476)) | A logit-lens "critical layer". Exiting early can raise accuracy, and "false induction heads" cause overthinking [snip] |
| 2023-03 | Tuned Lens ([2303.08112](https://arxiv.org/abs/2303.08112)) | The logit lens is biased or unreliable on some models. Affine probes fit per layer are more faithful [snip] |
| 2024-04 | LayerSkip ([2404.16710](https://arxiv.org/abs/2404.16710)) | Early-exit loss plus layer dropout with one shared exit. Self-speculative decoding at 1.8–2.2x [snip] |
| 2024-07 | Multi-exit gradient conflict: DFS ([2407.13986](https://arxiv.org/abs/2407.13986)); [2407.14320](https://arxiv.org/abs/2407.14320); RomeBERT ([2101.09755](https://arxiv.org/abs/2101.09755)) | Joint exit losses conflict on shared weights and can degrade the final exit [snip] |

### 2.3 TRACE: overthinking defined by utility (as the brief asked)

**Paper.** Zhang, Mohananey, Chronopoulou, Papalampidi, Gupta, Munkhdalai, Wang and Upadhyay (UMich and Google DeepMind), "Do LLMs Really Need 10+ Thoughts for 'Find
the Time 1000 Days Later'? Towards Structural Understanding of LLM Overthinking". arXiv **[2510.07880](https://arxiv.org/abs/2510.07880)**, 9 Oct 2025. A
secondary source says it was accepted to ACL 2026 (unverified).

About 30% of the full text was read [read]; the rest is [snip].

**Method.**
- TRACE stands for "Thought-process Reconstruction and Automated Clustering Engine".
- It splits a trace into "self-contained, complete and answer-bearing" sub-thoughts, labels each transition (verification,
  correction, backtrack, and so on), and builds a graph of how the answer progresses.

**Findings.**
- Thinking models are "five to twenty times longer at inference than non-thinking counterparts, with little or even no improvement".
- **Explorer:** "distributes the probability of correctness across multiple potential answers. It may find the correct answer early, but it
  continues to explore alternatives."
- **Late Landing:** "a convergent reasoning path where the probability of correctness is highly concentrated on the final, distinct
  answer." Its overthinking is over-verification.
- **Utility definition:** overthinking is "continuation of the thought process after the marginal return drops below a predefined
  threshold (ϵ)". The marginal return is ΔPerformance / ΔThought [snip for the formula].
- **Mitigations** [snip]:
  - Stop after k consecutive self-verifications. With k=2, Qwen3-235B's length roughly halves (2,722 → 1,315 words).
  - Stop when the model returns to an answer it gave before ("backtrack stop").
- TRACE says nothing about latent or looped computation.

**How it maps to a looped model** (a thought becomes one pass of the core) [est]:

| TRACE idea | Looped-model analogue |
|---|---|
| Landing | The loop after which the argmax at the readout never changes |
| Explorer token | The argmax visits ≥3 distinct values across loops |
| Late Landing token | Lands only at the last loop |
| Utility exit | Stop when the readout changes by less than τ (`EXIT_TAU`) |
| "k consecutive verifications" | PABEE-style patience |
| "Backtrack stop" | Stop when the argmax revisits an earlier value |

### 2.4 Looped and recurrent-depth transformers up to 2025

| Date | Work | Key finding |
|---|---|---|
| 2019-09 | ALBERT ([1909.11942](https://arxiv.org/abs/1909.11942)) | Sharing all layers costs −2.5 average points at E=768 and −1.5 at E=128. Most of the loss comes from **sharing the FFN**; sharing attention costs ≈0 [snip] |
| 2021-06 | Schwarzschild et al. ([2106.04537](https://arxiv.org/abs/2106.04537)) | Recurrent nets trained on easy problems solve hard ones with more iterations. They read the **most confident iteration** [snip] |
| 2022-02 | Bansal et al., "Extrapolation without Overthinking" ([2202.05826](https://arxiv.org/abs/2202.05826)) | **Recall** (re-inject the input) is required for extrapolation. **Progressive loss** (train from a detached, randomly advanced state) removes overthinking and halves the iterations needed. 0% → >97% on 512-bit prefix sums [snip] |
| 2022-11 | Path independence, Anil et al. ([2211.09961](https://arxiv.org/abs/2211.09961)) | Accuracy on harder or OOD inputs tracks **path independence** (reaching the same fixed point from any initial state). Path-independent models improve monotonically with test-time iterations [snip] |
| 2023-01 | Looped transformers as programmable computers, Giannou ([2301.13196](https://arxiv.org/abs/2301.13196), ICML'23) | Constant-depth blocks run in a loop emulate a computer; a looped 13-layer transformer runs a small instruction-set computer [snip] |
| 2024-05 | MoEUT ([2405.16039](https://arxiv.org/abs/2405.16039)) | Shared-layer transformers become competitive at language modeling once given MoE plus grouping and normalization tricks [snip] |
| 2024-09 | Looped transformers for length generalization, Fan et al. ([2409.15647](https://arxiv.org/abs/2409.15647)) | Looping with an **adaptive number of steps** generalizes in length on n-RASP-L tasks without scratchpads [snip] |
| 2024-10 | Relaxed Recursive Transformers ([2410.20672](https://arxiv.org/abs/2410.20672)) | **Per-loop LoRA** on shared layers, initialized by SVD. Recursive Gemma 1B beats a 1B trained from scratch by up to 13.5 points [snip] |
| 2025-02 | Huginn, Geiping et al. ([2502.05171](https://arxiv.org/abs/2502.05171)) | 3.5B recurrent depth. Loop count drawn from a log-normal Poisson with mean 32, truncated backprop through the last 8 loops. Exit per token works zero-shot, with criteria `entropy-diff`, `latent-diff`, `kl` (example threshold 5e-4) and `argmax-stability` ([model card](https://huggingface.co/tomg-group-umd/huginn-0125)). KV-cache budget `i mod k` works zero-shot [read and snip] |
| 2025-02 | Saunshi et al. ([2502.17416](https://arxiv.org/abs/2502.17416)) | A k-layer block looped L times ≈ a kL-layer model on reasoning, with **worse perplexity at matched compute but better reasoning** [snip] |
| 2025-09 | Two-scale latent dynamics ([2509.23314](https://arxiv.org/abs/2509.23314)) | Loop step sizes shrink within 5–10 loops. A **second-order ("acceleration") exit**, applied when below τ for 2 steps, beats step-norm and KL exits on latency at matched quality [snip] |
| 2025-10 | TRM, Jolicoeur-Martineau ([2510.04871](https://arxiv.org/abs/2510.04871)) | **7M** parameters, 2 layers, recursion with deep supervision. 44.6% on ARC-AGI-1 and 7.8% on ARC-AGI-2 [snip] |
| 2025-08 | ARC Prize HRM analysis ([blog](https://arcprize.org/blog/hrm-analysis)) | HRM's "hierarchical" design had minimal impact. The **outer refinement loop (deep supervision)** drove the gains [snip] |
| 2025-11 | Retrofitted recurrence, McLeish et al. ([2511.07384](https://arxiv.org/abs/2511.07384)) | A curriculum over loop count plus "healing" turns pretrained LMs into looped ones [snip] |
| 2025-11 | Think-at-Hard ([2511.08577](https://arxiv.org/abs/2511.08577), ICML'26) | **"Latent overthinking"**: most tokens are right after pass 1, and later passes flip some to wrong. An oracle per-token policy is worth **+7.3%**. A learned decider skips iterations on 93% of tokens and gains 3.8–4.4% [read] |

### 2.5 The 2026 looped-model cluster (abstracts read unless marked)

| Date | Work | Key finding |
|---|---|---|
| 2026-02 | LoopFormer ([2602.11451](https://arxiv.org/abs/2602.11451), ICLR'26) | "Shortcut-consistency training" aligns short and long loop trajectories. Conditions on time and step size |
| 2026-03 | Attention Residuals, Kimi ([2603.15031](https://arxiv.org/abs/2603.15031)) | Softmax over earlier layers' outputs replaces the fixed residual sum. Modest gains for ~4% training cost [snip]. Inkling 2 uses it |
| 2026-04 | Stability and generalization in looped transformers ([2604.15259](https://arxiv.org/abs/2604.15259), NeurIPS'26) | Without recall, a looped net has countable fixed points and weak dependence on the input. **Recall plus outer normalization** gives attractive, input-smooth fixed points |
| 2026-04 | Loop, Think & Generalize ([2604.07822](https://arxiv.org/abs/2604.07822)) | Train on ≤5 hops, test on 10. More test-time iterations reach deeper, but **overthinking caps extrapolation** [snip] |
| 2026-05 | LT2 ([2605.20670](https://arxiv.org/abs/2605.20670)) | Looping "uniquely synergizes" with linear and sparse attention |
| 2026-05 | Sparse layers in looped LMs ([2605.09165](https://arxiv.org/abs/2605.09165)) | Looped **MoE** scales and dense looped does not, because routing diverges across loops. Loop boundaries make good exit points [snip] |
| 2026-05 | LoopQ ([2605.16343](https://arxiv.org/abs/2605.16343)); LoopQuant ([2610.16368](https://arxiv.org/abs/2610.16368)); Loopy ([2610.05265](https://arxiv.org/abs/2610.05265)); Feedback exposure ([2609.30820](https://arxiv.org/abs/2609.30820)) | **Quantization error compounds across loops.** Standard post-training quantization (PTQ) shows error spikes at every loop boundary [snip] |
| 2026-06 | Readout blind spot ([2606.24898](https://arxiv.org/abs/2606.24898)) | 44M and 129M models: a per-loop cross-entropy through an RMSNorm readout cannot see state scale, and norms reach "thousands or tens of thousands". **"Dense supervision trains exits"**, but scale needs a separate loss |
| 2026-07 | DeepLoop ([2607.13491](https://arxiv.org/abs/2607.13491)) | Residual scaling must count *visits* to the shared block |
| 2026-08 | MixerLoop ([2608.18230](https://arxiv.org/abs/2608.18230)) | Loop the token mixer and apply the dense FFN once. Beats full-block looping at **15M** |
| 2026-08 | LoopMTP ([2608.03624](https://arxiv.org/abs/2608.03624)) | Targets "latent overthinking" by aligning loop t with the token t steps ahead |
| 2026-09 | What makes recurrence effective ([2609.36636](https://arxiv.org/abs/2609.36636)) | Recurrence helps reasoning and **hurts knowledge**. Harder instances do not consistently benefit more. A non-recurrent output layer adds robustness. History-state injection beats initial-state injection |
| 2026-09 | Right Direction, Wrong Step ([2609.16665](https://arxiv.org/abs/2609.16665)) | Failures after a finite number of steps come from the step scale, not the direction. A **quarter step** fixes 72–83% of the selected failures |
| 2026-09 | Beyond the training horizon ([2609.33144](https://arxiv.org/abs/2609.33144)) | Both the readout-state and propagation mechanisms "become unreliable at greater depths" |
| 2026-09 | Depth Control Protocol ([2609.19934](https://arxiv.org/abs/2609.19934)) | The slope from truncating depth mixes fewer applications, less distinct computation, and an **OOD residual reaching the readout** |
| 2026-09 | Looped DiT ([2609.40305](https://arxiv.org/abs/2609.40305)) | **Deep supervision across loops** fixed naive looping and "substantially reduces the penalty for early exit" [snip for the second clause] |
| 2026-10 | Looped Models Done Right II ([2610.06833](https://arxiv.org/abs/2610.06833)) | Near fixed points: **terminal KV sharing** with "almost no loss", truncated BPTT, orthogonal injection, a learned depth prior. At 1.6B a **3x smaller KV cache** matches full-cache fixed-depth training |
| 2026-10 | InfiLoop ([2610.11570](https://arxiv.org/abs/2610.11570)) | **"Latent overthinking"**, where later updates overwrite correct deductions. A loop-native residual (attention over past states plus temporal decay) lets a **7M** model reach 97.9% on Sudoku-Extreme and keep improving past 20,000 steps |
| 2026-10 | Learning to compute ≠ retain ([2610.19768](https://arxiv.org/abs/2610.19768)) | Trained to answer at loop 32, the model keeps only **22%** of its answers at loop 64. **Randomizing the training loop count helps; truncating the gradient does not.** "Retention, however, is not reuse" |
| 2026-10 | PAR ([2610.27628](https://arxiv.org/abs/2610.27628)) | Prediction at loop k is aligned with the state after k operations, plus **norm anchoring** (rescale to the loop-1 norm). Extrapolates 16x on S5 swaps. Removing the anchoring drops that to <2.2x. Has a learned completion readout |
| 2026-10 | Agreement, not normalisation ([2610.31943](https://arxiv.org/abs/2610.31943)) | 19M models: removing normalization is never worse, and the damage beyond the trained budget **sits in the readout and last layer; refitting that layer recovers it** |
| 2026-10 | Inside the Loop ([2610.24135](https://arxiv.org/abs/2610.24135), NeurIPS'26) | Permutation traversal: a looped 4-layer block run 8 times solves every seed. An untied model of the same depth never does without supervising intermediate layers. The looped model builds a **staircase** that adapts to its loop budget |
| 2026-10 | LoopCD ([2610.02185](https://arxiv.org/abs/2610.02185); EMNLP version [2609.24196](https://arxiv.org/abs/2609.24196) has different author lists) | Contrasting the final loop with loop 1 at decode time, with no training: Ouro-2.6B on AIME'24 goes 61.9 → 73.3. Halving the loops matches the full-depth baseline |
| 2026-10 | LoopOPD ([2610.10623](https://arxiv.org/abs/2610.10623)); LoOPSD ([2610.18162](https://arxiv.org/abs/2610.18162) [snip]) | The terminal loop is a teacher for intermediate loops (on-policy) |
| 2026-10 | How to loop your transformer ([2610.40325](https://arxiv.org/abs/2610.40325)) | Up to 1B parameters and 32 loops. "Adding loops is not uniformly beneficial." **Loop-specific low-rank adapters help, and more so as loops grow** |
| 2026-10 | Accordion ([2610.32492](https://arxiv.org/abs/2610.32492)) | Five fixed loop budgets trained jointly with **no** consistency loss. The middle budget captures 89% of the gain |
| 2026-10 | LoopFold ([2610.35556](https://arxiv.org/abs/2610.35556)) | Predict-then-correct parallel loops: −46% decode latency against 9 loops |
| 2026-11 | ARLoop ([2611.05234](https://arxiv.org/abs/2611.05234)) | Attention over depth instead of residuals, where **sliding-window depth attention ≥ full history**. 1.2B with 3 loops beats Huginn-3.5B |
| 2026-09 | Shared weights, selected computations ([2609.39892](https://arxiv.org/abs/2609.39892)) | A learned linear map on a loop's input **selects transitions the backbone already learned; it does not create new algorithms** [snip] |

### 2.6 Fixed points, stability, and older recurrent and dynamical-systems ideas

| Date | Work | Key finding |
|---|---|---|
| 2000s | Echo-state networks (contraction view: [2507.18467](https://arxiv.org/abs/2507.18467)) | The echo-state property plus Lipschitz dynamics give **fading memory**. A spectral radius <1 is a heuristic, not a guarantee [snip] |
| 2015 | LSTM forget-gate bias = 1, Jozefowicz et al. ([PMLR v37](https://proceedings.mlr.press/v37/jozefowicz15.html)) | Initializing the gate *open* closes most of the LSTM–GRU gap [snip] |
| 2017-05 | Stable architectures, Haber & Ruthotto ([1705.03341](https://arxiv.org/abs/1705.03341)) | A ResNet is an ODE discretization, so network stability equals ODE stability [snip] |
| 2019-09 | Deep Equilibrium Models ([1909.01377](https://arxiv.org/abs/1909.01377), NeurIPS'19) | Find the infinite-depth fixed point by root-finding, with constant memory (up to 88% memory reduction) [snip] |
| 2020-06 | Lipschitz RNN ([2006.12070](https://arxiv.org/abs/2006.12070)) | Global exponential stability conditions, building on antisymmetric RNNs [snip] |
| 2020-07 | Modern Hopfield networks ([2008.02217](https://arxiv.org/abs/2008.02217)) | Attention *is* the Hopfield update. Fixed points are a global average, **metastable averages of similar patterns**, or single patterns. Retrieval converges in one step when patterns are separated [snip] |
| 2021-03 | JFB ([2103.12803](https://arxiv.org/abs/2103.12803)); phantom gradients ([2111.05177](https://arxiv.org/abs/2111.05177)) | Cheap approximate gradients through fixed points: 1.4–1.7x faster backward passes [snip] |
| 2021-06 | DEQ Jacobian regularization ([2106.14342](https://arxiv.org/abs/2106.14342)) | Penalizing the Jacobian norm stabilizes forward and backward convergence, at only the cost of vector-Jacobian products [snip] |

### 2.7 Depth bounds, retrieval, and length generalization

| Date | Work | Key finding |
|---|---|---|
| 2022-09 | Induction heads ([2209.11895](https://arxiv.org/abs/2209.11895)); "What needs to go right" ([2404.07129](https://arxiv.org/abs/2404.07129)); Aoyama et al. ([2511.16893](https://arxiv.org/abs/2511.16893)) | Retrieval appears in a phase change made of interacting subcircuits. **Data properties, batch and context size set the timing** [snip] |
| 2023-05 | NoPE, Kazemnejad ([2305.19466](https://arxiv.org/abs/2305.19466)) | No positional encoding length-generalizes better than APE, T5, ALiBi or RoPE in decoders [snip] |
| 2023-10 | Binding IDs, Feng & Steinhardt ([2310.17191](https://arxiv.org/abs/2310.17191)) | LMs bind entities to attributes with additive binding-ID vectors [snip] |
| 2024-02 | Sanford, Hsu & Telgarsky ([2402.09268](https://arxiv.org/abs/2402.09268)) | **k-hop needs depth Θ(log k)**. The largest learnable k grows exponentially with depth in practice [snip] |
| 2024-05 | Abacus embeddings ([2405.17399](https://arxiv.org/abs/2405.17399)); Position coupling ([2405.20671](https://arxiv.org/abs/2405.20671)) | Structure-aware positions plus **input injection plus looping** give large length extrapolation: 20 → 100+ digits, and 30 → 200 [snip] |
| 2024-05 | Grokked transformers ([2405.15071](https://arxiv.org/abs/2405.15071)) | Implicit composition generalizes OOD only weakly. The authors suggest cross-layer sharing or recurrence [snip] |
| 2024-06 | Hopping too late ([2406.12775](https://arxiv.org/abs/2406.12775)) | The second hop happens too late in depth. Back-patching an earlier layer fixes up to 66% of failures [snip] |
| 2024-10 | Differential attention ([2410.05258](https://arxiv.org/abs/2410.05258)); SSMax ([2501.19399](https://arxiv.org/abs/2501.19399)); "softmax is not enough" ([2410.01104](https://arxiv.org/abs/2410.01104)) | Sharper or less noisy attention improves key-information retrieval. Adaptive temperature keeps attention sharp [snip] |
| 2025-03 | Log-depth transformers, Merrill & Sabharwal ([2503.03961](https://arxiv.org/abs/2503.03961)) | Depth Θ(log n) suffices for automata and connectivity. Scaling depth beats scaling width or CoT [snip] |

### 2.8 Engineering and training recipes

| Date | Work | Key finding |
|---|---|---|
| 2021-06 | Distillation, "patient and consistent" (Beyer et al., [2106.05237](https://arxiv.org/abs/2106.05237)) | Teacher and student must see **the same input** (function matching), over long schedules [snip] |
| 2023-08 | Re-warming the LR, Gupta et al. ([2308.04014](https://arxiv.org/abs/2308.04014)) | Re-warming a converged model spikes its loss. On a limited budget, the workshop version recommends **not re-warming** [snip] |
| 2024-10 → | Muon ([blog](https://kellerjordan.github.io/posts/muon/), [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)) | Orthogonalized momentum. Speedrun records also came from value embeddings, U-net skips and logit softcap [snip] |
| 2025-02 | Muon at scale ([2502.16982](https://arxiv.org/abs/2502.16982)) | ~2x compute efficiency. Whether a mismatched optimizer (an AdamW base fine-tuned with Muon) costs anything is **contested**: [2605.06654](https://arxiv.org/abs/2605.06654) says it does; [NeurIPS'26](https://neurips.cc/virtual/2026/172331) says the cost is ephemeral [snip] |
| 2025-11 | Intelligence per Watt ([2511.07885](https://arxiv.org/abs/2511.07885)) | Accuracy per unit of power, for local LMs [read] |
| 2026-03 | Karpathy autoresearch ([GitHub](https://github.com/karpathy/autoresearch)) | Fixed `prepare.py`, an agent-edited `train.py`, 5-minute runs, keep or discard by `val_bpb` [snip] |
| — | MLX `nn.quantize` ([docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html)) | Group size 64, 2–8 bits, quantizes Linear and Embedding in place [snip] |

---

## 3. First-principles analysis

### 3.1 What the base checkpoint actually does [local]

Rows are quick evals of exact match with 100 problems per hop, so the binomial SE is about ±5 points per cell. The source is
`~/.cache/inkling-autoresearch/base_progress.tsv`.

| train min | val_bpb | hop0 | hop1 | hop2 | hop4 | hop6 | hop8 (OOD) |
|---|---|---|---|---|---|---|---|
| 15 | 2.079 | .33 | .52 | .35 | .29 | .27 | .11 |
| 45 | 1.799 | .40 | .49 | .42 | .41 | .33 | .18 |
| 120 | 1.672 | .38 | .42 | .55 | .41 | .34 | .32 |
| 300 | 1.512 | .47 | .43 | .49 | .51 | .49 | .50 |
| 315 | 1.504 | .49 | .42 | .48 | .49 | .47 | .50 |
| 330 (after resume) | 1.585 | .35 | .39 | .51 | .46 | .43 | .52 |
| 345 | 1.565 | .47 | .51 | .49 | .57 | .44 | .43 |

**O1. The hop profile is flat by 5 hours.**
- Hops 0–8 all fall between 0.42 and 0.57.
- Hop 8, which is out of distribution, climbed from 0.11 to ~0.50.
- Hop 0 has sat near 0.45 since about 30 minutes.

The README's "accuracy falls with chain length" was true at 15–45 minutes and is now stale.

**O2. Direct lookup is weak.**
- A hop-0 problem is 3 statements, all of them digit literals, followed by `?x=`.
- The base gets only ≈0.47.
- With 3 digits present, guessing among them scores about 1/3, so the model is only modestly above a copy-a-digit heuristic.

This lines up with `results/calibration.md`: multi-query associative recall reached only 23% after 3,000 steps [local].

**O3. Resuming re-warms the LR, and that hurt.**
- AdamW was restarted and the LR warmed back up to about 8.5e-4.
- `val_bpb` went from 1.504 to 1.585 within 15 minutes, and hop 0 went from 0.49 to 0.35.
- Every 5-minute experiment does the same thing: LR 1e-3 with 100 warm-up steps out of about 255.

**O4. Without training for it, the loop count is not portable.**
- In the from-scratch 5-minute baseline, `val_bpb` was 2.332 at 4 loops.
- It was worse in both directions: 2.400 at 8 loops and 2.456 at 2 loops (`runs/20261123-060239-baseline/result.json`).
- The base was trained at a fixed 4 loops, so expect the same behavior until the loop sweep says otherwise.

**Candidate explanations for O2**, all testable with eval-only diagnostics (H1):
- **(i) Imprecise letter matching.** Answering needs a 3-gram induction: match `?k=` to an earlier `k=`, then copy the next byte. In
  Hopfield terms (Ramsauer, [2008.02217](https://arxiv.org/abs/2008.02217)), attention that is not sharp enough settles into a
  **metastable average over similar keys** (letters) and copies a blend.
- **(ii) A shift between the eval format and training.**
  - Eval lines start at position 0 with no preceding `\n`.
  - Complete training lines usually follow a `\n` and previous lines that use the same letters with other bindings.
  - A sink at position 0, or a learned "lines start after `\n`" heuristic, would land on the first statement.
- **(iii) The lookup is used once per query at the end**, after the chain has propagated through the statement positions. In that case
  its precision caps every hop equally, which matches the flat profile.

### 3.2 Why looping helps or hurts at 7M parameters

**Capacity against computation.** Knowledge scales with *unique* parameters:
- TML says looping "doesn't enable its weights to store more facts".
- Saunshi ([2502.17416](https://arxiv.org/abs/2502.17416)) finds worse perplexity at matched compute.
- Zhuang ([2609.36636](https://arxiv.org/abs/2609.36636)) finds that recurrence hurts knowledge.

At 7M parameters, `val_bpb` is limited by capacity, so loops cannot buy much there. Their value lies in serial,
algorithmic computation: the `reason_*` metrics.

**How much depth k-hop needs:**
- With pointer doubling it is Θ(log k) (Sanford, [2402.09268](https://arxiv.org/abs/2402.09268)). With one hop per layer it is Θ(k).
- The base applies a core layer 8 times per forward pass, plus the prelude and coda.
- One hop per core layer covers about 8 hops; doubling covers far more than 10.
- Parents always come before children in a line (`_forest` emits each chain in order), so causal attention allows either algorithm.
- The flat profile cannot tell the two apart. **The loop sweep can.** One hop per core layer predicts accuracy at hop h falls once
  2·loops < h. Doubling predicts it stays flat down to about loops ≈ ⌈log2 h⌉/2.

**Why weight tying helps OOD depth.** Weight tying is an inductive bias toward reusable steps:
- Inside the Loop's staircase ([2610.24135](https://arxiv.org/abs/2610.24135)).
- Schwarzschild ([2106.04537](https://arxiv.org/abs/2106.04537)).
- Bansal ([2202.05826](https://arxiv.org/abs/2202.05826)).

It only pays off if the dynamics are path-independent (Anil, [2211.09961](https://arxiv.org/abs/2211.09961)), which takes
variable-loop training (TML varloop; [2610.19768](https://arxiv.org/abs/2610.19768)).

**Costs:**
- **Wall clock.** 10 layer applications per step, against 4 for Plain-4L, so about 2.5x fewer steps per hour at equal parameters.
- **Gradients pile up on shared weights** ([2607.13491](https://arxiv.org/abs/2607.13491)).
- **Readout mismatch** away from the trained loop count ([2610.31943](https://arxiv.org/abs/2610.31943), [2606.24898](https://arxiv.org/abs/2606.24898)).
- **Latent overthinking** ([2511.08577](https://arxiv.org/abs/2511.08577), [2610.11570](https://arxiv.org/abs/2610.11570)).

### 3.3 What bounds OOD hop generalization in this harness

- **B1. The algorithm.** Is it iterative in depth or doubling (see §3.2)? Only the iterative kind needs more loops for more hops.
- **B2. Lookup precision.** If every hop needs a noisy lookup, accuracy falls roughly as pʰ. If one final lookup dominates, it is flat at
  about p. The data look like the second case with p ≈ 0.5.
- **B3. Depth is confounded with the number of variables.** `_forest` uses `n_dist = min(hops+2, 25−hops)`. That gives 15
  variables at hop 6 (the training maximum), 19 at hop 8 and 23 at hop 10 [local]. OOD failures can therefore come from
  *interference among more keys*, not from depth. An analysis script that varies depth at a fixed variable count can separate the
  two without touching the fixed `prepare.py`.
- **B4. Readout and coda mismatch** at any loop count other than 4 (O4; [2610.31943](https://arxiv.org/abs/2610.31943)).
- **B5. Path dependence**, which shows up as overthinking at 2x loops ([2211.09961](https://arxiv.org/abs/2211.09961), [2610.19768](https://arxiv.org/abs/2610.19768)).
- **B6. The 5-minute fine-tunes (about 255 steps) can select and reweight existing computations but rarely invent new circuits.**
  - "J selects computations learned by the backbone rather than creating new algorithms" ([2609.39892](https://arxiv.org/abs/2609.39892)).
  - Implicit composition takes grokking-length training ([2405.15071](https://arxiv.org/abs/2405.15071)).

### 3.4 Where compute is wasted

- **W1. Easy text bytes.** In byte-level text, most bytes inside a word are predictable. A convergence-based exit saved 38% of loops at a
  depth-8 model's quality at 135M ([2607.14427](https://arxiv.org/abs/2607.14427)). With only 4 fixed loops and a readout that only ever
  saw loop-4 states, expect less [est].
- **W2. Late Landing is partly an artifact here.**
  - `landing.py` reads loop r through the coda. That is exact: row r equals `model(x, loops=r+1)`.
  - But the coda was trained only on loop-4 states, so landing will look late until the readout is refit (H4) or consistency-trained (H6).
  - The logit lens itself can be biased (Tuned Lens, [2303.08112](https://arxiv.org/abs/2303.08112)). Add affine probes per loop to
    measure *information*, as distinct from *usable output*.
- **W3. Intermediates are not outputs.**
  - The exit rule judges a token by its own next-byte readout.
  - For the byte `k` in `b=k;`, the next byte is `;`, which is trivially predictable after one loop.
  - But that position's state is exactly where the chain value must arrive for later lookups.
  - Freezing it (`x = mx.where(active, x, h)` in `recur`, train.py:206) stops propagation for every token that reads it.
  - This is CALM's "missing hidden states" problem ([2207.07061](https://arxiv.org/abs/2207.07061)), with a twist: the problem is not
    missing states but *stale* ones.
  - **Prediction:** a KL exit at a τ that is harmless for `val_bpb` will cut deep-hop accuracy much more than hop 0 [est, falsifiable].
- **W4. Exit bookkeeping costs compute.** The KL metric runs coda plus head once per loop, about 0.5 loop of FLOPs. The state metric is
  nearly free. Mean-loop savings must be reported net of this.
- **W5. Flips are inflated by chance.**
  - With 10 answer classes, a near-random early readout is "right" 10% of the time per loop.
  - So P(some earlier loop right | final wrong) can be 20–30% with no real overthinking [est].
  - Count **confident flips** instead: p > 0.5, and correct for ≥2 consecutive loops. Compare them against a shuffled-loop null.

### 3.5 The memory budget behind "per GB" [local arithmetic]

| Item | Size | Note |
|---|---|---|
| Weights, bf16 | 13.9 MiB | |
| KV cache | 15 KiB/token, so **7.5 MiB at 512 bytes** | 35% of memory at full context |
| `KV_SHARE` | 6 KiB/token (3 MiB) | |
| Weights, W8 (group-64 affine, 8.5 bits/weight) | ≈7.4 MiB | |
| Weights, W4 (4.5 bits/weight) | ≈3.9 MiB | |

W4 plus shared KV comes to ≈6.9 MiB against 21.4 MiB today, **3.1x better per GB *if* accuracy holds**.

In looped models, quantization error compounds across loops (LoopQ [2605.16343](https://arxiv.org/abs/2605.16343), LoopQuant
[2610.16368](https://arxiv.org/abs/2610.16368)), so this must be measured. Note that `weights_mb_bf16` in the harness is
computed from the parameter count alone and cannot show quantization. Report quantized sizes separately.

### 3.6 Implications for ranking

1. **Run the diagnostics first** (H1). If depth does not bind on this probe, the lanes for hop-adaptive compute and test-time scaling
   will sit at the noise floor. In that case:
   - add deeper analysis hops (12–24, at a fixed variable count) to create headroom;
   - shift weight toward lookup precision (H8, H18) and the memory budget (H9, H14).
2. **Prefer eval-only and core-frozen experiments.** They cannot damage the base, and they are immune to the LR re-warm problem.
3. **Early exit in prefill must account for intermediates**, through the state metric, an answer-only scope, or patience (H2).
4. **Fix the recipe before crediting any architecture** (H3). Otherwise "improvements" may only mean "less damage from re-warming".

---

## 4. Ranked hypotheses

Ranking weighs expected information and gain against cost (eval-only first) and the risk of breaking the base.

Effect sizes are **priors [est]** and must be checked against seed noise:
- per hop, 400 problems gives SE ≈2.5 points;
- `reason_id` (1,200 problems) has SE ≈1.4 points;
- `reason_ood` (800) has SE ≈1.8 points;
- the keep threshold is 3 points, or 2x the seed SD.

### 4.1 Summary table

| # | Name | Test type | Target metric(s) | Expected effect [est] | No-op risk | LoC |
|---|---|---|---|---|---|---|
| H1 | Landing, loop sweep, propagation probe and format check *(a)* | eval-only | diagnosis | Decides whether depth binds; finds the hop-0 cause | none | ~60 (analysis) |
| H2 | Exits that account for intermediates *(b)* | eval-only | mean loops at equal quality | Text: −10–25% loops at Δbpb≤0.004. Reasoning: KL exit hurts; state, answer-only or patience exits do not | none (flag) | ~30 |
| H3 | Warm-start recipe: low LR, no re-warm, separate LR for new modules | 5-min | `val_bpb`, seed SD | Baseline stays within 0.01 of the base instead of +0.03–0.08; lower noise | none | ~15 |
| H4 | Readout refit across loop counts (core frozen) | 5-min | bpb at 1–8 loops; exits | bpb at 2 loops −0.05 or better; flat at 8 loops; ±0.003 at 4 | none (core frozen) | ~20 |
| H5 | Variable-loop fine-tune, mostly 4, plus progressive loss | 5-min | `reason_ood` at 2x loops; `scaling` | +0–5 points OOD at 8 loops; 2x-loop bpb no longer worse | none | ~12 |
| H6 | `LOOP_CONSIST` with zero-init exit adapters *(c)* | 5-min | exit frontier; final-loop metrics | −25% loops at equal bpb. Final path unchanged (adapters) or +0.005–0.02 bpb (without them) | none with adapters | ~20 |
| H7 | Hop-adaptive compute: oracle, answer-patience, halting *(d)* | eval-only, then 5-min | `reason_ood` per mean loop | 2x-loop OOD accuracy at ≤1.3x mean loops, *only if* H1 shows depth binds | eval-only none; `HALTING` not a no-op | ~25 |
| H8 | Attention temperature annealed by loop (per loop and head log-β) | eval pre-test, then 5-min | accuracy at hops 0–8 | +2–6 points if lookup sharpness binds; null otherwise | exact (log-β=0) | ~15 |
| H9 | Post-training W8/W4 quantization | eval-only | bytes, bpb, accuracy | W8: Δbpb≤0.003, 1.9x smaller. W4: +0.02–0.08 bpb, 3.6x smaller | none | ~25 (script) |
| H10 | Ensemble or most-confident loop readout | eval-only | reasoning accuracy | +0–2 points on the base; more after H4/H6 | none (flag) | ~10 |
| H11 | Norm anchoring beyond the trained loop count | eval-only | metrics at 8 loops | Halves degradation at 2x loops, if norm drift is the cause | exact at ≤4 loops | ~6 |
| H12 | Depth attention with temporal decay (`DEPTH_ATTN`) | 5-min | `scaling` at 2x; `reason_ood` | +2–5 points OOD at 8 loops | exact (zero `o`, γ=0) | ~8 |
| H13 | Per-loop LoRA adapters (relaxed recursion) | 5-min | bpb and reasoning per parameter | −0.005–0.015 bpb for +0.8M parameters (+11%); neutral per GB | exact (B=0) | ~30 |
| H14 | Terminal or convergence-aware KV reuse | eval emulation, then 5-min | KV KB/token at equal accuracy | 15 → 6 KiB per token within 2 points after fine-tuning | not a no-op | ~40 |
| H15 | LoopCD contrastive readout | eval-only | reasoning accuracy | +0–3 points at α 0.1–0.3; bpb worse | none (flag) | ~10 |
| H16 | Learned forget gate on the carried state, plus an Inkling-2-Mini control | 5-min / new base | 2x-loop stability | Neutral at 4 loops; better at 8 | exact (g=1+w, w=0) | ~6 |
| H17 | Contraction regularizer (state change or Jacobian) | 5-min | exits; 2x-loop stability | Exits sooner; risk −1–3 points reasoning | none (loss only) | ~15 |
| H18 | Differential attention in the core (zero λ) | 5-min | accuracy at hops 0–8 | +0–4 points if attention noise binds; +0.6M parameters | exact (λ=0) | ~15 |
| H19 | Progress-aligned loop supervision (PAR) | 5-min | `reason_ood`, extrapolation | High variance: big gain, or it destroys a doubling algorithm | none (loss only) | ~60 |

**Recommended order.**
1. Batch the eval-only items (H1, H2, H9, H10, H11, H15, and the pre-tests for H8 and H14) into one session of about an hour on the
   final base.
2. Run H3 to fix the recipe.
3. Then H4, H5 and H6, and re-run H2 on their outputs.
4. Pick lanes for H7, H8 and H12–H19 based on what H1 shows.

### 4.2 Details

Line numbers refer to `train.py` at commit `4fd5aa4`: `recur` at :206, `readout` at :187, `loss` at :290, the `Attention` call at :87,
and the LR schedule at :349. Every new switch defaults to off, and per `program.md` it goes into `CASES` in `tests/test_model.py`.

---

#### H1: Landing loop, loop sweep, propagation probe and format check *(must-include (a); mostly implemented)*

**Mechanism.** Measure before you optimize. Four measurements:
- **The exact landing loop:** the first loop after which the argmax never changes. It comes from `loop_readouts`.
- **Accuracy by (loop, hop):** from `eval_ckpt.py`, which sweeps loops 1–12 and hops 1–12.
- **A staircase probe:** query *every* variable of a line, exactly as in the training format, and read accuracy against the
  variable's depth at each loop. This tells us whether the algorithm is iterative in depth or doubling.
- **A format check:** the same eval problems with `"\n"` prepended. Optionally also prepend one solved distractor line.

**Evidence for:**
- Logit lens (Halawi, [2307.09476](https://arxiv.org/abs/2307.09476)).
- SDN overthinking (Kaya, [1810.07052](https://arxiv.org/abs/1810.07052)).
- TRACE landing ([2510.07880](https://arxiv.org/abs/2510.07880)).
- The staircase in Inside the Loop ([2610.24135](https://arxiv.org/abs/2610.24135)).
- Sanford's depth bound ([2402.09268](https://arxiv.org/abs/2402.09268)).

**Evidence against:**
- The logit lens can be biased (Tuned Lens, [2303.08112](https://arxiv.org/abs/2303.08112)).
- Flips are inflated by chance (§3.4, W5).

**Expected outcome.** This is a diagnosis, with these decision rules:
- If loops=2 keeps hop 8 near 0.5, the algorithm is doubling: deprioritize H7 and test-time scaling for hops ≤ 10.
- If the `\n` check lifts hop 0 by ≥10 points, part of O2 is an artifact of the format. Report it in the paper, and do not chase it
  with architecture.
- If accuracy at hops 10–12 rises with loops, H5 and H7 have headroom.

**Cheapest test.** Eval-only. Run `uv run swarm/landing.py --name base` and `uv run swarm/eval_ckpt.py --name base`, then the
add-ons in an analysis script (not part of the official metric).

**Sketch.** Additions to `landing.py`, about 60 lines:
- **Confident flips:** `correct[:, r] & (p[:, r] > .5) & correct[:, r+1]`, with a null in which loop readouts are shuffled across problems.
- **TRACE-style labels:** `explorer = n_distinct(argmax over loops) >= 3` and `late = land == L`, reported per hop.
- **The staircase:** generate lines with `prepare._forest`, append `?v=` for every variable as in training, read loop-r accuracy per depth.
- **Per-hop mean loops at the answer** in `exit_sweep`, which currently averages over hops.

**No-op risk.** None.

---

#### H2: Exits that account for intermediates *(must-include (b); `EXIT_TAU` exists)*

**Mechanism.** `EXIT_TAU` freezes a token once the KL between consecutive readouts falls below τ (or the relative L2 change, with
`state`). Four refinements, motivated by W3 and W4:

1. `EXIT_METRIC="both"`: exit only if the output KL < τ **and** the state change < τ_s. A token that is still absorbing chain
   information is still changing its state.
2. **An answer-only scope** (decode semantics): pass `exit_mask` (B,T,1) to `forward`. In the analysis script, only answer positions may
   exit; statements run every loop. This matches autoregressive decoding, where only the newest token's loops can be cut.
3. **Patience** (PABEE, [2006.04152](https://arxiv.org/abs/2006.04152); TRACE's "k consecutive verifications"; Huginn's
   `argmax-stability`): exit after k consecutive loops with an unchanged argmax.
4. **A revisit stop** (TRACE's backtrack stop): exit when the argmax returns to an earlier value. This is mainly useful at 8 or more loops.

The acceleration rule ([2509.23314](https://arxiv.org/abs/2509.23314)) needs at least 3 loops of history, so it only makes sense with a
maximum of 8 or more loops. Choose τ with Learn-then-Test ([2110.01052](https://arxiv.org/abs/2110.01052)), as CALM did, so that
"Δbpb ≤ 0.004 with 90% confidence" is a guarantee rather than a hand-tuned value.

**Evidence for:**
- 38% savings training-free at 135M ([2607.14427](https://arxiv.org/abs/2607.14427)).
- Huginn exits zero-shot (model card, KL threshold 5e-4).
- CALM ([2207.07061](https://arxiv.org/abs/2207.07061)).
- DeeBERT, BranchyNet.

**Evidence against:**
- W3 (stale intermediates).
- The 135M result trained with variable depth; our base did not.
- The early readout was never trained (W2).

**Expected effect [est].**
- On text, a KL exit should cut 10–25% of loops at Δbpb ≤ 0.004 on the base, and 25–40% after H4 or H6.
- On reasoning, a KL exit at the same τ should cost ≥5 points at deep hops.
- The `both` rule, answer-only scope and patience should cost ≤1 point.
- If the KL exit does *not* hurt reasoning, statement states converge early, which is itself informative about the algorithm.

**Cheapest test.** Eval-only: the τ sweep already in `landing.py`, plus the four variants.

**Sketch** (in `recur` at :206, about 30 lines):
```python
EXIT_PATIENCE = 0        # >0: also require k loops of unchanged argmax
EXIT_MASK = None         # analysis only: (B,T,1) bool, positions allowed to exit
...
if EXIT_METRIC in ("kl", "both", "patience"):
    d, lp = self._change(x, h, prev_lp, byte)
    a = mx.argmax(lp, -1, keepdims=True); same = a == prev_a; prev_a = a
    stable = mx.where(same, stable + 1, 0)
if EXIT_METRIC == "both":
    d = mx.maximum(d / exit_tau, state_change(x, h) / EXIT_TAU_STATE)   # exit only if both are below 1
ok = (d < thr) if not EXIT_PATIENCE else (stable >= EXIT_PATIENCE)
if EXIT_MASK is not None: ok = ok & EXIT_MASK
active = active & ~ok
```

**No-op risk.** None, because it is off by default.

---

#### H3: A warm-start recipe that does not re-warm the LR (recipe lane)

**Mechanism.**
- The base finishes its cosine at 5e-5.
- Each experiment restarts AdamW (bias-corrected updates are about LR in size, whatever the gradient) and ramps to **1e-3 within
  100 of about 255 steps**.
- That knocks the base off its minimum, and the baseline spends its 5 minutes recovering.
- Proposal:
  - a constant or short-cosine LR of 1e-4 to 3e-4 with ≤10 warm-up steps for the base parameters;
  - 10x that for new zero-init modules (MLX: two optimizers, or a learning-rate map per parameter group);
  - optionally a frozen core (see H4).

**Evidence for:**
- O3: −0.08 bpb after the resume re-warm [local].
- Gupta et al. ([2308.04014](https://arxiv.org/abs/2308.04014)) advise against re-warming on a small budget.
- `calibration.md`: retrieval suffers at LR > 1e-3 [local].

**Evidence against:**
- A higher LR may be needed for new modules to switch on within 255 steps. That is why they get a separate group.
- Switching the optimizer to Muon on an AdamW base risks extra forgetting; the evidence is contested ([2605.06654](https://arxiv.org/abs/2605.06654) against [NeurIPS'26](https://neurips.cc/virtual/2026/172331)).

**Expected effect [est].** The 1e-3 baseline sits 0.03–0.08 bpb above the base. The low-LR recipe stays within 0.01 of it,
with a lower seed SD and therefore lower keep thresholds for every lane.

**Cheapest test.** A 5-minute run, 3 seeds per arm.

**Sketch** (about 15 lines): `LR_NEW_MULT = 10`. Split `tree_flatten(model.parameters())` into the keys present in the base and new
keys, and step two `optim.AdamW` instances. Set `WARMUP_STEPS = 10` and `LR = 2e-4`.

**No-op risk.** None.

---

#### H4: Readout refit across loop counts, with the core frozen

**Mechanism.**
- Train only `coda + norm + head` on readouts of loop states at r ∈ {1..8}, keeping half the weight on r = 4.
- The core's function is untouched, so in-distribution behavior at 4 loops is preserved by construction, apart from the coda's own drift.
- Early loops become decodable, which makes landing statistics, exits (H2) and 2x-loop evaluation meaningful.

**Evidence for:**
- The damage beyond the trained budget "sits in the readout and last layer"; refitting recovers it ([2610.31943](https://arxiv.org/abs/2610.31943)).
- A non-recurrent output layer improves robustness to under-unrolling ([2609.36636](https://arxiv.org/abs/2609.36636)).
- SDN's internal classifiers ([1810.07052](https://arxiv.org/abs/1810.07052)).
- Readout confound ([2609.19934](https://arxiv.org/abs/2609.19934)).

**Evidence against.** One coda block shared across eight input distributions may underfit, and the loop-4 quality could drift by about
+0.003 bpb.

**Expected effect [est].** bpb at 2 loops improves by ≥0.05; bpb at 8 loops no longer exceeds bpb at 4; Δ ≤ 0.003 at 4 loops. The H2
frontier improves.

**Cheapest test.** A 5-minute fine-tune, followed by `landing.py` and `eval_ckpt.py` on the result.

**Sketch** (about 20 lines):
```python
READOUT_REFIT = False
# in main(): if READOUT_REFIT: model.freeze(); [m.unfreeze() for m in (model.coda, model.norm, model.head)]
# in loss():
_, _, states, byte = self.forward(batch[:, :-1], 8, return_states=True)
rs = [3] + list(rng.choice(8, 2, replace=False))          # loop 4 plus two random loops
ce = sum(w * nn.losses.cross_entropy(self.readout(mx.stop_gradient(states[r]), byte), y).mean()
         for r, w in zip(rs, (0.5, 0.25, 0.25)))
```
Running 8 loops forward without a core gradient costs about 1.6x a normal step.

**No-op risk.** None: there are no new parameters, and the core is frozen.

---

#### H5: Variable-loop fine-tune with progressive loss

**Mechanism.**
- Make the core path-independent, so that more loops do not hurt and can help OOD.
- Sample loops from an Inkling-style mix: P(4) = 0.5, the rest uniform over {2, 3, 5, 6, 8}.
- Optionally add Bansal's progressive loss: run n ~ U[0, k) loops under `stop_gradient`, then k − n loops with gradient.

**Evidence for:**
- TML varloop: N-Hop keeps improving to 10 loops, twice its training ([TML Mini post](https://thinkingmachines.ai/news/inkling-2-mini/)).
- Randomizing the loop count rescues retention, and truncation does not ([2610.19768](https://arxiv.org/abs/2610.19768)).
- Path independence ([2211.09961](https://arxiv.org/abs/2211.09961)).
- Progressive loss ([2202.05826](https://arxiv.org/abs/2202.05826)).
- Huginn's recipe ([2502.05171](https://arxiv.org/abs/2502.05171)).

**Evidence against:**
- If H1 shows doubling, there is little OOD headroom at ≤10 hops.
- 255 steps may be too few to change the dynamics (B6).
- The existing `RANDOM_LOOPS` (uniform over 1..7) puts heavy weight on 1–2 loops and may damage the base.

**Expected effect [est].** +0–5 points of `reason_ood` at 8 loops. The main measurable effect is that bpb at 8 loops is no worse than
at 4.

**Cheapest test.** A 5-minute fine-tune, compared at *matched mean loops* (about 4.5).

**Sketch** (about 12 lines): `LOOP_DIST = "inkling"`. In `train()`, draw
`k = 4 if rng.random() < .5 else rng.choice([2,3,5,6,8])`. With `PROGRESSIVE`, pass `n` into `recur` and wrap the first n loops in
`stop_gradient`. Each new k compiles its own graph; there are only 5 values.

**No-op risk.** None: there are no parameters.

---

#### H6: `LOOP_CONSIST` with zero-init exit adapters *(must-include (c); base version exists)*

**Mechanism.**
- The current implementation applies soft cross-entropy from the stop-gradient final readout to the readout of one random earlier loop,
  through the *shared* coda.
- Two refinements:
  - **(i)** distill into *all* earlier loops with weights that rise with r (LayerSkip style);
  - **(ii)** route early readouts through a per-loop zero-init low-rank adapter, `coda(h_r + B_r A_r h_r)`. The final (loop-L) path never
    uses an adapter, so final-loop behavior is protected from the multi-exit gradient conflict.

**Evidence for:**
- SDN ([1810.07052](https://arxiv.org/abs/1810.07052)), LayerSkip ([2404.16710](https://arxiv.org/abs/2404.16710)).
- LoopFormer's stop-gradient shortcut consistency ([2602.11451](https://arxiv.org/abs/2602.11451)).
- Deep supervision reduces the early-exit penalty ([2609.40305](https://arxiv.org/abs/2609.40305)).
- TRM and HRM's deep supervision ([2510.04871](https://arxiv.org/abs/2510.04871), [ARC Prize](https://arcprize.org/blog/hrm-analysis)).
- LoopOPD ([2610.10623](https://arxiv.org/abs/2610.10623)).
- The teacher and student see the same input, which is Beyer's "consistent teacher" ([2106.05237](https://arxiv.org/abs/2106.05237)).

**Evidence against:**
- Gradient conflict across exits degrades the final exit (DFS, [2407.13986](https://arxiv.org/abs/2407.13986); RomeBERT, [2101.09755](https://arxiv.org/abs/2101.09755)).
- Accordion got its gains with *no* consistency loss ([2610.32492](https://arxiv.org/abs/2610.32492)).
- Pushing early loops to copy the final one may encourage early landing and blunt the benefit of extra loops (open question 4).
- Per-loop cross-entropy cannot see state scale ([2606.24898](https://arxiv.org/abs/2606.24898)), so watch hidden-state norms.

**Expected effect [est].** −25% mean loops at Δbpb ≤ 0.004 with H2. Final-loop metrics stay within noise with adapters, and lose
+0.005–0.02 bpb without them.

**Cheapest test.** A 5-minute fine-tune with `LOOP_CONSIST` ∈ {0.1, 0.3}, run with and without adapters.

**Sketch** (about 20 lines): `self.exit_A = [nn.Linear(d, 16, bias=False) for _ in range(LOOPS)]` and
`self.exit_B = [zero-init nn.Linear(16, d, bias=False) …]`. Then `readout_early(h, r) = self.readout(h + self.exit_B[r](self.exit_A[r](h)))`,
used only in the consistency term and at exit time. Parameters: 4 × 2 × 384 × 16 ≈ 49k.

**No-op risk.** Exact: B = 0 and the final path is unchanged.

---

#### H7: Hop-adaptive compute *(must-include (d))*

**Mechanism.** Give deep queries more loops and shallow ones fewer, in three steps:
- **(i) Oracle, eval-only.** From the `eval_ckpt.py` sweep, the smallest loop count that reaches each hop's maximum accuracy gives the
  achievable frontier.
- **(ii) Training-free policy.** Patience exit at the answer position only (H2's answer-only scope plus patience), with a maximum of 12 at
  eval. Deep hops keep changing for longer, so they automatically get more loops.
- **(iii) Learned policy.** `HALTING`, which already exists, or an Ouro-style entropy-regularized exit gate.

**Evidence for:**
- Snell's compute-optimal test-time scaling that adapts to difficulty ([2408.03314](https://arxiv.org/abs/2408.03314)).
- Adaptive steps for length generalization (Fan, [2409.15647](https://arxiv.org/abs/2409.15647)).
- Think-at-Hard's oracle at +7.3% ([2511.08577](https://arxiv.org/abs/2511.08577)).
- Online halting tracks execution state (ANIRA, [2602.08864](https://arxiv.org/abs/2602.08864)).

**Evidence against:**
- ACT is brittle (Graves [1603.08983](https://arxiv.org/abs/1603.08983); PonderNet [2107.05407](https://arxiv.org/abs/2107.05407)).
- Exits "collapse" (Ouro [2510.25741](https://arxiv.org/abs/2510.25741)).
- A router trained on convergence labels saved nothing ([2607.14427](https://arxiv.org/abs/2607.14427)).
- "Harder instances do not consistently benefit more" ([2609.36636](https://arxiv.org/abs/2609.36636)).
- Allocation that tracks difficulty does not guarantee generalization (ANIRA).
- The user's own earlier MORPH work found random depth beating learned depth on Boolean reasoning (project memory, not re-verified).
- If H1 shows doubling, there is nothing to adapt at ≤10 hops.

**Expected effect [est].** Only if H1 shows accuracy at hops 10–12 rising with loops: the policy should match the OOD accuracy of 2x
loops at ≤1.3x mean loops. Otherwise the effect is null.

**Cheapest test.** Steps (i) and (ii) are eval-only. Step (iii) is a 5-minute fine-tune, and it is not a no-op.

**Sketch.** Steps (i) and (ii) reuse H1 and H2 code, about 25 lines in the analysis script. For step (iii), sweep `PONDER_COST`
∈ {1e-3, 3e-3} and initialize the halting bias at −4 so that soft halting starts nearer to a no-op (p ≈ 0.018 per loop instead of 0.12).

**No-op risk.** `HALTING` changes the computation at init. The −4 bias reduces but does not remove the change.

---

#### H8: Attention temperature annealed by loop (new idea grounded in Hopfield theory)

**Mechanism.**
- In modern Hopfield terms, retrieval converges to a *metastable average* when β times the separation between patterns is small, and
  to a *single pattern* when it is large (Ramsauer, [2008.02217](https://arxiv.org/abs/2008.02217)).
- Give each core attention head a learnable log-β per loop, `q ← q·exp(logβ[r, blk, head])`.
- This lets later loops sharpen the letter lookup that O2 identifies as the bottleneck.

**Evidence for:**
- Scaling attention logits improves key-information retrieval (SSMax, [2501.19399](https://arxiv.org/abs/2501.19399)).
- Cancelling attention noise helps retrieval (Differential Transformer, [2410.05258](https://arxiv.org/abs/2410.05258)).
- Adaptive temperature keeps attention sharp OOD ([2410.01104](https://arxiv.org/abs/2410.01104)).

**Evidence against:**
- Sharpening at decode time is dataset-dependent ([2610.02185](https://arxiv.org/abs/2610.02185) [snip]).
- Learned temperatures can drift, so clip them [snip, source unclear].
- No paper tests a per-loop β schedule (searched; none found).

**Expected effect [est].** +2–6 points across hops 0–8 if lookup sharpness binds; null otherwise; bpb about neutral.

**Cheapest test.**
- **Pre-test, eval-only:** multiply `q` in the core blocks by a single global factor β ∈ {1.0, 1.25, 1.5, 2.0} at 4 and 8 loops. If hop-0
  accuracy rises by ≥3 points, move on.
- Then a 5-minute fine-tune with the learnable version.

**Sketch** (about 15 lines): `self.loop_logbeta = mx.zeros((16, CORE, N_HEADS))`. Pass `r` into `Block.__call__` and on into
`Attention.__call__` (:87). After RoPE, apply
`q = q * mx.exp(mx.clip(logbeta[min(r, train_loops-1)], -1, 1.5))[None, :, None, None]`. Because `mx.fast.scaled_dot_product_attention`
takes a scalar scale, scaling q per head is the equivalent.

**No-op risk.** Exact, since exp(0) = 1.

---

#### H9: Post-training quantization: the direct "per GB" lever

**Mechanism.** Quantize the base to W8 or W4 with `mlx.nn.quantize` (group 64; [docs](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.nn.quantize.html)).
Try mixed precision, with core layers at 8 bits and the rest at 4, and the reverse. Report bytes.

**Evidence for:**
- Inkling 2 Mini itself ships NVFP4 experts with MXFP8 in the last 4 layers to hit ~20 GB ([TML Mini post](https://thinkingmachines.ai/news/inkling-2-mini/)).
- Not all shared layers are equally sensitive ([2609.30820](https://arxiv.org/abs/2609.30820)).

**Evidence against.** Error compounds across loops (LoopQ [2605.16343](https://arxiv.org/abs/2605.16343); LoopQuant [2610.16368](https://arxiv.org/abs/2610.16368)),
and a 7M model has less redundancy than a large one [est].

**Expected effect [est].**
- W8: Δbpb ≤ 0.003, Δacc ≤ 1 point, 13.9 → 7.4 MiB.
- W4: +0.02–0.08 bpb, −0–5 points, 3.9 MiB.
- Mixed precision should fall in between.

**Cheapest test.** Eval-only: `swarm/quant_eval.py` calls `nn.quantize(model, 64, bits, class_predicate=...)` and then
`prepare.evaluate`. `prepare.py` is not edited.

**Sketch** (about 25 lines): load the base, quantize, evaluate, and write
`bytes = Σ(bits/8·n + 4·n/64)` for the quantized layers plus 2·n for the rest.

**No-op risk.** None.

---

#### H10: Ensemble or most-confident loop readout

**Mechanism.** Undo destructive flips without any training. Two options:
- output the mean log-probability over loops r0..L;
- or take the loop with the highest maximum probability, as Schwarzschild did.

**Evidence for:**
- Voting across internal classifiers ([2105.13792](https://arxiv.org/abs/2105.13792)).
- SDN's confusion signal.
- The most-confident iteration ([2106.04537](https://arxiv.org/abs/2106.04537)).
- Think-at-Hard's oracle headroom.

**Evidence against.** The early readouts of the base are off-distribution (W2), so an ensemble may dilute the final answer, and bpb gets
worse with any extra readout.

**Expected effect [est].** +0–2 points on the base, and +1–4 after H4 or H6, but only if confident flips (H1) are ≥3% of problems.

**Cheapest test.** Eval-only.

**Sketch** (about 10 lines): `READOUT_ENSEMBLE = 0`. In `forward`, when > 0, return
`logsumexp(stack([log_softmax(readout(s)) for s in states[-K:]]), 0) - log K`.

**No-op risk.** None, because it is off by default.

---

#### H11: Norm anchoring beyond the trained loop count

**Mechanism.** At loops r ≥ 4, rescale each token's state to its norm at the last trained loop, `x ← x·‖h₃‖/‖x‖`. This caps radial
drift, which a coda trained at loop 4 has never seen.

**Evidence for:**
- PAR: removing norm anchoring cuts extrapolation from 16x to below 2.2x ([2610.27628](https://arxiv.org/abs/2610.27628)).
- Recall plus outer normalization gives stable fixed points ([2604.15259](https://arxiv.org/abs/2604.15259)).
- Norm blow-up through RMSNorm readouts ([2606.24898](https://arxiv.org/abs/2606.24898)).

**Evidence against.** "Agreement rather than normalisation": normalization was not what made depth reusable ([2610.31943](https://arxiv.org/abs/2610.31943)).
Our pre-norm blocks may already be norm-robust [est].

**Expected effect [est].** If degradation at 2x loops comes from drift, this halves it. Otherwise it is null. It costs nothing at 4 loops.

**Cheapest test.** Eval-only, read from the `scaling` metric at 8 loops.

**Sketch** (about 6 lines, in `recur` at :206): `if NORM_ANCHOR and r >= self.train_loops: x = x * (ref / (norm(x) + 1e-6))`, with
`ref = norm(h)` captured at `r == self.train_loops - 1`.

**No-op risk.** Exact at ≤4 loops.

---

#### H12: Depth attention with temporal decay (`DEPTH_ATTN` lane, refined)

**Mechanism.** Each loop attends over its own recent loop states. Already implemented, with a zero-init output projection. Add a learned
decay bias `−γ·age`, with γ starting at 0, as in InfiLoop.

**Evidence for:**
- ARLoop: sliding-window depth attention ≥ full history ([2611.05234](https://arxiv.org/abs/2611.05234)).
- Kimi Attention Residuals ([2603.15031](https://arxiv.org/abs/2603.15031)), which Inkling 2 uses.
- InfiLoop at 7M keeps improving out to 20,000 steps ([2610.11570](https://arxiv.org/abs/2610.11570)).
- Think-at-Hard's depth-dimension attention.

**Evidence against.** At 4 loops there is little history to attend over, and the gains reported at 140M+ may not transfer [est].

**Expected effect [est].** Flatter curve at 2x loops; +2–5 points `reason_ood` at 8 loops, mostly when combined with H5.

**Cheapest test.** A 5-minute fine-tune with K ∈ {2, 3}.

**Sketch** (about 8 lines): in `DepthAttention.__call__`, add `- softplus(self.gamma) * age` to the logits, with `self.gamma` starting
at −10 (≈0 decay) and `age = arange(K)[::-1]`.

**No-op risk.** Exact (zero `o`).

---

#### H13: Per-loop LoRA adapters (relaxed recursion)

**Mechanism.** Give each loop its own zero-init low-rank deltas (rank 8–16) on the core's qkv, o, gate, up and down projections. Each
loop can then specialize its transition while the bulk of the weights stays shared.

**Evidence for:**
- Relaxed Recursive Transformers ([2410.20672](https://arxiv.org/abs/2410.20672)).
- "Loop-specific low-rank adapters improve quality, with larger gains as loops are added" ([2610.40325](https://arxiv.org/abs/2610.40325)).
- Think-at-Hard's LoRA by iteration depth.
- A selector on a loop's input chooses among learned transitions ([2609.39892](https://arxiv.org/abs/2609.39892)).
- `LOOP_EMBED` (a bias indexed by loop) is the zero-rank version.

**Evidence against.**
- It adds unique parameters, which cost bytes.
- Loops at test time beyond the trained count have no adapter of their own, which hurts test-time scaling.
- ALBERT: sharing the FFN is where quality is lost ([1909.11942](https://arxiv.org/abs/1909.11942)), so the deltas should go on the FFN first.

**Expected effect [est].** −0.005 to −0.015 bpb and +1–3 points for about +0.8M parameters at rank 16 (+11%). Per GB, that is about
neutral.

**Cheapest test.** A 5-minute fine-tune.

**Sketch** (about 30 lines): a `LoRA(d_in, d_out, r)` module with B zeroed; `core_lora[r][blk][name]`. Pass `r` into
`Block.__call__`. For r ≥ the trained loop count, reuse the last adapter, or use none.

**No-op risk.** Exact (B = 0).

---

#### H14: Terminal or convergence-aware KV reuse (kv-reuse lane)

**Mechanism.** Cut the KV cache from 15 to 6 KiB per token by letting later loops reuse one loop's K/V:
- the existing `KV_SHARE` reuses loop 1;
- a "terminal" variant reuses the last loop (Huang Part II), which needs a decode-faithful emulation done in two passes;
- or per token, reuse once the state has converged (`program.md` lane 4).

**Evidence for:**
- Terminal KV sharing with "almost no loss" near fixed points, and a 3x smaller KV cache at 1.6B ([2610.06833](https://arxiv.org/abs/2610.06833)).
- Huginn's KV budget `i mod k` works zero-shot ([2502.05171](https://arxiv.org/abs/2502.05171)).
- MoR's recursive KV sharing ([2507.10524](https://arxiv.org/abs/2507.10524)).

**Evidence against.**
- Our base is probably not near a fixed point after 4 loops (O4).
- Sharing loop-1 KV removes the propagated chain information that later lookups need (W3).
- MoR notes the KV I/O bottleneck.

**Expected effect [est].** Zero-shot on the base, ≥5 points of reasoning lost. After a 5-minute `KV_SHARE` fine-tune, within 2 points
at 2.5x smaller KV, and better combined with H5.

**Cheapest test.** An eval-only zero-shot sweep: first versus last loop, and budgets k ∈ {1, 2}. Then a 5-minute fine-tune of the best.

**Sketch.** About 40 lines for the decode-faithful two-pass emulation in an analysis script: pass 1 computes the final K/V of every
position; pass 2 recomputes each loop using past positions' terminal K/V, and the current position's own K/V for the diagonal.

**No-op risk.** Not a no-op.

---

#### H15: LoopCD contrastive readout

**Mechanism.** `logits = (1+α)·lp_L − α·lp_1`, restricted to tokens with `p_L ≥ 0.1·max p_L` (the usual plausibility mask).

**Evidence for.** Without training: Ouro-2.6B on AIME'24 goes 61.9 → 73.3, and Huginn on HumanEval goes 22.6 → 31.7 ([2610.02185](https://arxiv.org/abs/2610.02185)).

**Evidence against.**
- Those results are from multi-billion-parameter models in thinking mode.
- Our loop-1 readout is off-distribution (W2).
- Distorting the distribution harms bpb by construction, so it is only usable for accuracy.

**Expected effect [est].** +0–3 points of reasoning at α ∈ {0.1, 0.3}; bpb worse.

**Cheapest test.** Eval-only.

**Sketch.** About 10 lines in `forward`, behind `LOOPCD_ALPHA = 0.0`.

**No-op risk.** None, because it is off by default.

---

#### H16: Learned forget gate on the carried state, plus an Inkling-2-Mini control

**Mechanism.**
- Our injection is `x_r = h_{r−1} + e`. Make it `x_r = e + g ⊙ h_{r−1}` with `g = 1 + w`, w a per-channel vector starting at 0.
- The model can then learn g < 1, which gives contraction and fading memory (the echo-state property) and therefore stable fixed points.
- The *control* is Inkling 2 Mini's own wiring: restart each pass from e and add a gated "message" starting at g = 0. That needs a new base
  (§1.5a).

**Evidence for:**
- TML's gate that starts closed ([TML Mini post](https://thinkingmachines.ai/news/inkling-2-mini/)).
- LSTM forget-gate initialization ([PMLR v37](https://proceedings.mlr.press/v37/jozefowicz15.html)).
- Echo-state contraction ([2507.18467](https://arxiv.org/abs/2507.18467)).
- Recall ([2202.05826](https://arxiv.org/abs/2202.05826)).
- Closing the Loop's single learned input-mixing scalar ([2610.00673](https://arxiv.org/abs/2610.00673)).

**Evidence against.** At 4 loops, contraction may slow down useful propagation [est].

**Expected effect [est].** Neutral at 4 loops; better bpb and accuracy at 8 loops.

**Cheapest test.** A 5-minute fine-tune for the gate. The control needs a new base, which is a human decision.

**Sketch.** About 6 lines at :221: `x = e + (1 + self.carry_w) * h if r > 0 else h`.

**No-op risk.** Exact.

---

#### H17: Contraction regularizer

**Mechanism.** Add `λ·mean(‖h_L − h_{L−1}‖² / ‖h_{L−1}‖²)`, or a Jacobian Frobenius estimate from one `mx.jvp` with a random vector.
This pushes the last loops toward a fixed point, which makes exits earlier and 2x loops safer.

**Evidence for:**
- DEQ Jacobian regularization ([2106.14342](https://arxiv.org/abs/2106.14342)).
- Lipschitz RNN ([2006.12070](https://arxiv.org/abs/2006.12070)).
- Path independence ([2211.09961](https://arxiv.org/abs/2211.09961)).
- Haber & Ruthotto ([1705.03341](https://arxiv.org/abs/1705.03341)).

**Evidence against.**
- Penalizing change suppresses useful late corrections. Looped-DiT reports that deeper loops correct earlier mistakes
  ([2609.40305](https://arxiv.org/abs/2609.40305)), and "Right Direction, Wrong Step" suggests scaling steps rather than suppressing
  them ([2609.16665](https://arxiv.org/abs/2609.16665)).

**Expected effect [est].** The state change at loop 4 drops 2–5x and H2 saves more loops; −1 to −3 points of reasoning is possible.

**Cheapest test.** A 5-minute fine-tune with λ ∈ {1e-3, 1e-2}.

**Sketch.** About 15 lines in `loss` at :290.

**No-op risk.** None: it is a loss term only.

---

#### H18: Differential attention in the core, starting at zero

**Mechanism.** Use `attn = SDPA(q,k,v) − λ_h·SDPA(q₂,k₂,v)` with new projections q₂ and k₂ and a per-head λ_h starting at 0. Subtracting
a second map cancels attention noise that is common to both, which sharpens lookups.

**Evidence for.** Better key-information retrieval and less distraction ([2410.05258](https://arxiv.org/abs/2410.05258)).

**Evidence against.**
- About +0.59M parameters (+8%).
- While λ = 0, the gradients to q₂ and k₂ are zero, so they wake up slowly.
- The gains were shown on LLM pretraining from scratch.

**Expected effect [est].** +0–4 points across hops if attention noise binds. Run H8 first; it is cheaper.

**Cheapest test.** A 5-minute fine-tune, with the new modules on H3's higher LR.

**Sketch.** About 15 lines in `Attention`.

**No-op risk.** Exact (λ = 0).

---

#### H19: Progress-aligned loop supervision (PAR)

**Mechanism.** Supervise the readout at loop k so that, at query positions, it predicts the queried variable's k-th ancestor. The targets
can be parsed from the reasoning rows inside `train.py` (rows `[0, mix)`) without touching the data. Add norm anchoring as well.

**Evidence for.** 16x length extrapolation on S5, and removing the alignment hurts ([2610.27628](https://arxiv.org/abs/2610.27628)).
The staircase emerges on its own anyway ([2610.24135](https://arxiv.org/abs/2610.24135)).

**Evidence against.**
- It forces one hop per loop: at most 4 hops at 4 loops, below the 6 hops seen in training.
- If the base uses doubling (H1), this *destroys* a better algorithm.
- An alternative is to align loop k with ancestor 2^k (doubling progress).
- The parsing costs about 60 lines, against the simplicity criterion.

**Expected effect [est].** High variance. Run it only after H1 shows the algorithm is iterative in depth.

**Cheapest test.** A 5-minute fine-tune.

**Sketch.** About 60 lines of numpy target construction plus an extra cross-entropy at query positions for loops 1..L.

**No-op risk.** None: it is a loss only. It does, however, strongly change training.

---

## 5. Known failures to avoid

1. **Measuring a capability that has not formed yet.** Five-minute from-scratch runs leave retrieval at chance ([local]
   `results/calibration.md`). Induction circuits form in a phase change whose timing depends on data, batch and context
   ([2404.07129](https://arxiv.org/abs/2404.07129), [2511.16893](https://arxiv.org/abs/2511.16893)). Never compare architectures
   trained from scratch for 5 minutes.
2. **Re-warming the LR on a converged base.** −0.08 bpb within 15 minutes after the resume (O3, [local]); see Gupta
   ([2308.04014](https://arxiv.org/abs/2308.04014)). It applies to `pretrain_base.py --resume` *and* to every experiment. See H3.
3. **Brittle learned halting.**
   - ACT's sensitivity to the time penalty ([1603.08983](https://arxiv.org/abs/1603.08983)) and its instability ([2107.05407](https://arxiv.org/abs/2107.05407)).
   - Exit collapse ([2510.25741](https://arxiv.org/abs/2510.25741)).
   - A router trained on convergence labels that saved nothing ([2607.14427](https://arxiv.org/abs/2607.14427)).
   - Random depth beating learned depth in the user's earlier MORPH work (project memory, unverified).
   - Prefer training-free, online convergence rules.
4. **Expecting test-time scaling from fixed-loop training.**
   - Retention at 2x loops fell to 22% ([2610.19768](https://arxiv.org/abs/2610.19768)).
   - The from-scratch baseline got worse at both 2 and 8 loops (O4).
   - Path dependence ([2211.09961](https://arxiv.org/abs/2211.09961)).
5. **Freezing intermediates with per-token exit** (W3). A token's own next-byte readout says nothing about whether other tokens still
   need its state.
6. **Trusting raw flip counts or logit-lens landing on a fixed-loop model** (W2, W5). Correct for chance and refit the readout first.
7. **Claiming wall-clock savings.** Frozen tokens are still computed (README, `program.md`). The KL exit adds a coda pass per loop (W4).
8. **Multi-exit gradient conflict** degrading the final readout ([2407.13986](https://arxiv.org/abs/2407.13986),
   [2101.09755](https://arxiv.org/abs/2101.09755), [2407.14320](https://arxiv.org/abs/2407.14320)). Prefer stop-gradient teachers and
   adapters only on the exit path (H6).
9. **Uncontrolled hidden-state norms** under per-loop losses through RMSNorm readouts ([2606.24898](https://arxiv.org/abs/2606.24898)).
   Log ‖h_r‖ for every run.
10. **KV sharing from loop 1 without training** removes propagated information (H14). MoR also notes the KV I/O bottleneck
    ([2507.10524](https://arxiv.org/abs/2507.10524)).
11. **Quantizing looped weights per layer without checking how error compounds across loops** ([2605.16343](https://arxiv.org/abs/2605.16343),
    [2610.16368](https://arxiv.org/abs/2610.16368)).
12. **Optimizer switches on a warm start.** Moving to Muon from an AdamW base may cause extra forgetting; the evidence is contested
    ([2605.06654](https://arxiv.org/abs/2605.06654) against [NeurIPS'26](https://neurips.cc/virtual/2026/172331)). Test it in the recipe
    lane with seeds before letting other lanes depend on it.
13. **Reading the OOD split as pure depth.** Hops 8 and 10 also have 19 and 23 variables against ≤15 in training (B3).
14. **Small-sample evals.** Quick evals at 100 per hop carry ±5 points; the full eval at 400 per hop carries ±2.5 points per hop. The
    README's claim of a "depth-limited" profile came from these and is stale (O1).
15. **Name collisions.** CALM (Confident Adaptive LM, 2022) is not CALM (Continuous Autoregressive LM, 2025). TML's Inkling is not this
    repo's Inkling.
16. **The ratio form of "intelligence per GB"** rewards tiny models trivially. Report frontiers with a capability floor (§1.4).

---

## 6. Open questions for debate

1. **Does depth bind at all on this probe?** If H1 shows doubling or a format artifact, should the project:
   - add deeper analysis hops (12–24) at a fixed variable count;
   - or move the paper's claim from "loops buy serial depth" to "loops buy lookup precision per byte"?
2. **What does per-token exit mean in a prefill evaluation**, where tokens are intermediates? Should official exit numbers follow
   decode semantics (answer-only) and treat statement tokens as non-exitable?
3. **What exactly is the goal metric?**
   - A ratio or a Pareto frontier?
   - Should KV count at full 512-byte context or at the probe's typical length (about 100 bytes)?
   - How should compute (loops) be priced against bytes?
   - What capability floor should apply?
4. **Do loop consistency and test-time scaling conflict?** Pushing early loops toward the final answer may train the model to land early
   and stop using extra loops. LoopFormer argues they align ([2602.11451](https://arxiv.org/abs/2602.11451)); Accordion did without a
   consistency loss ([2610.32492](https://arxiv.org/abs/2610.32492)). Which should we test first?
5. **Can a 255-step warm start teach new algorithms**, or only select among existing ones ([2609.39892](https://arxiv.org/abs/2609.39892))?
   If only select, should the base itself be retrained with variable loops (the TML varloop recipe), which is a 6-hour human decision,
   before running H5, H7 or H12?
6. **Should the paper include the Inkling-2-Mini-recipe control** (§1.5a), at the cost of another base? Is replicating TML's
   "middle beats last" result at 7M worth a base?
7. **The chunked core against lookup precision.** `CHUNK=K` pools K bytes before the core. Does that blur single-byte identifiers (`k=q;`
   spans chunk boundaries), which are exactly what O2 says is fragile?
8. **Is hop-0 weakness an optimization problem or a capacity problem?** Hop-0 accuracy has hardly moved since about 30 minutes while bpb
   kept falling. Would a short reasoning-weighted phase fix it more cheaply than any architecture change, and would that count as
   changing the data (forbidden) or the recipe (allowed in `train.py`)?
9. **External baselines (§1.5c).** Is it worth the disk (~4.8 GB free) and the user's approval to run Ouro-1.4B or Huginn on our probe as
   reference points, given that they are tokenized and 200–500x larger?

---

*Search log: 81 web searches. About 55 pages were read via a fetch tool: TML Inkling 2 Mini and Small pages, LM Studio, the HF docs and
blog, TRACE's HTML, and about 45 arXiv abstracts. All 2026 abstracts in §2.5 were read except those marked [snip].*
