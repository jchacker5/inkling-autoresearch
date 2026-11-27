# Papers the user asked us to use (input to the debate stage)

## 1. TRACE: "Do LLMs Really Need 10+ Thoughts for 'Find the Time 1000 Days Later'? Towards Structural Understanding of LLM Overthinking"
Zhang, Mohananey, Chronopoulou, Papalampidi, Gupta, Munkhdalai, Wang, Upadhyay (UMich + Google DeepMind). Only the abstract and page 1 were read.
- Thinking models are 5-20x slower on simple queries with little accuracy gain.
- Thoughts are split into minimal sub-thoughts and a thought-progression graph. Two patterns cause overthinking:
  Explorer (over-exploration) and Late Landing (over-verification after the answer is fixed).
- Overthinking is defined by utility (does a further step change or improve the answer), not by length.
- Already implemented in PR #1: landing-loop diagnostic (swarm/landing.py), EXIT_TAU utility exit, LOOP_CONSIST.

## 2. CALM: "Continuous Autoregressive Language Models", arXiv 2510.27688
Shao, Li, Meng, Zhou (WeChat AI, Tencent), 31 Oct 2025. Code: github.com/shaochenze/calm.
- A high-fidelity VAE compresses a chunk of K tokens into one continuous vector (K=4, latent 128, >99.9% reconstruction).
  The LM predicts the next vector, so it takes K times fewer autoregressive steps.
- AE details: beta=0.001 KL with per-dim KL clipping at 0.5 (plain KL collapsed 71/128 dims); latent dropout 0.15;
  token masking 0.15. Input to the LM is the previous chunk's discrete tokens (2-layer MLP compressor); feeding
  latents back in was worse (4.70 vs 3.25 BrierLM).
- Likelihood-free generative head trained with the energy score (N=8 model samples, M=100 targets, alpha=1).
  Single step, beats diffusion and matches flow matching.
- BrierLM: a likelihood-free metric that correlates with CE (r = -0.966).
- K=4 CALM-M (371M) is about Transformer-S (281M) at 44% fewer train FLOPs and 34% fewer inference FLOPs.
  K=8 degrades. (These table numbers came from a summary of the truncated HTML; check them against the PDF before citing.)
- Relevance to us. A softmax byte head must stay, or bits per byte cannot be measured. The transferable idea is
  to run the looped core over K-byte chunks (an hourglass): byte prelude -> pool K bytes -> looped core on L/K
  positions -> unpool -> byte coda with causal intra-chunk decoding. Expected effect: core FLOPs and KV cache
  divided by K, so more loops or width fit in the same budget. Risk: blurring identifiers in the variable-tracking task.
  This needs a new base and cannot be a zero-init warm start. Sweep K in {1,2,4} at matched wall clock.
