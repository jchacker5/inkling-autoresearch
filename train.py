"""
Inkling looped transformer (MLX). THE ONLY FILE AGENTS EDIT.

Architecture: prelude blocks -> shared core blocks looped LOOPS times -> coda blocks -> head.
Experiment switches below map to the research directions in program.md:
  DEPTH_ATTN      attention over the previous K recurrent states            (ARLoop-style)
  HALTING         per-token learned halting with a ponder cost             (MoR / ACT-style)
  HISTORY         history-state injection across loops                     (history-aware recurrence)
  LOOP_EMBED      loop-index (timestep) conditioning                       (history-aware recurrence)
  KV_SHARE        core loops >1 reuse loop-1 keys/values: smaller KV cache (fixed-point / KV reuse)
  EXIT_TAU        training-free early exit: a token stops looping once one more loop
                  changes its prediction by < tau (utility-based, TRACE-style overthinking cut)
  LOOP_CONSIST    self-distillation: earlier loops learn to predict the final loop's output,
                  so early exits lose less (cf. Shallow-Deep Networks, CALM)
Usage: uv run train.py > run.log 2>&1
"""
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from prepare import BASE_CKPT, SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, Batches, evaluate

# MLX keeps freed GPU buffers for reuse. With the variable-length eval batches that cache grew to 15 GB during
# the 6-hour base run, pushed macOS into swap and filled the disk until the kernel killed training. Cap it.
mx.set_cache_limit(2 * 2**30)

# ---------------------------------------------------------------------------
# Hyperparameters and experiment switches
# ---------------------------------------------------------------------------
SEED = int(os.environ.get("INKLING_SEED", "0"))   # swarm/run.py --seed sets this
D_MODEL = 384
N_HEADS = 6
MLP_MULT = 4
PRELUDE, CORE, CODA = 1, 2, 1     # unique blocks; effective depth = PRELUDE + CORE*LOOPS + CODA
LOOPS = 4                          # recurrent iterations at train time (and default at eval)
RANDOM_LOOPS = False               # sample loops in [1, 2*LOOPS-1] per step (mean LOOPS) for test-time scaling
BPTT_LOOPS = None                  # backprop only through the last k loops (None = full)
INJECTION = "add"                  # "add": re-inject prelude output each loop; "none"
LOOP_EMBED = False                 # add projected sinusoidal loop-index embedding each loop
HISTORY = "none"                   # "none" | "mean" | "gated": inject mean of earlier loop states
DEPTH_ATTN = 0                     # >0: each loop attends per-token over the last K loop states
DEPTH_ATTN_DIM = 64
HALTING = False                    # per-token ACT-style halting (dense in training, hard at eval)
PONDER_COST = 0.01                 # penalty per expected loop (in nats)
HALT_THRESHOLD = 0.5               # eval: token freezes once cumulative halt prob crosses this
KV_SHARE = False                   # core loops >1 attend with loop-1 K/V (KV cache ~ 1 loop)
EXIT_TAU = 0.0                     # eval only: >0 freezes a token once its loop-to-loop change falls below tau
EXIT_METRIC = "kl"                 # "kl": KL(new || previous readout through coda+head); "state": relative L2 change
LOOP_CONSIST = 0.0                 # train: weight of soft CE from final-loop readout to one random earlier loop

INIT_FROM_BASE = True              # continue from the shared base checkpoint (see pretrain_base.py)
BATCH_SIZE = 32
LR = 1e-3                          # peak LR (from-scratch base pretraining also uses 1e-3)
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
FINAL_LR_FRAC = 0.05
DTYPE = mx.float32                 # activation dtype; params are fp32 (bf16 only helps if params are cast too)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.rope = nn.RoPE(self.hd)

    def kv(self, x):
        B, T, _ = x.shape
        _, k, v = mx.split(self.qkv(x), 3, axis=-1)
        k = self.rope(k.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3))
        return k, v.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)

    def __call__(self, x, kv=None):
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = self.rope(q.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3))
        if kv is None:
            k = self.rope(k.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3))
            v = v.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)
        else:
            k, v = kv
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask="causal")
        return self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D)), (k, v)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = Attention(d, h)
        hidden = MLP_MULT * d * 2 // 3          # SwiGLU with ~4x-MLP parameter count
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def __call__(self, x, kv=None):
        a, kv = self.attn(self.n1(x), kv)
        x = x + a
        y = self.n2(x)
        return x + self.down(nn.silu(self.gate(y)) * self.up(y)), kv


class DepthAttention(nn.Module):
    """Per-token attention over the stack of previous recurrent states (ARLoop-style)."""

    def __init__(self, d, dk):
        super().__init__()
        self.n = nn.RMSNorm(d)
        self.q, self.k, self.v = nn.Linear(d, dk, bias=False), nn.Linear(d, dk, bias=False), nn.Linear(d, dk, bias=False)
        self.o = nn.Linear(dk, d, bias=False)
        self.o.weight = mx.zeros_like(self.o.weight)  # starts as identity
        self.dk = dk

    def __call__(self, h, states):
        S = mx.stack([self.n(s) for s in states], axis=2)               # B,T,K,D
        q = self.q(self.n(h))[:, :, None, :]                           # B,T,1,dk
        w = mx.softmax((q * self.k(S)).sum(-1) / math.sqrt(self.dk), axis=-1)  # B,T,K
        return h + self.o((w[..., None] * self.v(S)).sum(2))


def loop_features(r, d):
    freqs = mx.exp(-math.log(100.0) * mx.arange(d // 2) / (d // 2))
    return mx.concatenate([mx.sin(r * freqs), mx.cos(r * freqs)])


class Inkling(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.train_loops = LOOPS
        self.emb = nn.Embedding(VOCAB_SIZE, d)
        self.prelude = [Block(d, N_HEADS) for _ in range(PRELUDE)]
        self.core = [Block(d, N_HEADS) for _ in range(CORE)]
        self.coda = [Block(d, N_HEADS) for _ in range(CODA)]
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        if LOOP_EMBED:
            self.loop_proj = nn.Linear(d, d, bias=False)
            self.loop_proj.weight = mx.zeros_like(self.loop_proj.weight)
        if HISTORY != "none":
            self.hist_scale = mx.zeros((d,))
            if HISTORY == "gated":
                self.hist_gate = nn.Linear(d, d)
        if DEPTH_ATTN:
            self.depth_attn = DepthAttention(d, DEPTH_ATTN_DIM)
        if HALTING:
            self.halt = nn.Linear(d, 1)
            self.halt.bias = mx.array([-2.0])   # start by mostly continuing
        self.mean_loops_per_token = float(LOOPS)

    def kv_bytes_per_token(self):
        core_layers = CORE * (1 if KV_SHARE else self.train_loops)
        return 2 * D_MODEL * 2 * (PRELUDE + core_layers + CODA)   # K and V, bf16

    def readout(self, h):
        """Decode a loop state through the coda and head (an exact 'stop after this loop' logit lens)."""
        for blk in self.coda:
            h, _ = blk(h)
        return self.head(self.norm(h)).astype(mx.float32)

    def _change(self, x, h, prev_lp):
        """Per-token change from one loop to the next. Returns (change (B,T,1), log-probs of x's readout)."""
        if EXIT_METRIC == "state":
            num = mx.sqrt(mx.sum(mx.square((x - h).astype(mx.float32)), axis=-1, keepdims=True))
            den = mx.sqrt(mx.sum(mx.square(h.astype(mx.float32)), axis=-1, keepdims=True)) + 1e-6
            return num / den, None
        lp = nn.log_softmax(self.readout(x), axis=-1)
        if prev_lp is None:
            return None, lp
        return mx.sum(mx.exp(lp) * (lp - prev_lp), axis=-1, keepdims=True), lp

    def recur(self, e, loops, hard_halt=False, exit_tau=0.0):
        """Run the recurrent core. Returns (state to decode, expected loops per token, per-loop states)."""
        h, states, kvs = e, [], [None] * CORE
        B, T, _ = e.shape
        use_exit = exit_tau > 0 and not HALTING
        prev_lp = None
        active = mx.ones((B, T, 1), dtype=mx.bool_)          # hard halting: tokens still computing
        remain = mx.ones((B, T, 1), dtype=mx.float32)        # soft halting: prob mass not yet halted
        out = mx.zeros_like(e)
        expected = mx.zeros((B, T), dtype=mx.float32)
        for r in range(loops):
            if BPTT_LOOPS is not None and r == loops - BPTT_LOOPS:
                h = mx.stop_gradient(h)
            x = h + e if (INJECTION == "add" and r > 0) else h
            if LOOP_EMBED:
                x = x + self.loop_proj(loop_features(float(r), D_MODEL).astype(x.dtype))
            if HISTORY != "none" and states:
                m = sum(states) / len(states)
                g = mx.sigmoid(self.hist_gate(m)) if HISTORY == "gated" else 1.0
                x = x + self.hist_scale.astype(x.dtype) * g * m
            if DEPTH_ATTN and states:
                x = self.depth_attn(x, states[-DEPTH_ATTN:])
            for i, blk in enumerate(self.core):
                x, kv = blk(x, kvs[i] if (KV_SHARE and r > 0) else None)
                if KV_SHARE and r == 0:
                    kvs[i] = kv
            if use_exit:
                expected = expected + active[..., 0].astype(mx.float32)
                x = mx.where(active, x, h)                     # exited tokens keep their exit state
                if r < loops - 1:
                    d, prev_lp = self._change(x, h, prev_lp)
                    if d is not None and r > 0:                # r=0 compares against the prelude output: skip
                        active = active & (d >= exit_tau)
                h = x
                states.append(h)
                continue
            if not HALTING:
                h = x
                states.append(h)
                continue
            last = r == loops - 1
            if hard_halt:
                expected = expected + active[..., 0].astype(mx.float32)
                h = mx.where(active, x, h)                     # frozen tokens keep their exit state
                if not last:
                    active = active & (mx.sigmoid(self.halt(self.norm(h)).astype(mx.float32)) < HALT_THRESHOLD)
            else:
                h = x
                p = mx.ones(remain.shape) if last else mx.sigmoid(self.halt(self.norm(h)).astype(mx.float32))
                lam = remain * p
                out = out + lam.astype(h.dtype) * h
                expected = expected + (r + 1) * lam[..., 0]
                remain = remain * (1 - p)
            states.append(h)
        if HALTING and not hard_halt:
            return out, expected, states
        if not HALTING and not use_exit:
            expected = expected + loops
        return h, expected, states

    def forward(self, x, loops=None, hard_halt=False, exit_tau=0.0, return_states=False):
        loops = loops or self.train_loops
        h = self.emb(x).astype(DTYPE)
        for blk in self.prelude:
            h, _ = blk(h)
        h, expected, states = self.recur(h, loops, hard_halt, exit_tau)
        logits = self.readout(h)
        return (logits, expected, states) if return_states else (logits, expected)

    def loop_readouts(self, x, loops):
        """Logits after each loop 1..loops (row r equals model(x, loops=r+1) for this architecture)."""
        _, _, states = self.forward(x, loops, return_states=True)
        return [self.readout(s) for s in states]

    def __call__(self, x, loops=None):
        logits, expected = self.forward(x, loops, hard_halt=HALTING, exit_tau=EXIT_TAU)
        self.mean_loops_per_token = float(expected.mean().item())
        return logits

    def loss(self, batch, loops, consist_loop=-1):
        """consist_loop: earlier loop (0-based, < loops-1) distilled from the final loop this step; -1 = none.
        It is sampled outside the compiled step (one compiled graph per value), costing ~1 extra coda pass."""
        if LOOP_CONSIST > 0 and not HALTING and 0 <= consist_loop < loops - 1:
            logits, expected, states = self.forward(batch[:, :-1], loops, return_states=True)
            r = consist_loop
            teacher = mx.stop_gradient(mx.softmax(logits, axis=-1))
            early = nn.log_softmax(self.readout(states[r]), axis=-1)
            consist = -mx.sum(teacher * early, axis=-1).mean()
        else:
            logits, expected = self.forward(batch[:, :-1], loops)
            consist = 0.0
        ce = nn.losses.cross_entropy(logits, batch[:, 1:]).mean()
        return ce + (PONDER_COST * expected.mean() if HALTING else 0.0) + LOOP_CONSIST * consist, ce


# ---------------------------------------------------------------------------
# Training (fixed wall-clock budget)
# ---------------------------------------------------------------------------
def load_base(model, path):
    """Warm-start from the shared base checkpoint. New modules keep their (zero) init; shapes must match."""
    base = dict(mx.load(path).items())
    own = dict(tree_flatten(model.parameters()))
    bad = [k for k, v in base.items() if k in own and own[k].shape != v.shape]
    if bad:
        raise SystemExit(f"FAIL: base checkpoint shape mismatch for {bad[:5]}; keep the backbone shape or set INIT_FROM_BASE=False")
    model.load_weights([(k, v) for k, v in base.items() if k in own], strict=False)
    new = sorted(k for k in own if k not in base)
    print(f"warm-started from {path}; {len(new)} new tensors at init: {new[:8]}")


def train(model, budget, lr_peak, seed=SEED, log_every=50, on_tick=None, stop_after=None):
    """Train `model` for `budget` seconds of wall clock (compile warm-up excluded). Returns train_info.

    stop_after: stop early after this many seconds while keeping the LR schedule of the full budget
    (used to snapshot control models at a matched point of the same schedule)."""
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr_peak, weight_decay=WEIGHT_DECAY, betas=[0.9, 0.95])
    grad_fn = nn.value_and_grad(model, lambda b, k, c: model.loss(b, k, c))
    state = [model.state, opt.state]
    steps_fns = {}

    def step_fn(k, c=-1):
        if (k, c) not in steps_fns:
            def step(batch):
                (loss, ce), grads = grad_fn(batch, k, c)
                grads, _ = optim.clip_grad_norm(grads, 1.0)
                opt.update(model, grads)
                return loss, ce
            steps_fns[(k, c)] = mx.compile(step, inputs=state, outputs=state)
        return steps_fns[(k, c)]

    model.train()
    data = Batches(BATCH_SIZE, seed=seed)
    step, tokens, t_train, ce_ema = 0, 0, 0.0, None
    end = min(budget, stop_after) if stop_after else budget
    while t_train < end:
        t0 = time.perf_counter()
        progress = t_train / budget
        lr = lr_peak * min(1.0, (step + 1) / WARMUP_STEPS) * (FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * progress)))
        opt.learning_rate = lr
        k = int(rng.integers(1, 2 * LOOPS)) if RANDOM_LOOPS else LOOPS
        c = int(rng.integers(0, k - 1)) if (LOOP_CONSIST > 0 and k > 1) else -1
        loss, ce = step_fn(k, c)(mx.array(next(data)))
        mx.eval(loss, ce, state)
        dt = time.perf_counter() - t0
        if step >= 2:                      # exclude compilation warm-up from the budget
            t_train += dt
        tokens += BATCH_SIZE * SEQ_LEN
        ce_ema = ce.item() if ce_ema is None else 0.98 * ce_ema + 0.02 * ce.item()
        if step % log_every == 0:
            print(f"step {step:5d} | t {t_train:6.1f}s | lr {lr:.2e} | loops {k} | train_bpb {ce_ema/math.log(2):.4f} | {BATCH_SIZE*SEQ_LEN/dt/1e3:.1f}k tok/s", flush=True)
        if not math.isfinite(loss.item()):
            raise SystemExit("FAIL: non-finite loss")
        step += 1
        if on_tick:
            on_tick(step, t_train)
    return {"training_seconds": t_train, "steps": step, "tokens_M": tokens / 1e6}


def main():
    mx.random.seed(SEED)
    model = Inkling()
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"params: {n_params/1e6:.2f}M  effective depth: {PRELUDE + CORE*LOOPS + CODA}")
    if INIT_FROM_BASE:
        if not os.path.exists(BASE_CKPT):
            raise SystemExit(f"FAIL: {BASE_CKPT} missing; run `uv run pretrain_base.py` once (or set INIT_FROM_BASE=False)")
        load_base(model, BASE_CKPT)
    info = train(model, TIME_BUDGET, LR)
    info["init_from_base"] = INIT_FROM_BASE
    evaluate(model, info)


if __name__ == "__main__":
    main()
