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
                  so early exits lose less (cf. Shallow-Deep Networks, CALM early exit)
  CHUNK           run the looped core over K-byte chunks (hourglass; cf. CALM, arXiv 2510.27688): byte
                  prelude -> pool K bytes -> core on T/K positions -> shift + unpool -> byte coda.
                  Core FLOPs and core KV cache / K. Needs its own base (not a warm start).
Usage: uv run train.py > run.log 2>&1
"""
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map

from prepare import BASE_CKPT, BASE_NAME, REASONING_MIX, SEQ_LEN, TIME_BUDGET, VOCAB_SIZE, Batches, evaluate

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
LOOP_CHOICES = None                 # e.g. (4,)*7+(8,): sample the train loop count per step (depth lane; helps 8-loop bpb)
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
CANON = 4                          # >0: Canon layers (Allen-Zhu, Physics of LMs 4.1): a residual causal depthwise
                                   # conv of this kernel size on each block's attention and MLP inputs. Zero-init,
                                   # so warm starts begin as a no-op. Gives every position its neighbours, so an
                                   # induction match on a 2-byte key ("r=") needs no previous-token head.
                                   # Default since Nov 24: from scratch on pure 8-key lookup, retrieval forms by step
                                   # 1000 with it (.998) and not by step 2500 without (.30) (results/sanity-*.txt).
SLEEP = 0                          # >0: sleep window W (Lee et al. 2026, "Sample-Efficient In-Context Adaptation via
                                   # Sleep"). Predictions come from a one-pass "wake" stream (prelude, core once, coda)
                                   # that attends to the context of earlier W-byte windows as refined by the full
                                   # LOOPS-deep "sleep" stream, and to its own window's wake states. Inference: decode
                                   # each byte with one pass; when a window fills, run the loops over it ("sleep").
DREAM = 0.0                        # >0 (needs SLEEP): consolidation. Soft CE from the sleep stream's own deep
                                   # prediction (stop-gradient) into the wake prediction, so the fast path absorbs
                                   # what the loops compute; the sleep readout also gets its CE at this weight.
CHUNK = 1                          # >1: core runs on K-byte chunks (changes the architecture; new base needed)
ANSWER_W = 10.0                    # loss weight on reasoning answer digits ("?v=D"; unweighted they are ~2.4% of loss)
FIRST_ANSWER_W = 10.0              # weight on each line's FIRST answer: a chain end, the only deep, question-dependent
                                   # query (later answers can copy earlier ones). The reported ce stays unweighted.

INIT_FROM_BASE = True              # continue from the shared base checkpoint (see pretrain_base.py)
BATCH_SIZE = 32
LR = 1e-3                          # peak LR (from-scratch base pretraining also uses 1e-3)
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 100
FINAL_LR_FRAC = 0.05
# Warm-start optimizer recipe (main() only; pretrain_base.py always uses "legacy" with one LR).
#   "R": AdamW with three parameter groups -- tensors loaded from the base at R_LR (with weight decay), new
#        matrices at R_LR_NEW and new vectors/gates/scalars at R_LR_NEW_VEC (no weight decay on new tensors);
#        R_WARMUP steps of warm-up, constant, then linear decay to 0 over the last R_DECAY_FRAC of the budget.
#   "legacy": one group at LR with WARMUP_STEPS warm-up and cosine decay to FINAL_LR_FRAC (re-warms the base).
RECIPE = "R"
R_LR = 1e-4                        # tensors loaded from the base (below base-v2 final LR ~5e-5)
R_LR_NEW = 1e-3                    # new matrices (absent from the base checkpoint)
R_LR_NEW_VEC = 1e-2                # new 1-D tensors: gates, scales, biases
R_LR_CORE = None                   # base tensors of the shared looped core (None: same as R_LR)
R_WARMUP, R_DECAY_FRAC = 10, 1.0
R_INTERP = 0.5                     # <1: after training, base tensors become base + R_INTERP*(trained - base) (WiSE-FT)
BF16 = False                       # True: bf16 weights/activations, fp32 master weights (+57% tok/s, -49% memory at 400 steps; #8). Off until a long run matches fp32
DTYPE = mx.float32                 # activation dtype when BF16 is False


def act_dtype():
    return mx.bfloat16 if BF16 else DTYPE


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

    def __call__(self, x, kv=None, mem=None):
        """mem = (k_mem, v_mem, mask): extra keys/values (same positions, RoPE applied) placed before this
        stream's own; mask (T, 2T) bool, True = attend (SLEEP's wake stream)."""
        B, T, D = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = self.rope(q.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3))
        if kv is None:
            k = self.rope(k.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3))
            v = v.reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)
        else:
            k, v = kv
        if mem is None:
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd ** -0.5, mask="causal")
        else:
            km, vm, mask = mem
            a = mx.fast.scaled_dot_product_attention(q, mx.concatenate([km, k], axis=2), mx.concatenate([vm, v], axis=2),
                                                     scale=self.hd ** -0.5, mask=mask)
        return self.o(a.transpose(0, 2, 1, 3).reshape(B, T, D)), (k, v)


def causal_conv(h, w):
    """Depthwise causal conv over time: out[t] = sum_i w[i] * h[t - i] (zero before the start). h: (B, T, D), w: (K, D)."""
    T = h.shape[1]
    out = w[0] * h
    for i in range(1, w.shape[0]):
        out = out + w[i] * mx.pad(h, [(0, 0), (i, 0), (0, 0)])[:, :T]
    return out


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.attn = Attention(d, h)
        hidden = MLP_MULT * d * 2 // 3          # SwiGLU with ~4x-MLP parameter count
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        if CANON > 0:
            self.canon_a = mx.zeros((CANON, d))
            self.canon_c = mx.zeros((CANON, d))

    def __call__(self, x, kv=None, mem=None):
        h = self.n1(x)
        if "canon_a" in self:
            h = h + causal_conv(h, self.canon_a)
        a, kv = self.attn(h, kv, mem)
        x = x + a
        y = self.n2(x)
        if "canon_c" in self:
            y = y + causal_conv(y, self.canon_c)
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


def answer_mask(batch, with_first=False):
    """(B, T) bool over targets: the digit after "?<letter>=" in a reasoning row (rows [0, mix) of Batches),
    only once the row has seen a newline, so the line's statements are inside the window (answers in a line
    cut off at the window start are unpredictable noise)."""
    x, y = batch[:, :-1], batch[:, 1:]
    B, T = x.shape
    eq = x == ord("=")
    letter = mx.concatenate([mx.zeros((B, 1), dtype=mx.bool_), (x[:, :-1] >= ord("a")) & (x[:, :-1] <= ord("z"))], axis=1)
    qm = mx.concatenate([mx.zeros((B, 2), dtype=mx.bool_), x[:, :-2] == ord("?")], axis=1)
    digit = (y >= ord("0")) & (y <= ord("9"))
    seen_nl = mx.cummax((x == ord("\n")).astype(mx.int32), axis=1) > 0
    mix = max(1, round(REASONING_MIX * B / 32))
    rows = mx.arange(B)[:, None] < mix
    ans = eq & letter & qm & digit & seen_nl & rows
    if not with_first:
        return ans
    nl = (x == ord("\n")).astype(mx.int32)
    line = mx.cumsum(nl, axis=1)                                   # line index within the row
    c = mx.cumsum(ans.astype(mx.int32), axis=1)                    # answers so far in the row
    start = mx.cummax(mx.where(nl > 0, c, 0), axis=1)              # answer count at the latest newline
    first = ans & ((c - start) == 1) & (line > 0)
    return ans, first


def loop_features(r, d):
    freqs = mx.exp(-math.log(100.0) * mx.arange(d // 2) / (d // 2))
    return mx.concatenate([mx.sin(r * freqs), mx.cos(r * freqs)])


class Inkling(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.train_loops = LOOPS
        self.align = CHUNK                    # eval prompts end on a chunk boundary (prepare.encode_prompt)
        self.emb = nn.Embedding(VOCAB_SIZE, d)
        self.prelude = [Block(d, N_HEADS) for _ in range(PRELUDE)]
        self.core = [Block(d, N_HEADS) for _ in range(CORE)]
        self.coda = [Block(d, N_HEADS) for _ in range(CODA)]
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, VOCAB_SIZE, bias=False)
        if CHUNK > 1:
            self.down = nn.Linear(CHUNK * d, d)        # K byte states -> one core position
            self.up = nn.Linear(d, CHUNK * d)          # one core state -> a vector per byte offset
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
        if BF16:
            self.set_dtype(mx.bfloat16)

    def kv_bytes_per_token(self, loops=None):
        """KV cache bytes per byte of context at `loops` recurrent iterations (default: the trained count)."""
        core_layers = CORE * (1 if KV_SHARE else (loops or self.train_loops))
        return 2 * D_MODEL * 2 * (PRELUDE + core_layers / CHUNK + CODA)   # K and V, bf16, per byte

    def chunk(self, b):
        """(B, T, d) byte states -> (B, ceil(T/K), d) chunk states (right-padded; padding only feeds later chunks)."""
        B, T, d = b.shape
        pad = (-T) % CHUNK
        if pad:
            b = mx.concatenate([b, mx.zeros((B, pad, d), dtype=b.dtype)], axis=1)
        return self.down(b.reshape(B, (T + pad) // CHUNK, CHUNK * d))

    def unchunk(self, h, byte):
        """Chunk j's core state is added to the bytes of chunk j+1 only (shift by one chunk keeps it causal)."""
        B, T, d = byte.shape
        h = mx.concatenate([mx.zeros((B, 1, d), dtype=h.dtype), h[:, :-1]], axis=1)
        u = self.up(h).reshape(B, h.shape[1] * CHUNK, d)[:, :T]
        return byte + u

    def readout(self, h, byte=None):
        """Decode a loop state through the coda and head (an exact 'stop after this loop' logit lens)."""
        if CHUNK > 1:
            h = self.unchunk(h, byte)
        for blk in self.coda:
            h, _ = blk(h)
        return self.head(self.norm(h)).astype(mx.float32)

    def _change(self, x, h, prev_lp, byte=None):
        """Per-token change from one loop to the next. Returns (change (B,T,1), log-probs of x's readout)."""
        if EXIT_METRIC == "state":
            num = mx.sqrt(mx.sum(mx.square((x - h).astype(mx.float32)), axis=-1, keepdims=True))
            den = mx.sqrt(mx.sum(mx.square(h.astype(mx.float32)), axis=-1, keepdims=True)) + 1e-6
            return num / den, None
        lp = nn.log_softmax(self.readout(x, byte), axis=-1)
        if prev_lp is None:
            return None, lp
        return mx.sum(mx.exp(lp) * (lp - prev_lp), axis=-1, keepdims=True), lp

    def recur(self, e, loops, hard_halt=False, exit_tau=0.0, byte=None, collect=None):
        """Run the recurrent core. Returns (state to decode, expected loops per token, per-loop states).
        collect: a list that receives each core block's (k, v) from the final loop (SLEEP)."""
        h, states, kvs = e, [], [None] * CORE
        B, T, _ = e.shape
        use_exit = exit_tau > 0 and not HALTING
        if use_exit and CHUNK > 1 and EXIT_METRIC != "state":
            raise ValueError("with CHUNK > 1 use EXIT_METRIC='state' (KL is per byte, exits are per chunk)")
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
                if collect is not None and r == loops - 1:
                    collect.append(kv)
            if use_exit:
                expected = expected + active[..., 0].astype(mx.float32)
                x = mx.where(active, x, h)                     # exited tokens keep their exit state
                if r < loops - 1:
                    d, prev_lp = self._change(x, h, prev_lp, byte)
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

    def sleep_wake(self, x, loops=None):
        """SLEEP forward. Returns (wake logits, sleep logits, expected passes per byte)."""
        if CHUNK > 1 or HALTING or KV_SHARE or EXIT_TAU > 0 or DEPTH_ATTN or HISTORY != "none" or LOOP_EMBED:
            raise ValueError("SLEEP is implemented for the plain looped core only")
        loops = loops or self.train_loops
        e = self.emb(x).astype(act_dtype())
        for blk in self.prelude:                       # the prelude does not loop: identical in both streams
            e, _ = blk(e)
        core_kv = []
        h, _, _ = self.recur(e, loops, collect=core_kv)          # sleep stream: every byte, all loops
        coda_kv = []
        for blk in self.coda:
            h, kv = blk(h)
            coda_kv.append(kv)
        sleep_logits = self.head(self.norm(h)).astype(mx.float32)
        T = x.shape[1]
        w = mx.arange(T) // SLEEP
        mem_ok = w[None, :] < w[:, None]                          # slept: earlier windows only
        own_ok = (w[None, :] == w[:, None]) & (mx.arange(T)[None, :] <= mx.arange(T)[:, None])
        mask = mx.concatenate([mem_ok, own_ok], axis=1)
        h = e                                                     # wake stream: one pass (loop 1 has no injection)
        for blk, (k, v) in zip(self.core, core_kv):
            h, _ = blk(h, mem=(k, v, mask))
        for blk, (k, v) in zip(self.coda, coda_kv):
            h, _ = blk(h, mem=(k, v, mask))
        wake_logits = self.head(self.norm(h)).astype(mx.float32)
        expected = mx.full(x.shape, float(loops + 1))             # each byte: one wake pass + `loops` sleep passes
        return wake_logits, sleep_logits, expected

    def forward(self, x, loops=None, hard_halt=False, exit_tau=0.0, return_states=False):
        if SLEEP > 0 and not return_states:
            wake, _, expected = self.sleep_wake(x, loops)
            return wake, expected
        loops = loops or self.train_loops
        h = self.emb(x).astype(act_dtype())
        for blk in self.prelude:
            h, _ = blk(h)
        byte = h
        if CHUNK > 1:
            h = self.chunk(h)
        h, expected, states = self.recur(h, loops, hard_halt, exit_tau, byte)
        logits = self.readout(h, byte)
        return (logits, expected, states, byte) if return_states else (logits, expected)

    def loop_readouts(self, x, loops):
        """Logits after each loop 1..loops (row r equals model(x, loops=r+1) for this architecture)."""
        _, _, states, byte = self.forward(x, loops, return_states=True)
        return [self.readout(s, byte) for s in states]

    def __call__(self, x, loops=None):
        logits, expected = self.forward(x, loops, hard_halt=HALTING, exit_tau=EXIT_TAU)
        self.mean_loops_per_token = float(expected.mean().item())
        return logits

    def logits(self, x, loops=None):
        """Same output as __call__ without the host sync, so the harness can mx.compile it for latency."""
        return self.forward(x, loops, hard_halt=HALTING, exit_tau=EXIT_TAU)[0]

    def loss(self, batch, loops, consist_loop=-1):
        """consist_loop: earlier loop (0-based, < loops-1) distilled from the final loop this step; -1 = none.
        It is sampled outside the compiled step (one compiled graph per value), costing ~1 extra coda pass."""
        dream = 0.0
        if SLEEP > 0:
            logits, sleep_logits, expected = self.sleep_wake(batch[:, :-1], loops)
            consist = 0.0
            if DREAM > 0:
                teacher = mx.stop_gradient(mx.softmax(sleep_logits, axis=-1))
                dream = -mx.sum(teacher * nn.log_softmax(logits, axis=-1), axis=-1).mean() \
                    + nn.losses.cross_entropy(sleep_logits, batch[:, 1:]).mean()
        elif LOOP_CONSIST > 0 and not HALTING and 0 <= consist_loop < loops - 1:
            logits, expected, states, byte = self.forward(batch[:, :-1], loops, return_states=True)
            r = consist_loop
            teacher = mx.stop_gradient(mx.softmax(logits, axis=-1))
            early = nn.log_softmax(self.readout(states[r], byte), axis=-1)
            consist = -mx.sum(teacher * early, axis=-1).mean()
        else:
            logits, expected = self.forward(batch[:, :-1], loops)
            consist = 0.0
        tok = nn.losses.cross_entropy(logits, batch[:, 1:])
        ce = tok.mean()
        if ANSWER_W != 1.0 or FIRST_ANSWER_W != 1.0:
            ans, first = answer_mask(batch, with_first=True)
            w = mx.where(first, FIRST_ANSWER_W, mx.where(ans, ANSWER_W, 1.0)).astype(tok.dtype)
            main = (w * tok).sum() / w.sum()
        else:
            main = ce
        return main + (PONDER_COST * expected.mean() if HALTING else 0.0) + LOOP_CONSIST * consist + DREAM * dream, ce


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
    model.load_weights([(k, v.astype(own[k].dtype)) for k, v in base.items() if k in own], strict=False)
    new = sorted(k for k in own if k not in base)
    print(f"warm-started from {path}; {len(new)} new tensors at init: {new[:8]}")
    return set(new)


def lr_scale(recipe, step, progress):
    """LR multiplier at optimizer step `step` (0-based) and budget fraction `progress` in [0, 1]."""
    if recipe == "R":
        return min(1.0, (step + 1) / R_WARMUP) * max(0.0, min(1.0, (1 - progress) / R_DECAY_FRAC))
    return min(1.0, (step + 1) / WARMUP_STEPS) * (FINAL_LR_FRAC + (1 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * progress)))


def param_groups(new_keys):
    """Recipe R groups as (name, lr, weight_decay, filter(path, array)); the last group takes everything else."""
    new_keys = set(new_keys)
    return [("new_vec", R_LR_NEW_VEC, 0.0, lambda p, a: p in new_keys and a.ndim <= 1),
            ("new_mat", R_LR_NEW, 0.0, lambda p, a: p in new_keys),
            *([("core", R_LR_CORE, WEIGHT_DECAY, lambda p, a: p.startswith("core."))] if R_LR_CORE else []),
            ("base", R_LR, WEIGHT_DECAY, None)]


class MasterAdamW(optim.AdamW):
    """AdamW keeping fp32 master weights and moments in optimizer state for low-precision parameters."""

    def init_single(self, parameter: mx.array, state: dict):
        p32 = parameter.astype(mx.float32)
        super().init_single(p32, state)
        state["master"] = p32

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        if "master" not in state:
            return super().apply_single(gradient.astype(mx.float32), parameter, state)
        master = super().apply_single(gradient.astype(mx.float32), state["master"], state)
        state["master"] = master
        return master.astype(parameter.dtype)


def make_optimizer(model, recipe, lr_peak, new_keys=()):
    """Returns (optimizer, [(sub-optimizer, peak lr)]). The caller sets each sub-optimizer's LR every step
    (MultiOptimizer's own learning_rate setter would give every group the same LR)."""
    opt_cls = MasterAdamW if BF16 else optim.AdamW
    if recipe != "R":
        opt = opt_cls(learning_rate=lr_peak, weight_decay=WEIGHT_DECAY, betas=[0.9, 0.95])
        opt.init(model.trainable_parameters())
        return opt, [(opt, lr_peak)]
    groups = param_groups(new_keys)
    subs = [opt_cls(learning_rate=lr, weight_decay=wd, betas=[0.9, 0.95]) for _, lr, wd, _ in groups]
    opt = optim.MultiOptimizer(subs, filters=[f for _, _, _, f in groups[:-1]])
    opt.init(model.trainable_parameters())          # before mx.compile captures opt.state
    return opt, [(o, lr) for o, (_, lr, _, _) in zip(subs, groups)]


def train(model, budget, lr_peak, seed=SEED, log_every=50, on_tick=None, stop_after=None, recipe="legacy",
          new_keys=()):
    """Train `model` for `budget` seconds of wall clock (compile warm-up excluded). Returns train_info.

    stop_after: stop early after this many seconds while keeping the LR schedule of the full budget
    (used to snapshot control models at a matched point of the same schedule).
    recipe: "legacy" (one LR, lr_peak, cosine; pretrain_base.py) or "R" (warm starts; see RECIPE). new_keys are
    the parameter paths absent from the base checkpoint (load_base's return value); lr_peak is unused under R."""
    rng = np.random.default_rng(seed)
    opt, groups = make_optimizer(model, recipe, lr_peak, new_keys)
    grad_fn = nn.value_and_grad(model, lambda b, k, c: model.loss(b, k, c))
    state = [model.state, opt.state]
    steps_fns = {}

    def step_fn(k, c=-1):
        if (k, c) not in steps_fns:
            def step(batch):
                (loss, ce), grads = grad_fn(batch, k, c)
                if BF16:
                    grads = tree_map(lambda g: g.astype(mx.float32), grads)
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
        s = lr_scale(recipe, step, progress)
        for o, lr0 in groups:
            o.learning_rate = lr0 * s
        lr = groups[-1][1] * s             # logged: the base / single group
        k = int(rng.integers(1, 2 * LOOPS)) if RANDOM_LOOPS else LOOPS
        if LOOP_CHOICES:
            k = int(LOOP_CHOICES[int(rng.integers(0, len(LOOP_CHOICES)))])
        c = int(rng.integers(0, k - 1)) if (LOOP_CONSIST > 0 and k > 1) else -1
        fresh = (k, c) not in steps_fns
        loss, ce = step_fn(k, c)(mx.array(next(data)))
        mx.eval(loss, ce, state)
        dt = time.perf_counter() - t0
        if step >= 2 and not fresh:        # exclude compilation (first steps, each new loop count) from the budget
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
    return {"training_seconds": t_train, "steps": step, "tokens_M": tokens / 1e6, "recipe": recipe}


def main():
    mx.random.seed(SEED)
    model = Inkling()
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"params: {n_params/1e6:.2f}M  effective depth: {PRELUDE + CORE*LOOPS + CODA}")
    new_keys = set()
    if INIT_FROM_BASE:
        if not os.path.exists(BASE_CKPT):
            raise SystemExit(f"FAIL: {BASE_CKPT} missing; run `uv run pretrain_base.py` once (or set INIT_FROM_BASE=False)")
        new_keys = load_base(model, BASE_CKPT)
    recipe = RECIPE if INIT_FROM_BASE else "legacy"    # recipe R is defined for warm starts only
    base0 = {k: v.astype(mx.float32) for k, v in tree_flatten(model.parameters()) if k not in new_keys}
    info = train(model, TIME_BUDGET, LR, recipe=recipe, new_keys=new_keys)
    if INIT_FROM_BASE and R_INTERP < 1:
        model.load_weights([(k, (base0[k] + R_INTERP * (v.astype(mx.float32) - base0[k])).astype(v.dtype))
                            for k, v in tree_flatten(model.parameters()) if k in base0], strict=False)
        mx.eval(model.parameters())
    info["init_from_base"] = INIT_FROM_BASE
    info["base"] = BASE_NAME if INIT_FROM_BASE else "scratch"
    evaluate(model, info)


if __name__ == "__main__":
    main()
