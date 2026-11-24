"""Every experiment switch must build, train, stay causal, and (where it adds parameters) start as a no-op."""
import importlib
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import prepare as P

TINY = dict(D_MODEL=64, N_HEADS=2, DEPTH_ATTN_DIM=16)

CASES = {
    "baseline": {},
    "depth_attn": {"DEPTH_ATTN": 3},
    "halting": {"HALTING": True},
    "history_gated": {"HISTORY": "gated"},
    "history_mean": {"HISTORY": "mean"},
    "loop_embed": {"LOOP_EMBED": True},
    "kv_share": {"KV_SHARE": True},
    "random_loops_bptt": {"RANDOM_LOOPS": True, "BPTT_LOOPS": 2},
    "utility_exit_kl": {"EXIT_TAU": 0.01},
    "utility_exit_state": {"EXIT_TAU": 0.05, "EXIT_METRIC": "state"},
    "loop_consist": {"LOOP_CONSIST": 0.5},
    "chunk2": {"CHUNK": 2},
    "chunk4": {"CHUNK": 4},
    "chunk4_exit": {"CHUNK": 4, "EXIT_TAU": 0.05, "EXIT_METRIC": "state"},
    "combined": {"DEPTH_ATTN": 2, "HALTING": True, "LOOP_EMBED": True, "KV_SHARE": True},
}
# Switches that add parameters but must not change the function at init, so warm-starting is safe.
NO_OP_AT_INIT = ["depth_attn", "history_gated", "history_mean", "loop_embed", "loop_consist"]


def build(**switches):
    import train
    importlib.reload(train)
    for k, v in {**TINY, **switches}.items():
        setattr(train, k, v)
    mx.random.seed(0)
    model = train.Inkling()
    mx.eval(model.parameters())
    return train, model


def batch(B=2, T=48, seed=0):
    return mx.array(np.random.default_rng(seed).integers(0, 256, (B, T + 1)).astype(np.int32))


@pytest.mark.parametrize("name", list(CASES))
def test_switch_trains_and_scales(name):
    _, m = build(**CASES[name])
    x = batch()
    (loss, ce), grads = nn.value_and_grad(m, lambda b: m.loss(b, 4))(x)
    assert math.isfinite(loss.item()) and abs(ce.item() - math.log(256)) < 1.0
    gnorm = math.sqrt(sum((g * g).sum().item() for _, g in tree_flatten(grads)))
    assert math.isfinite(gnorm) and gnorm > 0
    for k in (1, 2, 4, 8):
        assert m(x[:, :-1], loops=k).shape == (2, 48, 256)


@pytest.mark.parametrize("name", list(CASES))
def test_switch_is_causal(name):
    _, m = build(**CASES[name])
    x = np.array(batch())
    y = x.copy()
    y[:, 30] = (y[:, 30] + 1) % 256
    a, b = m(mx.array(x[:, :-1])), m(mx.array(y[:, :-1]))
    assert mx.abs(a[:, :30] - b[:, :30]).max().item() < 1e-5
    assert mx.abs(a[:, 30:] - b[:, 30:]).max().item() > 0


@pytest.mark.parametrize("name", NO_OP_AT_INIT)
def test_new_modules_start_as_no_op(name, tmp_path):
    train, base = build()
    path = str(tmp_path / "base.safetensors")
    base.save_weights(path)
    x = batch()[:, :-1]
    ref = base(x)
    train, m = build(**CASES[name])
    train.load_base(m, path)
    assert mx.abs(m(x) - ref).max().item() < 1e-5


def test_kv_share_shrinks_cache():
    # train.py reads switches as module globals at call time, so measure before rebuilding the module.
    plain = build()[1].kv_bytes_per_token()
    shared = build(KV_SHARE=True)[1].kv_bytes_per_token()
    assert shared < plain


def test_default_config_fits_param_budget():
    import train
    importlib.reload(train)
    m = train.Inkling()
    n = sum(v.size for _, v in tree_flatten(m.parameters()))
    assert n <= P.MAX_PARAMS


def test_loop_readouts_equal_truncated_runs():
    """The logit lens used by swarm/landing.py is exact: readout after loop r == running r+1 loops."""
    _, m = build()
    x = batch()[:, :-1]
    outs = m.loop_readouts(x, 4)
    assert len(outs) == 4
    for r, lo in enumerate(outs):
        assert mx.abs(lo - m(x, loops=r + 1)).max().item() < 1e-4


@pytest.mark.parametrize("metric", ["kl", "state"])
def test_utility_exit_limits(metric):
    """A vanishing tau never exits (identical to no exit); a huge tau exits every token after loop 2."""
    train, m = build(EXIT_METRIC=metric)
    x = batch()[:, :-1]
    ref = m(x)
    train.EXIT_TAU = 1e-30
    assert mx.abs(m(x) - ref).max().item() < 1e-5 and abs(m.mean_loops_per_token - 4) < 1e-6
    train.EXIT_TAU = 1e30
    m(x)
    assert abs(m.mean_loops_per_token - 2) < 1e-6
    assert mx.abs(m(x) - m.loop_readouts(x, 2)[1]).max().item() < 1e-4


def test_loop_consistency_loss():
    """The consistency term is off unless a loop is sampled, and then adds a finite, non-negative penalty."""
    train, m = build(LOOP_CONSIST=0.5)
    x = batch()
    off, ce_off = m.loss(x, 4, -1)
    on, ce_on = m.loss(x, 4, 1)
    assert abs(off.item() - ce_off.item()) < 1e-6 and abs(ce_on.item() - ce_off.item()) < 1e-6
    assert on.item() > off.item() and math.isfinite(on.item())


@pytest.mark.parametrize("k", [2, 4])
def test_chunk_handles_ragged_length_and_shrinks_cache(k):
    """Lengths not divisible by K work (right padding never leaks backwards) and the core KV per byte drops."""
    plain = build()[1].kv_bytes_per_token()
    _, m = build(CHUNK=k)
    assert m.kv_bytes_per_token() < plain
    x = batch(T=47)[:, :-1]
    full = m(x)
    assert full.shape == (2, 47, 256)
    assert mx.abs(m(x[:, :30]) - full[:, :30]).max().item() < 1e-4


def test_chunk_readouts_equal_truncated_runs():
    _, m = build(CHUNK=4)
    x = batch()[:, :-1]
    for r, lo in enumerate(m.loop_readouts(x, 3)):
        assert mx.abs(lo - m(x, loops=r + 1)).max().item() < 1e-4


def test_answer_mask_marks_only_answer_digits_after_a_newline():
    import train
    importlib.reload(train)
    row = b"x;?q=5\na=3;b=a;?b=3?a=3\n?c=7"
    x = np.frombuffer(row, dtype=np.uint8).astype(np.int32)
    batch = mx.array(np.stack([x] * 32))
    m = np.array(train.answer_mask(batch))
    tgt = row[1:]
    hits = [i for i in range(len(tgt)) if m[0, i]]
    assert [chr(tgt[i]) for i in hits] == ["3", "3", "7"]     # "?q=5" precedes the first newline: excluded
    assert not m[8:].any()                                   # text rows (>= mix) never weighted


def test_first_answer_mask_marks_each_lines_first_answer():
    import train
    importlib.reload(train)
    row = b"x;?q=5\na=3;b=a;?b=3?a=3\nc=7;?c=7"
    x = np.frombuffer(row, dtype=np.uint8).astype(np.int32)
    ans, first = train.answer_mask(mx.array(np.stack([x] * 32)), with_first=True)
    tgt = row[1:]
    f = [i for i in range(len(tgt)) if np.array(first)[0, i]]
    assert len(f) == 2 and all(chr(tgt[i]) in "37" for i in f)
    assert f[0] == row.index(b"?b=3") + 2                     # first answer of line 2, not "?a=3"


def test_answer_weight_leaves_reported_ce_unweighted():
    train, m = build(ANSWER_W=10.0)
    x = batch()
    loss, ce = m.loss(x, 4)
    train.ANSWER_W = train.FIRST_ANSWER_W = 1.0
    loss1, ce1 = m.loss(x, 4)
    assert abs(ce.item() - ce1.item()) < 1e-6 and abs(loss1.item() - ce1.item()) < 1e-6
