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
    "combined": {"DEPTH_ATTN": 2, "HALTING": True, "LOOP_EMBED": True, "KV_SHARE": True},
}
# Switches that add parameters but must not change the function at init, so warm-starting is safe.
NO_OP_AT_INIT = ["depth_attn", "history_gated", "history_mean", "loop_embed"]


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
