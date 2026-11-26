"""Round-2 harness: D8 metric fixes, D3 score fields, D2 warm-start recipe R, D4 screening and confirmation."""
import importlib
import math
import os
import sys

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

import prepare as P
from test_model import batch, build

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "swarm"))
import confirm as C  # noqa: E402
import run as R  # noqa: E402


# ---------------------------------------------------------------- D8 / D3: evaluation metrics
def test_eval_constants_and_provenance():
    assert P.EVAL_PER_HOP == (40 if P.SMOKE else 800)
    assert len(P.HARNESS_SHA) == 10 and int(P.HARNESS_SHA, 16) >= 0
    assert P.HOLDOUT_EVAL_SEED != P.EVAL_SEED
    assert P.kappa(P.GROUP_BOUND) == 0 and P.kappa(1.0) == 1


def test_eval_seed_selects_a_different_set(monkeypatch):
    a, b = P.reasoning_eval_set(), P.reasoning_eval_set(P.HOLDOUT_EVAL_SEED)
    assert a != b and set(a) == set(b)
    monkeypatch.setenv("INKLING_EVAL_SEED", str(P.HOLDOUT_EVAL_SEED))
    try:
        importlib.reload(P)
        assert P.EVAL_SEED == P.HOLDOUT_EVAL_SEED and P.reasoning_eval_set() == b
    finally:
        monkeypatch.delenv("INKLING_EVAL_SEED")
        importlib.reload(P)
    assert P.reasoning_eval_set() == a


class _Varying:
    """Reports a different mean_loops_per_token on every call, like a halting model."""

    def __init__(self, vals):
        self.vals, self.i = list(vals), 0

    def __call__(self, x, loops=None):
        self.mean_loops_per_token = self.vals[self.i]
        self.i += 1
        return x


def test_mean_loops_is_token_weighted_over_all_calls():
    P._LOOPS_ACC.clear()
    m = _Varying([1.0, 3.0, 4.0])
    P._call(m, mx.zeros((1, 10)), None, ("text", None))
    P._call(m, mx.zeros((1, 30)), None, ("text", None))
    P._call(m, mx.zeros((2, 5)), None, ("reason", None))
    assert P._mean_loops(("text", None)) == pytest.approx((10 * 1 + 30 * 3) / 40)
    assert P._mean_loops(("text", None), ("reason", None)) == pytest.approx((10 + 90 + 40) / 50)
    assert P._mean_loops(("text", 8)) is None


def test_kv_bytes_count_the_evaluated_loops():
    _, m = build()
    assert m.kv_bytes_per_token() == m.kv_bytes_per_token(m.train_loops)
    assert m.kv_bytes_per_token(8) > m.kv_bytes_per_token(4) > m.kv_bytes_per_token(2)
    _, s = build(KV_SHARE=True)
    assert s.kv_bytes_per_token(8) == s.kv_bytes_per_token(2)


@pytest.mark.parametrize("sw", [{}, {"HALTING": True}, {"EXIT_TAU": 0.05, "EXIT_METRIC": "state"}])
def test_logits_matches_call_and_compiles(sw):
    _, m = build(**sw)
    x = batch()[:, :-1]
    assert mx.abs(m.logits(x) - m(x)).max().item() < 1e-5
    assert mx.abs(mx.compile(m.logits)(x) - m(x)).max().item() < 1e-4
    ms, compiled = P._latency_ms(m, mx, T=32, reps=2)
    assert compiled and ms > 0


def test_evaluate_reports_round2_fields(monkeypatch, tmp_path):
    """Full evaluate() on a tiny CPU model with a tiny fake validation set."""
    train, m = build()
    rng = np.random.default_rng(0)
    fake_val = rng.integers(32, 127, 40_000).astype(np.uint8)
    monkeypatch.setattr(P, "_load", lambda name: fake_val)
    monkeypatch.setattr(P, "VAL_WINDOWS", 16)
    monkeypatch.setattr(P, "EVAL_PER_HOP", 8)
    monkeypatch.setattr(P, "LATENCY_T", (100, 512))
    monkeypatch.chdir(tmp_path)
    res = P.evaluate(m, {"steps": 1, "training_seconds": 0.0, "base": "unit"})
    params = sum(v.size for _, v in tree_flatten(m.parameters()))
    kv = m.kv_bytes_per_token(m.train_loops)
    assert res["bytes_f"] == 2 * params + kv * P.SEQ_LEN
    assert res["kv_kb_per_token"] == kv / 1024
    assert res["scaling"][8]["kv_kb_per_token"] == m.kv_bytes_per_token(8) / 1024
    assert res["scaling"][8]["mean_loops_per_token"] == pytest.approx(8)
    assert res["mean_loops_per_token"] == pytest.approx(4)
    for h, a in res["reason_by_hop"].items():
        assert res["kappa_by_hop"][h] == pytest.approx(P.kappa(a))
    assert res["kappa_id"] == pytest.approx(np.mean([res["kappa_by_hop"][h] for h in P.EVAL_HOPS_ID]))
    assert res["kappa_ood"] == pytest.approx(np.mean([res["kappa_by_hop"][h] for h in P.EVAL_HOPS_OOD]))
    assert res["peak_mem_gb"] == max(res["train_peak_mem_gb"], res["eval_peak_mem_gb"])
    assert res["latency_ms_b1"] == res["latency_ms_b1_t512"] and res["latency_ms_b1_t100"] > 0
    assert res["harness"] == P.HARNESS_SHA and res["base"] == "unit" and res["eval_seed"] == P.EVAL_SEED
    assert (tmp_path / "runs" / "last.json").exists()


# ---------------------------------------------------------------- D2: recipe R
def _groups_of(model, new_keys):
    train = sys.modules["train"]
    opt, groups = train.make_optimizer(model, "R", 1e-3, new_keys)
    parts = opt._split_dictionary(model.trainable_parameters())
    return opt, groups, [set(k for k, _ in tree_flatten(p)) for p in parts]


def test_recipe_r_assigns_parameter_groups(tmp_path):
    train, base = build()
    path = str(tmp_path / "base.safetensors")
    base.save_weights(path)
    train, m = build(LOOP_EMBED=True, HISTORY="gated", R_LR_CORE=None)
    new = train.load_base(m, path)
    assert new == {"loop_proj.weight", "hist_scale", "hist_gate.weight", "hist_gate.bias"}
    opt, groups, (vec, mat, old) = _groups_of(m, new)
    assert vec == {"hist_scale", "hist_gate.bias"}
    assert mat == {"loop_proj.weight", "hist_gate.weight"}
    assert old == set(k for k, _ in tree_flatten(m.trainable_parameters())) - new
    assert [lr for _, lr in groups] == [train.R_LR_NEW_VEC, train.R_LR_NEW, train.R_LR]
    assert [o.weight_decay for o, _ in groups] == [0.0, 0.0, train.WEIGHT_DECAY]


def test_recipe_r_core_group():
    train, m = build(R_LR_CORE=1e-4)
    opt, groups, (vec, mat, core, old) = _groups_of(m, set())
    assert core and all(k.startswith("core.") for k in core)
    assert not any(k.startswith("core.") for k in old)
    assert [lr for _, lr in groups][2:] == [1e-4, train.R_LR]


def test_lr_schedules():
    import train
    importlib.reload(train)
    s = train.lr_scale
    w, d = train.R_WARMUP, train.R_DECAY_FRAC
    assert s("R", 0, 0.0) == pytest.approx(1 / w)
    assert s("R", w - 1, 0.0) == 1.0 and s("R", 500, 0.5 * (1 - d)) == 1.0 and s("R", 500, 1 - d) == 1.0
    assert s("R", 500, 1 - d / 2) == pytest.approx(0.5)
    assert s("R", 500, 1.0) == 0.0 and s("R", 500, 1.2) == 0.0
    # legacy is the old formula exactly
    for step, prog in ((0, 0.0), (49, 0.1), (99, 0.0), (300, 0.5), (900, 1.0)):
        old = min(1.0, (step + 1) / train.WARMUP_STEPS) * (
            train.FINAL_LR_FRAC + (1 - train.FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))
        assert s("legacy", step, prog) == old
    assert train.RECIPE == "R"


class _FakeBatches:
    def __init__(self, batch_size, seed=0):
        self.rng, self.B = np.random.default_rng(seed), batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.rng.integers(0, 256, (self.B, 33)).astype(np.int32)


@pytest.mark.parametrize("recipe", ["R", "legacy"])
def test_train_runs_with_each_recipe(recipe, tmp_path, monkeypatch):
    train, base = build()
    path = str(tmp_path / "base.safetensors")
    base.save_weights(path)
    train, m = build(LOOP_EMBED=True, BATCH_SIZE=2)
    monkeypatch.setattr(train, "Batches", _FakeBatches)
    new = train.load_base(m, path)
    before = {k: np.array(v) for k, v in tree_flatten(m.parameters())}
    seen = []
    info = train.train(m, 0.3, 1e-3, log_every=10**6, recipe=recipe, new_keys=new,
                       on_tick=lambda s, t: seen.append(s))
    assert info["recipe"] == recipe and info["steps"] >= 3
    after = dict(tree_flatten(m.parameters()))
    assert np.abs(np.array(after["loop_proj.weight"]) - before["loop_proj.weight"]).max() > 0
    assert np.abs(np.array(after["head.weight"]) - before["head.weight"]).max() > 0


def test_pretrain_base_keeps_the_legacy_recipe():
    import inspect

    import pretrain_base
    src = inspect.getsource(pretrain_base.main)
    assert "recipe" not in src                          # train()'s default recipe is legacy
    import train
    assert inspect.signature(train.train).parameters["recipe"].default == "legacy"


# ---------------------------------------------------------------- D4: screening
def _row(lane="baseline", bpb=1.50, k=0.10, valid="True", verdict="baseline", **kw):
    r = dict(lane=lane, val_bpb=str(bpb), kappa_id=str(k), valid=valid, verdict=verdict, harness="h1", base="b",
             eval_seed=R.SCREEN_EVAL_SEED, commit="c0", desc="d", primary="kappa_id", t_primary="")
    r.update({a: str(b) for a, b in kw.items()})
    return r


def _res(bpb=1.50, k=0.10, valid=True):
    return dict(val_bpb=bpb, kappa_id=k, valid=valid, violations=[])


def test_screening_needs_noise_then_flags_candidates():
    two = [_row(k=0.10), _row(k=0.12)]
    assert R.verdict(_res(k=0.5), two)[0] == "pending-noise"
    assert R.verdict(_res(k=0.5), [])[0] == "pending-noise"
    six = [_row(k=x) for x in (0.10, 0.12, 0.08, 0.11, 0.09, 0.10)]
    m = np.mean([0.10, 0.12, 0.08, 0.11, 0.09, 0.10])
    sd = np.std([0.10, 0.12, 0.08, 0.11, 0.09, 0.10], ddof=1)
    v, _, t = R.verdict(_res(k=m + 0.95 * sd), six)
    assert v == "candidate" and t == pytest.approx(0.95)
    assert R.verdict(_res(k=m + 0.85 * sd), six)[0] == "discard"
    # the val_bpb gate overrides a reasoning gain
    assert R.verdict(_res(k=0.9, bpb=1.50 + 0.021), six)[0] == "discard"
    assert R.verdict(_res(k=0.9, bpb=1.50 + 0.019), six)[0] == "candidate"
    assert R.verdict(_res(k=0.9, valid=False), six)[0] == "discard"


def test_val_bpb_primary_lane():
    six = [_row(bpb=b, k=0.10) for b in (1.50, 1.51, 1.49, 1.50, 1.52, 1.48)]
    sd = np.std([1.50, 1.51, 1.49, 1.50, 1.52, 1.48], ddof=1)
    assert R.verdict(_res(bpb=1.50 - sd, k=0.10), six, "val_bpb")[0] == "candidate"
    assert R.verdict(_res(bpb=1.50 - sd, k=0.05), six, "val_bpb")[0] == "discard"   # kappa_id non-inferiority
    assert R.verdict(_res(bpb=1.50, k=0.10), six, "val_bpb")[0] == "discard"


def test_baseline_pool_matches_harness_base_and_eval_set():
    rows = [_row(), _row(harness="h2"), _row(base="other"), _row(eval_seed=str(P.HOLDOUT_EVAL_SEED)),
            _row(valid="False"), _row(lane="depth")]
    assert len(R.baseline_rows(rows)) == 4
    assert len(R.baseline_rows(rows, "h1", "b", R.SCREEN_EVAL_SEED)) == 1
    assert R.FIELDS[:21][-1] == "power" and "kappa_id" in R.FIELDS and "harness" in R.FIELDS


META = dict(time="t", lane="x", commit="c", dirty="False", desc="d", hypothesis="h", seed="0", power="AC",
            primary="kappa_id")


def test_screen_and_ledger_row(monkeypatch):
    monkeypatch.delenv("INKLING_EVAL_SEED", raising=False)
    six = [_row(k=x) for x in (0.10, 0.12, 0.08, 0.11, 0.09, 0.10)]
    res = dict(_res(k=0.2), harness="h1", base="b", eval_seed=int(R.SCREEN_EVAL_SEED), reason_id=.4, reason_ood=.3,
               kappa_ood=.05, params=7, bytes_f=99, latency_ms_b1_t100=1.5, train_peak_mem_gb=2.0,
               eval_peak_mem_gb=1.0, recipe="R", scaling={"2": {"reason_ood": .2, "val_bpb": 1.6}})
    v, why, t, seed = R.screen(res, six, "lane")
    assert v == "candidate" and t > 0.92 and seed == R.SCREEN_EVAL_SEED
    assert R.screen(res, six, "baseline")[0] == "baseline"
    assert R.screen(dict(res, harness="other"), six, "lane")[0] == "pending-noise"   # no matching pool
    assert R.screen(dict(res, eval_seed=P.HOLDOUT_EVAL_SEED), six, "lane-confirm")[0] == "holdout"
    row = R.ledger_row(res, v, t, seed, **META)
    assert list(row) and set(row) == set(R.FIELDS) and row["kappa_id"] == "0.2000" and row["harness"] == "h1"
    crash = {"valid": False, "violations": ["exit 1"], "val_bpb": float("nan"), "reason_id": 0, "reason_ood": 0, "params": 0}
    v, why, t, seed = R.screen(crash, six, "lane")
    assert v == "discard" and R.ledger_row(crash, v, t, seed, **META)["kappa_id"] == ""


# ---------------------------------------------------------------- D4: confirmation
def test_student_t_tail():
    assert C.t_sf(0.0, 7) == pytest.approx(0.5)
    assert C.t_sf(2.0, 10) == pytest.approx(0.036694, abs=1e-5)
    assert C.t_sf(0.92, 5) == pytest.approx(0.19987, abs=2e-3)
    assert C.t_sf(-2.0, 10) == pytest.approx(1 - 0.036694, abs=1e-5)
    assert C.t_sf(1.0, 1) == pytest.approx(0.25, abs=1e-6)       # Cauchy


def test_welch_and_holm():
    t, df, p = C.welch([1, 2, 3, 4], [0, 1, 2, 3])
    assert t == pytest.approx(1 / math.sqrt(2 * (5 / 3) / 4)) and df == pytest.approx(6)
    assert C.welch([1, 2, 3, 4], [0, 1, 2, 3], sign=-1)[2] == pytest.approx(1 - p)
    h = C.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert h["a"] == (pytest.approx(0.05 / 3), True)
    assert h["c"][1] is False and h["b"][1] is False and h["d"] == (None, False)
    h = C.holm({"a": 0.01, "b": 0.02})
    assert h["a"][1] and h["b"][1]


def test_confirm_table_promotes_only_a_real_gain(tmp_path):
    hs = str(P.HOLDOUT_EVAL_SEED)
    rows = [_row(lane="good", verdict="candidate", commit="g1", t_primary=1.5),
            _row(lane="good", verdict="candidate", commit="g0", t_primary=1.0),
            _row(lane="null", verdict="candidate", commit="n1", t_primary=1.1),
            _row(lane="none", verdict="discard", commit="x")]
    base_k = [0.10, 0.12, 0.08, 0.11, 0.09, 0.10]
    rows += [_row(lane=C.BASELINE_LANE, k=k, eval_seed=hs, verdict="holdout") for k in base_k]
    rows += [_row(lane="good-confirm", k=k + 0.08, eval_seed=hs, commit="g1", verdict="holdout") for k in base_k]
    rows += [_row(lane="good-confirm", k=0.0, eval_seed=hs, commit="g0", verdict="holdout")]   # other commit: ignored
    rows += [_row(lane="null-confirm", k=k + 0.001, eval_seed=hs, commit="n1", verdict="holdout") for k in base_k]
    table = C.confirm_table(rows, ["good", "null", "none"], P.HOLDOUT_EVAL_SEED)
    by = {r["lane"]: r for r in table}
    assert set(by) == {"good", "null"}
    assert by["good"]["commit"] == "g1" and by["good"]["n_cand"] == 6 and by["good"]["promoted"]
    assert not by["null"]["promoted"]
    C.write_tsv(table, tmp_path / "confirm.tsv")
    lines = (tmp_path / "confirm.tsv").read_text().splitlines()
    assert lines[0].split("\t") == C.FIELDS and len(lines) == 3


def test_campaign_runs_six_baseline_seeds():
    import campaign
    assert campaign.BASELINE_SEEDS == (0, 1, 2, 3, 4, 5)
    assert not set(C.CONFIRM_SEEDS) & set(campaign.BASELINE_SEEDS)
