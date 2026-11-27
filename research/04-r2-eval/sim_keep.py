"""Monte Carlo of the swarm keep/promotion rules (pure numpy).

Model: every run's metric = true effect + N(0, sigma^2) (seed + item churn). 4 lanes x 15 experiments.
Global null: every idea has effect 0. Alternative: lane 0's 5th idea has a real effect delta (persists if kept, lost if discarded).

OLD = swarm/run.py as written (3 baseline seeds, thresholds max(default, 2 sd) / max(default, sd), the lane's
last kept run as reference, keep on bpb OR max(id, ood)) + campaign.py confirmation (2 seeds) + program.md
promotion ("mean beats the baseline mean by > 2x seed std", on any metric), with/without the discovery run.
NEW = 6 pooled baseline seeds, one primary metric, prediction-interval t-test vs the pooled baseline
(no ratchet), bpb guardrail, finalist = lane's last kept commit, confirmation with n_c fresh seeds vs
baseline (6 old + 3 fresh interleaved) by one-sided Welch t, Holm over finalists at FWER .05.
"""
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tdist  # noqa: E402

RNG = np.random.default_rng(20261124)
NS = int(os.environ.get("NSIM", "20000"))
LANES, EXPS = 4, 15


def mvn(n_shape, s_bpb, s_acc, rho):
    """(…, 3) draws: bpb (lower better), id, ood (higher better); id-ood correlated."""
    z = RNG.standard_normal(n_shape + (3,))
    bpb = s_bpb * z[..., 0]
    idv = s_acc * z[..., 1]
    ood = s_acc * (rho * z[..., 1] + np.sqrt(1 - rho**2) * z[..., 2])
    return np.stack([bpb, idv, ood], -1)


def old_rule(s_bpb, s_acc, rho=0.5, delta=None, metric="id", excl_disc=True):
    """Returns (P(>=1 lane promoted), E[keeps], P(real lane promoted)) for delta (in sigma units, on `metric`)."""
    mi = {"bpb": 0, "id": 1, "ood": 2}[metric]
    sig = np.array([s_bpb, s_acc, s_acc])
    sign = np.array([-1, 1, 1])
    base = mvn((NS, 3), s_bpb, s_acc, rho)                      # 3 baseline seeds
    bm = base.mean(1)
    sd = base.std(1, ddof=1)
    sd_acc = np.maximum(sd[:, 1], sd[:, 2])
    g_bpb, s_bpb_ = np.maximum(0.004, 2 * sd[:, 0]), np.maximum(0.004, sd[:, 0])
    g_acc, s_acc_ = np.maximum(0.03, 2 * sd_acc), np.maximum(0.02, sd_acc)
    promoted_any = np.zeros(NS, bool)
    real_prom = np.zeros(NS, bool)
    keeps = np.zeros(NS)
    for lane in range(LANES):
        ref = bm.copy()
        eff = np.zeros((NS, 3))
        last_kept_eff = np.full((NS, 3), np.nan)
        last_kept_val = np.full((NS, 3), np.nan)
        for e in range(EXPS):
            cand_eff = eff.copy()
            if delta is not None and lane == 0 and e == 4:
                cand_eff[:, mi] += sign[mi] * delta * sig[mi]
            x = cand_eff + mvn((NS,), s_bpb, s_acc, rho)
            d = x - ref
            k = ((d[:, 0] <= -g_bpb) & (d[:, 2] >= -s_acc_)) | \
                ((np.maximum(d[:, 1], d[:, 2]) >= g_acc) & (d[:, 0] <= s_bpb_))
            keeps += k
            ref = np.where(k[:, None], x, ref)
            eff = np.where(k[:, None], cand_eff, eff)
            last_kept_eff = np.where(k[:, None], cand_eff, last_kept_eff)
            last_kept_val = np.where(k[:, None], x, last_kept_val)
        has = ~np.isnan(last_kept_val[:, 0])
        y = np.nan_to_num(last_kept_eff)[:, None, :] + mvn((NS, 2), s_bpb, s_acc, rho)
        runs = y if excl_disc else np.concatenate([y, np.nan_to_num(last_kept_val)[:, None, :]], 1)
        m = runs.mean(1)
        better = (bm[:, 0] - m[:, 0] > 2 * sd[:, 0]) | (m[:, 1] - bm[:, 1] > 2 * sd[:, 1]) | (m[:, 2] - bm[:, 2] > 2 * sd[:, 2])
        prom = has & better
        promoted_any |= prom
        if lane == 0 and delta is not None:
            real_prom = prom & (np.abs(np.nan_to_num(last_kept_eff[:, mi])) > 0)
    return promoted_any.mean(), keeps.mean(), real_prom.mean()


def new_rule(delta=None, n_b=6, alpha_screen=0.10, n_c=5, n_fresh_b=3, fwer=0.05, guard=True, n_lanes=LANES):
    """Primary metric in sigma units (higher = better). Guardrail: an independent bpb draw must not be worse than
    +2 sigma_bpb (approximates the +.004 slack at sigma_bpb=.002). Returns P(any false promotion),
    E[keeps], P(real lane promoted), E[GPU runs]."""
    tcrit = tdist.t_ppf(1 - alpha_screen, n_b - 1)
    b = RNG.standard_normal((NS, n_b))
    m_b, s_b = b.mean(1), b.std(1, ddof=1)
    keeps = np.zeros(NS)
    pvals = np.ones((NS, n_lanes))
    has_real = np.zeros(NS, bool)
    fin_eff = np.zeros((NS, n_lanes))
    has = np.zeros((NS, n_lanes), bool)
    for lane in range(n_lanes):
        eff = np.zeros(NS)
        for e in range(EXPS):
            ce = eff.copy()
            real = delta is not None and lane == 0 and e == 4
            if real:
                ce = ce + delta
            x = ce + RNG.standard_normal(NS)
            t = (x - m_b) / (s_b * np.sqrt(1 + 1 / n_b))
            k = t >= tcrit
            if guard:
                k &= RNG.standard_normal(NS) <= 2.0
            keeps += k
            eff = np.where(k, ce, eff)
            has[:, lane] |= k
        fin_eff[:, lane] = eff
    # confirmation: n_c fresh finalist seeds vs 6 old + n_fresh_b fresh baseline seeds, Welch one-sided
    bb = np.concatenate([b, RNG.standard_normal((NS, n_fresh_b))], 1)
    for lane in range(n_lanes):
        y = fin_eff[:, lane, None] + RNG.standard_normal((NS, n_c))
        ym, yv = y.mean(1), y.var(1, ddof=1) / n_c
        bm_, bv = bb.mean(1), bb.var(1, ddof=1) / bb.shape[1]
        se = np.sqrt(yv + bv)
        dfw = (yv + bv) ** 2 / (yv**2 / (n_c - 1) + bv**2 / (bb.shape[1] - 1))
        tt = (ym - bm_) / se
        p = tdist.t_sf(tt, dfw)
        pvals[:, lane] = np.where(has[:, lane], p, 1.0)
    # Holm over the finalists that exist
    nfin = has.sum(1)
    order = np.argsort(pvals, 1)
    ps = np.take_along_axis(pvals, order, 1)
    rej = np.zeros_like(ps, bool)
    alive = np.ones(NS, bool)
    for i in range(n_lanes):
        thr = fwer / np.maximum(nfin - i, 1)
        r = alive & (ps[:, i] <= thr) & (i < nfin)
        rej[:, i] = r
        alive &= r
    rejected = np.zeros_like(rej)
    np.put_along_axis(rejected, order, rej, 1)
    false_any = (rejected & (fin_eff <= 0)).any(1).mean()
    real = (rejected[:, 0] & (fin_eff[:, 0] > 0)).mean() if delta is not None else float("nan")
    runs = n_b + EXPS * n_lanes + nfin.mean() * n_c + n_fresh_b
    return false_any, keeps.mean(), real, runs


if __name__ == "__main__":
    print("## OLD rule, global null (60 experiments, 4 lanes); sigma_bpb .002; rho(id,ood) .5")
    print("sigma_acc  P(>=1 promoted) excl.disc | incl.disc   E[keeps]")
    for s in (0.015, 0.025, 0.04):
        a = old_rule(0.002, s, excl_disc=True)
        b = old_rule(0.002, s, excl_disc=False)
        print(f"  {s:.3f}     {a[0]:.3f}                   {b[0]:.3f}       {a[1]:.2f}")
    print("OLD rule power, true effect on reason_id of delta*sigma in lane 0's 5th idea (sigma_acc .025, excl. discovery):")
    for dl in (1, 1.5, 2, 3):
        print(f"  delta {dl}: P(promoted) {old_rule(0.002, 0.025, delta=dl, metric='id')[2]:.3f}")

    print("\n## NEW rule: global null and power (delta in sigma units of the primary metric)")
    for a_s in (0.05, 0.10, 0.20):
        for n_c in (4, 5, 6):
            f, k, _, runs = new_rule(None, alpha_screen=a_s, n_c=n_c)
            pw = [new_rule(dl, alpha_screen=a_s, n_c=n_c)[2] for dl in (1, 1.5, 2, 3)]
            print(f"alpha_screen {a_s:.2f} n_c {n_c}: P(false promotion) {f:.3f}  E[keeps|null] {k:.2f}  "
                  f"power d=1/1.5/2/3: {pw[0]:.2f}/{pw[1]:.2f}/{pw[2]:.2f}/{pw[3]:.2f}  E[runs|null] {runs:.0f}")
    print("\nsingle-run screen power vs pooled 6-seed baseline (P(keep) for one attempt):")
    for a_s in (0.05, 0.10, 0.20):
        tc = tdist.t_ppf(1 - a_s, 5)
        for dl in (1, 2, 3):
            b = RNG.standard_normal((200000, 6))
            x = dl + RNG.standard_normal(200000)
            t = (x - b.mean(1)) / (b.std(1, ddof=1) * np.sqrt(1 + 1 / 6))
            print(f"  alpha {a_s:.2f} delta {dl}: {np.mean(t >= tc):.3f}", end="")
        print()
