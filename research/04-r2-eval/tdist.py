"""Student-t sf/ppf without scipy (regularized incomplete beta, Lentz continued fraction)."""
import math
import numpy as np

_lg = np.frompyfunc(math.lgamma, 1, 1)


def lgamma(x):
    return np.asarray(_lg(np.asarray(x, float)), float)


def _betacf(a, b, x, iters=300):
    a, b, x = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(x, float))
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = np.ones_like(x)
    d = 1 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1 / d
    h = d.copy()
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1 + aa / c; c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1 + aa / c; c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1 / d; de = d * c; h *= de
        if np.all(np.abs(de - 1) < 1e-12):
            break
    return h


def betainc(a, b, x):
    a, b, x = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.clip(np.asarray(x, float), 0, 1))
    out = np.zeros_like(x)
    inner = (x > 0) & (x < 1)
    out[x >= 1] = 1.0
    ai, bi, xi = a[inner], b[inner], x[inner]
    lbt = lgamma(ai + bi) - lgamma(ai) - lgamma(bi) + ai * np.log(xi) + bi * np.log1p(-xi)
    bt = np.exp(lbt)
    sw = xi < (ai + 1) / (ai + bi + 2)
    r = np.where(sw, bt * _betacf(ai, bi, xi) / ai, 1 - bt * _betacf(bi, ai, 1 - xi) / bi)
    out[inner] = r
    return out


def t_sf(t, df):
    t, df = np.broadcast_arrays(np.asarray(t, float), np.asarray(df, float))
    x = df / (df + t * t)
    half = 0.5 * betainc(df / 2, 0.5, x)
    return np.where(t > 0, half, 1 - half)


def t_ppf(q, df):
    """Scalar quantile by bisection."""
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - t_sf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


if __name__ == "__main__":
    print(t_ppf(0.95, 5), "expect 2.015")
    print(t_ppf(0.90, 5), "expect 1.476")
    print(t_ppf(0.975, 10), "expect 2.228")
    print(t_sf(np.array([2.015, 1.476, 0.0, -1.0]), 5), "expect .05 .10 .5 .818")
    print(t_sf(2.0, 7.5), "welch df")
