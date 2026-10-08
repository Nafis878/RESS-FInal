"""Statistical tools: Clopper-Pearson interval, exact binomial test, Tango score interval and TOST for paired proportions,
paired bootstrap of cost-rate ratios, Holm adjustment."""
import numpy as np
from scipy.stats import beta, binomtest, norm


def clopper_pearson(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def binom_greater(k, n, p0):
    """one-sided exact p-value for H0: rate <= p0 against rate > p0"""
    return float(binomtest(int(k), int(n), p0, alternative='greater').pvalue) if n else 1.0


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def tango_z(b, c, n, d0):
    """Tango (1998) score statistic for H0: p10 - p01 = d0 (b = n10, c = n01); restricted MLE of p01 from
    2n q^2 - B q - c d0 (1 - d0) = 0 with B = b + c + d0 (b - c - 2n)"""
    B = b + c + d0 * (b - c - 2 * n)
    q = (B + np.sqrt(max(B * B + 8 * n * c * d0 * (1 - d0), 0.0))) / (4 * n)
    v = n * (2 * q + d0 * (1 - d0))
    if v <= 0: return np.inf * np.sign(b - c - n * d0) if (b - c - n * d0) != 0 else 0.0
    return (b - c - n * d0) / np.sqrt(v)


def tango_ci(b, c, n, level=0.90):
    """score interval for p10 - p01 by inverting the Tango statistic (bisection on each side)"""
    z = norm.ppf(0.5 + level / 2); est = (b - c) / n

    def root(lo, hi, target):
        f = lambda d: tango_z(b, c, n, d) - target
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0: hi = mid
            else: lo = mid
        return 0.5 * (lo + hi)
    lower = root(-1 + 1e-12, est, z) if tango_z(b, c, n, -1 + 1e-12) > z else -1.0
    upper = root(est, 1 - 1e-12, -z) if tango_z(b, c, n, 1 - 1e-12) < -z else 1.0
    return lower, upper


def tost_paired(b, c, n, margin):
    """two one-sided score tests of |p10 - p01| >= margin; returns p_TOST (max of the two one-sided p-values)"""
    p_low = 1 - norm.cdf(tango_z(b, c, n, -margin))      # H0: diff <= -margin
    p_up = norm.cdf(tango_z(b, c, n, margin))            # H0: diff >= +margin
    return float(max(p_low, p_up))


def boot_ratio(cA, uA, cB, uB, B=2000, seed=2027):
    """paired bootstrap of (sum cA / sum uA) / (sum cB / sum uB) over units; returns point, lo, hi"""
    cA, uA, cB, uB = map(np.asarray, (cA, uA, cB, uB)); n = len(cA); rng = np.random.default_rng(seed)
    W = np.vstack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)
    r = ((W @ cA) / (W @ uA)) / ((W @ cB) / (W @ uB))
    return float((cA.sum() / uA.sum()) / (cB.sum() / uB.sum())), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))
