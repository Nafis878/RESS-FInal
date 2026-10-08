"""Unit tests of the primitives and of the guarantees (synthetic random signals, used ONLY to test the code).
Usage: python test_core.py"""
import numpy as np
import core as C
import stats as S


def brute_alarm(s, k, H):
    for j in range(len(s)):
        if j >= k - 1 and np.all(s[j - k + 1:j + 1] <= H): return j
    return len(s)


def random_unit(rng, L=None):
    L = L or int(rng.integers(60, 200)); r = np.arange(L - 1, 0, -1).astype(float)
    s = np.minimum(r, 60) + rng.normal(0, 3, len(r)).cumsum() * 0.3 + rng.normal(0, 2, len(r))
    return r, C.smooth(s, 3)


def test_first_alarm_and_critical_level(rng):
    for _ in range(300):
        r, s = random_unit(rng); k = int(rng.integers(1, 5)); lead = int(rng.integers(0, 20))
        rm = C.debounced_running_min(s, k)
        for H in rng.uniform(-5, 70, 5):
            assert int(C.first_alarm(rm, H)[0]) == brute_alarm(s, k, H)
        V = C.critical_level(rm, r, lead)
        for H in rng.uniform(-5, 70, 20):                 # Lemma 2: notice > lead  <=>  H >= V
            j = brute_alarm(s, k, H); ok = j < len(s) and r[j] > lead
            assert ok == (H >= V), (H, V)


def test_conformal_guarantee(rng, alpha=0.1, n=30, trials=3000, lead=10):
    fails = 0
    for _ in range(trials):
        units = [random_unit(rng) for _ in range(n + 1)]
        V = np.array([C.critical_level(C.debounced_running_min(s, 2), r, lead) for r, s in units[:n]])
        H = C.conformal_level(V, alpha); r, s = units[n]
        j = int(C.first_alarm(C.debounced_running_min(s, 2), H)[0]); fails += not (j < len(s) and r[j] > lead)
    rate = fails / trials; se = np.sqrt(alpha * (1 - alpha) / trials)
    assert rate <= alpha + 3 * se, rate
    assert rate >= alpha - 1 / (n + 1) - 3 * se, rate        # upper coverage bound 1 - alpha + 1/(n+1)
    return rate


def test_theorem1(rng):
    a, b = -13, 2
    for _ in range(2000):
        Y = rng.uniform(5, 30); c = rng.uniform(5, 30); R = int(rng.integers(0, 50)); D = Y - c
        lhs = abs(int(a <= Y - R <= b) - int(a <= c - R <= b))
        if D >= 0: rhs = int(c - b <= R < c - b + D) + int(c - a < R <= c - a + D)
        else: rhs = int(c - b + D <= R < c - b) + int(c - a + D < R <= c - a)
        assert lhs <= rhs


def test_causal_features(rng):
    """leakage check: features at row t must not change when rows after t are removed"""
    Z = rng.normal(size=(120, 5)).cumsum(0); F = C.causal_feats(Z)
    for t in (4, 17, 39, 80, 119):
        assert np.allclose(C.causal_feats(Z[:t + 1])[-1], F[t])
    # trend baseline is causal as well
    import policies as Pp
    q = 1.1 - np.abs(rng.normal(0, 0.001, 200)).cumsum(); full = Pp.trend_rul(q, 20, 3)
    for t in (10, 50, 150): assert np.isclose(Pp.trend_rul(q[:t + 1], 20, 3)[-1], full[t])


def test_tango():
    lo, hi = S.tango_ci(0, 0, 50, 0.90); z = 1.6448536269514722
    assert abs(hi - z * z / (50 + z * z)) < 1e-6 and abs(lo + hi) < 1e-9


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    test_first_alarm_and_critical_level(rng); test_theorem1(rng); test_causal_features(rng); test_tango()
    print('conformal failure rate (alpha 0.10, n 30):', round(test_conformal_guarantee(rng), 4))
    print('ALL TESTS PASSED')
