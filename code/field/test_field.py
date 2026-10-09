"""Code tests of the field-data pipeline on SYNTHETIC FIXTURES (not data; nothing here is a result).
Usage (from code/): python -m field.test_field"""
import numpy as np, pandas as pd
from field import records as R
from field import backblaze as B
from field import run_field as F


def test_truncation():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(5, 200)); t = np.cumsum(rng.integers(1, 4, n)).astype(float)
        raw = np.cumsum(rng.poisson(0.2, (n, 4)), 0).astype(float); raw[rng.random((n, 4)) < 0.05] = np.nan
        assert R.truncation_test(t, raw), 'causal features use future rows'


def test_cohort():
    days = pd.date_range('2022-01-01', '2022-12-31')
    rows = []
    def add(sn, dates, fail_on=None):
        for d in dates: rows.append(dict(date=d, serial_number=sn, model='M1', capacity=1e12, failure=int(fail_on is not None and d == fail_on), s9=24.0 * 100))
    add('A', days[:59], fail_on=days[58])                      # fails on 28 Feb (day 59)
    add('B', days[:151])                                       # removed after 31 May: censored
    add('C', days)                                             # survives the year
    add('D', days[31:])                                        # enters on 1 Feb: not in the cohort
    add('E', days[:100], fail_on=days[50])                     # reports after its failure record: excluded
    U, D = B.cohort_units(pd.DataFrame(rows), 2022)
    assert set(U.index) == {'A', 'B', 'C', 'E'}
    assert U.loc['A', 'event'] and U.loc['A', 'T'] == 59 and np.isnan(U.loc['A', 'c'])
    assert not U.loc['B', 'event'] and U.loc['B', 'c'] == 151 and U.loc['B', 'censored_before_end']
    assert not U.loc['C', 'event'] and not U.loc['C', 'censored_before_end'] and U.loc['C', 'c'] == 365
    assert U.loc['E', 'after_failure']


def fixture_units(n, rng, c0=365.0):
    units = []
    for i in range(n):
        T = rng.weibull(1.5) * 900; C_ = rng.uniform(0, 2 * c0) if rng.random() < 0.3 else np.inf
        end = min(T, C_, c0 + 40); t = np.arange(0.0, np.floor(end - 1e-9) + 1)
        raw = np.cumsum(rng.poisson(np.where(T - t < 30, 2.0, 0.05)[:, None], (len(t), 3)), 0).astype(float)
        ev = T <= min(C_, c0 + 40)
        units.append(dict(unit=i, group='g%d' % (i % 2), Z=np.array([i % 2, rng.random()]), t=t, X=R.causal_features(t, raw),
                          event=bool(ev), T=float(T) if ev else np.nan, c=np.nan if ev else float(min(C_, c0 + 40))))
    return units


def test_pipeline_runs():
    rng = np.random.default_rng(1); units = fixture_units(400, rng)
    idx = rng.permutation(len(units))
    res = F.run_design(units, idx[:120], idx[120:300], idx[300:], c0=365.0, cap=60.0, lead=14, spacing=1.0, alphas=[0.05])
    assert {'T2', 'CC', 'E2HIST_G1', 'E2BASE_Ginf', 'LBall_K1'} <= set(res.method.astype(str))
    w = F.shift_weights(units[:200], units[200:], np.arange(0, 100), np.arange(100, 200)); assert np.all(np.isfinite(w(np.array([[0, 0.5]]))))


if __name__ == '__main__':
    test_truncation(); test_cohort(); test_pipeline_runs(); print('FIELD CODE TESTS PASSED (synthetic fixtures)')
