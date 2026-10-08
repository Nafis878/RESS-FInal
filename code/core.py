"""Data loading, causal features, regressor and alarm primitives.

Everything here is a re-implementation, in vectorised form, of the instrument of the original study (gradient-boosted
regressor on causal sensor features, smoothed and debounced threshold trigger). Units are integers or strings; every series
is indexed by observation rows, and every remaining life is expressed in PHYSICAL cycles (engine cycles, flight cycles,
charge-discharge cycles). Battery cells are aggregated into blocks of KAPPA_BAT = 3 cycles, fixed a priori (the value of
the original battery protocol); no life of any unit enters the time scale.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
KAPPA_BAT = 3
COLS = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
SENS21 = [f's{i}' for i in range(1, 22)]
NCM_SENS = ['T24', 'T30', 'T48', 'T50', 'Nf', 'Nc']
BAT_FEAT = ['QDischarge', 'QCharge', 'IR', 'Tavg', 'Tmin', 'Tmax', 'chargetime']
BAT_RANGES = {'QDischarge': (0.5, 1.2, True), 'QCharge': (0.5, 1.2, True), 'IR': (0.0, 0.05, False), 'Tavg': (15, 60, True),
              'Tmin': (15, 60, True), 'Tmax': (15, 60, True), 'chargetime': (0.0, 200.0, False)}
BAT_SPLIT_CELLS = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4', 'b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']   # split tests
BAT_NO_EOL = ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']                                           # censored / no life


# ----------------------------------------------------------------------------------------------- raw data
def load_engines(name):
    path = os.path.join(DATA, 'phm08', 'train.txt.gz') if name == 'PHM08' else os.path.join(DATA, 'cmapss', f'train_{name}.txt.gz')
    df = pd.read_csv(path, sep=r'\s+', header=None, names=COLS)
    df['regime'] = df.os1.round(0).astype(int) * 1000 + df.os3.round(0).astype(int)
    df = df.sort_values(['unit', 'cycle']).reset_index(drop=True)
    df['life'] = df.groupby('unit').cycle.transform('max'); df['rul'] = df.life - df.cycle
    df['group'] = 0
    return df


def load_ncmapss():
    R = pd.read_csv(os.path.join(DATA, 'ncmapss', 'ncmapss_cycle_summary.csv.gz'))
    key = R.subset + '|' + R.part + '|' + R.unit.astype(int).astype(str)
    ids = {k: i + 1 for i, k in enumerate(sorted(key.unique()))}
    R['unit'] = key.map(ids); R['regime'] = R.Fc.round().astype(int)
    R = R.sort_values(['unit', 'cycle']).reset_index(drop=True)
    R['life'] = R.groupby('unit').cycle.transform('max'); R['rul'] = R.life - R.cycle
    R['group'] = R.subset.astype('category').cat.codes
    return R


def load_battery():
    """per-cycle summaries of the 123 cells retained by the original protocol; implausible values set missing and filled
    forward within the cell (never backward, never across cells)"""
    S = pd.read_csv(os.path.join(DATA, 'battery', 'severson_cycle_summary.csv.gz'))
    M = pd.read_csv(os.path.join(DATA, 'battery', 'severson_cells.csv')).set_index('cell')
    keep = [c for c in S.cell.unique() if np.isfinite(M.cycle_life_official[c]) and c not in BAT_SPLIT_CELLS and c not in BAT_NO_EOL]
    S = S[S.cell.isin(keep)].sort_values(['cell', 'cycle']).copy()
    S['L'] = S.cell.map(M.cycle_life_official).astype(int); S = S[S.cycle <= S.L]
    for f, (lo, hi, inc) in BAT_RANGES.items():
        v = S[f].astype(float); ok = np.isfinite(v) & (v <= hi) & ((v >= lo) if inc else (v > lo))
        S[f] = v.where(ok); S[f] = S.groupby('cell')[f].transform(lambda s: s.ffill())
    S['block'] = (S.cycle - 1) // KAPPA_BAT + 1; S = S[S.block <= S.L // KAPPA_BAT]
    B = S.groupby(['cell', 'block'])[BAT_FEAT].mean().reset_index()
    B['life'] = B.cell.map(S.groupby('cell').L.first())
    B['cycle'] = B.block * KAPPA_BAT                       # cycle at the end of the block
    B['rul'] = B.life - B.cycle                            # cycles left at the end of the block
    B['group'] = B.cell.str[1].astype(int)                 # batch 1, 2, 3
    B = B.rename(columns={'cell': 'unit'})
    return B


# ----------------------------------------------------------------------------------------------- causal features
def causal_feats(Z, base=20):
    """per-row causal features of one unit (rows 0..i only): early-life baseline (mean of the first min(20, i+1) rows),
    EWM (span 10, 30) of the baseline-removed signals, least-squares slopes over the last 15 and 40 rows (EWM span 5;
    zero until 5 rows are available), and the age in rows. Vectorised version of feats.causal_feats of the original code."""
    n, p = Z.shape
    cs = np.cumsum(Z, 0); cnt = np.minimum(np.arange(1, n + 1), base)
    b = cs[cnt - 1] / cnt[:, None]
    zc = pd.DataFrame(Z - b)
    feats = [zc.ewm(span=10).mean().values, zc.ewm(span=30).mean().values, b]
    x = np.arange(n, dtype=float)
    S0 = np.concatenate([[0.0], np.cumsum(np.ones(n))]); S1 = np.concatenate([[0.0], np.cumsum(x)]); S2 = np.concatenate([[0.0], np.cumsum(x * x)])
    Zc = np.vstack([np.zeros((1, p)), np.cumsum(Z, 0)]); XZ = np.vstack([np.zeros((1, p)), np.cumsum(x[:, None] * Z, 0)])
    for W in (15, 40):
        i = np.arange(n); a = np.maximum(0, i - W + 1); m = (i - a + 1).astype(float)
        sx = S1[i + 1] - S1[a]; sxx = S2[i + 1] - S2[a]; sz = Zc[i + 1] - Zc[a]; sxz = XZ[i + 1] - XZ[a]
        den = sxx - sx * sx / m
        with np.errstate(invalid='ignore', divide='ignore'):
            sl = (sxz - sx[:, None] * sz / m[:, None]) / den[:, None]
        sl[m < 5] = 0.0
        feats.append(pd.DataFrame(sl).ewm(span=5).mean().values)
    return np.hstack(feats + [np.arange(1, n + 1)[:, None]])


def build_features(df, kind, stat_units):
    """features of every unit, with normalisation statistics from stat_units only.
    Returns {unit: dict(X, r (cycles left per row), t (cycle per row), L (life), group)}"""
    sub = df[df.unit.isin(stat_units)]
    if kind == 'engine' or kind == 'ncmapss':
        sens = SENS21 if kind == 'engine' else NCM_SENS
        st = {}
        for s in sens:
            g = sub.groupby('regime')[s].agg(['mean', 'std'])
            if kind == 'ncmapss' or (g['std'] > 1e-6).all(): st[s] = g
        cols = list(st)
        Zall = np.column_stack([(df[s].values - df.regime.map(st[s]['mean']).values) / df.regime.map(st[s]['std']).values.clip(1e-6) for s in cols])
        Zall = np.nan_to_num(Zall)
    else:   # battery: one regime; first-block gaps filled with the training-cell median of block 1
        D = df.copy()
        for f in BAT_FEAT:
            if D[f].isna().any():
                D.loc[D[f].isna(), f] = sub[sub.block == 1][f].median()
        mu = D[D.unit.isin(stat_units)][BAT_FEAT].mean(); sd = D[D.unit.isin(stat_units)][BAT_FEAT].std().clip(1e-9)
        Zall = ((D[BAT_FEAT] - mu) / sd).values
    out = {}
    for u, idx in df.groupby('unit', sort=True).indices.items():
        g = df.iloc[idx]
        out[u] = dict(X=causal_feats(Zall[idx]), r=g.rul.values.astype(float), t=g.cycle.values.astype(float),
                      L=float(g.life.iloc[0]), group=int(g.group.iloc[0]))
        if kind == 'battery': out[u]['Q'] = g.QDischarge.values.astype(float)
    return out


def gbm(X, y, cap, seed=0):
    return HistGradientBoostingRegressor(loss='absolute_error', max_iter=500, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, random_state=seed).fit(X, np.minimum(y, cap))


# ----------------------------------------------------------------------------------------------- alarm primitives
def smooth(p, span):
    return p if span == 1 else pd.Series(p).ewm(span=span).mean().values


def debounced_running_min(s, k):
    """rm[j] = min_{i<=j} max(s[i-k+1..i]); alarm at threshold H is the first j with rm[j] <= H"""
    mk = pd.Series(s).rolling(k, min_periods=k).max().values
    mk = np.where(np.isnan(mk), np.inf, mk)
    return np.minimum.accumulate(mk)


def first_alarm(rm, H):
    """index of the first row with rm <= H for each threshold in H (len(rm) if never)"""
    return np.searchsorted(-rm, -np.atleast_1d(np.asarray(H, float)), side='left')


def critical_level(rm, r, lead):
    """V = smallest threshold whose alarm leaves strictly more than `lead` cycles: min of rm over rows with r > lead
    (rows are in time order and r decreases, so the admissible rows form a prefix). +inf if no admissible row."""
    ok = np.flatnonzero(r > lead)
    return float(rm[ok[-1]]) if len(ok) else np.inf


def conformal_level(V, alpha):
    """split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest critical level (inf if that exceeds n)"""
    V = np.sort(np.asarray(V, float)); n = len(V); k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(V[k - 1])


def outer_folds(units, k=5, seed=2026):
    return np.array_split(np.random.default_rng(seed).permutation(np.sort(units)), k)
