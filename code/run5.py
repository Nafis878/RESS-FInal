"""Amendment v5 (protocol_v5.md): robustness of the notice guarantee to the learner and to the alarm settings.

Repetition 0 of the five-fold cross-validation of the main protocol, with exactly the same outer folds and the same
proper-training / calibration split (seed 1000 + split index). For every split:
  * learners on the same proper-training units, target min(r, cap) with the CNA cap (60 cycles; battery 180):
      GBM    histogram gradient boosting of the main protocol (reference; must reproduce the main run)
      RIDGE  ridge regression (alpha 1) on standardised causal features
      RF     random forest (200 trees, min 20 rows per leaf, 1/3 of the features per split) on the causal features
      WMLP   multilayer perceptron (64-32, ReLU, Adam, early stopping) on the raw normalised sensors of the last 30 rows
  * settings of the GBM signal: cap x span x debounce = {40,60,90} (battery {120,180,270}) x {1,3,5} x {1,2,3}
For every learner (default settings: span 3, debounce 2) and every setting: CNA, LB-all and LB-near at alpha 0.05, 0.10,
0.20 for the four lead times. Writes results/raw5/<DS>_robust.csv.gz (one row per test unit, policy and lead time).
Usage: python run5.py DATASET [smoke]
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import core as C
import policies as P
from run import CFG, ALPHAS, load, splits, series

OUT = os.path.join(C.ROOT, 'results', 'raw5'); os.makedirs(OUT, exist_ok=True)
SPANS5, KS5 = [1, 3, 5], [1, 2, 3]
WIN = 30


def windows(Z, W=WIN):
    """causal windows of the last W rows of one unit (rows before the first are copies of the first row), flattened"""
    n = len(Z); idx = np.arange(n)[:, None] - np.arange(W)[::-1][None, :]
    return Z[np.maximum(idx, 0)].reshape(n, -1)


def raw_matrix(df, kind, stat_units):
    """normalised raw sensors per unit, statistics from stat_units only (as in core.build_features)"""
    sub = df[df.unit.isin(stat_units)]
    if kind in ('engine', 'ncmapss'):
        sens = C.SENS21 if kind == 'engine' else C.NCM_SENS
        st = {}
        for s in sens:
            g = sub.groupby('regime')[s].agg(['mean', 'std'])
            if kind == 'ncmapss' or (g['std'] > 1e-6).all(): st[s] = g
        Z = np.column_stack([(df[s].values - df.regime.map(st[s]['mean']).values) / df.regime.map(st[s]['std']).values.clip(1e-6) for s in st])
        Z = np.nan_to_num(Z)
    else:
        D = df.copy()
        for f in C.BAT_FEAT:
            if D[f].isna().any(): D.loc[D[f].isna(), f] = sub[sub.block == 1][f].median()
        mu = D[D.unit.isin(stat_units)][C.BAT_FEAT].mean(); sd = D[D.unit.isin(stat_units)][C.BAT_FEAT].std().clip(1e-9)
        Z = ((D[C.BAT_FEAT] - mu) / sd).values
    return {u: windows(Z[idx]) for u, idx in df.groupby('unit', sort=True).indices.items()}


def fit_predict(name, Xtr, ytr, cap, Xs):
    yc = np.minimum(ytr, cap)
    if name == 'GBM':
        m = C.gbm(Xtr, ytr, cap)
    elif name == 'RIDGE':
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, yc)
    elif name == 'RF':
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, max_features=1 / 3, random_state=0, n_jobs=1).fit(Xtr, yc)
    elif name == 'WMLP':
        m = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=1e-3, batch_size=256, max_iter=60,
                                                         early_stopping=True, n_iter_no_change=8, random_state=0)).fit(Xtr, yc / cap)
        return [m.predict(X) * cap for X in Xs]
    return [m.predict(X) for X in Xs]


def evaluate(rows, base, S_cal, S_te, cal, tes, cap, k, leads):
    for lead in leads:
        V = P.cna_levels(S_cal, cal, k, lead)
        for a in ALPHAS:
            pols = {f'CNA_a{a:.2f}': C.conformal_level(V, a),
                    f'LBall_a{a:.2f}': lead + 1 + P.lb_quantile(S_cal, cal, cap, a, False),
                    f'LBnear_a{a:.2f}': lead + 1 + P.lb_quantile(S_cal, cal, cap, a, True)}
            for pol, H in pols.items():
                J = P.alarm_rows(S_te, k, [H])
                succ, use, notice = P.outcome(J, tes, lead)
                for i, d in enumerate(tes):
                    rows.append(dict(base, policy=pol, alpha=a, lead=lead, H=H, unit=d['unit'], L=d['L'], fail=not succ[i, 0], notice=notice[i, 0]))


def run_split(df, kind, name, tr, te, sidx, smoke=False):
    cfg = CFG[kind]; t0 = time.time(); rep = 0
    rng = np.random.default_rng(1000 + 100 * rep + sidx); perm = rng.permutation(tr)          # identical to run.run_split
    ncal = int(round(0.4 * len(perm))); cal_u = np.sort(perm[:ncal]); fit_u = np.sort(perm[ncal:])
    assert not (set(cal_u) & set(fit_u)) and not (set(tr) & set(te)), 'disjoint units'
    F = C.build_features(df, kind, fit_u)
    cal, tes = series(F, cal_u), series(F, te)
    X = np.vstack([F[u]['X'] for u in fit_u]); y = np.concatenate([F[u]['r'] for u in fit_u])
    Xc = [F[d['unit']]['X'][d['mask']] for d in cal]; Xt = [F[d['unit']]['X'][d['mask']] for d in tes]
    Wd = raw_matrix(df, kind, fit_u)
    WX = np.vstack([Wd[u] for u in fit_u])
    Wc = [Wd[d['unit']][d['mask']] for d in cal]; Wt = [Wd[d['unit']][d['mask']] for d in tes]
    cc, sp, kk = cfg['cna_cap'], cfg['cna_span'], cfg['cna_k']
    rows = []; base = dict(split=name, rep=rep, n_fit=len(fit_u), n_cal=len(cal_u))
    learners = ['GBM'] if smoke else ['GBM', 'RIDGE', 'RF', 'WMLP']
    for ln in learners:
        Xtr, Xs = (WX, Wc + Wt) if ln == 'WMLP' else (X, Xc + Xt)
        p = fit_predict(ln, Xtr, y, cc, Xs); pc, pt = p[:len(cal)], p[len(cal):]
        evaluate(rows, dict(base, learner=ln, cap=cc, span=sp, k=kk, study='learner'),
                 [C.smooth(s, sp) for s in pc], [C.smooth(s, sp) for s in pt], cal, tes, cc, kk, cfg['leads'])
        print(f'  {name} {ln} {time.time() - t0:.0f}s', flush=True)
    if not smoke:
        for cap in cfg['caps']:
            p = fit_predict('GBM', X, y, cap, Xc + Xt); pc, pt = p[:len(cal)], p[len(cal):]
            for s_ in SPANS5:
                Sc = [C.smooth(s, s_) for s in pc]; St = [C.smooth(s, s_) for s in pt]
                for k_ in KS5:
                    evaluate(rows, dict(base, learner='GBM', cap=cap, span=s_, k=k_, study='setting'), Sc, St, cal, tes, cap, k_, [cfg['leads'][2]])
        print(f'  {name} settings {time.time() - t0:.0f}s', flush=True)
    return rows


def main(ds, smoke=False):
    df, kind = load(ds)
    S = [s for s in splits(df, kind) if s[0] == 'cv']
    out = f'{OUT}/{ds}_robust{"_smoke" if smoke else ""}.csv.gz'
    if os.path.exists(out) and not smoke:
        print('exists', out); return
    rows = []
    for sidx, (an, name, tr, te) in enumerate(S):
        rows += run_split(df, kind, name, tr, te, sidx, smoke)
        if smoke: break
    R = pd.DataFrame(rows); R['dataset'] = ds
    R.to_csv(out, index=False)
    print('written', out, len(R))


if __name__ == '__main__':
    main(sys.argv[1], smoke=len(sys.argv) > 2 and sys.argv[2] == 'smoke')
