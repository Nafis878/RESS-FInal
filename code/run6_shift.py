"""Amendment v6, Part 4 (protocol_v6.md): Theorem 7 (covariate-shift-weighted critical level) on held-out battery
batches and N-CMAPSS subsets.

Leave-one-group-out splits of run.py, repetitions 0-2, GBM signal and calibration units of run.py, primary lead time.
Covariates known before deployment: battery - first-step C-rate, switch state of charge, second-step C-rate (from the
charging policy); N-CMAPSS - flight class (one-hot) and mean altitude, Mach number and throttle angle of the first 10
flights. Weights: logistic regression (C = 1, standardised) of target vs source, source = proper-training units,
target = held-out units of the other half of a two-fold split of the held-out group (cross-fitting).
Output: results/raw6/<DS>_shift.csv.gz, one row per test unit, alpha and method (static / weighted).
Usage: python run6_shift.py DATASET [REPS]
"""
import os, re, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import core as C
import policies as P
from run import CFG, load, splits, series

OUT = os.path.join(C.ROOT, 'results', 'raw6'); os.makedirs(OUT, exist_ok=True)
ALPHAS = [0.05, 0.10, 0.20]


def parse_policy(p):
    """'5.4C(40%)-3.6C' -> (5.4, 0.40, 3.6); trailing '-newstructure' ignored (no overlap; not used)"""
    m = re.match(r'([\d.]+)C\((\d+)%\)-([\d.]+)', p)
    return float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))


def covariates(df, kind):
    if kind == 'battery':
        M = pd.read_csv(os.path.join(C.DATA, 'battery', 'severson_cells.csv')).set_index('cell')
        units = np.sort(df.unit.unique())
        return {u: np.array(parse_policy(M.policy[u])) for u in units}
    out = {}
    for u, g in df.groupby('unit', sort=True):
        h = g.sort_values('cycle').head(10)
        fc = int(round(h.Fc.iloc[0]))
        out[u] = np.r_[[float(fc == 1), float(fc == 2), float(fc == 3)], h.alt.mean(), h.Mach.mean(), h.TRA.mean()]
    return out


def weighted_levels(V, w, w_tests, alpha):
    V = np.asarray(V, float); w = np.asarray(w, float); o = np.argsort(V, kind='stable'); Vs = V[o]; cum = np.cumsum(w[o])
    thr = (1 - alpha) * (w.sum() + np.asarray(w_tests, float))
    i = np.searchsorted(cum, thr * (1 - 1e-12), side='left')
    return np.where(np.isfinite(thr) & (i < len(V)), Vs[np.minimum(i, len(V) - 1)], np.inf)


def run_split(df, kind, an, name, tr, te, rep, sidx, Z):
    cfg = CFG[kind]; t0 = time.time(); cc, sp, kk = cfg['cna_cap'], cfg['cna_span'], cfg['cna_k']
    lead = 30 if kind == 'battery' else 10
    rng = np.random.default_rng(1000 + 100 * rep + sidx); perm = rng.permutation(tr)          # identical to run.run_split
    ncal = int(round(0.4 * len(perm))); cal_u = np.sort(perm[:ncal]); fit_u = np.sort(perm[ncal:])
    assert not (set(cal_u) & set(fit_u)) and not (set(tr) & set(te)), 'disjoint units'
    F = C.build_features(df, kind, fit_u)
    X = np.vstack([F[u]['X'] for u in fit_u]); y = np.concatenate([F[u]['r'] for u in fit_u])
    m = C.gbm(X, y, cc)
    cal, tes = series(F, cal_u), series(F, te)
    RMc = [C.debounced_running_min(C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp), kk) for d in cal]
    RMt = [C.debounced_running_min(C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp), kk) for d in tes]
    Vc = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMc, cal)])
    Vt = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMt, tes)])
    Zs = np.array([Z[u] for u in fit_u]); Zc = np.array([Z[d['unit']] for d in cal]); Zt = np.array([Z[d['unit']] for d in tes])
    halves = np.array_split(np.random.default_rng(7900 + 100 * rep + sidx).permutation(len(tes)), 2)
    out = []
    for h in (0, 1):
        ev, fit_t = halves[h], halves[1 - h]                       # evaluate on one half, fit weights with the other
        sc = StandardScaler().fit(np.r_[Zs, Zt[fit_t]])
        lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(np.r_[Zs, Zt[fit_t]]), np.r_[np.zeros(len(Zs)), np.ones(len(fit_t))])
        odds = lambda Zx: np.exp(lr.decision_function(sc.transform(Zx))) * len(Zs) / len(fit_t)
        wc, wt = odds(Zc), odds(Zt[ev]); ess = wc.sum() ** 2 / (wc ** 2).sum()
        for a in ALPHAS:
            Hs = C.conformal_level(Vc, a); Hw = weighted_levels(Vc, wc, wt, a)
            for meth, H in (('static', np.full(len(ev), Hs)), ('weighted', Hw)):
                for e_i, Hh in zip(ev, H):
                    d = tes[e_i]; j = int(C.first_alarm(RMt[e_i], Hh)[0]) if np.isfinite(Hh) else 0
                    notice = d['r'][j] if j < len(d['r']) else np.nan
                    out.append(dict(analysis=an, split=name, rep=rep, unit=d['unit'], group=d['group'], alpha=a, method=meth, H=float(Hh),
                                    fail=bool(Vt[e_i] > Hh), notice=float(notice), infinite=bool(np.isinf(Hh)), ess=float(ess),
                                    w_test=float(wt[list(ev).index(e_i)]) if meth == 'weighted' else 1.0, n_cal=len(cal_u)))
    print(f'  {name} rep {rep} {time.time() - t0:.0f}s', flush=True)
    return out


def main(ds, reps):
    out = f'{OUT}/{ds}_shift.csv.gz'
    if os.path.exists(out):
        print('exists', out); return
    df, kind = load(ds); Z = covariates(df, kind)
    S = splits(df, kind); R = []
    for rep in reps:
        for sidx, (an, name, tr, te) in enumerate(S):
            if an != 'group': continue
            R += run_split(df, kind, an, name, tr, te, rep, sidx, Z)
    pd.DataFrame(R).assign(dataset=ds).to_csv(out, index=False)
    print(ds, 'written', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2])
