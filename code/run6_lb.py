"""Amendment v6, Part 1 (protocol_v6.md): action level and debounce of the per-cycle lower-bound alarms (H14) and
Theorem 5 on the data (H15).

Same outer splits, proper-training/calibration splits, seeds and GBM signal as run.py (so the ORIG variant must
reproduce run.py exactly). Variants of LB-all / LB-near:
    ORIG  debounce 2, action level l + 1          (main protocol)
    K1    debounce 1, action level l + Delta      (Delta = row spacing: 1 cycle; battery 3 cycles)
    K2C   debounce 2, action level l + 2 Delta
Repetition 0 also: learners of amendment v5 (RIDGE, RF, WMLP; K1) and the Theorem 5 diagnostics (GBM, alpha 0.10,
primary lead, K1). Outputs in results/raw6/:
    <DS>_lb.csv.gz          one row per test unit, policy, alpha, lead, variant (GBM, repetitions 0-2)
    <DS>_lb_learners.csv.gz one row per test unit, learner, policy, alpha, lead (K1, repetition 0)
    <DS>_b1_units.csv.gz    per test unit: failure, critical-row exceedance and miss, products for the bounds
    <DS>_b1_rows.csv.gz     per remaining-life value: rows, exceedances of the action level, misses
Usage: python run6_lb.py DATASET [REPS, e.g. 0,1,2]
"""
import os, sys, time
import numpy as np, pandas as pd
import core as C
import policies as P
from run import CFG, ALPHAS, load, splits, series
import run5

OUT = os.path.join(C.ROOT, 'results', 'raw6'); os.makedirs(OUT, exist_ok=True)


def spacing(kind):
    return 3.0 if kind == 'battery' else 1.0


def variants(lead, dlt):
    return {'ORIG': (2, lead + 1.0), 'K1': (1, lead + dlt), 'K2C': (2, lead + 2.0 * dlt)}


def lb_rows(base, S_cal, S_te, cal, tes, cc, leads, dlt, which=('ORIG', 'K1', 'K2C')):
    out = []
    for lead in leads:
        for a in ALPHAS:
            for near in (False, True):
                q = P.lb_quantile(S_cal, cal, cc, a, near)
                for vn, (k, act) in variants(lead, dlt).items():
                    if vn not in which: continue
                    J = P.alarm_rows(S_te, k, [act + q])
                    succ, use, notice = P.outcome(J, tes, lead)
                    for i, d in enumerate(tes):
                        out.append(dict(base, policy=f'LB{"near" if near else "all"}', alpha=a, lead=lead, variant=vn, q=q,
                                        unit=d['unit'], L=d['L'], fail=not succ[i, 0], notice=notice[i, 0]))
    return out


def acf_lag(res_list, max_lag=60, thr=0.2):
    """mean autocorrelation of the residual series across units; first lag with ACF < thr (max_lag if none)"""
    ac = np.zeros(max_lag + 1); cnt = np.zeros(max_lag + 1)
    for e in res_list:
        e = e - e.mean(); v = (e * e).sum()
        if len(e) < 3 or v <= 0: continue
        for h in range(min(max_lag, len(e) - 1) + 1):
            ac[h] += (e[:len(e) - h] * e[h:]).sum() / v; cnt[h] += 1
    ac = ac / np.maximum(cnt, 1)
    below = np.flatnonzero(ac < thr)
    return (int(below[0]) if len(below) else max_lag), ac


def b1_diagnostics(base, S_cal, S_te, cal, tes, cc, lead, dlt):
    """Theorem 5 on the data: K1 variant, alpha 0.10, primary lead; per unit and per remaining life."""
    units, rows = [], []
    for near in (False, True):
        pol = f'LB{"near" if near else "all"}'
        q = P.lb_quantile(S_cal, cal, cc, 0.10, near); act = lead + dlt
        J = P.alarm_rows(S_te, 1, [act + q]); succ, _, _ = P.outcome(J, tes, lead)
        LB = [s - q for s in S_te]
        # per remaining life: exceedance of the action level and miss (LB > r), pooled over test rows
        R = np.concatenate([d['r'] for d in tes]); LBc = np.concatenate(LB)
        tgt = np.concatenate([np.minimum(d['r'], cc) for d in tes])
        Rk = np.round(R).astype(int)
        df = pd.DataFrame(dict(r=Rk, exceed=LBc > act, miss=LBc > R, miss_target=LBc > tgt))
        g = df.groupby('r').agg(rows=('exceed', 'size'), exceed=('exceed', 'sum'), miss=('miss', 'sum'), miss_target=('miss_target', 'sum')).reset_index()
        for rr in g.itertuples():
            rows.append(dict(base, policy=pol, r=rr.r, rows=rr.rows, exceed=int(rr.exceed), miss=int(rr.miss), miss_target=int(rr.miss_target)))
        pi = dict(zip(g.r, g.exceed / g.rows))
        res = [s - np.minimum(d['r'], cc) for s, d in zip(S_te, tes)]
        m_lag, _ = acf_lag(res)
        for i, d in enumerate(tes):
            A = np.flatnonzero(d['r'] > lead)
            if not len(A): continue
            jl = A[-1]
            p_all = np.array([pi.get(int(round(d['r'][j])), np.nan) for j in A])
            idx_m = jl - np.arange(0, len(A), m_lag + 1)
            p_m = np.array([pi.get(int(round(d['r'][j])), np.nan) for j in idx_m])
            units.append(dict(base, policy=pol, unit=d['unit'], fail=not succ[i, 0], r_crit=float(d['r'][jl]),
                              crit_exceed=bool(LB[i][jl] > act), crit_miss=bool(LB[i][jl] > d['r'][jl]),
                              prod_indep=float(np.nanprod(p_all)), min_bound=float(np.nanmin(p_all)), prod_mdep=float(np.nanprod(p_m)),
                              m_lag=m_lag, pooled_miss_target=float(df.miss_target.mean()), pooled_miss=float(df.miss.mean())))
    return units, rows


def run_split(df, kind, an, name, tr, te, rep, sidx, learners=False):
    cfg = CFG[kind]; t0 = time.time(); dlt = spacing(kind)
    rng = np.random.default_rng(1000 + 100 * rep + sidx); perm = rng.permutation(tr)          # identical to run.run_split
    ncal = int(round(0.4 * len(perm))); cal_u = np.sort(perm[:ncal]); fit_u = np.sort(perm[ncal:])
    assert not (set(cal_u) & set(fit_u)) and not (set(tr) & set(te)), 'disjoint units'
    F = C.build_features(df, kind, fit_u)
    cal, tes = series(F, cal_u), series(F, te)
    X = np.vstack([F[u]['X'] for u in fit_u]); y = np.concatenate([F[u]['r'] for u in fit_u])
    cc, sp = cfg['cna_cap'], cfg['cna_span']
    m = C.gbm(X, y, cc)
    S_cal = [C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp) for d in cal]
    S_te = [C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp) for d in tes]
    base = dict(analysis=an, split=name, rep=rep, n_cal=len(cal_u))
    lb = lb_rows(dict(base, learner='GBM'), S_cal, S_te, cal, tes, cc, cfg['leads'], dlt)
    lrn, b1u, b1r = [], [], []
    if rep == 0 and an == 'cv':
        lp = 30 if kind == 'battery' else 10
        b1u, b1r = b1_diagnostics(base, S_cal, S_te, cal, tes, cc, lp, dlt)
        if learners:
            Xc = [F[d['unit']]['X'][d['mask']] for d in cal]; Xt = [F[d['unit']]['X'][d['mask']] for d in tes]
            Wd = run5.raw_matrix(df, kind, fit_u); WX = np.vstack([Wd[u] for u in fit_u])
            Wc = [Wd[d['unit']][d['mask']] for d in cal]; Wt = [Wd[d['unit']][d['mask']] for d in tes]
            for ln in ['RIDGE', 'RF', 'WMLP']:
                Xtr, Xs = (WX, Wc + Wt) if ln == 'WMLP' else (X, Xc + Xt)
                p = run5.fit_predict(ln, Xtr, y, cc, Xs); pc, pt = p[:len(cal)], p[len(cal):]
                lrn += lb_rows(dict(base, learner=ln), [C.smooth(s, sp) for s in pc], [C.smooth(s, sp) for s in pt], cal, tes, cc,
                               cfg['leads'], dlt, which=('K1',))
    print(f'  {name} rep {rep} {time.time() - t0:.0f}s', flush=True)
    return lb, lrn, b1u, b1r


def main(ds, reps):
    df, kind = load(ds)
    S = [s for s in splits(df, kind) if s[0] == 'cv']
    for rep in reps:
        out = f'{OUT}/{ds}_rep{rep}_lb.csv.gz'
        if os.path.exists(out):
            print('exists', out); continue
        LB, LR, BU, BR = [], [], [], []
        for sidx, (an, name, tr, te) in enumerate(S):
            a, b, c, d = run_split(df, kind, an, name, tr, te, rep, sidx, learners=(rep == 0)); LB += a; LR += b; BU += c; BR += d
        if rep == 0:
            pd.DataFrame(LR).assign(dataset=ds).to_csv(f'{OUT}/{ds}_lb_learners.csv.gz', index=False)
            pd.DataFrame(BU).assign(dataset=ds).to_csv(f'{OUT}/{ds}_b1_units.csv.gz', index=False)
            pd.DataFrame(BR).assign(dataset=ds).to_csv(f'{OUT}/{ds}_b1_rows.csv.gz', index=False)
        pd.DataFrame(LB).assign(dataset=ds).to_csv(out, index=False)
        print(ds, 'rep', rep, 'written', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2])
