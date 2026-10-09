"""Amendment v6, Part 3 (protocol_v6.md): Theorem 6 (probabilistic censored-as-failed, E2) and Proposition 3 (IPCW)
on the existing datasets, with a horizon and four censoring mechanisms imposed on the calibration units.

Repetition 0, five-fold cross-validation, GBM signal and calibration units of run.py; primary lead time.
Horizon c0 = median life of the split's proper-training units. Target: failure within the horizon before the planned
replacement (V° > H). Output: results/raw6/<DS>_cens.csv.gz, one row per split, mechanism, draw, method and alpha.
Usage: python run6_cens.py DATASET
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import core as C
import policies as P
from run import CFG, load, splits, series
from run2 import upper_level

OUT = os.path.join(C.ROOT, 'results', 'raw6'); os.makedirs(OUT, exist_ok=True)
ALPHAS = [0.05, 0.10, 0.20]
GAMMAS = [1.0, 2.0, 4.0, np.inf]
N_DRAWS = 50


def km_censoring(times, is_cens):
    """reverse Kaplan-Meier: returns G(u) = P(C >= u) (left-continuous) as a function"""
    times = np.asarray(times, float); is_cens = np.asarray(is_cens, bool)
    ev = np.unique(times[is_cens]); surv = []; s = 1.0
    for u in ev:
        n_risk = (times >= u).sum(); d = ((times == u) & is_cens).sum()
        s *= (1 - d / n_risk) if n_risk else 1.0; surv.append(s)
    ev = np.array(ev); surv = np.array(surv)

    def G(u):
        u = np.atleast_1d(np.asarray(u, float))
        if not len(ev): return np.ones(len(u))                                                 # no censoring event
        idx = np.searchsorted(ev, u, side='left') - 1                                          # events strictly before u
        return np.where(idx >= 0, surv[np.maximum(idx, 0)], 1.0)
    return G


def weighted_level(V, w, w_test, alpha):
    V = np.asarray(V, float); w = np.asarray(w, float)
    if not len(V) or not np.isfinite(w_test): return np.inf
    o = np.argsort(V, kind='stable'); cum = np.cumsum(w[o]); thr = (1 - alpha) * (w.sum() + w_test)
    i = int(np.searchsorted(cum, thr * (1 - 1e-12), side='left'))
    return float(V[o][i]) if i < len(V) else np.inf


def row_features(t, rho, s, c0, cap):
    return np.column_stack([t / c0, np.minimum(rho, cap) / cap, s / cap])


def run_split(df, kind, an, name, tr, te, sidx):
    cfg = CFG[kind]; t0 = time.time(); cc, sp, kk = cfg['cna_cap'], cfg['cna_span'], cfg['cna_k']
    lead = 30 if kind == 'battery' else 10
    rng = np.random.default_rng(1000 + sidx); perm = rng.permutation(tr)                     # repetition 0, as run.py
    ncal = int(round(0.4 * len(perm))); cal_u = np.sort(perm[:ncal]); fit_u = np.sort(perm[ncal:])
    assert not (set(cal_u) & set(fit_u)) and not (set(tr) & set(te)), 'disjoint units'
    F = C.build_features(df, kind, fit_u)
    X = np.vstack([F[u]['X'] for u in fit_u]); y = np.concatenate([F[u]['r'] for u in fit_u])
    m = C.gbm(X, y, cc)
    cal, tes, fit = series(F, cal_u), series(F, te), series(F, fit_u)
    S_cal = [C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp) for d in cal]
    S_te = [C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp) for d in tes]
    RMc = [C.debounced_running_min(s, kk) for s in S_cal]; RMt = [C.debounced_running_min(s, kk) for s in S_te]
    Lfit = np.array([d['L'] for d in fit]); c0 = float(np.round(np.median(Lfit)))
    Lc = np.array([d['L'] for d in cal]); Lt = np.array([d['L'] for d in tes])
    Vc = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMc, cal)])
    Vt = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMt, tes)])
    Voc = np.where(Lc <= c0, Vc, -np.inf); Vot = np.where(Lt <= c0, Vt, -np.inf)

    # ---- p0 for E2-HIST: logistic regression on out-of-fold signals of the proper-training units (never censored)
    hrng = np.random.default_rng(7800 + sidx); halves = np.array_split(hrng.permutation(fit_u), 2)
    feats, labs = [], []
    for k_, hu in enumerate(halves):
        other = halves[1 - k_]
        Xo = np.vstack([F[u]['X'] for u in other]); yo = np.concatenate([F[u]['r'] for u in other]); mo = C.gbm(Xo, yo, cc)
        for u in hu:
            d = next(dd for dd in fit if dd['unit'] == u)
            s = C.smooth(mo.predict(F[u]['X'])[d['mask']], sp); rho = C.debounced_running_min(s, kk)
            ok = d['t'] < c0
            if ok.any():
                feats.append(row_features(d['t'][ok], rho[ok], s[ok], c0, cc)); labs.append(np.full(ok.sum(), float(d['L'] <= c0)))
    Xh = np.vstack(feats); yh = np.concatenate(labs)
    hist = LogisticRegression(C=1.0, max_iter=2000).fit(Xh, yh) if 0 < yh.mean() < 1 else None
    surv_fit = lambda x: (Lfit[None, :] > np.asarray(x, float)[:, None]).mean(1)                # empirical survival of training lives

    crng = np.random.default_rng(7700 + sidx)
    recs, Hs = [], []
    base = dict(analysis=an, split=name, n_cal=len(cal_u), c0=c0, lead=lead)
    for mech in ('M1', 'M2', 'M3', 'M4'):
        for draw in range(N_DRAWS):
            if mech == 'M1':
                subj = crng.random(len(cal)) < 0.5; c = np.where(subj, crng.uniform(0, c0, len(cal)), np.inf)
            elif mech == 'M2':
                c = c0 - crng.uniform(0, c0, len(cal))
            elif mech == 'M3':
                Hinc = np.median(Vc) + crng.normal(0, 1.0) * np.std(Vc[np.isfinite(Vc)]) * 0.25
                c = np.full(len(cal), np.inf)
                for i, (rm, d) in enumerate(zip(RMc, cal)):
                    j = int(C.first_alarm(rm, Hinc)[0])
                    if j < len(d['r']) and d['r'][j] > lead: c[i] = d['t'][j] + lead
            else:
                subj = crng.random(len(cal)) < np.where(Lc <= c0, 0.8, 0.2); c = np.where(subj, crng.uniform(0, c0, len(cal)), np.inf)
            cens = c < np.minimum(Lc, c0)
            ci = np.where(cens, c, 0.0)
            U = np.array([upper_level(RMc[i], cal[i], lead, ci[i]) if cens[i] else Voc[i] for i in range(len(cal))])
            assert np.all(U[cens] >= Vc[cens] - 1e-9), 'Lemma 3'
            # probabilities of failing within the horizon for censored units
            pb = np.zeros(len(cal)); ph = np.zeros(len(cal))
            idx = np.flatnonzero(cens)
            if len(idx):
                Sc = surv_fit(c[idx]); S0 = surv_fit([c0])[0]
                pb[idx] = np.where(Sc > 0, 1 - S0 / np.maximum(Sc, 1e-12), 1.0)
                ph[idx] = pb[idx]                                    # no model, or no row observed yet: baseline probability
                if hist is not None:
                    has = [i for i in idx if (cal[i]['t'] <= c[i]).any()]
                    if has:
                        fr = []
                        for i in has:
                            d = cal[i]; jl = int(np.flatnonzero(d['t'] <= c[i])[-1])
                            fr.append(row_features(d['t'][jl:jl + 1], RMc[i][jl:jl + 1], S_cal[i][jl:jl + 1], c0, cc)[0])
                        ph[has] = hist.predict_proba(np.array(fr))[:, 1]
            u = crng.random(len(cal))
            # IPCW: reverse Kaplan-Meier of the censoring times of the calibration units (no covariates)
            obs_t = np.where(cens, c, np.minimum(Lc, c0)); G = km_censoring(obs_t, cens)
            comp = ~cens; Gc = G(np.minimum(Lc[comp], c0)); Gc0 = float(G([c0])[0])
            for a in ALPHAS:
                lv = {'ORACLE': C.conformal_level(Voc, a), 'T2': C.conformal_level(np.where(cens, U, Voc), a),
                      'CC': C.conformal_level(Voc[comp], a) if comp.any() else np.inf,
                      'IPCW': weighted_level(Voc[comp], 1 / np.maximum(Gc, 1e-300), (1 / Gc0) if Gc0 > 0 else np.inf, a)}
                for G_ in GAMMAS:
                    for nm, pp in (('E2BASE', pb), ('E2HIST', ph)):
                        pt = np.ones(len(cal)) if np.isinf(G_) else G_ * pp / (1 + (G_ - 1) * pp)
                        lv[f'{nm}_G{G_:g}'] = C.conformal_level(np.where(cens, np.where(u <= pt, U, -np.inf), Voc), a)
                for meth, h in lv.items():
                    recs.append(dict(base, mechanism=mech, draw=draw, method=meth, alpha=a, frac_cens=float(cens.mean()),
                                     cens_time_mean=float(c[cens].mean()) if cens.any() else np.nan)); Hs.append(h)
    Hs = np.array(Hs)
    fail = (Vot[:, None] > Hs[None, :])                                                     # Lemma 2 (horizon form)
    J = np.vstack([C.first_alarm(rm, Hs) for rm in RMt])
    tJ = np.vstack([np.where(J[i] < len(d['t']), d['t'][np.minimum(J[i], len(d['t']) - 1)], np.inf) for i, d in enumerate(tes)])
    rJ = np.vstack([np.where(J[i] < len(d['r']), d['r'][np.minimum(J[i], len(d['r']) - 1)], np.nan) for i, d in enumerate(tes)])
    first_row = (J <= kk - 1)                                                                # infinite level: alarm at row 0
    hz = Lt <= c0
    with np.errstate(all='ignore'):
        notice = np.nanmean(np.where(hz[:, None] & ~fail, rJ, np.nan), 0)
        false_alarm = (tJ[~hz] < c0).mean(0) if (~hz).any() else np.full(len(Hs), np.nan)
    out = []
    for r, h, f, n_, fr_, fa in zip(recs, Hs, fail.mean(0), notice, first_row.mean(0), false_alarm):
        r.update(H=h, fail=float(f), notice=float(n_), first_row=float(fr_), false_alarm=float(fa), n_test=len(tes),
                 n_test_horizon=int(hz.sum())); out.append(r)
    print(f'  {name} {time.time() - t0:.0f}s', flush=True)
    return out


def main(ds):
    out = f'{OUT}/{ds}_cens.csv.gz'
    if os.path.exists(out):
        print('exists', out); return
    df, kind = load(ds)
    S = [s for s in splits(df, kind) if s[0] == 'cv']
    R = []
    for sidx, (an, name, tr, te) in enumerate(S): R += run_split(df, kind, an, name, tr, te, sidx)
    pd.DataFrame(R).assign(dataset=ds).to_csv(out, index=False)
    print(ds, 'written', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
