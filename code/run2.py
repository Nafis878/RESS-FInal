"""Amendment experiments (protocol_v2.md): censored calibration data, cross-fitted calibration (CV+), and the decision
setting of Kamariotis et al. (decision grid, no lead time, metric M).

Usage: python run2.py DATASET [REPS, e.g. 0 or 0,1,2]
Same outer splits, proper-training/calibration splits and seeds as run.py, so the split-conformal policies here reproduce
those of run.py unit by unit. Outputs in results/raw2/: <DS>_rep<r>_censor.csv.gz, <DS>_rep<r>_cvplus.csv.gz, <DS>_rep<r>_grid.csv.gz
"""
import os, sys, time
import numpy as np, pandas as pd
import core as C
import policies as P
from run import CFG, RHOS, load, splits, series

OUT = os.path.join(C.ROOT, 'results', 'raw2'); os.makedirs(OUT, exist_ok=True)
ALPHAS = [0.05, 0.10, 0.20]
ALPHA_MAX = 0.10
PIS = [0.25, 0.50, 0.75]
N_DRAWS = 50
GRID_DT = 10                        # decision interval of Kamariotis et al. (cycles); C-MAPSS/PHM08 only
GRID_RATIOS = [0.1, 0.02]           # c_p / c_c (0.1 primary, as in their paper)


# ------------------------------------------------------------------------------------------------- helpers
def upper_level(rm, d, lead, c):
    """upper bound on the critical level of a unit known to be alive at cycle c (life > c): the critical level computed as
    if the unit had failed at c (Lemma 3). +inf if no row has t <= c - lead."""
    ok = np.flatnonzero(d['t'] <= c - lead)
    return float(rm[ok[-1]]) if len(ok) else np.inf


def eval_level(RM, ser, H, lead):
    J = np.array([[int(C.first_alarm(rm, H)[0])] for rm in RM])
    succ, use, notice = P.outcome(J, ser, lead)
    return succ[:, 0], use[:, 0], notice[:, 0]


def cvplus_counts(rho_k, Vsorted_k):
    """C_t = sum_k #{i in fold k : V_i >= rho^(k)_t}; nondecreasing in t"""
    tot = 0
    for rho, Vs in zip(rho_k, Vsorted_k):
        tot = tot + (len(Vs) - np.searchsorted(Vs, rho, side='left'))
    return tot


# ------------------------------------------------------------------------------------------------- one split
def run_split(df, kind, an, name, tr, te, rep, sidx):
    cfg = CFG[kind]; t0 = time.time(); cc, sp, kk = cfg['cna_cap'], cfg['cna_span'], cfg['cna_k']
    rng = np.random.default_rng(1000 + 100 * rep + sidx); perm = rng.permutation(tr)
    ncal = int(round(0.4 * len(perm))); cal_u = np.sort(perm[:ncal]); fit_u = np.sort(perm[ncal:])
    assert not (set(cal_u) & set(fit_u)) and not (set(tr) & set(te)), 'E1 disjoint units'
    base = dict(analysis=an, split=name, rep=rep, n_cal=len(cal_u), n_train=len(tr))
    F = C.build_features(df, kind, fit_u)
    X = np.vstack([F[u]['X'] for u in fit_u]); y = np.concatenate([F[u]['r'] for u in fit_u])
    m = C.gbm(X, y, cc)
    cal, tes = series(F, cal_u), series(F, te)
    S_cal = [C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp) for d in cal]
    S_te = [C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp) for d in tes]
    RMc = [C.debounced_running_min(s, kk) for s in S_cal]; RMt = [C.debounced_running_min(s, kk) for s in S_te]
    cens, cvp, grid = [], [], []

    # ---------------------------------------------------------------- A. censored calibration data (split CNA)
    crng = np.random.default_rng(7000 + 100 * rep + sidx)
    L = np.array([d['L'] for d in cal])
    for lead in cfg['leads']:
        V = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMc, cal)])
        Vte = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMt, tes)])
        recs, Hs = [], []                                                          # evaluate all levels of this lead at once
        for a in ALPHAS:                                                             # reference: no censoring
            recs.append(dict(base, lead=lead, alpha=a, mechanism='none', pi=0.0, draw=-1, method='oracle', frac_cens=0.0)); Hs.append(C.conformal_level(V, a))
        for mech in ('random', 'administrative', 'policy'):
            for pi in (PIS if mech != 'policy' else [np.nan]):
                for draw in range(N_DRAWS):
                    if mech == 'random':                       # independent: censored w.p. pi at a uniform time in [0.3L, L)
                        cz = crng.random(len(cal)) < pi; c = np.where(cz, crng.uniform(0.3, 1.0, len(cal)) * L, np.inf)
                    elif mech == 'administrative':             # staggered entry, common end of observation: long lives censored
                        e = crng.uniform(0, L.max(), len(cal)); T_end = np.quantile(e + L, 1 - pi)
                        cz = e + L > T_end; c = np.where(cz, T_end - e, np.inf)
                    else:                                      # history under an incumbent alarm (level = median critical level
                        Hinc = np.median(V) + crng.normal(0, 1.0) * np.std(V) * 0.25   # of these units, jittered): replaced units censored
                        c = np.full(len(cal), np.inf)
                        for i, (rm, d) in enumerate(zip(RMc, cal)):
                            j = int(C.first_alarm(rm, Hinc)[0])
                            if j < len(d['r']) and d['r'][j] > lead: c[i] = d['t'][j] + lead   # replaced alive at alarm + lead
                        cz = np.isfinite(c)
                    U = np.array([upper_level(RMc[i], cal[i], lead, c[i]) if cz[i] else V[i] for i in range(len(cal))])
                    assert np.all(U >= V - 1e-12), 'Lemma 3 violated'
                    for a in ALPHAS:
                        for meth, Hh in (('censored_as_failure', C.conformal_level(U, a)),
                                         ('complete_case', C.conformal_level(V[~cz], a) if (~cz).sum() else np.inf)):
                            recs.append(dict(base, lead=lead, alpha=a, mechanism=mech, pi=pi, draw=draw, method=meth, frac_cens=float(cz.mean())))
                            Hs.append(Hh)
        Hs = np.array(Hs)
        fail = (Vte[:, None] > Hs[None, :])                                          # Lemma 2: failure iff V_test > H
        J = np.vstack([C.first_alarm(rm, Hs) for rm in RMt])                         # alarm rows for every level
        notice = np.vstack([np.where(J[i] < len(d['r']), d['r'][np.minimum(J[i], len(d['r']) - 1)], np.nan) for i, d in enumerate(tes)])
        assert np.array_equal(fail, ~(np.nan_to_num(notice, nan=-1) > lead)), 'Lemma 2 check'
        nt = np.where(fail, np.nan, notice)
        with np.errstate(all='ignore'):
            mn = np.nanmean(nt, 0)
        for r, h, f, m_ in zip(recs, Hs, fail.mean(0), mn):
            r.update(H=h, fail=float(f), notice=float(m_), n_test=len(tes)); cens.append(r)

    # ---------------------------------------------------------------- B. cross-fitted calibration on all training units (CV+)
    folds = np.array_split(np.random.default_rng(3000 + 100 * rep + sidx).permutation(tr), 5)
    trs = series(C.build_features(df, kind, tr), tr)                        # r, t, L of training units (features unused)
    trs_by_u = {d['unit']: d for d in trs}
    RHO_te, RMtr, kof = [], {}, {}
    for k, fu in enumerate(folds):
        fit_k = np.setdiff1d(tr, fu); Fk = C.build_features(df, kind, fit_k)
        Xk = np.vstack([Fk[u]['X'] for u in fit_k]); yk = np.concatenate([Fk[u]['r'] for u in fit_k]); mk = C.gbm(Xk, yk, cc)
        for u in fu:
            d = trs_by_u[u]; RMtr[u] = C.debounced_running_min(C.smooth(mk.predict(Fk[u]['X'])[d['mask']], sp), kk); kof[u] = k
        RHO_te.append([C.debounced_running_min(C.smooth(mk.predict(Fk[d['unit']]['X'])[d['mask']], sp), kk) for d in tes])
    n = len(tr)
    for lead in cfg['leads']:
        Vtr = {u: C.critical_level(RMtr[u], trs_by_u[u]['r'], lead) for u in tr}
        Vs_k = [np.sort([Vtr[u] for u in fu]) for fu in folds]
        CT = [cvplus_counts([RHO_te[k][i] for k in range(5)], Vs_k) for i in range(len(tes))]   # count series per test unit
        # approximate calibration use for the decision-aware choice: training unit i with its own fold model against all other V
        Vall = np.sort(np.array([Vtr[u] for u in tr]))
        CTtr = {u: (n - np.searchsorted(Vall, RMtr[u], side='left')) - (Vtr[u] >= RMtr[u]) for u in tr}

        def rows_for(cthr, counts, ser):
            return np.array([[int(np.searchsorted(cnt, cthr, side='left'))] for cnt in counts])   # first t with count >= cthr

        for a in ALPHAS:
            q = int(np.ceil((1 - a) * (n + 1))); cthr = n + 1 - q
            J = rows_for(cthr, CT, tes); s_, u_, n_ = P.outcome(J, tes, lead)
            for rho in RHOS:
                for i, d in enumerate(tes):
                    cvp.append(dict(base, policy=f'CNAplus_a{a:.2f}', lead=lead, rho=rho, unit=d['unit'], group=d['group'], L=d['L'], notice=n_[i, 0],
                                    fail=not s_[i, 0], use=u_[i, 0], cost=1.0 if s_[i, 0] else float(rho), cthr=cthr))
        q0 = int(np.ceil((1 - ALPHA_MAX) * (n + 1))); cmax = n + 1 - q0
        cands = np.arange(1, cmax + 1)                                           # smaller threshold = earlier alarm
        Jtr = np.vstack([np.searchsorted(CTtr[d['unit']], cands, side='left') for d in trs])
        str_, utr, _ = P.outcome(Jtr, trs, lead)
        for rho in RHOS:
            bound = (cands) / (n + 1)                                            # risk proxy (n+1-q)/(n+1) with q = n+1-c
            obj = (1.0 + (rho - 1.0) * bound) / utr.mean(0); cbest = int(cands[int(np.argmin(obj))])
            J = rows_for(cbest, CT, tes); s_, u_, n_ = P.outcome(J, tes, lead)
            for i, d in enumerate(tes):
                cvp.append(dict(base, policy='CNAplus_DA', lead=lead, rho=rho, unit=d['unit'], group=d['group'], L=d['L'], notice=n_[i, 0],
                                fail=not s_[i, 0], use=u_[i, 0], cost=1.0 if s_[i, 0] else float(rho), cthr=cbest))

    # ---------------------------------------------------------------- C. Kamariotis et al. setting: decision grid, no lead time, metric M
    if kind == 'engine':
        def gridify(ser, sig):
            out_s, out_x = [], []
            for d, s in zip(ser, sig):
                g = (d['t'] % GRID_DT) == 0
                out_s.append(dict(unit=d['unit'], r=d['r'][g], t=d['t'][g], L=d['L'], group=d['group'])); out_x.append(s[g])
            return out_s, out_x
        raw_cal = [m.predict(F[d['unit']]['X'])[d['mask']] for d in cal]; raw_te = [m.predict(F[d['unit']]['X'])[d['mask']] for d in tes]
        gc, gS = gridify(cal, S_cal); gt, gT = gridify(tes, S_te); _, grc = gridify(cal, raw_cal); _, grt = gridify(tes, raw_te)
        gRMc = [C.debounced_running_min(s, 1) for s in gS]; gRMt = [C.debounced_running_min(s, 1) for s in gT]   # one decision = one row
        Vg = np.array([C.critical_level(rm, d['r'], 0) for rm, d in zip(gRMc, gc)])
        pr = P.PRes(grc, gc, 0, GRID_DT, min_rows=20)
        Ltr = np.array([F[u]['L'] for u in tr])
        taus = cfg['taus']
        Jc_cal = np.hstack([P.alarm_rows([C.smooth(p, s_) for p in grc], 1, taus) for s_ in P.SPANS])
        Jc_te = np.hstack([P.alarm_rows([C.smooth(p, s_) for p in grt], 1, taus) for s_ in P.SPANS])
        succ_c, use_c, not_c = P.outcome(Jc_cal, gc, 0)
        Wb = P.boot_weights(len(gc), 4026 + sidx + 100 * rep)
        for ratio in GRID_RATIOS:
            rho = 1.0 / ratio
            pol = {}
            Hda, _, _ = P.cna_da(Vg, gS, gc, 1, 0, rho, ALPHA_MAX); pol['CNA_DA'] = P.alarm_rows(gT, 1, [Hda])[:, 0]
            pol['CNA_a0.05'] = P.alarm_rows(gT, 1, [C.conformal_level(Vg, 0.05)])[:, 0]
            pol['KAM_P1_heur'] = P.pres_rows(pr, grt, [ratio])[:, 0]
            pol['KAM_P1_opt'] = P.pres_rows(pr, grt, [P.pres_optimise(pr, grc, gc, 0, rho)])[:, 0]
            pol['STW_cost_1se'] = Jc_te[:, P.select_cost(succ_c, use_c, not_c, rho, '1se', Wb)]
            for nm, J in pol.items():
                s_, u_, n_ = P.outcome(np.asarray(J).reshape(-1, 1), gt, 0)
                for i, d in enumerate(gt):
                    grid.append(dict(base, policy=nm, ratio=ratio, unit=d['unit'], L=d['L'], fail=not s_[i, 0], use=u_[i, 0],
                                     cost=1.0 if s_[i, 0] else rho))
            Ts = np.arange(GRID_DT, int(Ltr.max()) + GRID_DT, GRID_DT)
            cr = np.array([np.where(Ltr > T, 1.0, rho).sum() / np.minimum(Ltr, T).sum() for T in Ts]); T = Ts[int(np.argmin(cr))]
            for d in gt:
                ok = d['L'] > T; last = GRID_DT * np.floor((d['L'] - 1) / GRID_DT)
                grid.append(dict(base, policy='age', ratio=ratio, unit=d['unit'], L=d['L'], fail=not ok, use=T if ok else d['L'], cost=1.0 if ok else rho))
                grid.append(dict(base, policy='perfect_grid', ratio=ratio, unit=d['unit'], L=d['L'], fail=False, use=last, cost=1.0))
    print(f'  {name} rep {rep} ({time.time() - t0:.0f}s)', flush=True)
    return cens, cvp, grid


def main(ds, reps):
    """one output file set per (dataset, repetition); existing checkpoints are skipped"""
    df, kind = load(ds); S = splits(df, kind)
    for rep in reps:
        tag = f'{OUT}/{ds}_rep{rep}'
        if os.path.exists(f'{tag}_cvplus.csv.gz'): print(ds, 'rep', rep, 'exists'); continue
        A, B, G = [], [], []
        for sidx, (an, name, tr, te) in enumerate(S):
            a, b, g = run_split(df, kind, an, name, tr, te, rep, sidx); A += a; B += b; G += g
        pd.DataFrame(A).assign(dataset=ds).to_csv(f'{tag}_censor.csv.gz', index=False)
        if G: pd.DataFrame(G).assign(dataset=ds).to_csv(f'{tag}_grid.csv.gz', index=False)
        pd.DataFrame(B).assign(dataset=ds).to_csv(f'{tag}_cvplus.csv.gz', index=False)
        print(ds, 'rep', rep, 'written', flush=True)


if __name__ == '__main__':
    if os.environ.get('SMOKE'):
        df, kind = load(sys.argv[1]); S = splits(df, kind)
        a, b, g = run_split(df, kind, *S[0], 0, 0)
        print(pd.DataFrame(a).groupby(['mechanism', 'method', 'alpha']).agg(fail=('fail', 'mean'), notice=('notice', 'mean'), fc=('frac_cens', 'mean')).round(3))
        print(pd.DataFrame(b).groupby(['policy', 'lead']).fail.mean().unstack().round(3))
        if g:
            G = pd.DataFrame(g); print(G.groupby(['ratio', 'policy']).agg(fail=('fail', 'mean'), c=('cost', 'sum'), u=('use', 'sum')).assign(R=lambda x: x.c / x.u))
    else:
        main(sys.argv[1], [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2])
