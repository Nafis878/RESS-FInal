"""Main experiment: conformal notice-calibrated alarms and every comparison policy, on one dataset.

Usage: python run.py DATASET [REPEATS]
  DATASET in PHM08, FD001, FD002, FD003, FD004, NCMAPSS, BATTERY

Protocol (protocol.md): outer splits by unit (5-fold, seed 2026) and, for NCMAPSS and BATTERY, leave-one-group-out
splits (flight-data subset / test batch). Inside each outer training set the units are split at random into PROPER
TRAINING units (60 %, fit the regressors and the normalisation) and CALIBRATION units (40 %, every selection and
calibration step). Outer test units are touched once. REPEATS (default 3) repeats the proper-training/calibration split with
seeds 1000 + 100 * rep + split index.
Outputs in results/raw/: <DS>_decisions.csv.gz, <DS>_window.csv.gz, <DS>_levels.csv.gz, <DS>_pilot.csv.gz
"""
import os, sys, time
import numpy as np, pandas as pd
import core as C
import policies as P

OUT = os.path.join(C.ROOT, 'results', 'raw'); os.makedirs(OUT, exist_ok=True)
RHOS = [5, 10, 20, 50]
ALPHAS = [0.05, 0.10, 0.20]
ALPHA_MAX = 0.10
CFG = {
    'engine': dict(lo=-13.0, hi=2.0, Hw=20.0, Hs=[10.0, 15.0, 20.0, 30.0, 40.0], caps=[40.0, 60.0, 90.0], margin=20.0,
                   shifts=np.arange(-12, 8.01, 0.5), taus=np.arange(1.0, 60.01, 1.0), leads=[0, 5, 10, 15], cna_cap=60.0,
                   dt=1.0, step=1.0, cna_span=3, cna_k=2),
    'battery': dict(lo=-39.0, hi=6.0, Hw=60.0, Hs=[30.0, 45.0, 60.0, 90.0, 120.0], caps=[120.0, 180.0, 270.0], margin=60.0,
                    shifts=np.arange(-36, 24.01, 1.5), taus=np.arange(3.0, 180.01, 3.0), leads=[0, 15, 30, 45], cna_cap=180.0,
                    dt=3.0, step=3.0, cna_span=3, cna_k=2),
}
CFG['ncmapss'] = CFG['engine']
TREND_W = (10, 20, 40)                      # rows of 3 cycles: 30, 60, 120 cycles
CNA_TREND_W = 20


def load(ds):
    if ds == 'NCMAPSS': return C.load_ncmapss(), 'ncmapss'
    if ds == 'BATTERY': return C.load_battery(), 'battery'
    return C.load_engines(ds), 'engine'


def splits(df, kind):
    units = np.sort(df.unit.unique()); out = []
    for f, te in enumerate(C.outer_folds(units), 1):
        out.append(('cv', f'cv{f}', np.setdiff1d(units, te), np.sort(te)))
    if kind in ('battery', 'ncmapss'):
        g = df.groupby('unit').group.first()
        for gi in sorted(g.unique()):
            te = np.sort(g.index[g == gi].values); out.append(('group', f'g{gi}', np.setdiff1d(units, te), te))
    return out


def series(F, units, rmin=1.0):
    """rows with at least rmin cycle left (the unit is alive and an alarm can still act)"""
    out = []
    for u in units:
        d = F[u]; m = d['r'] >= rmin
        out.append(dict(unit=u, r=d['r'][m], t=d['t'][m], L=d['L'], group=d['group'], mask=m))
    return out


def run_split(df, kind, an, name, tr, te, rep, sidx):
    cfg = CFG[kind]; t0 = time.time()
    rng = np.random.default_rng(1000 + 100 * rep + sidx); perm = rng.permutation(tr)
    ncal = int(round(0.4 * len(perm))); cal_u = np.sort(perm[:ncal]); fit_u = np.sort(perm[ncal:])
    assert not (set(cal_u) & set(fit_u)) and not (set(tr) & set(te)), 'E1 disjoint units'
    F = C.build_features(df, kind, fit_u)                          # normalisation from proper-training units only
    X = np.vstack([F[u]['X'] for u in fit_u]); y = np.concatenate([F[u]['r'] for u in fit_u])
    cal, tes = series(F, cal_u), series(F, te)
    pred = {}
    for c in cfg['caps']:
        m = C.gbm(X, y, c)
        pred[c] = {'cal': [m.predict(F[d['unit']]['X'])[d['mask']] for d in cal], 'te': [m.predict(F[d['unit']]['X'])[d['mask']] for d in tes]}
    dec, win, lev = [], [], []
    base = dict(analysis=an, split=name, rep=rep, n_fit=len(fit_u), n_cal=len(cal_u))

    def add(policy, lead, rho, J, ser, extra=None):
        succ, use, notice = P.outcome(np.asarray(J).reshape(-1, 1), ser, lead)
        for i, d in enumerate(ser):
            row = dict(base, policy=policy, lead=lead, rho=rho, unit=d['unit'], group=d['group'], L=d['L'],
                       notice=notice[i, 0], fail=not succ[i, 0], use=use[i, 0], cost=1.0 if succ[i, 0] else float(rho))
            if extra: row.update(extra)
            dec.append(row)

    # ------------------------------------------------ window-tuned STW, per warning level (and the window analysis at Hw)
    wsel = {H: P.window_select(cal, {c: pred[c]['cal'] for c in cfg['caps']}, H, cfg['lo'], cfg['hi'], cfg['shifts'], cfg['caps'], cfg['margin'])
            for H in cfg['Hs']}
    wJ_cal, wJ_te = {}, {}
    for H, s in wsel.items():
        wJ_cal[H] = P.window_apply(pred[s['cap']]['cal'], cal, H, s); wJ_te[H] = P.window_apply(pred[s['cap']]['te'], tes, H, s)
    Hw = cfg['Hw']; sw = wsel[Hw]
    const = P.constant_select(wJ_cal[Hw], cal, Hw, cfg['lo'], cfg['hi'], cfg['step'])
    # calibration-set estimates of the pmf of r at the alarm near the two edges of the constant window (Theorem 1)
    r_cal = np.array([d['r'][a['row']] for a, d in zip(wJ_cal[Hw], cal) if a['row'] < len(d['r'])])
    est_cal = np.array([a['est'] for a in wJ_cal[Hw]]); est_cal = est_cal[np.isfinite(est_cal)]
    hw = 3 * cfg['step']
    pm = lambda x: ((r_cal >= x - hw) & (r_cal <= x + hw)).mean() / (2 * hw + 1)        # pmf per cycle
    p_b, p_a = pm(const - cfg['hi']), pm(const - cfg['lo'])
    Dcal = np.abs(est_cal - const)
    pred_bound = (p_a + p_b) * np.mean(np.ceil(Dcal)) if len(Dcal) else np.nan
    for a, d in zip(wJ_te[Hw], tes):
        has = a['row'] < len(d['r']); r = d['r'][a['row']] if has else np.nan
        e_m = a['est'] - r if has else np.nan; e_c = const - r if has else np.nan
        ok_m = bool(has and cfg['lo'] <= e_m <= cfg['hi']); ok_c = bool(has and cfg['lo'] <= e_c <= cfg['hi'])
        D = a['est'] - const if has else np.nan
        # symmetric-difference region of the two windows (half-open intervals; Theorem 1 (i))
        if has:
            lo1, hi1 = sorted((const - cfg['hi'], a['est'] - cfg['hi'])); lo2, hi2 = sorted((const - cfg['lo'], a['est'] - cfg['lo']))
            sd = (lo1 <= r < hi1) or (lo2 < r <= hi2)
        else: sd = False
        win.append(dict(base, unit=d['unit'], group=d['group'], L=d['L'], H=Hw, cap=sw['cap'], span=sw['span'], k=sw['k'], shift=sw['shift'],
                        const=const, alarm=has, r=r, est=a['est'], ok_model=ok_m, ok_const=ok_c, D=D, symdiff=sd,
                        maxdrop=a['maxdrop'], immediate=a['immediate'], p_a=p_a, p_b=p_b, pred_bound=pred_bound,
                        cal_correct=sw['cal_correct'], cal_sd_est=float(np.nanstd(est_cal)), cal_mean_absD=float(np.mean(Dcal)) if len(Dcal) else np.nan))

    # ------------------------------------------------ fixed-setting signal for the conformal policies
    cc, sp, kk = cfg['cna_cap'], cfg['cna_span'], cfg['cna_k']
    S_cal = [C.smooth(p, sp) for p in pred[cc]['cal']]; S_te = [C.smooth(p, sp) for p in pred[cc]['te']]
    # ------------------------------------------------ cost-tuned STW tables (caps x spans x debounce x thresholds)
    keys = [(c, s_, k_) for c in cfg['caps'] for s_ in P.SPANS for k_ in P.KS]
    Jc_cal = np.hstack([P.alarm_rows([C.smooth(p, s_) for p in pred[c]['cal']], k_, cfg['taus']) for c, s_, k_ in keys])
    Jc_te = np.hstack([P.alarm_rows([C.smooth(p, s_) for p in pred[c]['te']], k_, cfg['taus']) for c, s_, k_ in keys])
    Wb = P.boot_weights(len(cal), 2026 + sidx + 100 * rep)
    # ------------------------------------------------ capacity trend (battery): nothing fitted, all training cells used
    if kind == 'battery':
        trn = series(F, tr)
        TQ = {w: [P.trend_rul(F[d['unit']]['Q'][d['mask']], w, C.KAPPA_BAT) for d in trn] for w in TREND_W}
        TQte = {w: [P.trend_rul(F[d['unit']]['Q'][d['mask']], w, C.KAPPA_BAT) for d in tes] for w in TREND_W}
        tsel = {H: P.window_select(trn, {w: TQ[w] for w in TREND_W}, H, cfg['lo'], cfg['hi'], cfg['shifts'], list(TREND_W), -1e9) for H in cfg['Hs']}
        tJ_tr = {H: P.window_apply(TQ[s['cap']], trn, H, s) for H, s in tsel.items()}
        tJ_te = {H: P.window_apply(TQte[s['cap']], tes, H, s) for H, s in tsel.items()}
        tkeys = [(w, s_, k_) for w in TREND_W for s_ in P.SPANS for k_ in P.KS]
        Jt_tr = np.hstack([P.alarm_rows([C.smooth(q, s_) for q in TQ[w]], k_, cfg['taus']) for w, s_, k_ in tkeys])
        Jt_te = np.hstack([P.alarm_rows([C.smooth(q, s_) for q in TQte[w]], k_, cfg['taus']) for w, s_, k_ in tkeys])
        Wt = P.boot_weights(len(trn), 3026 + sidx + 100 * rep)
        ST_tr = [C.smooth(q, sp) for q in TQ[CNA_TREND_W]]; ST_te = [C.smooth(q, sp) for q in TQte[CNA_TREND_W]]
        # window analysis of the trend at Hw (accuracy and notice)
        for a, d in zip(tJ_te[Hw], tes):
            has = a['row'] < len(d['r']); r = d['r'][a['row']] if has else np.nan
            win.append(dict(base, unit=d['unit'], group=d['group'], L=d['L'], H=Hw, cap=np.nan, span=tsel[Hw]['span'], k=tsel[Hw]['k'], shift=tsel[Hw]['shift'],
                            const=np.nan, alarm=has, r=r, est=a['est'], ok_model=bool(has and cfg['lo'] <= a['est'] - r <= cfg['hi']), ok_const=np.nan,
                            method='capacity_trend'))
    Lall = np.array([F[u]['L'] for u in tr])

    for lead in cfg['leads']:
        # conformal notice alarm, fixed alpha (independent of rho)
        V = P.cna_levels(S_cal, cal, kk, lead)
        Hfix = {a: C.conformal_level(V, a) for a in ALPHAS}
        Jfix = {a: P.alarm_rows(S_te, kk, [Hfix[a]])[:, 0] for a in ALPHAS}
        # per-cycle split-conformal lower-bound alarms (interval calibration, same signal and trigger)
        qlb = {(a, near): P.lb_quantile(S_cal, cal, cc, a, near) for a in ALPHAS for near in (False, True)}
        Jlb = {key: P.alarm_rows(S_te, kk, [lead + 1 + q])[:, 0] for key, q in qlb.items()}
        for a in ALPHAS:
            lev.append(dict(base, lead=lead, policy=f'CNA_a{a:.2f}', H=Hfix[a], n_V=len(V), n_inf=int(np.isinf(V).sum())))
            for near in (False, True):
                lev.append(dict(base, lead=lead, policy=f'LB{"near" if near else "all"}_a{a:.2f}', H=lead + 1 + qlb[(a, near)], q=qlb[(a, near)]))
        if kind == 'battery':
            VT = P.cna_levels(ST_tr, trn, kk, lead); HTf = {a: C.conformal_level(VT, a) for a in ALPHAS}
            JTf = {a: P.alarm_rows(ST_te, kk, [HTf[a]])[:, 0] for a in ALPHAS}
            for a in ALPHAS: lev.append(dict(base, lead=lead, policy=f'CNAtrend_a{a:.2f}', H=HTf[a], n_V=len(VT), n_inf=int(np.isinf(VT).sum())))
        # Kamariotis et al. policy 1 with P-res, built on the raw cap-60 (battery: cap-180) estimate
        pr = P.PRes(pred[cc]['cal'], cal, lead, cfg['dt'])
        succ_c, use_c, not_c = P.outcome(Jc_cal, cal, lead); succ_t, use_t, not_t = P.outcome(Jc_te, tes, lead)
        if kind == 'battery':
            ts_c, tu_c, tn_c = P.outcome(Jt_tr, trn, lead)
        for rho in RHOS:
            for a in ALPHAS:
                add(f'CNA_a{a:.2f}', lead, rho, Jfix[a], tes, dict(H=Hfix[a]))
                for near in (False, True):
                    add(f'LB{"near" if near else "all"}_a{a:.2f}', lead, rho, Jlb[(a, near)], tes)
            Hda, kda, bda = P.cna_da(V, S_cal, cal, kk, lead, rho, ALPHA_MAX)
            add('CNA_DA', lead, rho, P.alarm_rows(S_te, kk, [Hda])[:, 0], tes, dict(H=Hda, bound=bda))
            lev.append(dict(base, lead=lead, rho=rho, policy='CNA_DA', H=Hda, k_sel=kda, bound=bda, n_V=len(V)))
            # window-tuned STW: warning level by the calibration cost rate
            crH = {H: P.cost_rate(*P.outcome(np.array([[a_['row']] for a_ in wJ_cal[H]]), cal, lead)[:2], rho)[0] for H in cfg['Hs']}
            Hb = min(cfg['Hs'], key=lambda H: crH[H])
            add('STW_window', lead, rho, [a_['row'] for a_ in wJ_te[Hb]], tes, dict(H=Hb))
            for rule in ('plain', '1se'):
                j = P.select_cost(succ_c, use_c, not_c, rho, rule, Wb)
                add(f'STW_cost_{rule}', lead, rho, Jc_te[:, j], tes, dict(H=float(cfg['taus'][j % len(cfg['taus'])])))
            for nm, thr in (('KAM_P1_heur', 1.0 / rho), ('KAM_P1_opt', P.pres_optimise(pr, pred[cc]['cal'], cal, lead, rho))):
                add(nm, lead, rho, P.pres_rows(pr, pred[cc]['te'], [thr])[:, 0], tes, dict(H=thr))
            T = P.age_T(Lall, rho)
            for d in tes:
                ok = d['L'] > T
                dec.append(dict(base, policy='age', lead=lead, rho=rho, unit=d['unit'], group=d['group'], L=d['L'], notice=np.nan,
                                fail=not ok, use=T if ok else d['L'], cost=1.0 if ok else float(rho), H=T))
                dec.append(dict(base, policy='perfect', lead=lead, rho=rho, unit=d['unit'], group=d['group'], L=d['L'], notice=1.0,
                                fail=False, use=d['L'] - 1, cost=1.0))
            if kind == 'battery':
                for a in ALPHAS: add(f'CNAtrend_a{a:.2f}', lead, rho, JTf[a], tes, dict(H=HTf[a]))
                HdaT, _, bT = P.cna_da(VT, ST_tr, trn, kk, lead, rho, ALPHA_MAX)
                add('CNAtrend_DA', lead, rho, P.alarm_rows(ST_te, kk, [HdaT])[:, 0], tes, dict(H=HdaT, bound=bT))
                crT = {H: P.cost_rate(*P.outcome(np.array([[a_['row']] for a_ in tJ_tr[H]]), trn, lead)[:2], rho)[0] for H in cfg['Hs']}
                HbT = min(cfg['Hs'], key=lambda H: crT[H])
                add('trend_window', lead, rho, [a_['row'] for a_ in tJ_te[HbT]], tes, dict(H=HbT))
                j = P.select_cost(ts_c, tu_c, tn_c, rho, '1se', Wt)
                add('trend_cost_1se', lead, rho, Jt_te[:, j], tes, dict(H=float(cfg['taus'][j % len(cfg['taus'])])))

    # ------------------------------------------------ pilot recalibration on the held-out group (transfer splits only)
    pil = []
    if an == 'group':
        prng = np.random.default_rng(5000 + sidx + 100 * rep)
        ms = [5, 10, 20] if kind == 'battery' else [5]
        sigs = [('CNA', S_te)] + ([('CNAtrend', ST_te)] if kind == 'battery' else [])
        for lead in cfg['leads']:
            for nm, S_ in sigs:
                Vte = P.cna_levels(S_, tes, kk, lead); RM = [C.debounced_running_min(s, kk) for s in S_]
                for m in ms:
                    if m >= len(tes) - 2: continue
                    for draw in range(300):
                        pi = prng.choice(len(tes), m, replace=False); rest = np.setdiff1d(np.arange(len(tes)), pi)
                        for a in (0.10, 0.20):
                            Hp = C.conformal_level(Vte[pi], a)
                            J = np.array([[int(C.first_alarm(RM[i], Hp)[0])] for i in rest])
                            succ, use, notice = P.outcome(J, [tes[i] for i in rest], lead)
                            pil.append(dict(base, signal=nm, lead=lead, m=m, alpha=a, draw=draw, H=Hp, n_test=len(rest),
                                            fail_rate=float(1 - succ.mean()), mean_notice=float(np.nanmean(np.where(succ, notice, np.nan))) if succ.any() else np.nan,
                                            sum_use=float(use.sum()), n_fail=int((~succ).sum())))
    print(f'  {name} rep {rep}: fit {len(fit_u)} cal {len(cal_u)} test {len(te)}; window cap {sw["cap"]} span {sw["span"]} k {sw["k"]} '
          f'shift {sw["shift"]:+.1f}; const {const:.1f} ({time.time() - t0:.0f}s)', flush=True)
    return dec, win, lev, pil


def main(ds, reps):
    df, kind = load(ds); S = splits(df, kind); D, Wn, Lv, Pl = [], [], [], []
    if os.environ.get('SMOKE'): S = [S[0]] + [x for x in S if x[0] == 'group'][:1]; reps = 1
    for rep in range(reps):
        for sidx, (an, name, tr, te) in enumerate(S):
            d, w, l, p = run_split(df, kind, an, name, tr, te, rep, sidx); D += d; Wn += w; Lv += l; Pl += p
    pd.DataFrame(D).assign(dataset=ds).to_csv(f'{OUT}/{ds}_decisions.csv.gz', index=False)
    pd.DataFrame(Wn).assign(dataset=ds).to_csv(f'{OUT}/{ds}_window.csv.gz', index=False)
    pd.DataFrame(Lv).assign(dataset=ds).to_csv(f'{OUT}/{ds}_levels.csv.gz', index=False)
    if Pl: pd.DataFrame(Pl).assign(dataset=ds).to_csv(f'{OUT}/{ds}_pilot.csv.gz', index=False)
    print(ds, 'written', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 3)
