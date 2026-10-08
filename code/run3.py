"""Amendment v3 (protocol_v3.md): equal-information comparison for the cross-fitted conformal alarm.
The comparators are rebuilt on exactly the same five fold models as CNA+ (run2.py, same seeds): selection on the
out-of-fold predictions of all training units, application to a test unit through the average of the five fold-model
predictions (cross-fitted ensemble). CNA+ is recomputed and checked against run2.py.

Usage: python run3.py DATASET [REPS, e.g. 0 or 0,1,2]
Output: results/raw3/<DS>_rep<r>_cf.csv.gz (one row per policy, lead, cost ratio and test unit)
"""
import os, sys, time
import numpy as np, pandas as pd
import core as C
import policies as P
from run import CFG, RHOS, load, splits, series
from run2 import cvplus_counts, ALPHA_MAX

OUT = os.path.join(C.ROOT, 'results', 'raw3'); os.makedirs(OUT, exist_ok=True)


def run_split(df, kind, an, name, tr, te, rep, sidx):
    cfg = CFG[kind]; t0 = time.time(); cc, sp, kk = cfg['cna_cap'], cfg['cna_span'], cfg['cna_k']
    base = dict(analysis=an, split=name, rep=rep, n_train=len(tr))
    folds = np.array_split(np.random.default_rng(3000 + 100 * rep + sidx).permutation(tr), 5)      # as run2.py
    trs = series(C.build_features(df, kind, tr), tr); trs_by_u = {d['unit']: d for d in trs}
    tes = None; oof = {}; te_pred = []
    for k, fu in enumerate(folds):
        fit_k = np.setdiff1d(tr, fu); Fk = C.build_features(df, kind, fit_k)
        mk = C.gbm(np.vstack([Fk[u]['X'] for u in fit_k]), np.concatenate([Fk[u]['r'] for u in fit_k]), cc)
        if tes is None: tes = series(Fk, te)
        for u in fu: oof[u] = mk.predict(Fk[u]['X'])[trs_by_u[u]['mask']]
        te_pred.append([mk.predict(Fk[d['unit']]['X'])[d['mask']] for d in tes])
    order = [d['unit'] for d in trs]; Ptr = [oof[u] for u in order]
    Pte_avg = [np.mean([te_pred[k][i] for k in range(5)], axis=0) for i in range(len(tes))]
    rows = []

    def add(policy, lead, rho, J, extra=None):
        succ, use, notice = P.outcome(np.asarray(J).reshape(-1, 1), tes, lead)
        for i, d in enumerate(tes):
            r = dict(base, policy=policy, lead=lead, rho=rho, unit=d['unit'], group=d['group'], L=d['L'], notice=notice[i, 0],
                     fail=not succ[i, 0], use=use[i, 0], cost=1.0 if succ[i, 0] else float(rho))
            if extra: r.update(extra)
            rows.append(r)

    # cost-tuned STW on the cross-fitted ensemble: spans x debounce x thresholds, selected on out-of-fold predictions
    keys = [(s_, k_) for s_ in P.SPANS for k_ in P.KS]
    J_tr = np.hstack([P.alarm_rows([C.smooth(p, s_) for p in Ptr], k_, cfg['taus']) for s_, k_ in keys])
    J_te = np.hstack([P.alarm_rows([C.smooth(p, s_) for p in Pte_avg], k_, cfg['taus']) for s_, k_ in keys])
    Wb = P.boot_weights(len(trs), 6026 + sidx + 100 * rep)
    # CNA+ (check against run2.py) from the per-fold running minima
    RMtr = {u: C.debounced_running_min(C.smooth(oof[u], sp), kk) for u in tr}
    RHO_te = [[C.debounced_running_min(C.smooth(te_pred[k][i], sp), kk) for i in range(len(tes))] for k in range(5)]
    n = len(tr)
    for lead in cfg['leads']:
        succ_c, use_c, not_c = P.outcome(J_tr, trs, lead)
        pr = P.PRes(Ptr, trs, lead, cfg['dt'])
        Vtr = {u: C.critical_level(RMtr[u], trs_by_u[u]['r'], lead) for u in tr}
        Vs_k = [np.sort([Vtr[u] for u in fu]) for fu in folds]
        CT = [cvplus_counts([RHO_te[k][i] for k in range(5)], Vs_k) for i in range(len(tes))]
        q = int(np.ceil(0.9 * (n + 1))); cthr = n + 1 - q
        Jcna = [int(np.searchsorted(c_, cthr, side='left')) for c_ in CT]
        for rho in RHOS:
            for rule in ('plain', '1se'):
                j = P.select_cost(succ_c, use_c, not_c, rho, rule, Wb)
                add(f'STW_cost_{rule}_CF', lead, rho, J_te[:, j], dict(setting=j))
            add('KAM_P1_opt_CF', lead, rho, P.pres_rows(pr, Pte_avg, [P.pres_optimise(pr, Ptr, trs, lead, rho)])[:, 0])
            add('KAM_P1_heur_CF', lead, rho, P.pres_rows(pr, Pte_avg, [1.0 / rho])[:, 0])
            add('CNAplus_a0.10_check', lead, rho, Jcna)
    print(f'  {name} rep {rep} ({time.time() - t0:.0f}s)', flush=True)
    return rows


def main(ds, reps):
    df, kind = load(ds); S = splits(df, kind)
    for rep in reps:
        f = f'{OUT}/{ds}_rep{rep}_cf.csv.gz'
        if os.path.exists(f): print(ds, 'rep', rep, 'exists'); continue
        R = []
        for sidx, (an, name, tr, te) in enumerate(S): R += run_split(df, kind, an, name, tr, te, rep, sidx)
        pd.DataFrame(R).assign(dataset=ds).to_csv(f, index=False); print(ds, 'rep', rep, 'written', flush=True)


if __name__ == '__main__':
    if os.environ.get('SMOKE'):
        df, kind = load(sys.argv[1]); S = splits(df, kind); R = pd.DataFrame(run_split(df, kind, *S[0], 0, 0))
        print(R[R.rho == 10].groupby(['policy', 'lead']).fail.mean().unstack().round(3))
    else:
        main(sys.argv[1], [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2])
