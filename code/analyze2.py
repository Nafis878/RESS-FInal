"""Analysis of the amendment experiments (protocol_v2.md): censoring (H6), cross-fitting (H7), metric M (H8).
Usage: python analyze2.py   (after run.py and run2.py)"""
import os, json
import numpy as np, pandas as pd
import core as C
import stats as S

RAW, RAW2, OUT = (os.path.join(C.ROOT, 'results', x) for x in ('raw', 'raw2', ''))
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS', 'BATTERY']
lp = lambda d: 30 if d == 'BATTERY' else 10


def load(kind, folder=RAW2):
    import glob
    if folder == RAW2:
        fs = sorted(f for d in DS for f in glob.glob(f'{folder}/{d}_rep*_{kind}.csv.gz'))
    else:
        fs = [f'{folder}/{d}_{kind}.csv.gz' for d in DS if os.path.exists(f'{folder}/{d}_{kind}.csv.gz')]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def censoring():
    A = load('censor'); A = A[A.analysis == 'cv']
    A['nfail'] = A.fail * A.n_test
    g = A.groupby(['dataset', 'rep', 'lead', 'alpha', 'mechanism', 'pi', 'method'], dropna=False).agg(
        nfail=('nfail', 'sum'), ntest=('n_test', 'sum'), notice=('notice', 'mean'), frac_cens=('frac_cens', 'mean'),
        infeasible=('H', lambda h: float(np.isinf(h).mean()))).reset_index()
    g['fail'] = g.nfail / g.ntest
    g.to_csv(f'{OUT}/censoring.csv', index=False)
    n_units = A[(A.method == 'oracle')].groupby(['dataset', 'rep', 'lead', 'alpha']).n_test.sum().groupby('dataset').max()
    h = g[(g.rep == 0) & (g.alpha == 0.10) & (g.method == 'censored_as_failure')]
    h = h[[r.lead == lp(r.dataset) for r in h.itertuples()]].copy()
    h['tol'] = [0.10 + 2 * np.sqrt(0.09 / n_units[d]) for d in h.dataset]; h['ok'] = h.fail <= h.tol
    dec = dict(H6=dict(cells=len(h), ok=int(h.ok.sum()), worst=float(h.fail.max()), supported=bool(h.ok.all()),
                       failing=h[~h.ok][['dataset', 'mechanism', 'pi', 'fail', 'tol']].round(4).to_dict('records')))
    return g, dec


def cvplus():
    B = load('cvplus'); D = load('decisions', RAW)
    rows = []
    for (ds, an, rep, pol, lead), x in B[B.rho == 10].groupby(['dataset', 'analysis', 'rep', 'policy', 'lead']):
        k, n = int(x.fail.sum()), len(x)
        a = float(pol.split('_a')[1]) if '_a' in pol else 0.10
        rows.append(dict(dataset=ds, analysis=an, rep=rep, policy=pol, lead=lead, n=n, failures=k, rate=k / n, p_exceed=S.binom_greater(k, n, a),
                         notice_median=float(x.notice.median()), wasted_mean=float((x[~x.fail].notice - lead).mean())))
    V = pd.DataFrame(rows); V.to_csv(f'{OUT}/cvplus_validity.csv', index=False)
    h = V[(V.analysis == 'cv') & (V.rep == 0) & (V.policy == 'CNAplus_a0.10')]
    h = h[[r.lead == lp(r.dataset) for r in h.itertuples()]].copy()
    h['tol'] = 0.10 + 2 * np.sqrt(0.09 / h.n); h['ok'] = h.rate <= h.tol
    # cost: CNA+-DA against the policies of run.py (same units, repetition 0)
    cmp_rows, cells = [], []
    comps = ['CNA_DA', 'STW_cost_1se', 'STW_window', 'KAM_P1_opt', 'age', 'trend_cost_1se', 'trend_window']
    for (ds, an, lead, rho), x in B[(B.rep == 0) & (B.policy == 'CNAplus_DA')].groupby(['dataset', 'analysis', 'lead', 'rho']):
        base = x.sort_values('unit'); d = D[(D.dataset == ds) & (D.analysis == an) & (D.rep == 0) & (D.lead == lead) & (D.rho == rho)]
        pf = d[d.policy == 'perfect']; rp = pf.cost.sum() / pf.use.sum()
        cells.append(dict(dataset=ds, analysis=an, lead=lead, rho=rho, policy='CNAplus_DA', rel_cost=(base.cost.sum() / base.use.sum()) / rp, failures=int(base.fail.sum())))
        for pol in comps:
            o = d[d.policy == pol].sort_values('unit')
            if not len(o): continue
            assert (o.unit.values == base.unit.values).all()
            r, lo, hi = S.boot_ratio(base.cost.values, base.use.values, o.cost.values, o.use.values)
            cmp_rows.append(dict(dataset=ds, analysis=an, lead=lead, rho=rho, comparator=pol, ratio=r, lo=lo, hi=hi, better=hi < 1, worse=lo > 1))
    Q = pd.DataFrame(cmp_rows); Q.to_csv(f'{OUT}/cvplus_cost_vs.csv', index=False); pd.DataFrame(cells).to_csv(f'{OUT}/cvplus_cost_cells.csv', index=False)
    summ = []
    for (ds, an, pol), x in Q.groupby(['dataset', 'analysis', 'comparator']):
        nb, nw = int(x.better.sum()), int(x.worse.sum())
        summ.append(dict(dataset=ds, analysis=an, comparator=pol, better=nb, worse=nw, ratio_median=float(x.ratio.median()),
                         verdict='CNA+-DA cheaper' if nb >= 8 and nw == 0 else ('CNA+-DA dearer' if nw >= 8 and nb == 0 else 'mixed')))
    Sm = pd.DataFrame(summ); Sm.to_csv(f'{OUT}/cvplus_cost_summary.csv', index=False)
    vs_split = Sm[(Sm.analysis == 'cv') & (Sm.comparator == 'CNA_DA')]
    # mean excess over the cheapest policy, including CNA+-DA
    R = pd.read_csv(f'{OUT}/cost_cells.csv'); R = R[(R.rep == 0)]
    pols = ['CNA_DA', 'STW_window', 'STW_cost_1se', 'STW_cost_plain', 'KAM_P1_heur', 'KAM_P1_opt', 'age', 'trend_window', 'trend_cost_1se', 'CNAtrend_DA']
    X = pd.concat([R[R.policy.isin(pols)][['dataset', 'analysis', 'lead', 'rho', 'policy', 'rel_cost', 'failures']], pd.DataFrame(cells)])
    X['excess'] = X.rel_cost / X.groupby(['dataset', 'analysis', 'lead', 'rho']).rel_cost.transform('min') - 1
    E = X.groupby(['dataset', 'analysis', 'policy']).agg(mean_excess=('excess', 'mean'), max_excess=('excess', 'max')).reset_index()
    E.to_csv(f'{OUT}/cost_excess_with_cvplus.csv', index=False)
    dec = dict(H7a=dict(table=h[['dataset', 'lead', 'n', 'failures', 'rate', 'tol']].round(4).to_dict('records'), supported=bool(h.ok.all())),
               H7b=dict(cheaper_than_split_on=vs_split[vs_split.verdict == 'CNA+-DA cheaper'].dataset.tolist(),
                        dearer_than_split_on=vs_split[vs_split.verdict == 'CNA+-DA dearer'].dataset.tolist(),
                        supported=int((vs_split.verdict == 'CNA+-DA cheaper').sum()) >= 4))
    return V, Sm, E, dec


def metric_m():
    G = load('grid'); G = G[(G.analysis == 'cv') & (G.rep == 0)]
    rows, diff = [], []
    rng = np.random.default_rng(2027)
    for (ds, ratio), g in G.groupby(['dataset', 'ratio']):
        pf = g[g.policy == 'perfect_grid'].sort_values('unit'); units = pf.unit.values; n = len(units)
        W = np.vstack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(2000)]).astype(float)
        Rp = pf.cost.values.sum() / pf.use.values.sum(); Rpb = (W @ pf.cost.values) / (W @ pf.use.values)
        Mb = {}
        for pol, x in g.groupby('policy'):
            if pol == 'perfect_grid': continue
            x = x.sort_values('unit'); assert (x.unit.values == units).all()
            M = (x.cost.sum() / x.use.sum()) / Rp - 1; Mb[pol] = ((W @ x.cost.values) / (W @ x.use.values)) / Rpb - 1
            rows.append(dict(dataset=ds, ratio=ratio, policy=pol, M_pct=100 * M, lo=100 * np.quantile(Mb[pol], .025), hi=100 * np.quantile(Mb[pol], .975),
                             failures=int(x.fail.sum()), n=n))
        for pol in Mb:
            if pol == 'CNA_DA': continue
            dlt = 100 * (Mb['CNA_DA'] - Mb[pol]); point = [r for r in rows if r['dataset'] == ds and r['ratio'] == ratio]
            pm = {r['policy']: r['M_pct'] for r in point}
            lo, hi = np.quantile(dlt, [.025, .975])
            diff.append(dict(dataset=ds, ratio=ratio, comparator=pol, diff_pp=pm['CNA_DA'] - pm[pol], lo=lo, hi=hi,
                             verdict='CNA-DA better' if hi < 0 else ('CNA-DA worse' if lo > 0 else 'unclear')))
    M = pd.DataFrame(rows); Dd = pd.DataFrame(diff)
    M.to_csv(f'{OUT}/metric_M.csv', index=False); Dd.to_csv(f'{OUT}/metric_M_diff.csv', index=False)
    return M, Dd


if __name__ == '__main__':
    g, d6 = censoring(); V, Sm, E, d7 = cvplus(); M, Dd = metric_m()
    dec = {**d6, **d7}; json.dump(dec, open(f'{OUT}/hypotheses_v2.json', 'w'), indent=1, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'table'} for k, v in dec.items()}, indent=1, default=float))
