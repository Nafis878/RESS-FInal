"""Summaries and hypothesis decisions (protocol.md, Section 5) from results/raw/*. Writes CSV tables to results/.
Usage: python analyze.py
"""
import os, json
import numpy as np, pandas as pd
import core as C
import stats as S

RAW = os.path.join(C.ROOT, 'results', 'raw'); OUT = os.path.join(C.ROOT, 'results')
DSETS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS', 'BATTERY']
PRIMARY_LEAD = {'BATTERY': 30}
ALPHA_MAX = 0.10
COMPARATORS = ['age', 'STW_window', 'STW_cost_1se', 'STW_cost_plain', 'KAM_P1_heur', 'KAM_P1_opt', 'LBnear_a0.10',
               'trend_window', 'trend_cost_1se', 'CNAtrend_DA']


def load(kind):
    parts = [pd.read_csv(f'{RAW}/{d}_{kind}.csv.gz') for d in DSETS if os.path.exists(f'{RAW}/{d}_{kind}.csv.gz')]
    return pd.concat(parts, ignore_index=True)


def lead_p(ds):
    return PRIMARY_LEAD.get(ds, 10)


def alpha_of(policy):
    return float(policy.split('_a')[1]) if '_a' in policy else (ALPHA_MAX if policy.endswith('_DA') else np.nan)


# ------------------------------------------------------------------------------------------------ failure rates
def validity(D):
    rows = []
    sub = D[~D.policy.isin(['perfect'])]
    for (ds, an, rep, pol, lead), g in sub[sub.rho == 10].groupby(['dataset', 'analysis', 'rep', 'policy', 'lead']):
        n = len(g); k = int(g.fail.sum()); lo, hi = S.clopper_pearson(k, n); a = alpha_of(pol)
        sc = g[~g.fail]
        rows.append(dict(dataset=ds, analysis=an, rep=rep, policy=pol, lead=lead, n=n, failures=k, rate=k / n, ci_lo=lo, ci_hi=hi,
                         alpha=a, p_exceed=S.binom_greater(k, n, a) if np.isfinite(a) else np.nan,
                         notice_median=float(g.notice.median()), notice_p05=float(g.notice.quantile(.05)), notice_p95=float(g.notice.quantile(.95)),
                         wasted_mean=float((sc.notice - lead).mean()) if len(sc) else np.nan))
    V = pd.DataFrame(rows); V.to_csv(f'{OUT}/validity.csv', index=False)
    # repetition-averaged rates (each unit appears once per repetition)
    A = V.groupby(['dataset', 'analysis', 'policy', 'lead']).agg(rate_mean=('rate', 'mean'), rate_min=('rate', 'min'), rate_max=('rate', 'max'),
                                                                   notice_median=('notice_median', 'mean'), wasted_mean=('wasted_mean', 'mean')).reset_index()
    A.to_csv(f'{OUT}/validity_reps.csv', index=False)
    dec = {}
    # H1
    h1 = V[(V.analysis == 'cv') & (V.rep == 0) & (V.policy == 'CNA_a0.10')]
    h1 = h1[[r.lead == lead_p(r.dataset) for r in h1.itertuples()]].copy(); h1['p_holm'] = S.holm(h1.p_exceed.values)
    dec['H1'] = dict(table=h1[['dataset', 'lead', 'n', 'failures', 'rate', 'ci_lo', 'ci_hi', 'p_exceed', 'p_holm']].round(4).to_dict('records'),
                     rejected_on=h1[h1.p_holm < 0.05].dataset.tolist(), supported=bool((h1.p_holm >= 0.05).all()))
    # H2
    h2 = V[(V.analysis == 'cv') & (V.rep == 0) & (V.policy == 'LBall_a0.10')]
    h2 = h2[[r.lead == lead_p(r.dataset) for r in h2.itertuples()]]
    nexc = int((h2.p_exceed < 0.05).sum())
    dec['H2'] = dict(table=h2[['dataset', 'lead', 'n', 'failures', 'rate', 'p_exceed']].round(4).to_dict('records'), n_exceed=nexc, supported=nexc >= 4)
    # H4b
    h4b = V[(V.analysis == 'cv') & (V.rep == 0) & (V.policy == 'CNA_DA') & (V.lead > 0)]
    rows_b = []
    for (ds, lead), g in D[(D.analysis == 'cv') & (D.rep == 0) & (D.policy == 'CNA_DA') & (D.lead > 0)].groupby(['dataset', 'lead']):
        for rho, gg in g.groupby('rho'):
            k, n = int(gg.fail.sum()), len(gg); rows_b.append(dict(dataset=ds, lead=lead, rho=rho, n=n, failures=k, rate=k / n, p=S.binom_greater(k, n, ALPHA_MAX)))
    B = pd.DataFrame(rows_b)
    dec['H4b'] = dict(max_rate=float(B.rate.max()), cells_rejected=int((B.p < 0.05).sum()), n_cells=len(B), supported=bool((B.p >= 0.05).all()))
    B.to_csv(f'{OUT}/cna_da_failures.csv', index=False)
    return V, dec


# ------------------------------------------------------------------------------------------------ equivalence (Theorem 1)
def equivalence(W):
    rows = []
    Wm = W[W.get('method').isna()] if 'method' in W else W
    for (ds, an, rep), g in Wm.groupby(['dataset', 'analysis', 'rep']):
        n = len(g); b = int((g.ok_model & ~g.ok_const.astype(bool)).sum()); c = int((~g.ok_model & g.ok_const.astype(bool)).sum())
        lo, hi = S.tango_ci(b, c, n, 0.90); diff = (b - c) / n
        al = g[g.alarm]
        lemma_lb = al.H - al.k * al.maxdrop.clip(lower=0)
        lemma_ok = ((al.est <= al.H + 1e-9) & ((al.est >= lemma_lb - 1e-9) | al.immediate)).mean()
        pb = g.groupby('split').pred_bound.first().mean()
        rows.append(dict(dataset=ds, analysis=an, rep=rep, n=n, model_pct=100 * g.ok_model.mean(), const_pct=100 * g.ok_const.astype(bool).mean(),
                         only_model=b, only_const=c, diff_pp=100 * diff, ci90_lo_pp=100 * lo, ci90_hi_pp=100 * hi,
                         p_tost=S.tost_paired(b, c, n, 0.05), equivalent_5pp=bool(lo > -0.05 and hi < 0.05),
                         abs_diff_pp=100 * abs(diff), discordant_pp=100 * (b + c) / n, symdiff_pp=100 * g.symdiff.mean(), pred_bound_pp=100 * pb,
                         bound_holds=bool(100 * pb >= 100 * abs(diff)), bound_holds_discordance=bool(pb >= (b + c) / n), sd_est=float(al.est.std()), sd_notice=float(al.r.std()),
                         mean_absD=float(al.D.abs().mean()), lemma_ok=float(lemma_ok), immediate=int(al.immediate.sum()),
                         notice_median=float(al.r.median()), missed=int((~g.alarm).sum())))
    E = pd.DataFrame(rows); E.to_csv(f'{OUT}/equivalence.csv', index=False)
    p = E[(E.analysis == 'cv') & (E.rep == 0)]
    dec = dict(H3=dict(equivalent_on=p[p.equivalent_5pp].dataset.tolist(), not_equivalent_on=p[~p.equivalent_5pp].dataset.tolist(),
                       supported=bool(p.equivalent_5pp.all()), bound_holds_on=int(p.bound_holds.sum()), bound_holds_discordance_on=int(p.bound_holds_discordance.sum()), n=len(p)))
    return E, dec


# ------------------------------------------------------------------------------------------------ cost
def cost(D):
    rows, cmp_rows = [], []
    for (ds, an, rep, lead, rho), g in D.groupby(['dataset', 'analysis', 'rep', 'lead', 'rho']):
        pf = g[g.policy == 'perfect']; rp = pf.cost.sum() / pf.use.sum()
        for pol, gg in g.groupby('policy'):
            rows.append(dict(dataset=ds, analysis=an, rep=rep, lead=lead, rho=rho, policy=pol, rel_cost=(gg.cost.sum() / gg.use.sum()) / rp,
                             failures=int(gg.fail.sum()), n=len(gg)))
        if rep != 0: continue
        base = g[g.policy == 'CNA_DA'].sort_values('unit')
        for pol in COMPARATORS:
            o = g[g.policy == pol].sort_values('unit')
            if not len(o): continue
            assert (o.unit.values == base.unit.values).all()
            r, lo, hi = S.boot_ratio(base.cost.values, base.use.values, o.cost.values, o.use.values)
            cmp_rows.append(dict(dataset=ds, analysis=an, lead=lead, rho=rho, comparator=pol, ratio=r, lo=lo, hi=hi,
                                 cna_better=hi < 1, cna_worse=lo > 1))
    R = pd.DataFrame(rows); R.to_csv(f'{OUT}/cost_cells.csv', index=False)
    Q = pd.DataFrame(cmp_rows); Q.to_csv(f'{OUT}/cost_vs_cna_da.csv', index=False)
    summ = []
    for (ds, an, pol), g in Q.groupby(['dataset', 'analysis', 'comparator']):
        nb, nw = int(g.cna_better.sum()), int(g.cna_worse.sum())
        verdict = 'CNA_DA cheaper' if nb >= 8 and nw == 0 else ('CNA_DA dearer' if nw >= 8 and nb == 0 else 'mixed')
        summ.append(dict(dataset=ds, analysis=an, comparator=pol, cells=len(g), cna_better=nb, cna_worse=nw,
                         ratio_median=float(g.ratio.median()), ratio_min=float(g.ratio.min()), ratio_max=float(g.ratio.max()), verdict=verdict))
    Sm = pd.DataFrame(summ); Sm.to_csv(f'{OUT}/cost_summary.csv', index=False)
    # excess over the cheapest policy in each cell (descriptive; rep 0, CV)
    pols = ['CNA_DA', 'STW_window', 'STW_cost_1se', 'STW_cost_plain', 'KAM_P1_heur', 'KAM_P1_opt', 'age', 'trend_window', 'trend_cost_1se', 'CNAtrend_DA']
    X = R[(R.rep == 0) & R.policy.isin(pols)].copy()
    X['excess'] = X.rel_cost / X.groupby(['dataset', 'analysis', 'lead', 'rho']).rel_cost.transform('min') - 1
    E = X.groupby(['dataset', 'analysis', 'policy']).agg(mean_excess=('excess', 'mean'), max_excess=('excess', 'max'),
                                                         failures=('failures', 'sum'), cells=('excess', 'size')).reset_index()
    E.to_csv(f'{OUT}/cost_excess.csv', index=False)
    return R, Q, Sm, E


# ------------------------------------------------------------------------------------------------ transfer
def transfer(D, Pl):
    rows = []
    G = D[(D.analysis == 'group') & (D.rho == 10)]
    for (ds, rep, split, pol, lead), g in G.groupby(['dataset', 'rep', 'split', 'policy', 'lead']):
        n, k = len(g), int(g.fail.sum()); a = alpha_of(pol)
        rows.append(dict(dataset=ds, rep=rep, split=split, policy=pol, lead=lead, n=n, failures=k, rate=k / n,
                         p_exceed=S.binom_greater(k, n, a) if np.isfinite(a) else np.nan, notice_median=float(g.notice.median())))
    T = pd.DataFrame(rows); T.to_csv(f'{OUT}/transfer_groups.csv', index=False)
    if not len(T): return T, {}
    bt = T[(T.dataset == 'BATTERY') & (T.rep == 0) & (T.policy == 'CNA_a0.10') & (T.lead == 30)]
    dec = dict(H5a=dict(table=bt[['split', 'n', 'failures', 'rate', 'p_exceed']].round(4).to_dict('records'), supported=bool((bt.p_exceed < 0.05).any())))
    if Pl is not None and len(Pl):
        P_ = Pl.groupby(['dataset', 'rep', 'split', 'signal', 'lead', 'm', 'alpha']).agg(
            draws=('fail_rate', 'size'), fail_mean=('fail_rate', 'mean'), fail_sd=('fail_rate', 'std'), infeasible=('H', lambda h: float(np.isinf(h).mean())),
            notice_mean=('mean_notice', 'mean')).reset_index()
        P_['mc_se'] = P_.fail_sd / np.sqrt(P_.draws)
        P_.to_csv(f'{OUT}/pilot_summary.csv', index=False)
        q = P_[(P_.dataset == 'BATTERY') & (P_.rep == 0) & (P_.m == 10) & (P_.alpha == 0.10) & (P_.signal == 'CNA')]
        ok = q.fail_mean <= 0.10 + 2 * q.mc_se
        dec['H5b'] = dict(cells=len(q), ok=int(ok.sum()), worst=float(q.fail_mean.max()), supported=bool(ok.all()))
    return T, dec


def main():
    D, W, Pl = load('decisions'), load('window'), (load('pilot') if any(os.path.exists(f'{RAW}/{d}_pilot.csv.gz') for d in DSETS) else None)
    V, d1 = validity(D); E, d3 = equivalence(W); R, Q, Sm, Ex = cost(D); T, d5 = transfer(D, Pl)
    dec = {**d1, **d3, **d5}
    json.dump(dec, open(f'{OUT}/hypotheses.json', 'w'), indent=1, default=float)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'table'} for k, v in dec.items()}, indent=1, default=float))


if __name__ == '__main__':
    main()
