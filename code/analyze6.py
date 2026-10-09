"""Analysis of amendment v6 (protocol_v6.md), Parts 1-4. Reads results/raw6/*, results/raw/*, results/sim6/*.
Writes results/lb_variants.csv, lb_tests.csv, lb_learners.csv, b1_data.csv, cens6.csv, shift6.csv, hypotheses_v6.json.
Usage: python analyze6.py
"""
import glob, json, os
import numpy as np, pandas as pd
import core as C
from stats import binom_greater, holm, clopper_pearson

RES = os.path.join(C.ROOT, 'results'); RAW6 = os.path.join(RES, 'raw6')
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS', 'BATTERY']
lp = lambda ds: 30 if ds == 'BATTERY' else 10
tol = lambda a, n: a + 2 * np.sqrt(a * (1 - a) / n)


def cat(pattern):
    fs = sorted(glob.glob(os.path.join(RAW6, pattern)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


# ------------------------------------------------------------------------------------------------ Part 1: H14
def part1():
    L = cat('*_rep*_lb.csv.gz')
    # H14a: ORIG reproduces run.py
    chk = {}
    for ds in DS:
        M = pd.read_csv(f'{RES}/raw/{ds}_decisions.csv.gz')
        M = M[(M.analysis == 'cv') & M.policy.str.match(r'^LB(all|near)_a') & (M.rho == M.rho.min())].copy()
        M['pol'] = M.policy.str.split('_').str[0]; M['alpha'] = M.policy.str.split('_a').str[1].astype(float)
        O = L[(L.dataset == ds) & (L.variant == 'ORIG')]
        J = O.merge(M[['rep', 'split', 'pol', 'alpha', 'lead', 'unit', 'fail', 'notice']], left_on=['rep', 'split', 'policy', 'alpha', 'lead', 'unit'],
                    right_on=['rep', 'split', 'pol', 'alpha', 'lead', 'unit'], suffixes=('', '_m'), how='outer', indicator=True)
        chk[ds] = bool((J._merge == 'both').all() and (J.fail.astype(bool) == J.fail_m.astype(bool)).all()
                       and np.allclose(J.notice.fillna(-1), J.notice_m.fillna(-1)))
    g = L.groupby(['dataset', 'variant', 'policy', 'alpha', 'lead', 'rep']).agg(n=('fail', 'size'), failures=('fail', 'sum'),
                                                                               notice_median=('notice', 'median')).reset_index()
    g['rate'] = g.failures / g.n; g.to_csv(f'{RES}/lb_variants.csv', index=False)
    P0 = g[(g.rep == 0) & (g.alpha == 0.10) & np.array([l == lp(d) for l, d in zip(g.lead, g.dataset)])].copy()
    P0['p'] = [binom_greater(k, n, 0.10) for k, n in zip(P0.failures, P0.n)]
    P0['p_holm'] = np.nan
    for (v, pol), idx in P0.groupby(['variant', 'policy']).groups.items():
        P0.loc[idx, 'p_holm'] = holm(P0.loc[idx, 'p'].values)
    P0['sig'] = P0.p_holm < 0.05
    P0.to_csv(f'{RES}/lb_tests.csv', index=False)
    k1all = P0[(P0.variant == 'K1') & (P0.policy == 'LBall')]; k1near = P0[(P0.variant == 'K1') & (P0.policy == 'LBnear')]
    # learners, repetition 0, K1
    Lr = cat('*_lb_learners.csv.gz')
    Lr = Lr[(Lr.alpha == 0.10) & np.array([l == lp(d) for l, d in zip(Lr.lead, Lr.dataset)])]
    gl = Lr.groupby(['learner', 'dataset', 'policy']).agg(n=('fail', 'size'), failures=('fail', 'sum'), notice_median=('notice', 'median')).reset_index()
    gl['rate'] = gl.failures / gl.n; gl['p'] = [binom_greater(k, n, 0.10) for k, n in zip(gl.failures, gl.n)]
    gl['p_holm'] = np.nan
    for (ln, pol), idx in gl.groupby(['learner', 'policy']).groups.items():
        gl.loc[idx, 'p_holm'] = holm(gl.loc[idx, 'p'].values)
    gl['sig'] = gl.p_holm < 0.05; gl.to_csv(f'{RES}/lb_learners.csv', index=False)
    sig_l = gl[gl.policy == 'LBall'].groupby('learner').sig.sum()
    return dict(H14a=dict(reproduces_run_py=chk, supported=all(chk.values())),
                H14b=dict(significant=int(k1all.sig.sum()), rates={r.dataset: round(r.rate, 4) for r in k1all.itertuples()},
                          supported=bool(k1all.sig.sum() >= 4)),
                H14c=dict(significant=int(k1near.sig.sum()), rates={r.dataset: round(r.rate, 4) for r in k1near.itertuples()},
                          no_dataset_significant=bool(k1near.sig.sum() == 0)),
                H14d=dict(significant_per_learner={k: int(v) for k, v in sig_l.items()}, supported=bool((sig_l >= 4).sum() >= 2)))


# ------------------------------------------------------------------------------------------------ Part 1: H15
def part1_b1():
    U = cat('*_b1_units.csv.gz'); rows = []
    for (ds, pol), g in U.groupby(['dataset', 'policy']):
        f, ce, cm = g.fail.mean(), g.crit_exceed.mean(), g.crit_miss.mean()
        rows.append(dict(dataset=ds, policy=pol, n=len(g), fail=f, crit_exceed=ce, crit_miss=cm,
                         p_fail_given_exceed=(g.fail & g.crit_exceed).sum() / max(g.crit_exceed.sum(), 1),
                         prod_indep=g.prod_indep.mean(), prod_mdep=g.prod_mdep.mean(), m_lag=float(g.m_lag.median()),
                         pooled_miss_target=g.pooled_miss_target.mean(), pathwise=bool((~g.fail.astype(bool) | g.crit_miss.astype(bool)).all())))
    B = pd.DataFrame(rows); B.to_csv(f'{RES}/b1_data.csv', index=False)
    A = B[B.policy == 'LBall']
    ratio_ok = A.crit_miss >= 2 * A.pooled_miss_target
    between = (B.fail >= B.prod_indep - 1e-12) & (B.fail <= B.crit_miss + 1e-12)
    return dict(H15a=dict(supported=bool(B.pathwise.all())),
                H15b=dict(count=int(ratio_ok.sum()), supported=bool(ratio_ok.sum() >= 5),
                          ratio={r.dataset: round(r.crit_miss / max(r.pooled_miss_target, 1e-9), 2) for r in A.itertuples()}),
                H15c=dict(count=int(between.sum()), of=len(B), all_LBall=bool(between[B.policy == 'LBall'].all())))


# ------------------------------------------------------------------------------------------------ Part 3: H17
def part3():
    D = cat('*_cens.csv.gz')
    D['nf'] = D.fail * D.n_test
    g = D.groupby(['dataset', 'mechanism', 'method', 'alpha']).agg(nf=('nf', 'sum'), nt=('n_test', 'sum'), notice=('notice', 'mean'),
                                                                 first_row=('first_row', 'mean'), false_alarm=('false_alarm', 'mean'),
                                                                 frac_cens=('frac_cens', 'mean'), inf_level=('H', lambda h: float(np.isinf(h).mean())),
                                                                 c0=('c0', 'mean')).reset_index()
    g['fail'] = g.nf / g.nt
    n_units = D[(D.mechanism == 'M1') & (D.draw == 0) & (D.method == 'ORACLE') & (D.alpha == 0.10)].groupby('dataset').n_test.sum()
    g['tol'] = [tol(a, n_units[d]) for a, d in zip(g.alpha, g.dataset)]; g['ok'] = g.fail <= g.tol
    g.to_csv(f'{RES}/cens6.csv', index=False)
    h = g[g.alpha == 0.10]
    ok = lambda mech, meth: h[(h.mechanism == mech) & (h.method == meth)].set_index('dataset').ok.reindex(DS)
    val = lambda mech, meth, col: h[(h.mechanism == mech) & (h.method == meth)].set_index('dataset')[col].reindex(DS)
    H17a = all(ok(m, e).all() for m in ('M1', 'M2') for e in ('E2HIST_G1', 'E2BASE_G1'))
    lower = {m: int((val(m, 'E2HIST_G1', 'notice') < val(m, 'T2', 'notice')).sum()) for m in ('M1', 'M2')}
    H17b = all(v >= 5 for v in lower.values())
    H17c = bool(ok('M3', 'E2HIST_G1').all())
    viol = {e: int((~ok('M4', e)).sum()) for e in ('E2HIST_G1', 'E2BASE_G1')}
    gam = {}
    for e in ('E2HIST', 'E2BASE'):
        for G_ in ('1', '2', '4', 'inf'):
            if ok('M4', f'{e}_G{G_}').all(): gam[e] = G_; break
    return dict(H17a=dict(supported=bool(H17a), cells_ok={f'{m}/{e}': int(ok(m, e).sum()) for m in ('M1', 'M2') for e in ('E2HIST_G1', 'E2BASE_G1')}),
                H17b=dict(supported=bool(H17b), datasets_lower_notice=lower),
                H17c=dict(supported=H17c, cells_ok=int(ok('M3', 'E2HIST_G1').sum())),
                H17d=dict(T2_ok_all=bool(ok('M4', 'T2').all()), violations=viol,
                          supported=bool(ok('M4', 'T2').all() and all(v >= 1 for v in viol.values())), smallest_gamma_valid=gam))


# ------------------------------------------------------------------------------------------------ Part 4: H18
def part4():
    S = cat('*_shift.csv.gz')
    g = S.groupby(['dataset', 'split', 'rep', 'alpha', 'method']).agg(n=('fail', 'size'), failures=('fail', 'sum'), infinite=('infinite', 'mean'),
                                                                   notice_median=('notice', 'median'), ess=('ess', 'mean')).reset_index()
    g['rate'] = g.failures / g.n; g['tol'] = [tol(a, n) for a, n in zip(g.alpha, g.n)]
    g.to_csv(f'{RES}/shift6.csv', index=False)
    h = g[(g.rep == 0) & (g.alpha == 0.10)]
    st = h[h.method == 'static'].set_index(['dataset', 'split']); wt = h[h.method == 'weighted'].set_index(['dataset', 'split'])
    bad = st[st.rate > st.tol].index
    res = {f'{d}/{s}': dict(static=round(st.loc[(d, s)].rate, 3), weighted=round(wt.loc[(d, s)].rate, 3), tol=round(st.loc[(d, s)].tol, 3),
                            weighted_ok=bool(wt.loc[(d, s)].rate <= wt.loc[(d, s)].tol), infinite=round(wt.loc[(d, s)].infinite, 3)) for d, s in bad}
    return dict(H18=dict(groups=res, supported=bool(len(res) and all(v['weighted_ok'] for v in res.values()))))


if __name__ == '__main__':
    out = {}
    for f in (part1, part1_b1, part3, part4):
        try:
            out.update(f())
        except Exception as e:                                   # report missing parts without failing the others
            out[f.__name__] = dict(error=repr(e))
    sims = json.load(open(f'{RES}/sim6/summary.json'))
    out['simulations'] = {k: bool(v.get('passed')) for k, v in sims.items()}
    json.dump(out, open(f'{RES}/hypotheses_v6.json', 'w'), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
