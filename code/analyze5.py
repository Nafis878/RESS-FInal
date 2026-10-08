"""Analysis of amendment v5 (protocol_v5.md): robustness to the learner and to the alarm settings.
Reads results/raw5/<DS>_robust.csv.gz and results/raw/<DS>_decisions.csv.gz; writes
    results/robust_learners.csv   rate, interval, p-values and notice per learner, dataset and policy (alpha 0.10, primary lead)
    results/robust_learners_all.csv  all alphas and lead times
    results/robust_settings.csv   pooled and per-dataset rates and notice per alarm setting (alpha 0.10, primary lead)
    results/hypotheses_v5.json    implementation check, H11-H13
    figures/Fig8_robustness.{pdf,png}
Usage: python analyze5.py
"""
import glob, json, os
import numpy as np, pandas as pd
import core as C
from stats import binom_greater, clopper_pearson, holm

RES = os.path.join(C.ROOT, 'results')
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS', 'BATTERY']
ALT = ['RIDGE', 'RF', 'WMLP']
lp = lambda ds: 30 if ds == 'BATTERY' else 10


def load():
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{RES}/raw5/*_robust.csv.gz'))], ignore_index=True)
    R['primary'] = [l == lp(d) for l, d in zip(R.lead, R.dataset)]
    return R


def implementation_check(R):
    out = {}
    for ds in DS:
        M = pd.read_csv(f'{RES}/raw/{ds}_decisions.csv.gz')
        M = M[(M.analysis == 'cv') & (M.rep == 0) & M.policy.str.match(r'^(CNA|LBall|LBnear)_a') & (M.rho == M.rho.min())]
        G = R[(R.dataset == ds) & (R.study == 'learner') & (R.learner == 'GBM')]
        key = ['split', 'policy', 'lead', 'unit']
        J = G.merge(M[key + ['fail', 'notice']], on=key, suffixes=('', '_main'), how='outer', indicator=True)
        same = (J._merge == 'both').all() and (J.fail.astype(bool) == J.fail_main.astype(bool)).all() and \
            np.allclose(J.notice.fillna(-1).values, J.notice_main.fillna(-1).values)
        out[ds] = bool(same)
    return out


def learners(R):
    L = R[R.study == 'learner']
    g = L.groupby(['learner', 'dataset', 'policy', 'alpha', 'lead']).agg(n=('fail', 'size'), failures=('fail', 'sum'),
                                                                        notice_median=('notice', 'median')).reset_index()
    g['rate'] = g.failures / g.n
    g.to_csv(f'{RES}/robust_learners_all.csv', index=False)
    P = g[np.array([l == lp(d) for l, d in zip(g.lead, g.dataset)]) & (g.alpha == 0.10).values].copy()
    P['ci_lo'], P['ci_hi'] = zip(*[clopper_pearson(k, n) for k, n in zip(P.failures, P.n)])
    P['p_greater'] = [binom_greater(k, n, 0.10) for k, n in zip(P.failures, P.n)]
    P['p_holm'] = np.nan
    for fam in ('CNA', 'LBall'):
        m = P.learner.isin(ALT) & (P.policy == f'{fam}_a0.10')
        P.loc[m, 'p_holm'] = holm(P.loc[m, 'p_greater'].values)
    P.to_csv(f'{RES}/robust_learners.csv', index=False)
    h11 = P[P.learner.isin(ALT) & (P.policy == 'CNA_a0.10')]
    h12 = P[P.learner.isin(ALT) & (P.policy == 'LBall_a0.10')]
    sig12 = h12.assign(sig=h12.p_holm < 0.05).groupby('learner').sig.sum()
    H11 = dict(supported=bool((h11.p_holm >= 0.05).all()), smallest_adjusted_p=float(h11.p_holm.min()),
               rates={f'{r.learner}/{r.dataset}': round(r.rate, 4) for r in h11.itertuples()})
    H12 = dict(significant_datasets_per_learner={k: int(v) for k, v in sig12.items()}, supported=bool((sig12 >= 4).sum() >= 2),
               rates={f'{r.learner}/{r.dataset}': round(r.rate, 4) for r in h12.itertuples()})
    return P, H11, H12


def settings(R):
    S = R[(R.study == 'setting') & (R.policy == 'CNA_a0.10')]
    S = S[np.array([l == lp(d) for l, d in zip(S.lead, S.dataset)])]
    rows = []
    for (cap_rank, span, k), g in S.assign(cap_rank=S.groupby('dataset').cap.rank(method='dense').astype(int)).groupby(['cap_rank', 'span', 'k']):
        n, f = len(g), int(g.fail.sum())
        row = dict(cap=('low', 'mid', 'high')[cap_rank - 1], span=span, k=k, n=n, failures=f, rate=f / n, p_greater=binom_greater(f, n, 0.10),
                   notice_mean_engines=g[g.dataset != 'BATTERY'].notice.mean(), notice_mean_battery=g[g.dataset == 'BATTERY'].notice.mean())
        for ds in DS:
            row[f'rate_{ds}'] = g[g.dataset == ds].fail.mean()
        rows.append(row)
    T = pd.DataFrame(rows); T['p_holm'] = holm(T.p_greater.values)
    T.to_csv(f'{RES}/robust_settings.csv', index=False)
    H13 = dict(supported=bool((T.p_holm >= 0.05).all()), pooled_rate_range=[float(T.rate.min()), float(T.rate.max())],
               smallest_adjusted_p=float(T.p_holm.min()), notice_engines_range=[float(T.notice_mean_engines.min()), float(T.notice_mean_engines.max())],
               notice_battery_range=[float(T.notice_mean_battery.min()), float(T.notice_mean_battery.max())])
    return T, H13


def figure(P, T):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figures as F
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.3))
    lrn = ['GBM'] + ALT; lab = {'GBM': 'GBM\n(reference)', 'RIDGE': 'ridge', 'RF': 'random\nforest', 'WMLP': 'MLP on raw\nwindows'}
    fams = [('CNA', 'CNA', F.CAT[0]), ('LBall', 'per-cycle bound, all rows', F.CAT[1]), ('LBnear', 'per-cycle bound, r ≤ cap', F.CAT[2])]
    for j, (fam, name, col) in enumerate(fams):
        for i, ln in enumerate(lrn):
            g = P[(P.learner == ln) & (P.policy == f'{fam}_a0.10')]
            x = i + (j - 1) * 0.25
            ax[0].plot([x] * len(g), g.rate.values, 'o', ms=3, color=col, alpha=0.6, mec='none')
            ax[0].plot([x - 0.1, x + 0.1], [g.rate.mean()] * 2, color=col, lw=2.0, label=name if i == 0 else None)
    ax[0].axhline(0.10, color=F.INK, lw=0.9, ls='--'); ax[0].set_xticks(range(len(lrn))); ax[0].set_xticklabels([lab[l] for l in lrn])
    ax[0].set_ylabel('failure before replacement\n(nominal 0.10, primary lead time)'); ax[0].set_ylim(-0.02, 1.0)
    ax[0].legend(fontsize=6.3, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.01), handlelength=1.2, columnspacing=1.0)
    ax[0].set_title('(a) learners (dots: datasets)', fontsize=8, loc='left')
    for kk, mk in zip((1, 2, 3), ('o', 's', '^')):
        g = T[T.k == kk]
        ax[1].plot(g.notice_mean_engines, g.rate, mk, color=F.CAT[0], ms=4, mec='white', label=f'debounce {kk}')
    ax[1].axhline(0.10, color=F.INK, lw=0.9, ls='--'); ax[1].set_ylim(0, max(0.2, T.rate.max() + 0.02))
    ax[1].set_xlabel('mean notice, engines (cycles)'); ax[1].set_ylabel('pooled failure rate')
    ax[1].set_title('(b) 27 alarm settings of the GBM signal', fontsize=8, loc='left'); ax[1].legend(fontsize=6.5, loc='lower right')
    F.save(fig, 'Fig8_robustness')


if __name__ == '__main__':
    R = load()
    chk = implementation_check(R)
    P, H11, H12 = learners(R)
    T, H13 = settings(R)
    out = dict(implementation_check=chk, H11=H11, H12=H12, H13=H13)
    json.dump(out, open(f'{RES}/hypotheses_v5.json', 'w'), indent=1, default=float)
    print(json.dumps({k: (v if k == 'implementation_check' else {kk: vv for kk, vv in v.items() if kk != 'rates'}) for k, v in out.items()}, indent=1, default=float))
    pd.set_option('display.width', 220)
    print(P[P.policy.isin(['CNA_a0.10', 'LBall_a0.10', 'LBnear_a0.10'])].pivot_table(index=['learner', 'policy'], columns='dataset', values='rate').round(3).to_string())
    print(T[['cap', 'span', 'k', 'rate', 'p_holm', 'notice_mean_engines', 'notice_mean_battery']].round(3).to_string(index=False))
    figure(P, T)
