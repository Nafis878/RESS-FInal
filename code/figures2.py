"""Figures of the amendment analyses (results/*.csv from analyze2.py -> figures/Fig6_censoring, Fig7_cvplus_M).
Usage: python figures2.py"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures import CAT, INK, INK2, GRID, DS, DSL, save, RES

lp = lambda d: 30 if d == 'BATTERY' else 10


def fig_censoring():
    G = pd.read_csv(f'{RES}/censoring.csv'); G = G[(G.alpha == 0.10)]
    G = G[[r.lead == lp(r.dataset) for r in G.itertuples()]]
    ref = G[G.method == 'oracle'].groupby(['dataset', 'rep']).agg(of=('fail', 'mean'), on=('notice', 'mean'))
    G = G.join(ref, on=['dataset', 'rep'])
    G['notice_ratio'] = G.notice / G.on
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.38))
    styles = {('random', 'censored_as_failure'): (CAT[0], '-', 'o', 'random, censored as failure'),
              ('random', 'complete_case'): (CAT[0], ':', 's', 'random, complete case'),
              ('administrative', 'censored_as_failure'): (CAT[1], '-', 'o', 'administrative, censored as failure'),
              ('administrative', 'complete_case'): (CAT[1], ':', 's', 'administrative, complete case')}
    for (mech, meth), (col, ls, mk, lab) in styles.items():
        g = G[(G.mechanism == mech) & (G.method == meth)].groupby('pi').agg(f=('fail', 'mean'), fmax=('fail', 'max'), nr=('notice_ratio', 'median')).reset_index()
        ax[0].plot(g.pi, g.f, ls, color=col, marker=mk, ms=4, label=lab)
        ax[1].plot(g.pi, g.nr, ls, color=col, marker=mk, ms=4, label=lab)
    o = G[G.method == 'oracle'].fail.mean(); ax[0].plot([0], [o], 'D', color=INK, ms=5, label='no censoring')
    ax[0].axhline(0.10, color=INK, ls='--', lw=0.9); ax[0].set_xlabel('fraction of calibration units censored'); ax[0].set_ylabel('failure before replacement')
    ax[0].set_title('(a) validity, α = 0.10', fontsize=8, loc='left'); ax[0].set_xticks([0, 0.25, 0.5, 0.75])
    ax[1].set_yscale('log'); ax[1].set_xlabel('fraction of calibration units censored'); ax[1].set_ylabel('notice / notice without censoring')
    ax[1].set_title('(b) price: notice (median over datasets)', fontsize=8, loc='left'); ax[1].set_xticks([0.25, 0.5, 0.75])
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    P = G[G.mechanism == 'policy']
    for j, meth in enumerate(['censored_as_failure', 'complete_case']):
        g = P[P.method == meth].groupby('dataset').agg(f=('fail', 'mean'), fc=('frac_cens', 'mean')).reindex(DS)
        ax[2].plot(np.arange(len(DS)) + (j - 0.5) * 0.3, g.f, 'o' if j == 0 else 's', color=CAT[2 + j * 4], ms=4.5, mec='white',
                   label='censored as failure' if j == 0 else 'complete case')
    o = G[G.method == 'oracle'].groupby('dataset').fail.mean().reindex(DS)
    ax[2].plot(np.arange(len(DS)), o.values, '_', color=INK, ms=9, mew=1.5, label='no censoring')
    ax[2].axhline(0.10, color=INK, ls='--', lw=0.9); ax[2].set_xticks(range(len(DS))); ax[2].set_xticklabels([DSL.get(d, d) for d in DS], rotation=55, ha='right', fontsize=6.5)
    ax[2].set_title('(c) censored by an earlier alarm', fontsize=8, loc='left'); ax[2].set_ylabel('failure before replacement')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=6.3, bbox_to_anchor=(0.36, -0.2))
    ax[2].legend(fontsize=6.2, loc='upper left')
    save(fig, 'Fig6_censoring')


def fig_cvplus_m():
    V = pd.read_csv(f'{RES}/validity.csv'); Vp = pd.read_csv(f'{RES}/cvplus_validity.csv')
    E = pd.read_csv(f'{RES}/cost_excess_with_cvplus.csv'); M = pd.read_csv(f'{RES}/metric_M.csv')
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.15]))
    x = np.arange(len(DS))
    for j, (src, pol, lab, col) in enumerate([(V, 'CNA_a0.10', 'split CNA', CAT[0]), (Vp, 'CNAplus_a0.10', 'cross-fitted CNA+', CAT[2])]):
        g = src[(src.analysis == 'cv') & (src.policy == pol)]
        g = g[[r.lead == lp(r.dataset) for r in g.itertuples()]].groupby('dataset').rate.mean().reindex(DS)
        ax[0].plot(x + (j - 0.5) * 0.3, g.values, 'o', color=col, ms=4.5, mec='white', label=lab)
    ax[0].axhline(0.10, color=INK, ls='--', lw=0.9); ax[0].set_xticks(x); ax[0].set_xticklabels([DSL.get(d, d) for d in DS], rotation=55, ha='right', fontsize=6.5)
    ax[0].set_ylabel('failure before replacement'); ax[0].set_title('(a) α = 0.10, primary lead time', fontsize=8, loc='left'); ax[0].legend(fontsize=6.3, loc='upper left')
    e = E[E.analysis == 'cv']
    for j, (pol, lab, col) in enumerate([('CNA_DA', 'CNA-DA (split)', CAT[0]), ('CNAplus_DA', 'CNA+-DA (cross-fitted)', CAT[2]), ('STW_cost_1se', 'STW cost-tuned 1-SE', CAT[1])]):
        g = e[e.policy == pol].set_index('dataset').reindex(DS)
        ax[1].bar(x + (j - 1) * 0.27, 100 * g.mean_excess.values, 0.26, color=col, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels([DSL.get(d, d) for d in DS], rotation=55, ha='right', fontsize=6.5); ax[1].set_axisbelow(True)
    ax[1].set_ylabel('mean excess over cheapest (%)'); ax[1].set_title('(b) cost, 16 cells per dataset', fontsize=8, loc='left'); ax[1].legend(fontsize=6.0, loc='upper left')
    m = M[M.ratio == 0.1]; dsm = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004']
    pols = [('CNA_DA', 'CNA-DA', CAT[0]), ('KAM_P1_opt', 'Kamariotis P1, optimised', CAT[6]), ('KAM_P1_heur', 'Kamariotis P1, p = c_p/c_c', CAT[4]),
            ('STW_cost_1se', 'STW 1-SE', CAT[1]), ('CNA_a0.05', 'CNA, α = 0.05', CAT[2])]
    for j, (pol, lab, col) in enumerate(pols):
        g = m[m.policy == pol].set_index('dataset').reindex(dsm); xx = np.arange(len(dsm)) + (j - 2) * 0.15
        ax[2].errorbar(xx, g.M_pct, yerr=[g.M_pct - g.lo, g.hi - g.M_pct], fmt='o', color=col, ms=3.5, lw=1, capsize=0, label=lab)
    ax[2].set_xticks(range(len(dsm))); ax[2].set_xticklabels(dsm, rotation=55, ha='right', fontsize=6.5); ax[2].set_yscale('symlog', linthresh=10)
    ax[2].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax[2].set_ylabel('metric M (%)'); ax[2].set_title('(c) setting of Kamariotis et al.', fontsize=8, loc='left'); ax[2].legend(fontsize=5.8, loc='upper left', ncol=1)
    save(fig, 'Fig7_cvplus_M')


if __name__ == '__main__':
    import matplotlib.ticker
    fig_censoring(); fig_cvplus_m()
