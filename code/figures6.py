"""Figures of amendment v6 (protocol_v6.md): Fig9_separation (Theorem 5 in data and simulation) and Fig10_censoring
(Theorem 6 / Proposition 3 on the existing data; exploratory event-rate sweep). Also re-draws Fig2_validity and the
learner panel of Fig8_robustness with the per-cycle comparators at the corrected action level (debounce 1, action l + spacing).
Usage: python figures6.py
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import figures as F

RES = F.RES; DS = F.DS; DSL = F.DSL; CAT = F.CAT
lp = lambda ds: 30 if ds == 'BATTERY' else 10


def fig_validity_corrected():
    """Fig. 2 with LB-all / LB-near at the corrected action level (K1); CNA unchanged (run.py)."""
    V = pd.read_csv(f'{RES}/validity.csv'); V = V[V.analysis == 'cv']
    L = pd.read_csv(f'{RES}/lb_variants.csv'); L = L[L.variant == 'K1']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    fams = [('CNA', 'conformal notice alarm (CNA)', CAT[0]), ('LBall', 'per-cycle bound, all rows', CAT[1]), ('LBnear', 'per-cycle bound, r ≤ cap', CAT[2])]
    for ax, a in zip(axes, [0.05, 0.10, 0.20]):
        for j, (fam, lab, col) in enumerate(fams):
            for i, ds in enumerate(DS):
                if fam == 'CNA':
                    g = V[(V.dataset == ds) & (V.policy == f'CNA_a{a:.2f}') & (V.lead == lp(ds))]; rates = g.rate.values
                else:
                    g = L[(L.dataset == ds) & (L.policy == fam) & (np.isclose(L.alpha, a)) & (L.lead == lp(ds))]; rates = g.rate.values
                if not len(rates): continue
                x = i + (j - 1) * 0.25
                ax.plot([x] * len(rates), rates, 'o', ms=3, color=col, alpha=0.55, mec='none')
                ax.plot([x - 0.1, x + 0.1], [rates.mean()] * 2, color=col, lw=2.0, label=lab if i == 0 else None)
        ax.axhline(a, color=F.INK, lw=0.9, ls='--'); ax.set_xticks(range(len(DS))); ax.set_xticklabels([DSL.get(d, d) for d in DS], rotation=55, ha='right')
        ax.set_title(f'nominal α = {a:.2f}', fontsize=8, loc='left')
    axes[0].set_ylabel('failure before replacement\n(test units, primary lead time)')
    axes[0].legend(loc='upper left', fontsize=6.5, bbox_to_anchor=(0, 1.0))
    F.save(fig, 'Fig2_validity')


def fig_separation():
    L = pd.read_csv(f'{RES}/lb_variants.csv'); B = pd.read_csv(f'{RES}/b1_data.csv'); S3 = pd.read_csv(f'{RES}/sim6/S3.csv')
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.45, width_ratios=[1.15, 1.15, 1]))
    x = np.arange(len(DS))
    h = L[(L.alpha == 0.10) & np.array([l == lp(d) for l, d in zip(L.lead, L.dataset)])]
    for j, (pol, var, lab, col, mk) in enumerate([('LBall', 'ORIG', 'all rows, main protocol', CAT[1], 'o'), ('LBall', 'K1', 'all rows, corrected', CAT[1], 's'),
                                                   ('LBnear', 'ORIG', 'r ≤ cap, main protocol', CAT[2], 'o'), ('LBnear', 'K1', 'r ≤ cap, corrected', CAT[2], 's')]):
        g = h[(h.policy == pol) & (h.variant == var)].groupby('dataset').rate.mean().reindex(DS)
        ax[0].plot(x + (j - 1.5) * 0.15, g.values, mk, color=col, ms=4, mfc=col if var == 'K1' else 'white', mec=col, label=lab)
    ax[0].axhline(0.10, color=F.INK, ls='--', lw=0.9); ax[0].set_xticks(x); ax[0].set_xticklabels([DSL.get(d, d) for d in DS], rotation=55, ha='right', fontsize=6.5)
    ax[0].set_ylabel('failure before replacement'); ax[0].set_title('(a) action level ℓ+1, k=2 vs corrected', fontsize=8, loc='left')
    ax[0].legend(fontsize=5.8, loc='upper right')
    A = B[B.policy == 'LBall'].set_index('dataset').reindex(DS)
    ax[1].plot(x, A.crit_miss, '_', color=F.INK2, ms=10, mew=1.6, label='miss at critical row (upper bound)')
    ax[1].plot(x, A.fail, 'o', color=CAT[1], ms=4.5, mec='white', label='realised failure')
    ax[1].plot(x, A.prod_indep, 'v', color=CAT[6], ms=4.5, mec='white', label='independent rows (lower bound)')
    ax[1].plot(x, A.pooled_miss_target, 'x', color=CAT[0], ms=4.5, label='pooled miss over rows')
    ax[1].set_yscale('symlog', linthresh=1e-3); ax[1].set_ylim(0, 1)
    ax[1].set_xticks(x); ax[1].set_xticklabels([DSL.get(d, d) for d in DS], rotation=55, ha='right', fontsize=6.5)
    ax[1].set_title('(b) Theorem 5 in data (all rows)', fontsize=8, loc='left'); ax[1].legend(fontsize=5.6, loc='lower left')
    s = S3[S3.errors.str.startswith('AR')].copy(); s['phi'] = s.errors.str[2:].astype(float)
    ax[2].plot(s.phi, s.pF, 'o-', color=CAT[1], ms=4, label='P(F), simulated')
    ax[2].axhline(float(s['product'].iloc[0]), color=CAT[6], ls=':', lw=1.2, label='product of π (independence)')
    ax[2].axhline(float(s['minimum'].iloc[0]), color=F.INK2, ls='--', lw=1.0, label='min π (nested errors)')
    ax[2].set_xlabel('AR(1) correlation of errors'); ax[2].set_title('(c) dependence (simulation)', fontsize=8, loc='left')
    ax[2].legend(fontsize=5.6, loc='center left')
    F.save(fig, 'Fig9_separation')


def fig_censoring6():
    g = pd.read_csv(f'{RES}/cens6.csv'); g = g[g.alpha == 0.10]
    S7 = pd.read_csv(f'{RES}/sim6/S7_exploratory_summary.csv')
    meths = [('ORACLE', 'no censoring', F.INK2, 'o'), ('T2', 'Theorem 2', CAT[3], 's'), ('CC', 'complete case', CAT[4], '^'),
             ('IPCW', 'IPCW (Prop. 3)', CAT[6], 'D'), ('E2HIST_G1', 'Theorem 6, Γ=1', CAT[0], 'v'), ('E2HIST_G4', 'Theorem 6, Γ=4', CAT[2], 'P')]
    mechs = ['M1', 'M2', 'M3', 'M4']; mlab = {'M1': 'random', 'M2': 'staggered', 'M3': 'policy', 'M4': 'informative'}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw=dict(wspace=0.45))
    for j, (m_, lab, col, mk) in enumerate(meths):
        for i, me in enumerate(mechs):
            v = g[(g.mechanism == me) & (g.method == m_)]
            xx = i + (j - 2.5) * 0.12
            ax[0].plot([xx] * len(v), v.fail, mk, color=col, ms=2.6, alpha=0.6, mec='none')
            ax[0].plot([xx - 0.05, xx + 0.05], [v.fail.max()] * 2, color=col, lw=1.6, label=lab if i == 0 else None)
            ax[1].plot([xx] * len(v), v.first_row, mk, color=col, ms=2.6, alpha=0.7, mec='none')
    ax[0].axhline(0.10, color=F.INK, ls='--', lw=0.9)
    for a_ in ax[:2]:
        a_.set_xticks(range(4)); a_.set_xticklabels([mlab[m] for m in mechs], fontsize=6.5, rotation=30, ha='right')
    ax[0].set_ylabel('failure within horizon\nbefore replacement'); ax[0].set_title('(a) validity (bar: worst dataset)', fontsize=8, loc='left')
    ax[1].set_ylabel('share alarmed at first row'); ax[1].set_title('(b) uninformative alarms', fontsize=8, loc='left')
    ax[0].legend(fontsize=6.0, loc='upper center', ncol=3, bbox_to_anchor=(1.25, -0.32))
    for m_, lab, col, mk in [('ORACLE', 'no censoring', F.INK2, 'o'), ('T2', 'Theorem 2', CAT[3], 's'), ('IPCW', 'IPCW', CAT[6], 'D'), ('E2_G1', 'Theorem 6, Γ=1', CAT[0], 'v')]:
        v = S7[S7.method == m_].sort_values('p_horizon')
        ax[2].plot(v.p_horizon, v.fail / v.alpha, mk + '-', color=col, ms=3.5, lw=1.1, label=lab)
    ax[2].axhline(1.0, color=F.INK, ls='--', lw=0.9); ax[2].set_xscale('log')
    pts = sorted(S7.p_horizon.unique()); ax[2].set_xticks(pts); ax[2].set_xticklabels([f'{100 * p:.0f}%' if p >= 0.1 else f'{100 * p:.1f}%' for p in pts], fontsize=6.5)
    ax[2].minorticks_off()
    ax[2].set_xlabel('P(fail within horizon)'); ax[2].set_ylabel('failure / α'); ax[2].set_title('(c) exploratory simulation', fontsize=8, loc='left')
    ax[2].legend(fontsize=5.8, loc='lower right', bbox_to_anchor=(1.0, 0.08))
    F.save(fig, 'Fig10_censoring_horizon')


def fig_robust_corrected():
    """Fig. 8(a) with the per-cycle comparators at the corrected action level; panel (b) as in analyze5."""
    import analyze5 as A5
    P5 = pd.read_csv(f'{RES}/robust_learners.csv'); T5 = pd.read_csv(f'{RES}/robust_settings.csv')
    Lr = pd.read_csv(f'{RES}/lb_learners.csv'); Lg = pd.read_csv(f'{RES}/lb_variants.csv')
    Lg = Lg[(Lg.variant == 'K1') & (Lg.rep == 0) & (Lg.alpha == 0.10) & np.array([l == lp(d) for l, d in zip(Lg.lead, Lg.dataset)])]
    rows = [dict(learner=r.learner, dataset=r.dataset, policy=f'{r.policy}_a0.10', rate=r.rate) for r in Lr.itertuples()]
    rows += [dict(learner='GBM', dataset=r.dataset, policy=f'{r.policy}_a0.10', rate=r.rate) for r in Lg.itertuples()]
    P = pd.concat([P5[P5.policy == 'CNA_a0.10'][['learner', 'dataset', 'policy', 'rate']], pd.DataFrame(rows)], ignore_index=True)
    A5.figure(P, T5)


if __name__ == '__main__':
    fig_validity_corrected(); fig_separation(); fig_censoring6(); fig_robust_corrected(); print('figures written')
