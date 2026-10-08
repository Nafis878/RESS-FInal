"""Figures of the manuscript (results/ CSVs -> figures/*.png and *.pdf). Usage: python figures.py"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import core as C
import policies as P

OUT = os.path.join(C.ROOT, 'figures'); RES = os.path.join(C.ROOT, 'results'); os.makedirs(OUT, exist_ok=True)
# categorical slots of the reference palette, fixed order (blue, orange, aqua, yellow, magenta, green, violet, red)
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS', 'BATTERY']
DSL = {'NCMAPSS': 'N-CMAPSS', 'BATTERY': 'Battery'}
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'lines.linewidth': 1.6, 'savefig.dpi': 300, 'font.family': 'DejaVu Sans'})


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', bbox_inches='tight'); fig.savefig(f'{OUT}/{name}.pdf', bbox_inches='tight'); plt.close(fig)


def lead_p(ds):
    return 30 if ds == 'BATTERY' else 10


# ---------------------------------------------------------------------------------------------- Fig. 1 method
def fig_method():
    df = C.load_engines('FD001'); units = np.sort(df.unit.unique()); te = C.outer_folds(units)[0]; tr = np.setdiff1d(units, te)
    perm = np.random.default_rng(1000).permutation(tr); ncal = int(round(0.4 * len(perm))); cal_u, fit_u = np.sort(perm[:ncal]), np.sort(perm[ncal:])
    F = C.build_features(df, 'engine', fit_u); X = np.vstack([F[u]['X'] for u in fit_u]); y = np.concatenate([F[u]['r'] for u in fit_u])
    m = C.gbm(X, y, 60.0); lead = 10
    V = []; tracks = {}
    for u in cal_u:
        d = F[u]; mk = d['r'] >= 1; s = C.smooth(m.predict(d['X'])[mk], 3); rm = C.debounced_running_min(s, 2)
        V.append(C.critical_level(rm, d['r'][mk], lead)); tracks[u] = (d['r'][mk], s, rm)
    V = np.array(V); Hh = C.conformal_level(V, 0.10)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    u = cal_u[int(np.argsort(V)[len(V) // 2])]; r, s, rm = tracks[u]; Vu = C.critical_level(rm, r, lead)
    ax[0].plot(r, s, color=CAT[0], label='smoothed estimate $s_t$')
    ax[0].plot(r, np.where(np.isfinite(rm), rm, np.nan), color=CAT[1], lw=1.2, ls='--', label='debounced running minimum $\\rho_t$')
    ax[0].axvline(lead, color=INK2, lw=0.8, ls=':'); ax[0].text(lead + 2, 3, f'lead time $\\ell$ = {lead}', color=INK2, fontsize=7, ha='right')
    ax[0].axhline(Vu, color=CAT[2], lw=1.0); ax[0].text(r.max() * 0.98, Vu - 4.2, f'critical level of this engine $V$ = {Vu:.1f}', color=INK, fontsize=7)
    ax[0].axhline(Hh, color=CAT[6], lw=1.0, ls='-.'); ax[0].text(r.max() * 0.98, Hh + 1.2, f'conformal level $\\hat H$ = {Hh:.1f} (α = 0.10)', color=INK, fontsize=7)
    jl = np.flatnonzero(r > lead)[-1]; ax[0].plot(r[jl], rm[jl], 'o', color=CAT[2], ms=5, mec='white', zorder=5)
    ax[0].invert_xaxis(); ax[0].set_xlabel('true remaining life $r_t$ (cycles)'); ax[0].set_ylabel('estimate (cycles)'); ax[0].set_ylim(0, 65)
    ax[0].legend(loc='center left', fontsize=7, bbox_to_anchor=(0, 0.62)); ax[0].set_title('(a) one calibration engine (FD001)', fontsize=8, loc='left')
    vv = np.sort(V[np.isfinite(V)])
    ax[1].set_axisbelow(True); ax[1].hist(vv, bins=15, color=CAT[0], edgecolor='white', linewidth=0.8)
    ax[1].axvline(Hh, color=CAT[6], lw=1.2, ls='-.'); ax[1].text(Hh + 0.3, ax[1].get_ylim()[1] * 0.75, '$\\hat H$ (90%\nconformal\nquantile)', fontsize=6.5, color=INK, ha='left')
    ax[1].set_xlabel(f'critical level $V_i$ ($\\ell$ = {lead})'); ax[1].set_ylabel('calibration engines')
    ax[1].set_title(f'(b) {len(V)} calibration engines', fontsize=8, loc='left')
    save(fig, 'Fig1_method')


# ---------------------------------------------------------------------------------------------- Fig. 2 validity
def fig_validity():
    V = pd.read_csv(f'{RES}/validity.csv'); V = V[(V.analysis == 'cv')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    fams = [('CNA', 'conformal notice alarm (CNA)', CAT[0]), ('LBall', 'per-cycle lower bound, all rows', CAT[1]), ('LBnear', 'per-cycle lower bound, r ≤ cap', CAT[2])]
    for ax, a in zip(axes, [0.05, 0.10, 0.20]):
        for j, (fam, lab, col) in enumerate(fams):
            for i, ds in enumerate(DS):
                g = V[(V.dataset == ds) & (V.policy == f'{fam}_a{a:.2f}') & (V.lead == lead_p(ds))]
                if not len(g): continue
                x = i + (j - 1) * 0.25; rates = g.rate.values
                ax.plot([x] * len(rates), rates, 'o', ms=3, color=col, alpha=0.55, mec='none')
                ax.plot([x - 0.1, x + 0.1], [rates.mean()] * 2, color=col, lw=2.0, label=lab if i == 0 else None)
        ax.axhline(a, color=INK, lw=0.9, ls='--'); ax.set_xticks(range(len(DS))); ax.set_xticklabels([DSL.get(d, d) for d in DS], rotation=55, ha='right')
        ax.set_title(f'nominal α = {a:.2f}', fontsize=8, loc='left')
    axes[0].set_ylabel('failure before replacement\n(test units, primary lead time)')
    axes[0].legend(loc='upper left', fontsize=6.5, bbox_to_anchor=(0, 1.0))
    save(fig, 'Fig2_validity')


# ---------------------------------------------------------------------------------------------- Fig. 3 equivalence
def fig_equivalence():
    E = pd.read_csv(f'{RES}/equivalence.csv'); E = E[(E.analysis == 'cv') & (E.rep == 0)].set_index('dataset').reindex(DS)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(wspace=0.35))
    y = np.arange(len(DS))[::-1]
    ax[0].axvspan(-5, 5, color=GRID, alpha=0.6, lw=0); ax[0].axvline(0, color=INK2, lw=0.8)
    for yi, (ds, r) in zip(y, E.iterrows()):
        col = CAT[0] if r.equivalent_5pp else CAT[1]
        ax[0].plot([r.ci90_lo_pp, r.ci90_hi_pp], [yi, yi], color=col, lw=2.2, solid_capstyle='round'); ax[0].plot(r.diff_pp, yi, 'o', color=col, ms=5, mec='white')
    ax[0].plot([], [], color=CAT[0], lw=2.2, label='equivalent (within ±5)'); ax[0].plot([], [], color=CAT[1], lw=2.2, label='not shown equivalent')
    ax[0].legend(fontsize=6.5, loc='lower left')
    ax[0].set_yticks(y); ax[0].set_yticklabels([DSL.get(d, d) for d in DS]); ax[0].set_xlabel('model − constant (percentage points)')
    ax[0].set_title('(a) window score difference, Tango 90% CI', fontsize=8, loc='left')
    xs = np.arange(len(DS))
    ax[1].bar(xs - 0.2, E.pred_bound_pp, 0.38, color=GRID, edgecolor=INK2, linewidth=0.6, label='Theorem 1 bound (calibration units)')
    ax[1].plot(xs + 0.2, E.discordant_pp, 'D', color=CAT[0], ms=4.5, mec='white', label='discordant units (test)')
    ax[1].plot(xs + 0.2, E.abs_diff_pp, 'o', color=CAT[1], ms=4.5, mec='white', label='|difference in score| (test)')
    ax[1].set_xticks(xs); ax[1].set_xticklabels([DSL.get(d, d) for d in DS], rotation=45, ha='right'); ax[1].set_ylabel('percentage points')
    ax[1].set_title('(b) bound vs realised discordance', fontsize=8, loc='left'); ax[1].legend(fontsize=6.3, loc='upper left'); ax[1].set_axisbelow(True)
    save(fig, 'Fig3_equivalence')


# ---------------------------------------------------------------------------------------------- Fig. 4 cost
def fig_cost():
    R = pd.read_csv(f'{RES}/cost_cells.csv'); R = R[(R.analysis == 'cv') & (R.rep == 0)]
    pols = [('CNA_DA', 'CNA, decision-aware', CAT[0], '-'), ('STW_window', 'STW, window-tuned', CAT[1], '-'), ('STW_cost_1se', 'STW, cost-tuned 1-SE', CAT[2], '-'),
            ('STW_cost_plain', 'STW, cost-tuned plain', CAT[3], ':'), ('KAM_P1_opt', 'Kamariotis P1, optimised', CAT[6], '--'), ('trend_cost_1se', 'capacity trend, 1-SE', CAT[5], '--')]
    fig, axes = plt.subplots(2, 7, figsize=(7.4, 3.6), sharex=False)
    for row, rho in enumerate([10, 50]):
        for c, ds in enumerate(DS):
            ax = axes[row, c]; g = R[(R.dataset == ds) & (R.rho == rho)]
            for pol, lab, col, ls in pols:
                h = g[g.policy == pol].sort_values('lead')
                if len(h): ax.plot(h.lead, h.rel_cost, ls, color=col, lw=1.4, marker='o', ms=2.5, label=lab)
            ax.set_yscale('log'); ax.set_ylim(1.0, 2.6 if rho == 10 else 8.0); ax.set_yticks([1, 1.5, 2, 3, 5, 8] if rho == 50 else [1, 1.25, 1.5, 2, 2.5]); ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g')); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_xticks(sorted(g.lead.unique())); ax.tick_params(labelsize=6)
            if row == 0: ax.set_title(DSL.get(ds, ds), fontsize=7.5)
            if c == 0: ax.set_ylabel(f'ρ = {rho}\ncost / perfect foresight', fontsize=7)
            if row == 1: ax.set_xlabel('lead time', fontsize=7)
    h, l = axes[0, 6].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=6, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, 'Fig4_cost')


# ---------------------------------------------------------------------------------------------- Fig. 5 transfer
def fig_transfer():
    T = pd.read_csv(f'{RES}/transfer_groups.csv'); Pp = pd.read_csv(f'{RES}/pilot_summary.csv')
    O = pd.read_csv(f'{RES}/online.csv') if os.path.exists(f'{RES}/online.csv') else None
    DR = pd.read_csv(f'{RES}/raw/BATTERY_decisions.csv.gz'); DR = DR[(DR.analysis == 'group') & (DR.policy == 'CNA_a0.10') & (DR.lead == 30) & (DR.rho == 10)]
    tg = T[(T.dataset == 'BATTERY') & (T.policy == 'CNA_a0.10') & (T.lead == 30)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.3))
    meths = [('static', 'level of the other batches (static)', 'o', CAT[1]), ('pilot', '10 pilot cells run to failure', 'D', CAT[0]),
             ('online', 'online, no pilot (η = 0.1 cap, g = 1)', '^', CAT[2]), ('online5', 'online, groups of 5 (g = 5)', 'v', CAT[5])]
    for i, batch in enumerate(['g1', 'g2', 'g3']):
        vals = {}
        g = tg[tg.split == batch]; dd = DR[(DR.split == batch) & ~DR.fail]
        vals['static'] = (g.rate.mean(), dd.groupby('rep').notice.mean().mean())
        q = Pp[(Pp.signal == 'CNA') & (Pp.split == batch) & (Pp.lead == 30) & (Pp.alpha == 0.10) & (Pp.m == 10) & (Pp.dataset == 'BATTERY')]
        vals['pilot'] = (q.fail_mean.mean(), q.notice_mean.mean())
        if O is not None:
            for key, gg in (('online', 1), ('online5', 5)):
                o = O[(O.dataset == 'BATTERY') & (O.split == batch) & (O.eta_frac == 0.10) & (O.group == gg)]
                vals[key] = (o.fail.mean(), o.notice.mean())
        for j, (key, lab, mk, col) in enumerate(meths):
            if key not in vals: continue
            f, nt = vals[key]
            ax[0].plot(i + (j - 1.5) * 0.17, f, mk, color=col, ms=5, mec='white', label=lab if i == 0 else None)
            ax[1].plot(f, nt, mk, color=col, ms=5, mec='white')
        ax[1].annotate(f'batch {batch[1]}', vals['static'], fontsize=6.5, color=INK2, xytext=(4, 3), textcoords='offset points')
    ax[0].axhline(0.10, color=INK, ls='--', lw=0.9); ax[0].set_xticks(range(3)); ax[0].set_xticklabels(['batch 1', 'batch 2', 'batch 3']); ax[0].set_xlabel('held-out batch', fontsize=7)
    ax[0].set_ylabel('failure before replacement'); ax[0].set_title('(a) α = 0.10, lead time 30 cycles', fontsize=8, loc='left')
    ax[0].legend(fontsize=6.2, loc='upper center', bbox_to_anchor=(1.15, -0.2), ncol=2)
    ax[1].axvline(0.10, color=INK, ls='--', lw=0.9); ax[1].set_xlabel('failure before replacement'); ax[1].set_ylabel('mean notice at planned\nreplacements (cycles)')
    ax[1].set_title('(b) failure vs notice (GBM signal)', fontsize=8, loc='left')
    save(fig, 'Fig5_transfer')


if __name__ == '__main__':
    fig_method()
    for f in (fig_validity, fig_equivalence, fig_cost, fig_transfer):
        try: f()
        except FileNotFoundError as e: print('skipped', f.__name__, e)
