"""Manuscript tables (markdown) from results/*.csv -> results/paper_tables.md and paper/sections/_tables.json.
Usage: python make_tables.py"""
import os, json
import numpy as np, pandas as pd
import core as C

RES = os.path.join(C.ROOT, 'results')
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS', 'BATTERY']
DSL = {'NCMAPSS': 'N-CMAPSS', 'BATTERY': 'Battery'}
lp = lambda d: 30 if d == 'BATTERY' else 10


def md(df):
    cols = list(df.columns); out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows(): out.append('| ' + ' | '.join(str(v) for v in r.values) + ' |')
    return '\n'.join(out)


def t_validity():
    V = pd.read_csv(f'{RES}/validity.csv'); A = pd.read_csv(f'{RES}/validity_reps.csv')
    rows = []
    for d in DS:
        v0 = V[(V.dataset == d) & (V.analysis == 'cv') & (V.rep == 0) & (V.lead == lp(d))].set_index('policy')
        a = A[(A.dataset == d) & (A.analysis == 'cv') & (A.lead == lp(d))].set_index('policy')
        r0 = v0.loc['CNA_a0.10']
        f = lambda p: f"{a.loc[p, 'rate_mean']:.3f}"
        rows.append({'Dataset': DSL.get(d, d), 'Test units': int(r0.n), 'ℓ': lp(d),
                     'CNA α=0.10, rep 0: failures (rate; 95% CI)': f"{int(r0.failures)} ({r0.rate:.3f}; {r0.ci_lo:.3f}–{r0.ci_hi:.3f})",
                     'CNA 0.05': f('CNA_a0.05'), 'CNA 0.10': f('CNA_a0.10'), 'CNA 0.20': f('CNA_a0.20'), 'CNA-DA': f('CNA_DA'),
                     'LB-all 0.10': f('LBall_a0.10'), 'LB-near 0.10': f('LBnear_a0.10'),
                     'Notice, CNA 0.10 (median)': f"{a.loc['CNA_a0.10', 'notice_median']:.0f}", 'Notice, LB-all 0.10': f"{a.loc['LBall_a0.10', 'notice_median']:.0f}"})
    return pd.DataFrame(rows)


def t_equiv():
    E = pd.read_csv(f'{RES}/equivalence.csv'); E = E[(E.rep == 0)]
    rows = []
    for d in DS:
        for an in ('cv', 'group'):
            e = E[(E.dataset == d) & (E.analysis == an)]
            if not len(e): continue
            e = e.iloc[0]
            rows.append({'Dataset': DSL.get(d, d) + (' (held-out group)' if an == 'group' else ''), 'n': int(e.n),
                         'Estimate (%)': f'{e.model_pct:.1f}', 'Constant (%)': f'{e.const_pct:.1f}',
                         'Discordant (est./const.)': f'{int(e.only_model)}/{int(e.only_const)}',
                         'Difference, 90% CI (points)': f'{e.diff_pp:+.1f} ({e.ci90_lo_pp:+.1f}, {e.ci90_hi_pp:+.1f})',
                         'TOST p (±5)': f'{e.p_tost:.3f}', 'Equivalent': 'yes' if e.equivalent_5pp else 'no',
                         'SD of estimate / notice at alarm': f'{e.sd_est:.1f} / {e.sd_notice:.1f}',
                         'Thm 1 bound / discordance (points)': f'{e.pred_bound_pp:.1f} / {e.discordant_pp:.1f}',
                         'Lemma (iii) holds': f'{100 * e.lemma_ok:.0f}%'})
    return pd.DataFrame(rows)


def t_cost():
    S = pd.read_csv(f'{RES}/cost_summary.csv'); X = pd.read_csv(f'{RES}/cost_excess.csv')
    comps = [('age', 'Age replacement'), ('STW_window', 'STW, window-tuned'), ('STW_cost_plain', 'STW, cost-tuned (plain)'),
             ('STW_cost_1se', 'STW, cost-tuned (1-SE)'), ('KAM_P1_heur', 'Kamariotis P1, p = 1/ρ'), ('KAM_P1_opt', 'Kamariotis P1, optimised'),
             ('LBnear_a0.10', 'LB-near alarm (α = 0.10)'), ('trend_window', 'Capacity trend, window-tuned'), ('trend_cost_1se', 'Capacity trend, 1-SE'),
             ('CNAtrend_DA', 'CNA-DA on the capacity trend')]
    cols = [(d, 'cv') for d in DS] + [('NCMAPSS', 'group'), ('BATTERY', 'group')]
    rows = []
    for c, lab in comps:
        r = {'CNA-DA vs': lab}
        for d, an in cols:
            s = S[(S.dataset == d) & (S.analysis == an) & (S.comparator == c)]
            r[DSL.get(d, d) + (' LOGO' if an == 'group' else '')] = '–' if not len(s) else f"{int(s.cna_better.iloc[0])}/{int(s.cna_worse.iloc[0])} ({s.ratio_median.iloc[0]:.2f})"
        rows.append(r)
    for pol, lab in (('CNA_DA', 'Mean excess over cheapest: CNA-DA'), ('STW_cost_1se', '… STW 1-SE'), ('KAM_P1_opt', '… Kamariotis P1 opt.'), ('STW_window', '… STW window-tuned')):
        r = {'CNA-DA vs': lab}
        for d, an in cols:
            x = X[(X.dataset == d) & (X.analysis == an) & (X.policy == pol)]
            r[DSL.get(d, d) + (' LOGO' if an == 'group' else '')] = f'{100 * x.mean_excess.iloc[0]:.0f}%' if len(x) else '–'
        rows.append(r)
    return pd.DataFrame(rows)


def t_transfer():
    T = pd.read_csv(f'{RES}/transfer_groups.csv'); P = pd.read_csv(f'{RES}/pilot_summary.csv')
    D = pd.read_csv(f'{RES}/raw/BATTERY_decisions.csv.gz')
    rows = []
    for d in ('BATTERY', 'NCMAPSS'):
        for g in sorted(T[T.dataset == d].split.unique()):
            t = T[(T.dataset == d) & (T.split == g) & (T.lead == lp(d))]
            fr = lambda p: f"{t[t.policy == p].rate.mean():.2f}" if (t.policy == p).any() else '–'
            pm = P[(P.dataset == d) & (P.split == g) & (P.lead == lp(d)) & (P.alpha == 0.10) & (P.signal == 'CNA')]
            m10 = pm[pm.m == 10]; m5 = P[(P.dataset == d) & (P.split == g) & (P.lead == lp(d)) & (P.alpha == 0.20) & (P.signal == 'CNA') & (P.m == 5)]
            rows.append({'Held-out group': f"{DSL.get(d, d)} {'batch' if d == 'BATTERY' else 'subset'} {g[1:] if d == 'BATTERY' else int(g[1:]) + 1}",
                         'Units': int(t.n.iloc[0]), 'CNA 0.10, other groups': fr('CNA_a0.10'), 'CNA-DA': fr('CNA_DA'),
                         'Pilot m=10, α=0.10': f"{m10.fail_mean.mean():.3f}" if len(m10) else '–',
                         'Pilot m=5, α=0.20': f"{m5.fail_mean.mean():.3f}" if len(m5) else '–',
                         'STW window': fr('STW_window'), 'STW 1-SE': fr('STW_cost_1se'), 'Kamariotis P1 opt.': fr('KAM_P1_opt'),
                         'Trend 1-SE': fr('trend_cost_1se'), 'CNA 0.10 on trend': fr('CNAtrend_a0.10')})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    tabs = {'validity': t_validity(), 'equivalence': t_equiv(), 'cost': t_cost(), 'transfer': t_transfer()}
    with open(f'{RES}/paper_tables.md', 'w') as f:
        for k, v in tabs.items(): f.write(f'## {k}\n\n{md(v)}\n\n')
    for k, v in tabs.items(): v.to_csv(f'{RES}/table_{k}.csv', index=False)
    json.dump({k: md(v) for k, v in tabs.items()}, open(os.path.join(C.ROOT, 'paper', 'sections', '_tables.json'), 'w'), indent=1)
    print(open(f'{RES}/paper_tables.md').read())
