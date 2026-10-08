"""LaTeX tables of the amendment analyses, generated from results/*.csv into paper_tex/ (tab_m.tex, tab_cost.tex).
Usage: python make_tables_tex.py"""
import os
import numpy as np, pandas as pd
import core as C

RES = os.path.join(C.ROOT, 'results'); TEX = os.path.join(C.ROOT, 'paper_tex')
DS = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004', 'NCMAPSS', 'BATTERY']
DSL = {'NCMAPSS': 'N-CMAPSS', 'BATTERY': 'Battery'}


def tab_m():
    M = pd.read_csv(f'{RES}/metric_M.csv'); m = M[M.ratio == 0.1]
    ds = ['PHM08', 'FD001', 'FD002', 'FD003', 'FD004']
    pols = [('CNA_DA', 'CNA-DA (certified, $\\alpha_{\\max}=0.10$)'), ('CNA_a0.05', 'CNA, $\\alpha=0.05$ (certified)'),
            ('KAM_P1_opt', 'Policy 1, optimised threshold'), ('KAM_P1_heur', 'Policy 1, threshold $c_p/c_c$'),
            ('STW_cost_1se', 'Cost-tuned STW, 1-SE'), ('age', 'Age replacement')]
    out = ['\\begin{tabular}{@{}l' + 'l' * len(ds) + '@{}}', '\\toprule', 'Policy & ' + ' & '.join(ds) + ' \\\\', '\\midrule']
    for p, lab in pols:
        cells = []
        for d in ds:
            r = m[(m.policy == p) & (m.dataset == d)].iloc[0]; cells.append(f'{r.M_pct:.1f} ({int(r.failures)})')
        out.append(lab + ' & ' + ' & '.join(cells) + ' \\\\')
    out += ['\\bottomrule', '\\end{tabular}']
    open(f'{TEX}/tab_m.tex', 'w').write('\n'.join(out) + '\n')


def tab_cost():
    E1 = pd.read_csv(f'{RES}/cost_excess_with_cvplus.csv'); E1 = E1[E1.analysis == 'cv']
    has3 = os.path.exists(f'{RES}/equalinfo_excess.csv')
    E3 = pd.read_csv(f'{RES}/equalinfo_excess.csv') if has3 else None
    if has3: E3 = E3[E3.analysis == 'cv']
    rows = [('CNA_DA', 'CNA-DA, split (certified)', E1), ('CNAplus_DA', 'CNA+-DA, cross-fitted (certified, $2\\alpha$ bound)', E1),
            ('CNAtrend_DA', 'CNA-DA on capacity trend (certified)', E1),
            ('STW_cost_1se', 'Cost-tuned STW, 1-SE, split', E1), ('STW_window', 'Window-tuned STW, split', E1), ('KAM_P1_opt', 'Policy 1, optimised, split', E1),
            ('trend_cost_1se', 'Capacity trend, 1-SE', E1), ('age', 'Age replacement', E1)]
    out = ['\\resizebox{\\textwidth}{!}{%', '\\begin{tabular}{@{}l' + 'l' * len(DS) + '@{}}', '\\toprule',
           'Policy & ' + ' & '.join(DSL.get(d, d) for d in DS) + ' \\\\', '\\midrule']
    def line(lab, E, p):
        c = []
        for d in DS:
            x = E[(E.policy == p) & (E.dataset == d)]
            c.append(f'{100 * x.mean_excess.iloc[0]:.1f}' if len(x) else '--')
        return lab + ' & ' + ' & '.join(c) + ' \\\\'
    for p, lab, E in rows: out.append(line(lab, E, p))
    if has3:
        out.append('\\midrule')
        out.append('\\multicolumn{%d}{@{}l}{\\emph{Equal information with CNA+ (same five fold models; excess over the cheapest of these six policies; policy 1 with threshold $1/\\rho$ not shown)}} \\\\' % (len(DS) + 1))
        for p, lab in [('CNAplus_DA', 'CNA+-DA (certified)'), ('STW_cost_1se_CF', 'Cost-tuned STW, 1-SE, CF'), ('STW_cost_plain_CF', 'Cost-tuned STW, plain, CF'),
                       ('KAM_P1_opt_CF', 'Policy 1, optimised, CF'), ('age', 'Age replacement')]:
            out.append(line(lab, E3, p))
    out += ['\\bottomrule', '\\end{tabular}}']
    open(f'{TEX}/tab_cost.tex', 'w').write('\n'.join(out) + '\n')


if __name__ == '__main__':
    tab_m(); tab_cost(); print(open(f'{TEX}/tab_m.tex').read()); print(open(f'{TEX}/tab_cost.tex').read())
