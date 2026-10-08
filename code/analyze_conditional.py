"""Training-conditional check (exploratory; not in the hashed protocols).

Corollary 2 of the manuscript: for i.i.d. units, the failure probability of the alarm given its calibration set,
pi = P(V_new > V_(q) | calibration set), is stochastically at most Beta(n+1-q, q), with equality for distinct critical
levels. The failures among the m test units of one outer fold then follow a Beta-binomial(m, n+1-q, q) law. This script
compares the observed fold-level failure counts of CNA (cross-validation, all repetitions and lead times) with that law
through randomised probability integral transforms (PIT) and writes
    results/conditional_check.csv    one row per (dataset, alpha, lead, repetition, fold)
    results/conditional_summary.csv  PIT summaries per alpha (all lead times; primary lead time)
    results/conditional_theory.csv   Beta quantiles of pi and the q needed for P(pi <= alpha) >= 1 - delta
    figures/Fig3_conditional.{pdf,png}
Folds of one repetition share calibration units and lead times share calibration sets, so the KS p-values are descriptive.
Usage: python analyze_conditional.py
"""
import glob, math, os
import numpy as np, pandas as pd
from scipy.stats import beta, betabinom, kstest
import core as C

RES = os.path.join(C.ROOT, 'results')
LEAD_P = {'BATTERY': 30}


def fold_counts():
    rows = []
    for f in sorted(glob.glob(f'{RES}/raw/*_decisions.csv.gz')):
        d = pd.read_csv(f)
        d = d[(d.analysis == 'cv') & d.policy.str.match(r'^CNA_a0\.\d+$')]
        d = d[d.rho == d.rho.min()]                 # CNA's alarms do not depend on the cost ratio
        for (pol, lead, rep, split), g in d.groupby(['policy', 'lead', 'rep', 'split']):
            a = float(pol.split('_a')[1]); n = int(g.n_cal.iloc[0]); q = math.ceil((n + 1) * (1 - a))
            if q > n:
                continue
            ds = g.dataset.iloc[0]
            rows.append(dict(dataset=ds, alpha=a, lead=int(lead), primary=int(lead) == LEAD_P.get(ds, 10), rep=int(rep),
                             split=split, n_cal=n, q=q, m=len(g), failures=int(g.fail.sum())))
    return pd.DataFrame(rows)


def add_pit(R, seed=2026):
    rng = np.random.default_rng(seed)
    a_, b_ = R.n_cal + 1 - R.q, R.q
    lo = np.where(R.failures > 0, betabinom.cdf(R.failures - 1, R.m, a_, b_), 0.0)
    hi = betabinom.cdf(R.failures, R.m, a_, b_)
    R['pit'] = lo + rng.uniform(size=len(R)) * (hi - lo)
    R['rate'] = R.failures / R.m
    R['pred_mean'] = a_ / (R.n_cal + 1)
    R['pred_var_rate'] = R.pred_mean * (1 - R.pred_mean) / R.m * (1 + (R.m - 1) / (R.n_cal + 2))
    R['band_lo'] = betabinom.ppf(0.05, R.m, a_, b_) / R.m
    R['band_hi'] = betabinom.ppf(0.95, R.m, a_, b_) / R.m
    return R


def summary(R):
    out = []
    for scope, S in [('all lead times', R), ('primary lead time', R[R.primary])]:
        for a, g in S.groupby('alpha'):
            out.append(dict(scope=scope, alpha=a, calibration_sets=len(g), mean_pit=g.pit.mean(), ks_p=kstest(g.pit, 'uniform').pvalue,
                            share_pit_above_095=(g.pit > 0.95).mean(), share_pit_below_005=(g.pit < 0.05).mean(),
                            var_rate_observed=g.rate.var(), var_rate_predicted=g.pred_var_rate.mean(),
                            share_outside_90band=((g.rate < g.band_lo) | (g.rate > g.band_hi)).mean()))
    return pd.DataFrame(out)


def theory(ns=(10, 32, 39, 70, 80, 83, 150, 300), alphas=(0.05, 0.10, 0.20), delta=0.10):
    out = []
    for n in ns:
        for a in alphas:
            q = math.ceil((n + 1) * (1 - a))
            if q > n:
                out.append(dict(n_cal=n, alpha=a, attainable=False)); continue
            A, B = n + 1 - q, q
            qs = [k for k in range(1, n + 1) if beta.cdf(a, n + 1 - k, k) >= 1 - delta]
            out.append(dict(n_cal=n, alpha=a, attainable=True, q=q, mean_pi=A / (n + 1), pi_q90=beta.ppf(0.90, A, B), pi_q95=beta.ppf(0.95, A, B),
                            P_pi_above_alpha_plus_005=1 - beta.cdf(a + 0.05, A, B),
                            q_conditional=qs[0] if qs else np.nan, marginal_rate_conditional=(n + 1 - qs[0]) / (n + 1) if qs else np.nan))
    return pd.DataFrame(out)


def figure(R):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import figures as F
    P = R[R.primary & (R.alpha == 0.10)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, ds in enumerate(F.DS):
        g = P[P.dataset == ds]
        if not len(g):
            continue
        # 90 % predictive band of the fold-level rate for the typical fold size of the dataset
        m = int(np.round(g.m.median())); n = int(g.n_cal.iloc[0]); q = int(g.q.iloc[0])
        lo, hi = betabinom.ppf([0.05, 0.95], m, n + 1 - q, q) / m
        ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color=F.GRID, lw=0))
        ax.plot([i - 0.3, i + 0.3], [(n + 1 - q) / (n + 1)] * 2, color=F.INK2, lw=1.0)
        jit = np.linspace(-0.18, 0.18, len(g))
        ax.plot(i + jit, g.rate.values, 'o', ms=2.8, color=F.CAT[0], alpha=0.75, mec='none')
    ax.axhline(0.10, color=F.INK, lw=0.8, ls='--')
    ax.set_xticks(range(len(F.DS))); ax.set_xticklabels([F.DSL.get(d, d) for d in F.DS], rotation=45, ha='right')
    ax.set_ylabel('failure rate in one test fold'); ax.set_xlim(-0.6, len(F.DS) - 0.4)
    F.save(fig, 'Fig3_conditional')


if __name__ == '__main__':
    R = add_pit(fold_counts())
    R.to_csv(f'{RES}/conditional_check.csv', index=False)
    S = summary(R); S.to_csv(f'{RES}/conditional_summary.csv', index=False)
    T = theory(); T.to_csv(f'{RES}/conditional_theory.csv', index=False)
    pd.set_option('display.width', 200)
    print(S.round(4).to_string(index=False)); print(T.round(4).to_string(index=False))
    figure(R)
