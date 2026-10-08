"""Amendment v4 (protocol_v4.md): online calibration of the warning level under batch shift.
Units of the new population arrive in a random order, in groups of g; all units of a group use the current level q; after
the group's outcomes (failure before the planned replacement, or not) are known, q <- q + eta * sum(fail - alpha).
q starts at the split-conformal level of the old population (run.py's calibration units). Theorem 5: for any sequence,
|mean failure - alpha| <= (B + g*eta) / (eta * T), B the range of the critical levels.

Usage: python run4.py DATASET [REPS]     (DATASET in BATTERY, NCMAPSS)
Output: results/raw4/<DS>_rep<r>_online.csv.gz
"""
import os, sys, time
import numpy as np, pandas as pd
import core as C
import policies as P
from run import CFG, load, splits, series

OUT = os.path.join(C.ROOT, 'results', 'raw4'); os.makedirs(OUT, exist_ok=True)
ALPHA = 0.10
ETAS = [0.05, 0.10, 0.20]          # step size as a fraction of the signal cap (0.10 primary)
GROUPS = [1, 5]                    # units per update (1 primary)
N_PERM = 200


def run_split(df, kind, an, name, tr, te, rep, sidx):
    cfg = CFG[kind]; t0 = time.time(); cc, sp, kk = cfg['cna_cap'], cfg['cna_span'], cfg['cna_k']
    rng = np.random.default_rng(1000 + 100 * rep + sidx); perm = rng.permutation(tr)      # as run.py
    ncal = int(round(0.4 * len(perm))); cal_u = np.sort(perm[:ncal]); fit_u = np.sort(perm[ncal:])
    F = C.build_features(df, kind, fit_u)
    m = C.gbm(np.vstack([F[u]['X'] for u in fit_u]), np.concatenate([F[u]['r'] for u in fit_u]), cc)
    cal, tes = series(F, cal_u), series(F, te)
    RMc = [C.debounced_running_min(C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp), kk) for d in cal]
    RMt = [C.debounced_running_min(C.smooth(m.predict(F[d['unit']]['X'])[d['mask']], sp), kk) for d in tes]
    prng = np.random.default_rng(9000 + 100 * rep + sidx); rows = []
    orders = [prng.permutation(len(tes)) for _ in range(N_PERM)]
    for lead in cfg['leads']:
        Vc = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMc, cal)])
        Vt = np.array([C.critical_level(rm, d['r'], lead) for rm, d in zip(RMt, tes)])
        q1 = C.conformal_level(Vc, ALPHA)
        allV = np.concatenate([Vc, Vt]); B = float(allV.max() - allV.min())
        st_fail = float((Vt > q1).mean())                                           # static: old level kept
        for eta_f in ETAS:
            eta = eta_f * cc
            for g in GROUPS:
                for pi_, o in enumerate(orders):
                    q = q1; fails = np.zeros(len(o), bool); notices = np.full(len(o), np.nan); qs = []
                    for s in range(0, len(o), g):
                        idx = o[s:s + g]; qs.append(q)
                        f = Vt[idx] > q; fails[s:s + g] = f
                        for j, i in enumerate(idx):
                            if not f[j]:
                                jr = int(C.first_alarm(RMt[i], q)[0]); notices[s + j] = tes[i]['r'][jr]
                        q = q + eta * (f.sum() - ALPHA * len(idx))
                    T = len(o); half = T // 2
                    rows.append(dict(analysis=an, split=name, rep=rep, lead=lead, eta_frac=eta_f, group=g, perm=pi_, T=T, B=B, eta=eta,
                                     bound=(B + g * eta) / (eta * T), q1=q1, q_final=q, q_max=max(qs), q_min=min(qs),
                                     fail=float(fails.mean()), fail_second_half=float(fails[half:].mean()), static_fail=st_fail,
                                     notice=float(np.nanmean(notices)) if (~fails).any() else np.nan))
    print(f'  {name} rep {rep} ({time.time() - t0:.0f}s)', flush=True)
    return rows


def main(ds, reps):
    df, kind = load(ds); S = splits(df, kind)
    for rep in reps:
        f = f'{OUT}/{ds}_rep{rep}_online.csv.gz'
        if os.path.exists(f): continue
        R = []
        for sidx, (an, name, tr, te) in enumerate(S): R += run_split(df, kind, an, name, tr, te, rep, sidx)
        pd.DataFrame(R).assign(dataset=ds).to_csv(f, index=False); print(ds, 'rep', rep, 'written', flush=True)


if __name__ == '__main__':
    if os.environ.get('SMOKE'):
        df, kind = load(sys.argv[1]); S = splits(df, kind); gi = [i for i, s in enumerate(S) if s[0] == 'group'][1]
        R = pd.DataFrame(run_split(df, kind, *S[gi], 0, gi))
        print(R[R.lead == (30 if kind == 'battery' else 10)].groupby(['eta_frac', 'group']).agg(fail=('fail', 'mean'), f2=('fail_second_half', 'mean'),
              static=('static_fail', 'mean'), notice=('notice', 'mean'), bound=('bound', 'mean')).round(3))
    else:
        main(sys.argv[1], [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0, 1, 2])
