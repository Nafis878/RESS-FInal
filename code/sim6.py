"""Amendment v6, Part 2 (protocol_v6.md): simulation checks of Theorems 5-7 and Proposition 3.

SYNTHETIC DATA, used only to check the theorems. Seeds fixed below. Outputs: results/sim6/<check>.csv and
results/sim6/summary.json (pass/fail of every pre-specified check: estimate <= nominal + 4 Monte Carlo SE, or the
stated equality within 4 SE).
Usage: python sim6.py [S1 S2 S3 S4 S6]   (default: all)
"""
import json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
import core as C

OUT = os.path.join(C.ROOT, 'results', 'sim6'); os.makedirs(OUT, exist_ok=True)
SUMMARY = f'{OUT}/summary.json'


def save_summary(key, val):
    S = json.load(open(SUMMARY)) if os.path.exists(SUMMARY) else {}
    S[key] = val; json.dump(S, open(SUMMARY, 'w'), indent=1, default=float)


def weighted_levels(V, w, w_tests, alpha):
    """vectorised weighted_level for many test weights"""
    V = np.asarray(V, float); w = np.asarray(w, float); o = np.argsort(V, kind='stable'); Vs = V[o]; cum = np.cumsum(w[o])
    thr = (1 - alpha) * (w.sum() + np.asarray(w_tests, float))
    i = np.searchsorted(cum, thr * (1 - 1e-12), side='left')
    return np.where(i < len(V), Vs[np.minimum(i, len(V) - 1)], np.inf)


def weighted_level(V, w, w_test, alpha):
    """smallest v with sum_{V_i <= v} w_i >= (1 - alpha)(sum_i w_i + w_test); +inf if none (Theorem 7, Proposition 3)"""
    V = np.asarray(V, float); w = np.asarray(w, float)
    if not len(V): return np.inf
    o = np.argsort(V, kind='stable'); cum = np.cumsum(w[o]); thr = (1 - alpha) * (w.sum() + w_test)
    i = int(np.searchsorted(cum, thr * (1 - 1e-12), side='left'))
    return float(V[o][i]) if i < len(V) else np.inf


# ---------------------------------------------------------------------------------------------- Theorem 5
def s1(N=200_000, alpha=0.10, lead=10, act=11, k=2, m=40, seed=601):
    """Theorem 5(a): per-cycle coverage exactly 1 - alpha at every cycle, yet every alarm fails."""
    rng = np.random.default_rng(seed)
    g = lambda x: x ** lead * (1 - x ** m)
    xs = (lead / (lead + m)) ** (1 / m)                                   # maximiser of g
    lo, hi = xs, 1 - 1e-12                                                 # larger root (realistic lives)
    for _ in range(200):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if g(mid) > alpha else (lo, mid)
    x = (lo + hi) / 2
    L = rng.geometric(1 - x, N)                                            # P(L > l) = x^l
    res, ok = [], True
    for t in (0, 10, 50, 100):
        al = L > t; r = L[al] - t; miss = (r > lead) & (r <= lead + m)
        se = np.sqrt(alpha * (1 - alpha) / al.sum()); est = miss.mean()
        res.append(dict(check='cycle', t=t, n_alive=int(al.sum()), miss=est, se=se, ok=abs(est - alpha) <= 4 * se)); ok &= abs(est - alpha) <= 4 * se
    a_u = np.minimum(m, np.maximum(0, L - lead)).astype(float)           # misses per unit (rows within a unit are not
    pooled = a_u.sum() / L.sum()                                          # independent): delta-method SE of a ratio of
    se = np.sqrt(np.var(a_u - pooled * L) / len(L)) / L.mean()            # unit-level sums (corrected after the first run)
    res.append(dict(check='pooled', miss=pooled, se=se, ok=abs(pooled - alpha) <= 4 * se)); ok &= abs(pooled - alpha) <= 4 * se
    # alarms simulated with the trigger code on 2,000 units: LB = r + M0 1{zone}, action level act, debounce k
    M0 = act - lead + 1; q = 0.0; fails = 0; sub = L[:2000]
    for Lu in sub:
        r = np.arange(Lu, 0, -1).astype(float); s = q + r + M0 * ((r > lead) & (r <= lead + m))
        j = int(C.first_alarm(C.debounced_running_min(s, k), act + q)[0])
        fails += not (j < len(r) and r[j] > lead)
    res.append(dict(check='alarm', n=len(sub), fail=fails / len(sub), ok=fails == len(sub))); ok &= fails == len(sub)
    # contrast (Theorem 1): CNA calibrated on 200 units of the same law, tested on 2,000
    def V_of(Lu):
        r = np.arange(Lu, 0, -1).astype(float); s = q + r + M0 * ((r > lead) & (r <= lead + m))
        return C.critical_level(C.debounced_running_min(s, k), r, lead)
    H = C.conformal_level(np.array([V_of(Lu) for Lu in L[2000:2200]]), alpha)
    Vt = np.array([V_of(Lu) for Lu in L[2200:4200]]); fc = float((Vt > H).mean())
    res.append(dict(check='cna_contrast', fail=fc, se=np.sqrt(alpha * (1 - alpha) / 2000), ok=fc <= alpha + 4 * np.sqrt(alpha * (1 - alpha) / 2000)))
    pd.DataFrame(res).assign(x=x, m=m, mean_life=float(L.mean())).to_csv(f'{OUT}/S1.csv', index=False)
    save_summary('S1', dict(passed=bool(ok and res[-1]['ok']), x=x, m=m, mean_life=float(L.mean()), cna_fail=fc))
    return res


def s2(N=100_000, n=100, alpha=0.005, lead=10, seed=602):
    """Theorem 5(c): sharpness, P(F) = min(1, alpha n) with pooled miss rate alpha."""
    rng = np.random.default_rng(seed)
    B = rng.random(N) < min(1.0, alpha * n)
    r = np.arange(n, 0, -1).astype(float); jl = int(np.flatnonzero(r > lead)[-1])
    fails = 0
    for b in B[:5000]:                                                    # trigger code on 5,000 units
        LB = r.copy(); LB[jl] = lead + 1 + b
        j = int(C.first_alarm(C.debounced_running_min(LB, 1), lead + 1)[0]); fails += not (j < n and r[j] > lead)
    pf = fails / 5000; se = np.sqrt(0.25 / 5000)
    pooled = B.mean() / n; se_p = np.sqrt(alpha * (1 - alpha) / (N * n))
    ok = abs(pf - min(1, alpha * n)) <= 4 * se and pooled <= alpha + 4 * se_p and abs(pf - B[:5000].mean()) < 1e-12
    pd.DataFrame([dict(pf=pf, se=se, pooled_miss=pooled, ok=ok)]).to_csv(f'{OUT}/S2.csv', index=False)
    save_summary('S2', dict(passed=bool(ok), pf=pf, pooled_miss=pooled))


def s3(N=200_000, nrow=150, lead=10, act=11, alpha=0.10, seed=603):
    """Theorem 5(d): product <= P(F) <= min; m-dependence bound for MA(5) errors."""
    rng = np.random.default_rng(seed)
    r = np.arange(nrow, 0, -1).astype(float); A = np.flatnonzero(r > lead)
    sig = 1.0 + 4.0 * (r <= 30)                                           # heteroscedastic: errors larger near failure
    # q: pooled (1 - alpha) quantile of eps = sig * Z over rows (exact, Gaussian mixture)
    f = lambda q: np.mean(norm.sf(q / sig)) - alpha
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2; lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    q = (lo + hi) / 2
    pi = norm.sf((act - r + q) / sig)                                      # P(LB_j > act), LB_j = r_j + eps_j - q
    res, ok = [], True
    for name, phi, ma in [('AR0', 0.0, 0), ('AR0.5', 0.5, 0), ('AR0.9', 0.9, 0), ('AR0.99', 0.99, 0), ('MA5', None, 5)]:
        Fcount = 0; nb = 0
        for b in range(N // 20000):
            if ma:
                eta = rng.standard_normal((20000, nrow + ma)); Z = np.stack([eta[:, i:i + nrow] for i in range(ma + 1)]).sum(0) / np.sqrt(ma + 1)
            else:
                Z = np.empty((20000, nrow)); Z[:, 0] = rng.standard_normal(20000)
                for j in range(1, nrow): Z[:, j] = phi * Z[:, j - 1] + np.sqrt(1 - phi ** 2) * rng.standard_normal(20000)
            LB = r[None, :] + sig[None, :] * Z - q
            Fcount += np.all(LB[:, A] > act, axis=1).sum(); nb += 20000
        pf = Fcount / nb; se = np.sqrt(max(pf * (1 - pf), 1e-12) / nb)
        prod = float(np.prod(pi[A])); mn = float(np.min(pi[A]))
        row = dict(errors=name, pF=pf, se=se, product=prod, minimum=mn, ok=(pf >= prod - 4 * se) and (pf <= mn + 4 * se))
        if ma:
            jl = A[-1]; idx = jl - np.arange(0, len(A), ma + 1); row['mdep_bound'] = float(np.prod(pi[idx])); row['ok'] &= pf <= row['mdep_bound'] + 4 * se
        res.append(row); ok &= row['ok']
    pd.DataFrame(res).assign(q=q, pooled_coverage=1 - np.mean(norm.sf(q / sig))).to_csv(f'{OUT}/S3.csv', index=False)
    save_summary('S3', dict(passed=bool(ok), rows=res))


# ---------------------------------------------------------------------------------------------- synthetic units
CAP, PHI, SD = 60.0, 0.8, 5.0
SD_ETA = SD * np.sqrt(1 - PHI ** 2)


def gen_units(n, lam, rng, maxlen):
    """lives T ~ Weibull(2, lam) (lam per unit), one row per cycle t = 0..ceil(T)-1, r_t = T - t,
    signal s_t = min(r_t, 60) + AR(1) noise (phi 0.8, sd 5). Rows beyond the life are +inf (padding)."""
    T = lam * rng.weibull(2.0, n)
    t = np.arange(maxlen, dtype=float)
    R = T[:, None] - t[None, :]
    E = np.empty((n, maxlen)); E[:, 0] = SD * rng.standard_normal(n)
    for j in range(1, maxlen): E[:, j] = PHI * E[:, j - 1] + SD_ETA * rng.standard_normal(n)
    S = np.where(R > 0, np.minimum(R, CAP) + E, np.inf)
    return T, S


def runmin(S, k=2):
    """debounced running minimum, row-wise (same definition as core.debounced_running_min)"""
    M = S.copy()
    for i in range(1, k): M[:, i:] = np.maximum(M[:, i:], S[:, :-i])
    M[:, :k - 1] = np.inf
    return np.minimum.accumulate(M, axis=1)


def crit(T, RM, lead):
    """V = rho at the last row with r = T - t > lead (+inf if that row is < k - 1)"""
    j = np.ceil(T - lead).astype(int) - 1
    out = np.full(len(T), np.inf); okj = (j >= 0) & (j < RM.shape[1])
    out[okj] = RM[np.flatnonzero(okj), j[okj]]
    return out


def upper(c, RM, lead):
    """U(c) = rho at the last row with t <= c - lead (+inf if none)"""
    j = np.floor(c - lead).astype(int)
    out = np.full(len(c), np.inf); okj = (j >= 0)
    jj = np.minimum(j, RM.shape[1] - 1)
    out[okj] = RM[np.flatnonzero(okj), jj[okj]]
    return out


def weib_sf(t, lam):
    return np.exp(-(np.maximum(t, 0) / lam) ** 2)


def posterior_p(c, s_obs, lam, c0, step=0.1):
    """P(T <= c0 | T > c, s_0..s_floor(c)) for the generative model of gen_units (exact on a grid)."""
    t = np.arange(len(s_obs), dtype=float); B = np.floor(c) + CAP
    Tg = np.arange(c + step / 2, B, step)

    def loglik(Tv):
        Ev = s_obs[None, :] - np.minimum(Tv[:, None] - t[None, :], CAP)
        inn = Ev[:, 1:] - PHI * Ev[:, :-1]
        return -(Ev[:, 0] ** 2) / (2 * SD ** 2) - (inn ** 2).sum(1) / (2 * SD_ETA ** 2)
    llg = loglik(Tg); ll_inf = float(loglik(np.array([B + 1.0]))[0])
    mx = max(llg.max() if len(llg) else -np.inf, ll_inf)
    dens = (2 * Tg / lam ** 2) * np.exp(-(Tg / lam) ** 2)                 # Weibull(2) density
    wg = dens * np.exp(llg - mx) * step
    tail = np.exp(ll_inf - mx) * weib_sf(B, lam)
    num = wg[Tg <= c0].sum() + (np.exp(ll_inf - mx) * max(weib_sf(B, lam) - weib_sf(c0, lam), 0.0) if c0 > B else 0.0)
    return float(num / (wg.sum() + tail))


def s4(reps=2000, n_cal=200, n_te=1000, c0=100.0, lead=10, alpha=0.10, seed=604):
    """Theorem 6 (E2) and Proposition 3 (IPCW), horizon c0, lives depending on a binary covariate."""
    rng = np.random.default_rng(seed); lam = {0: 120.0, 1: 80.0}; maxlen = int(c0) + 2
    rows = []
    for rep in range(reps):
        X = rng.random(n_cal + n_te) < 0.5; L = np.where(X, lam[1], lam[0])
        T, S = gen_units(n_cal + n_te, L, rng, maxlen); RM = runmin(S, 2)
        V = crit(T, RM, lead); Vo = np.where(T <= c0, V, -np.inf)
        Vc, Vt = Vo[:n_cal], Vo[n_cal:]; Tc, Xc = T[:n_cal], X[:n_cal]
        for mech in ('CAR', 'INF4', 'CARX'):
            if mech == 'CAR': subj = rng.random(n_cal) < 0.5
            elif mech == 'INF4': subj = rng.random(n_cal) < np.where(Tc <= c0, 0.8, 0.2)
            else: subj = rng.random(n_cal) < np.where(Xc, 0.7, 0.3)
            Cc = np.where(subj, rng.uniform(0, c0, n_cal), np.inf)
            cens = Cc < np.minimum(Tc, c0)
            U = upper(np.where(cens, Cc, 0.0), RM[:n_cal], lead)
            p0 = np.zeros(n_cal); pb = np.zeros(n_cal)
            for i in np.flatnonzero(cens):
                lm = lam[int(Xc[i])]; nobs = int(np.floor(Cc[i])) + 1
                p0[i] = posterior_p(Cc[i], S[i, :nobs], lm, c0)
                pb[i] = 1 - weib_sf(c0, lm) / weib_sf(Cc[i], lm)
            levels = {'ORACLE': C.conformal_level(Vc, alpha), 'T2': C.conformal_level(np.where(cens, U, Vc), alpha),
                      'CC': C.conformal_level(Vc[~cens], alpha)}
            u = rng.random(n_cal)
            for G in (1, 4, np.inf):
                for nm, pp in (('E2', p0), ('E2base', pb)):
                    pt = np.ones(n_cal) if np.isinf(G) else G * pp / (1 + (G - 1) * pp)
                    W = np.where(cens, np.where(u <= pt, U, -np.inf), Vc)
                    levels[f'{nm}_G{G:g}'] = C.conformal_level(W, alpha)
            # IPCW with the G of the CAR-X mechanism (known for CARX; misspecified otherwise): G(u|x) = 1 - p_x u / c0
            px = np.where(Xc, 0.7, 0.3) if mech == 'CARX' else np.full(n_cal, 0.5)
            comp = ~cens; Gm = 1 - px * np.minimum(Tc, c0) / c0
            xt = X[n_cal:]; pxt = np.where(xt, 0.7, 0.3) if mech == 'CARX' else np.full(n_te, 0.5)
            wbar = 1 / (1 - pxt)                                               # 1 / G(c0 | x), one level per test covariate
            for xv in (0, 1):
                levels[f'IPCW_x{xv}'] = weighted_level(Vc[comp], 1 / Gm[comp], float(wbar[xt == xv][0]) if (xt == xv).any() else 1.0, alpha)
            for nm, H in levels.items():
                if nm.startswith('IPCW'): continue
                rows.append(dict(rep=rep, mech=mech, method=nm, fail=float((Vt > H).mean()), frac_cens=float(cens.mean())))
            fi = np.where(xt, Vt > levels['IPCW_x1'], Vt > levels['IPCW_x0'])
            rows.append(dict(rep=rep, mech=mech, method='IPCW', fail=float(fi.mean()), frac_cens=float(cens.mean())))
    R = pd.DataFrame(rows); R.to_csv(f'{OUT}/S4.csv.gz', index=False)
    g = R.groupby(['mech', 'method']).fail.agg(['mean', 'std', 'count']).reset_index()
    g['se'] = g['std'] / np.sqrt(g['count']); g['ok'] = g['mean'] <= alpha + 4 * g['se']
    g.to_csv(f'{OUT}/S4_summary.csv', index=False)
    need = [('CAR', 'E2_G1'), ('INF4', 'E2_G4'), ('INF4', 'E2_Ginf'), ('CAR', 'T2'), ('INF4', 'T2'), ('CARX', 'IPCW')]
    passed = all(bool(g[(g.mech == a) & (g.method == b)].ok.iloc[0]) for a, b in need)
    save_summary('S4_S5', dict(passed=passed, table=g.round(4).to_dict('records')))


def s6(reps=2000, n_cal=200, n_te=1000, lead=10, alpha=0.10, shift=0.8, seed=606):
    """Theorem 7: covariate shift N(0,1) -> N(0.8,1), lives Weibull(2, 100 exp(0.3 x)); lifetime target."""
    rng = np.random.default_rng(seed); maxlen = 700
    w = lambda x: np.exp(shift * x - shift ** 2 / 2)                      # dQ_X/dP_X
    rows = []
    for rep in range(reps):
        Xs = rng.standard_normal(n_cal); Xq = shift + rng.standard_normal(n_te)
        Ts, Ss = gen_units(n_cal, 100 * np.exp(0.3 * Xs), rng, maxlen); Vs = crit(Ts, runmin(Ss), lead)
        Tq, Sq = gen_units(n_te, 100 * np.exp(0.3 * Xq), rng, maxlen); Vq = crit(Tq, runmin(Sq), lead)
        Tc, Sc = gen_units(n_te, 70 * np.exp(0.3 * Xq), rng, maxlen); Vcs = crit(Tc, runmin(Sc), lead)   # concept shift
        # estimated weights from independent samples (500 source, 500 target covariates)
        a_s = rng.standard_normal(500); a_t = shift + rng.standard_normal(500)
        lr = LogisticRegression(C=1e6).fit(np.r_[a_s, a_t][:, None], np.r_[np.zeros(500), np.ones(500)])
        wh = lambda x: np.exp(lr.intercept_[0] + lr.coef_[0, 0] * x)
        big = rng.standard_normal(200000); norm_c = wh(big).mean(); whn = lambda x: wh(x) / norm_c
        gap = 0.5 * np.mean(np.abs(whn(big) - w(big)))
        H_un = C.conformal_level(Vs, alpha)
        Hk = weighted_levels(Vs, w(Xs), w(Xq), alpha); He = weighted_levels(Vs, whn(Xs), whn(Xq), alpha)
        fw = Vq > Hk; fe = Vq > He; fc = Vcs > Hk
        ess = w(Xs).sum() ** 2 / (w(Xs) ** 2).sum()
        rows.append(dict(rep=rep, unweighted=float((Vq > H_un).mean()), weighted_known=float(fw.mean()), weighted_est=float(fe.mean()),
                         concept_shift=float(fc.mean()), gap_bound=gap, ess=ess))
    R = pd.DataFrame(rows); R.to_csv(f'{OUT}/S6.csv.gz', index=False)
    se = R.std() / np.sqrt(len(R)); mn = R.mean()
    ok_known = mn.weighted_known <= alpha + 4 * se.weighted_known
    ok_est = mn.weighted_est <= alpha + mn.gap_bound + 4 * se.weighted_est
    save_summary('S6', dict(passed=bool(ok_known and ok_est), mean=mn.round(4).to_dict(), se=se.round(5).to_dict()))


if __name__ == '__main__':
    which = sys.argv[1:] or ['S1', 'S2', 'S3', 'S4', 'S6']
    for w_ in which:
        t0 = time.time(); {'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4, 'S6': s6}[w_](); print(w_, f'{time.time() - t0:.0f}s', flush=True)
    print(json.dumps({k: v.get('passed') for k, v in json.load(open(SUMMARY)).items()}, indent=1))


def s7_exploratory(reps=300, n_cal=1000, n_te=2000, c0=100.0, lead=10, seed=607):
    """EXPLORATORY (added after the protocol-v6 results were seen; not a pre-specified check): how the gain of E2 over
    Theorem 2 depends on the probability of failing within the horizon. Lives Weibull(2, lam) with lam chosen for
    horizon-failure probabilities of about 2, 6, 22 and 63 %; censoring at random (subject w.p. 0.5, C ~ U(0, c0));
    alpha = half the horizon-failure probability (a 50 % recall target)."""
    rng = np.random.default_rng(seed); rows = []; maxlen = int(c0) + 2
    for lam in (800.0, 400.0, 200.0, 100.0):
        pf = 1 - weib_sf(c0, lam); alpha = pf / 2
        for rep in range(reps):
            T, S = gen_units(n_cal + n_te, np.full(n_cal + n_te, lam), rng, maxlen); RM = runmin(S, 2)
            Vo = np.where(T <= c0, crit(T, RM, lead), -np.inf); Vc, Vt, Tc = Vo[:n_cal], Vo[n_cal:], T[:n_cal]
            subj = rng.random(n_cal) < 0.5; Cc = np.where(subj, rng.uniform(0, c0, n_cal), np.inf)
            cens = Cc < np.minimum(Tc, c0); U = upper(np.where(cens, Cc, 0.0), RM[:n_cal], lead)
            p0 = np.zeros(n_cal)
            for i in np.flatnonzero(cens):
                p0[i] = posterior_p(Cc[i], S[i, :int(np.floor(Cc[i])) + 1], lam, c0)
            u = rng.random(n_cal); comp = ~cens; Gm = 1 - 0.5 * np.minimum(Tc, c0) / c0
            lv = {'ORACLE': C.conformal_level(Vc, alpha), 'T2': C.conformal_level(np.where(cens, U, Vc), alpha),
                  'CC': C.conformal_level(Vc[comp], alpha), 'IPCW': weighted_level(Vc[comp], 1 / Gm[comp], 2.0, alpha),
                  'E2_G1': C.conformal_level(np.where(cens, np.where(u <= p0, U, -np.inf), Vc), alpha),
                  'E2_G4': C.conformal_level(np.where(cens, np.where(u <= 4 * p0 / (1 + 3 * p0), U, -np.inf), Vc), alpha)}
            RMt = RM[n_cal:]
            for nm, H in lv.items():
                J = np.argmax(RMt <= H, axis=1); has = (RMt <= H).any(1)
                rows.append(dict(lam=lam, p_horizon=pf, alpha=alpha, rep=rep, method=nm, fail=float((Vt > H).mean()),
                                 first_row=float((has & (J <= 1)).mean()), inf_level=float(np.isinf(H)), frac_cens=float(cens.mean())))
    R = pd.DataFrame(rows); R.to_csv(f'{OUT}/S7_exploratory.csv.gz', index=False)
    g = R.groupby(['lam', 'p_horizon', 'alpha', 'method'])[['fail', 'first_row', 'inf_level']].mean().reset_index()
    g.to_csv(f'{OUT}/S7_exploratory_summary.csv', index=False)
    return g
