"""Field-data analysis of protocol_v6.md, Part 5 (H19-H23). NOT RUN in this repository: the data could not be
downloaded (network policy). Tested on a synthetic code fixture only (test_field.py); expect data-specific fixes when
it first meets the real files, each to be logged as a deviation.

Implemented: GBM signal on min(r, cap) and a hazard learner (alarm on -p); CNA at alpha with the censoring treatments
T2 (Theorem 2), CC, IPCW (Proposition 3, reverse Kaplan-Meier within strata), E2-BASE and E2-HIST (Theorem 6,
Gamma in {1, 2, 4, inf}); per-cycle LB-all / LB-near (debounce 1, action l + spacing); within-cohort five-fold
cross-validation (60/40 proper-training/calibration); cohort -> next cohort with and without Theorem 7 weights.
Not implemented (needs the data to fix design details, see protocol): CNA-DA, CNA+, online updating with weekly feedback.

Usage:
    python -m field.run_field backblaze 2022 2023       # within-cohort CV on 2022 and 2022 -> 2023
    python -m field.run_field scania
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import core as C
import policies as P
from run6_cens import km_censoring, weighted_level, row_features
from field.records import targets, horizon_status

OUT = os.path.join(C.ROOT, 'results', 'field'); os.makedirs(OUT, exist_ok=True)
GAMMAS = [1.0, 2.0, 4.0, np.inf]


def fit_signals(train, cap, lead, horizon_w, seed=0, neg_frac=0.1):
    """GBM regression on min(r, cap) (rows with known target; rows with target = cap subsampled with weights) and a
    hazard classifier of failure within lead + horizon_w (rows with known label; negatives subsampled with weights)."""
    rng = np.random.default_rng(seed); Xr, yr, wr, Xh, yh, wh = [], [], [], [], [], []
    for u in train:
        y, known = targets(u, cap); keep = known & ((y < cap) | (rng.random(len(y)) < neg_frac))
        Xr.append(u['X'][keep]); yr.append(y[keep]); wr.append(np.where(y[keep] < cap, 1.0, 1.0 / neg_frac))
        if u['event']:
            lab = (u['T'] - u['t']) <= lead + horizon_w; kn = np.ones(len(lab), bool)
        else:
            lab = np.zeros(len(u['t']), bool); kn = (u['c'] - u['t']) > lead + horizon_w
        keep = kn & (lab | (rng.random(len(lab)) < neg_frac))
        Xh.append(u['X'][keep]); yh.append(lab[keep]); wh.append(np.where(lab[keep], 1.0, 1.0 / neg_frac))
    Xr, yr, wr = np.vstack(Xr), np.concatenate(yr), np.concatenate(wr)
    reg = C.gbm(Xr, yr, cap).fit(Xr, np.minimum(yr, cap), sample_weight=wr)
    clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                         random_state=0).fit(np.vstack(Xh), np.concatenate(yh), sample_weight=np.concatenate(wh))
    return dict(GBM=lambda X: reg.predict(X), HAZ=lambda X: -clf.predict_proba(X)[:, 1])


def unit_levels(u, s, k, lead, c0):
    """running minimum; V (event within the horizon), U(c) (censored before the horizon), horizon status"""
    rm = C.debounced_running_min(s, k); st = horizon_status(u, c0)
    if st == 'fail':
        r = u['T'] - u['t']; return rm, st, C.critical_level(rm, r, lead), np.nan
    if st == 'survive': return rm, st, -np.inf, np.nan
    ok = np.flatnonzero(u['t'] <= u['c'] - lead)
    return rm, st, np.nan, (float(rm[ok[-1]]) if len(ok) else np.inf)


def cna_levels(cal, Sc, k, lead, c0, alpha, cap, p_base, p_hist, strata, rng):
    """levels of every censoring treatment for one calibration set"""
    info = [unit_levels(u, s, k, lead, c0) for u, s in zip(cal, Sc)]
    st = np.array([i[1] for i in info]); V = np.array([i[2] for i in info], float); U = np.array([i[3] for i in info], float)
    cens = st == 'censored'; Vo = np.where(cens, np.nan, V)
    lv = {'T2': C.conformal_level(np.where(cens, U, Vo), alpha),
          'CC': C.conformal_level(Vo[~cens], alpha) if (~cens).any() else np.inf}
    # IPCW within strata: reverse Kaplan-Meier of the censoring times per stratum
    obs = np.array([u['c'] if s_ == 'censored' else min(u['T'] if u['event'] else u['c'], c0) for u, s_ in zip(cal, st)])
    w = np.zeros(len(cal)); wbar = {}
    for g in np.unique(strata):
        m = strata == g; G = km_censoring(obs[m], cens[m])
        w[m & ~cens] = 1.0 / np.maximum(G(obs[m & ~cens]), 1e-300); g0 = float(G([c0])[0]); wbar[g] = (1 / g0) if g0 > 0 else np.inf
    lv['IPCW'] = {g: weighted_level(Vo[~cens], w[~cens], wbar[g], alpha) for g in wbar}
    u_ = rng.random(len(cal))
    for G_ in GAMMAS:
        for nm, pp in (('E2BASE', p_base), ('E2HIST', p_hist)):
            pt = np.ones(len(cal)) if np.isinf(G_) else G_ * pp / (1 + (G_ - 1) * pp)
            lv[f'{nm}_G{G_:g}'] = C.conformal_level(np.where(cens, np.where(u_ <= pt, U, -np.inf), Vo), alpha)
    return lv, info


def horizon_probs(fit_units, cal, info, Sc_fit, Sc_cal, k, c0, cap, strata_fit, strata_cal):
    """p0 for censored calibration units: E2-BASE from Kaplan-Meier of failure times within stratum; E2-HIST from an
    IPCW-weighted landmark logistic regression on (t/c0, rho_t/cap, s_t/cap) over proper-training rows."""
    pb = np.zeros(len(cal)); ph = np.zeros(len(cal))
    T_obs = np.array([u['T'] if u['event'] else u['c'] for u in fit_units]); ev = np.array([u['event'] for u in fit_units])
    for i, (u, inf) in enumerate(zip(cal, info)):
        if inf[1] != 'censored': continue
        m = strata_fit == strata_cal[i]
        S = km_censoring(T_obs[m], ev[m])                                   # same estimator, events = failures
        sc, s0 = float(S([u['c']])[0]), float(S([c0])[0]); pb[i] = 1 - s0 / sc if sc > 0 else 1.0
    Gc = km_censoring(T_obs, ~ev)
    feats, labs, wts = [], [], []
    for u, s in zip(fit_units, Sc_fit):
        st = horizon_status(u, c0)
        if st == 'censored': continue
        lab = float(st == 'fail'); rm = C.debounced_running_min(s, k); end = min(u['T'] if u['event'] else c0, c0)
        ok = u['t'] < c0
        if ok.any():
            feats.append(row_features(u['t'][ok], rm[ok], s[ok], c0, cap)); labs.append(np.full(ok.sum(), lab))
            wts.append(Gc(u['t'][ok]) / max(float(Gc([end])[0]), 1e-12))
    if feats and 0 < np.concatenate(labs).mean() < 1:
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(np.vstack(feats), np.concatenate(labs), sample_weight=np.concatenate(wts))
        for i, (u, s, inf) in enumerate(zip(cal, Sc_cal, info)):
            if inf[1] != 'censored': continue
            j = np.flatnonzero(u['t'] <= u['c'])
            if not len(j): ph[i] = pb[i]; continue
            rm = C.debounced_running_min(s, k); j = j[-1]
            ph[i] = lr.predict_proba(row_features(u['t'][j:j + 1], rm[j:j + 1], s[j:j + 1], c0, cap))[0, 1]
    else:
        ph = pb.copy()
    return pb, ph


def evaluate(tes, St, k, lead, c0, H, strata_te=None):
    """failure within the horizon before the planned replacement on test units: complete units exactly; censored
    units are a success if an alarm left lead time before c, otherwise unknown (reported as bounds)."""
    rows = []
    for i, (u, s) in enumerate(zip(tes, St)):
        h = H[strata_te[i]] if isinstance(H, dict) else H
        rm, st, V, Uc = unit_levels(u, s, k, lead, c0); j = int(C.first_alarm(rm, h)[0]) if np.isfinite(h) else 0
        alarm_t = u['t'][j] if j < len(u['t']) else np.inf
        if st == 'fail': out = float(V > h)
        elif st == 'survive': out = 0.0
        else: out = 0.0 if alarm_t + lead <= u['c'] else np.nan
        rows.append(dict(unit=u['unit'], status=st, fail=out, alarm_t=alarm_t, false_alarm=float(st == 'survive' and alarm_t < c0)))
    return pd.DataFrame(rows)


def run_design(units, cal_idx, fit_idx, te_idx, c0, cap, lead, spacing, alphas, k=2, seed=0, horizon_w=30):
    fit = [units[i] for i in fit_idx]; cal = [units[i] for i in cal_idx]; tes = [units[i] for i in te_idx]
    sig = fit_signals(fit, cap, lead, horizon_w, seed)
    strata = lambda us: np.array([u['group'] for u in us])
    res = []
    for ln, f in sig.items():
        Sf = [C.smooth(f(u['X']), 3) for u in fit]; Sc = [C.smooth(f(u['X']), 3) for u in cal]; St = [C.smooth(f(u['X']), 3) for u in tes]
        info0 = [unit_levels(u, s, k, lead, c0) for u, s in zip(cal, Sc)]
        pb, ph = horizon_probs(fit, cal, info0, Sf, Sc, k, c0, cap, strata(fit), strata(cal))
        for a in alphas:
            lv, _ = cna_levels(cal, Sc, k, lead, c0, a, cap, pb, ph, strata(cal), np.random.default_rng(seed + 1))
            for meth, H in lv.items():
                E = evaluate(tes, St, k, lead, c0, H, strata(tes))
                res.append(dict(learner=ln, alpha=a, method=meth, n_cal=len(cal), n_cal_censored=int(sum(i[1] == 'censored' for i in info0)),
                                fail_complete=E[E.status != 'censored'].fail.mean(), fail_upper=E.fail.fillna(1).mean(),
                                fail_lower=E.fail.fillna(0).mean(), false_alarm=E[E.status == 'survive'].false_alarm.mean(),
                                n_test=len(E), n_test_fail=int((E.status == 'fail').sum())))
            if ln == 'GBM':                                                     # per-cycle bounds, K1
                for near in (False, True):
                    cal_r = [dict(r=np.minimum(targets(u, cap)[0], cap)) for u in cal]
                    e = np.concatenate([(s - np.where(np.isnan(cr['r']), np.nan, cr['r']))[~np.isnan(cr['r']) & ((cr['r'] < cap) if near else True)]
                                        for s, cr in zip(Sc, cal_r)])
                    e = np.sort(e[np.isfinite(e)]); kq = int(np.ceil((len(e) + 1) * (1 - a)))
                    q = np.inf if kq > len(e) else e[kq - 1]
                    E = evaluate(tes, [s - q for s in St], 1, lead, c0, lead + spacing)
                    res.append(dict(learner='GBM', alpha=a, method=f'LB{"near" if near else "all"}_K1', fail_complete=E[E.status != 'censored'].fail.mean(),
                                    fail_upper=E.fail.fillna(1).mean(), fail_lower=E.fail.fillna(0).mean(), n_test=len(E)))
    return pd.DataFrame(res)


def shift_weights(src, tgt, ev_idx, fit_idx):
    """Theorem 7 weights: logistic regression of target (fit half) vs source covariates"""
    Zs = np.array([u['Z'] for u in src]); Zt = np.array([tgt[i]['Z'] for i in fit_idx])
    sc = StandardScaler().fit(np.r_[Zs, Zt])
    lr = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(np.r_[Zs, Zt]), np.r_[np.zeros(len(Zs)), np.ones(len(Zt))])
    return lambda Z: np.exp(lr.decision_function(sc.transform(np.atleast_2d(Z)))) * len(Zs) / len(Zt)


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'backblaze':
        from field import backblaze as B
        from field.records import causal_features
        y0 = int(sys.argv[2])
        D = B.load_year(y0); U, D = B.cohort_units(D, y0)
        print('cohort', y0, 'drives', len(U), 'failures', int(U.event.sum()), 'removed', int(U.censored_before_end.sum()),
              'excluded (report after failure)', int(U.after_failure.sum()))
        raise SystemExit('per-drive record construction and the runs are to be executed once the data are verified')
    elif which == 'scania':
        from field import scania as S
        print(S.describe())
        raise SystemExit('runs are to be executed once the data are verified')
