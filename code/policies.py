"""Alarm policies. Every policy is fitted/selected on CALIBRATION units only (models are fitted on proper-training units
elsewhere), and returns, for each target unit, the row index of its first alarm (len(r) = no alarm) and the reported
estimate at the alarm where the policy reports one.

Conventions (all quantities in physical cycles):
  series  dict(r=cycles left per row (>= 1), t=cycle per row, L=life, s=signal per row)
  alarm   at row j: planned replacement at t[j] + lead if r[j] > lead (cost 1, use t[j] + lead);
          otherwise, or if no alarm, failure at L (cost rho, use L).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from core import smooth, debounced_running_min, first_alarm, critical_level, conformal_level

SPANS = (1, 3, 6)
KS = (1, 2, 3, 5)


# ----------------------------------------------------------------------------------------------- generic tables
def alarm_rows(sig, k, H):
    """sig: list of signal arrays; returns (n_units x n_H) array of first-alarm rows (len = none)"""
    return np.vstack([first_alarm(debounced_running_min(s, k), H) for s in sig])


def outcome(J, ser, lead):
    """J: (n x m) alarm rows; returns success (planned before failure), use, notice (nan if none)"""
    n, m = J.shape; succ = np.zeros((n, m), bool); use = np.zeros((n, m)); notice = np.full((n, m), np.nan)
    for i, d in enumerate(ser):
        j = J[i]; has = j < len(d['r']); jj = np.minimum(j, len(d['r']) - 1)
        rr = np.where(has, d['r'][jj], np.nan); notice[i] = rr
        ok = has & (rr > lead); succ[i] = ok
        use[i] = np.where(ok, d['t'][jj] + lead, d['L'])
    return succ, use, notice


def cost_rate(succ, use, rho, w=None):
    c = np.where(succ, 1.0, rho)
    if w is None: return c.sum(0) / use.sum(0)
    return (w @ c) / (w @ use)


def boot_weights(n, seed, B=200):
    rng = np.random.default_rng(seed)
    return np.vstack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(float)


def select_cost(succ, use, notice, rho, rule, W):
    """index of the selected setting: plain minimum of the calibration cost rate, or one-standard-error rule (among the
    settings within one bootstrap SE of the minimum, the one with the largest mean notice; failures count as 0 notice)"""
    cr = cost_rate(succ, use, rho); j0 = int(np.argmin(cr))
    if rule == 'plain': return j0
    se = np.std(cost_rate(succ[:, [j0]], use[:, [j0]], rho, W))
    cand = np.flatnonzero(cr <= cr[j0] + se)
    mn = np.where(succ, np.nan_to_num(notice), 0.0).mean(0)
    return int(cand[np.argmax(mn[cand])])


# ----------------------------------------------------------------------------------------------- window-tuned STW
def window_select(cal, preds, H, lo, hi, shifts, caps, cap_margin):
    """cal: list of calibration series (r, t, L); preds: {cap: list of raw predictions aligned with cal}.
    Chooses cap, span, debounce and shift maximising the number of correct first warnings (ties: smaller |shift|)."""
    best = None
    for c in caps:
        if c < H + cap_margin: continue
        for sp in SPANS:
            S = [smooth(p, sp) for p in preds[c]]
            for k in KS:
                J = alarm_rows(S, k, H - shifts)                       # threshold on the unshifted signal
                tot = np.zeros(len(shifts))
                for i, d in enumerate(cal):
                    j = J[i]; has = j < len(d['r']); jj = np.minimum(j, len(d['r']) - 1)
                    e = S[i][jj] + shifts - d['r'][jj]
                    tot += has & (e >= lo) & (e <= hi)
                sc = tot - 1e-3 * np.abs(shifts) / np.abs(shifts).max()
                j = int(np.argmax(sc))
                if best is None or sc[j] > best[0]: best = (sc[j], c, sp, k, float(shifts[j]), int(tot[j]))
    return dict(cap=best[1], span=best[2], k=best[3], shift=best[4], cal_correct=best[5])


def window_apply(preds_c, ser, H, sel):
    """alarm rows, reported (smoothed, shifted) estimate and the largest one-step drop of the shifted signal over the
    k rows ending at the alarm (for the structural lemma)"""
    out = []
    for p, d in zip(preds_c, ser):
        s = smooth(p, sel['span']) + sel['shift']; rm = debounced_running_min(s, sel['k'])
        j = int(first_alarm(rm, H)[0])
        if j < len(s):
            a = max(1, j - sel['k'] + 1); drops = s[a - 1:j] - s[a:j + 1]
            out.append(dict(row=j, est=float(s[j]), maxdrop=float(drops.max()) if len(drops) else np.nan, immediate=j == sel['k'] - 1))
        else:
            out.append(dict(row=j, est=np.nan, maxdrop=np.nan, immediate=False))
    return out


def constant_select(cal_alarms, cal, H, lo, hi, step):
    """one constant reported at the alarm times of the window-tuned policy, maximising correct calibration warnings"""
    grid = np.arange(H - 15 * step, H + 5 * step + 1e-9, 0.5 * step)
    r = np.array([d['r'][a['row']] if a['row'] < len(d['r']) else np.nan for a, d in zip(cal_alarms, cal)])
    e = grid[None, :] - r[:, None]; n = np.nansum((e >= lo) & (e <= hi), 0) - 1e-3 * np.abs(grid - H) / step
    return float(grid[int(np.argmax(n))])


# ----------------------------------------------------------------------------------------------- conformal notice alarm
def cna_levels(S_cal, cal, k, lead):
    """critical levels V_i of the calibration units for a given lead time"""
    return np.array([critical_level(debounced_running_min(s, k), d['r'], lead) for s, d in zip(S_cal, cal)])


def cna_da(V, S_cal, cal, k, lead, rho, alpha_max):
    """decision-aware choice among the order statistics V_(k), k >= ceil((n+1)(1-alpha_max)): minimise the calibration
    cost rate in which the failure probability is replaced by its conformal bound (n+1-k)/(n+1). Returns (H, k, bound)."""
    n = len(V); Vs = np.sort(V); kmin = int(np.ceil((n + 1) * (1 - alpha_max)))
    if kmin > n: return np.inf, n + 1, 1.0 / (n + 1)
    ks = np.arange(kmin, n + 1); Hs = Vs[ks - 1]
    fin = np.isfinite(Hs); ks, Hs = ks[fin], Hs[fin]
    if not len(ks): return np.inf, n + 1, 0.0
    J = alarm_rows(S_cal, k, Hs); succ, use, _ = outcome(J, cal, lead)
    bnd = (n + 1 - ks) / (n + 1)
    obj = (1.0 + (rho - 1.0) * bnd) / use.mean(0)
    i = int(np.argmin(obj))
    return float(Hs[i]), int(ks[i]), float(bnd[i])


def lb_quantile(S_cal, cal, cap, alpha, near):
    """per-cycle split-conformal lower bound: (1-alpha) quantile of the residual s_t - min(r_t, cap) over calibration
    rows (all rows, or rows with r_t <= cap)"""
    e = np.concatenate([s - np.minimum(d['r'], cap) if not near else (s - d['r'])[d['r'] <= cap] for s, d in zip(S_cal, cal)])
    e = np.sort(e); N = len(e); kq = int(np.ceil((N + 1) * (1 - alpha)))
    return np.inf if kq > N else float(e[kq - 1])


# ----------------------------------------------------------------------------------------------- Kamariotis et al. policy 1 (P-res)
class PRes:
    """empirical distribution of the true remaining life in equal-count bins (>= 200 rows) of the raw cap-60 estimate,
    monotonicity in the estimate enforced by isotonic regression; probability of failure before the next decision
    (r <= lead + dt), as in policy 1 of Kamariotis et al. adapted to per-row decisions with a lead time"""
    def __init__(self, P_cal, cal, lead, dt, min_rows=200):
        y = np.concatenate(P_cal); r = np.concatenate([d['r'] for d in cal])
        nb = max(1, len(y) // min_rows); edges = np.unique(np.quantile(y, np.linspace(0, 1, nb + 1)[1:-1]))
        b = np.searchsorted(edges, y, side='right'); nbin = len(edges) + 1
        m = np.array([y[b == i].mean() if (b == i).any() else np.nan for i in range(nbin)])
        pr = np.array([(r[b == i] <= lead + dt).mean() if (b == i).any() else np.nan for i in range(nbin)])
        w = np.array([(b == i).sum() for i in range(nbin)]); ok = w > 0
        iso = IsotonicRegression(increasing=False).fit(m[ok], pr[ok], sample_weight=w[ok])
        self.edges = edges; self.p = np.full(nbin, 0.0); self.p[ok] = iso.predict(m[ok])
        for i in range(nbin):                                                      # empty bins: neighbour value
            if not ok[i]: self.p[i] = self.p[i - 1] if i else self.p[ok][0]

    def prob(self, y):
        return self.p[np.searchsorted(self.edges, y, side='right')]


def pres_rows(model, P, thr):
    """first row at which the failure probability reaches each threshold in thr"""
    out = []
    for p in P:
        pr = model.prob(p); cm = np.maximum.accumulate(pr)
        out.append(np.searchsorted(cm, np.atleast_1d(thr), side='left'))
    return np.vstack(out)


def pres_optimise(model, P_cal, cal, lead, rho):
    thr = np.unique(model.p[model.p > 0])
    if not len(thr): return 1.0
    J = pres_rows(model, P_cal, thr); succ, use, _ = outcome(J, cal, lead)
    cr = cost_rate(succ, use, rho); m = np.isclose(cr, cr.min(), rtol=1e-12, atol=1e-15)
    idx = np.flatnonzero(m); return float(thr[idx[len(idx) // 2]])          # middle of the tied range


# ----------------------------------------------------------------------------------------------- age replacement and trend
def age_T(L_train, rho):
    L = np.asarray(L_train, float); Ts = np.arange(1, int(L.max()) + 1)
    cr = np.array([np.where(L > T, 1.0, rho).sum() / np.minimum(L, T).sum() for T in Ts])
    return float(Ts[int(np.argmin(cr))])


def trend_rul(q, w, kappa, eol=0.88):
    """causal least-squares slope of block-mean discharge capacity over the last w rows, extrapolated to the end-of-life
    capacity 0.88 Ah; remaining life in cycles (capped at 3000); as in the original capacity-trend baseline"""
    n = len(q); out = np.full(n, 3000.0); x = np.arange(n, dtype=float)
    S1 = np.concatenate([[0], np.cumsum(x)]); S2 = np.concatenate([[0], np.cumsum(x * x)])
    Q1 = np.concatenate([[0], np.cumsum(q)]); XQ = np.concatenate([[0], np.cumsum(x * q)])
    i = np.arange(n); a = np.maximum(0, i - w + 1); m = (i - a + 1).astype(float)
    sx = S1[i + 1] - S1[a]; sxx = S2[i + 1] - S2[a]; sq = Q1[i + 1] - Q1[a]; sxq = XQ[i + 1] - XQ[a]
    with np.errstate(invalid='ignore', divide='ignore'):
        sl = (sxq - sx * sq / m) / (sxx - sx * sx / m)
        val = np.where((m >= 5) & (sl < 0), np.minimum(3000.0, np.maximum(0.0, (q - eol) / (-sl)) * kappa), 3000.0)
    out[:] = val
    return out
