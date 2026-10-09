"""Generic per-unit records for field data with irregular observation times and right censoring (protocol_v6.md,
Part 5). NOT RUN on field data in this repository (data unreachable); tested on code fixtures only (test_field.py).

A unit record is a dict:
    unit, group (stratum, e.g. drive model), Z (baseline covariates for Theorem 7 weights),
    t (observation times since the cohort start, increasing), X (causal features, one row per time),
    event (True if the failure was observed), T (failure time, nan if censored), c (last time seen alive, nan if event)
"""
import numpy as np, pandas as pd


def causal_features(t, raw, windows=(7, 30), spans=(7, 30)):
    """features at row i from rows <= i only: log1p of the raw counters (forward-filled within the unit), their
    increase over the last w time units for each window, EWMs of the per-row increments, and the elapsed time."""
    t = np.asarray(t, float); A = pd.DataFrame(np.asarray(raw, float)).ffill().fillna(0.0).values
    A = np.sign(A) * np.log1p(np.abs(A))
    out = [A]
    for w in windows:
        j = np.searchsorted(t, t - w, side='right') - 1            # last row at least w time units earlier
        base = np.where((j >= 0)[:, None], A[np.maximum(j, 0)], A[0][None, :])
        out.append(A - base)
    inc = np.vstack([np.zeros((1, A.shape[1])), np.diff(A, axis=0)])
    for s in spans:
        out.append(pd.DataFrame(inc).ewm(span=s).mean().values)
    out.append(t[:, None])
    return np.hstack(out)


def truncation_test(t, raw, n_checks=20, seed=0):
    """features of the rows up to a random cut must not change when the later rows are removed"""
    rng = np.random.default_rng(seed); full = causal_features(t, raw)
    for _ in range(n_checks):
        k = int(rng.integers(1, len(t) + 1))
        part = causal_features(t[:k], np.asarray(raw)[:k])
        if not np.allclose(full[:k], part, equal_nan=True): return False
    return True


def targets(u, cap):
    """regression target min(r, cap) where known: event units all rows; censored units rows with c - t >= cap"""
    if u['event']:
        r = u['T'] - u['t']; return np.minimum(r, cap), np.ones(len(r), bool)
    known = (u['c'] - u['t']) >= cap
    return np.where(known, cap, np.nan), known


def horizon_status(u, c0):
    """'fail' (event within the horizon), 'survive' (observed through the horizon), or 'censored' (lost before it)"""
    if u['event'] and u['T'] <= c0: return 'fail'
    if (u['event'] and u['T'] > c0) or (not u['event'] and u['c'] >= c0): return 'survive'
    return 'censored'
