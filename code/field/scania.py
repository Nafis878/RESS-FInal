"""SCANIA Component X (SND 2024-34, doi:10.5878/jvb5-d390): per-vehicle records (protocol_v6.md, Part 5).

NOT RUN in this repository: the data hosts were unreachable (network policy). Expected files in data/field/scania/raw
(names as described in the dataset paper, arXiv:2401.15199; verified at load time):
    train_operational_readouts.csv  vehicle_id, time_step, anonymised counters and histogram bins
    train_tte.csv                   vehicle_id, length_of_study_time_step, in_study_repair
    train_specifications.csv        vehicle_id, Spec_0 ... Spec_7 (categorical)
Failure = in_study_repair == 1 at length_of_study_time_step; censoring = in_study_repair == 0 (in service at the end
of its study). Rows at or after the failure time are dropped (remaining life must be positive).
Usage: python -m field.scania describe
"""
import os, sys
import numpy as np, pandas as pd
from field.records import causal_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, 'data', 'field', 'scania', 'raw')


def load():
    R = pd.read_csv(os.path.join(RAW, 'train_operational_readouts.csv'))
    E = pd.read_csv(os.path.join(RAW, 'train_tte.csv')).set_index('vehicle_id')
    Sp = pd.read_csv(os.path.join(RAW, 'train_specifications.csv')).set_index('vehicle_id')
    need = {'vehicle_id', 'time_step'}
    assert need <= set(R.columns), f'unexpected readout columns: {list(R.columns)[:10]}'
    assert {'length_of_study_time_step', 'in_study_repair'} <= set(E.columns), f'unexpected tte columns: {list(E.columns)}'
    return R, E, Sp


def records(windows=(6, 24), spans=(6, 24)):
    R, E, Sp = load()
    feat_cols = [c for c in R.columns if c not in ('vehicle_id', 'time_step')]
    spec_d = pd.get_dummies(Sp.astype(str), drop_first=False).astype(float)
    units = []
    for v, g in R.sort_values(['vehicle_id', 'time_step']).groupby('vehicle_id'):
        if v not in E.index: continue
        L, ev = float(E.length_of_study_time_step[v]), bool(E.in_study_repair[v] == 1)
        g = g[g.time_step < L] if ev else g[g.time_step <= L]
        if not len(g): continue
        t = g.time_step.values.astype(float)
        units.append(dict(unit=v, group=str(Sp.loc[v].iloc[0]) if v in Sp.index else 'NA',
                          Z=spec_d.loc[v].values if v in spec_d.index else np.zeros(spec_d.shape[1]),
                          t=t, X=causal_features(t, g[feat_cols].values, windows, spans), raw=g[feat_cols].values,
                          event=ev, T=L if ev else np.nan, c=np.nan if ev else L))
    return units


def describe():
    R, E, Sp = load()
    return dict(vehicles_readouts=int(R.vehicle_id.nunique()), readouts=len(R), vehicles_tte=len(E),
                repairs=int((E.in_study_repair == 1).sum()), censored=int((E.in_study_repair == 0).sum()),
                study_length_median=float(E.length_of_study_time_step.median()), feature_columns=len(R.columns) - 2)


if __name__ == '__main__':
    if sys.argv[1] == 'describe': print(describe())
