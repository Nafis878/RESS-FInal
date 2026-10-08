# Conformal notice-calibrated end-of-life alarms (RESS rewrite)

Rewrite of the evaluation study *"What does an end-of-life alarm score measure?"* (original manuscript in
`original_reference/`) around one new contribution: an end-of-life alarm whose warning level is calibrated by split
conformal prediction on the **notice at the alarm**, with a finite-sample guarantee
P(remaining life at the alarm > lead time) ≥ 1 − α under exchangeability, a decision-aware variant that keeps the guarantee,
and a supporting bound (with an equivalence test) on what the alarm-window score of a threshold-crossing alarm measures.

| Path | Content |
|---|---|
| `paper/manuscript.pdf`, `paper/manuscript.docx` | Manuscript (built from `paper/sections/*.md` by `paper/build.py`; assembled source `paper/manuscript.md`) |
| `protocol.md`, `protocol.sha256` | Protocol written and hashed before the final experiments (not a registry preregistration) |
| `DECISIONS.md` | Choice of contribution, what worked, what failed, remaining weaknesses, self-assessed rating |
| `code/` | `core.py` (data, features, trigger), `policies.py` (all policies), `run.py` (experiments), `analyze.py` (tables, hypothesis decisions), `stats.py`, `figures.py`, `test_core.py` |
| `results/` | CSV tables (`validity.csv`, `equivalence.csv`, `cost_*.csv`, `transfer_groups.csv`, `pilot_summary.csv`, `hypotheses.json`); `raw/` per-unit outputs; `logs/` |
| `figures/` | Figures 1–5 (PNG and PDF) |
| `data/` | Input data (C-MAPSS/PHM08 training sets, N-CMAPSS per-flight summary, Severson et al. cell summaries) with `SHA256SUMS` |

## Reproduce

```bash
pip install numpy pandas scikit-learn scipy matplotlib     # versions used: see code/ENVIRONMENT.txt
cd code
python test_core.py                                         # unit tests of the guarantee and the primitives
for d in PHM08 FD001 FD002 FD003 FD004 NCMAPSS BATTERY; do OMP_NUM_THREADS=1 python run.py $d 3; done   # ~1-2 CPU-hours
python analyze.py && python figures.py
cd ../paper && python build.py                              # needs pandoc and LibreOffice
```

All seeds are fixed (outer folds 2026; proper-training/calibration splits 1000 + 100·rep + split; bootstrap 2026/2027;
pilot draws 5000 + split). Gradient-boosted trees can differ slightly between scikit-learn versions.
