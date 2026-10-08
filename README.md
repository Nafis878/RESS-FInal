# Conformal notice-calibrated end-of-life alarms (RESS manuscript)

Rewrite of the evaluation study *"What does an end-of-life alarm score measure?"* (original manuscript and its LaTeX source
in `original_reference/`) around one contribution: an end-of-life alarm whose warning level is calibrated by conformal
prediction on the **notice at the alarm**, with a finite-sample guarantee P(remaining life at the alarm > lead time) ≥ 1 − α,
a decision-aware variant, a pathwise result that keeps the guarantee under **any censoring** of the calibration units,
a cross-fitted variant (CV+), and a supporting result on what alarm-window scores measure.

| Path | Content |
|---|---|
| `paper_tex/manuscript.pdf`, `paper_tex/*.tex` | **Submission version**: LaTeX in the Elsevier `cas-dc` class (RESS template), compiled PDF, `highlights.txt`; build with `python paper_tex/build_tex.py` |
| `paper/manuscript.{md,docx,pdf}` | First-round version of the manuscript (Markdown source, Word, PDF) |
| `protocol.md`, `protocol_v2.md` (+ `.sha256`) | Protocols written and hashed before the main experiments and before the amendment experiments (not registry preregistrations) |
| `DECISIONS.md` | Choice of contribution, what worked, what failed, remaining weaknesses, self-assessed rating |
| `code/` | `core.py` (data, causal features, trigger), `policies.py`, `run.py` (main experiments), `run2.py` (censoring, CV+, decision grid), `analyze.py`, `analyze2.py`, `stats.py`, `figures.py`, `figures2.py`, `make_tables.py`, `test_core.py` |
| `results/` | CSV tables and `hypotheses.json`, `hypotheses_v2.json`; per-unit outputs in `raw/` and `raw2/`; logs |
| `figures/` | Figures 1–7 (PNG and PDF) |
| `data/` | Input data (C-MAPSS/PHM08 training sets, N-CMAPSS per-flight summary, Severson et al. cell summaries) with `SHA256SUMS` |

## Reproduce

```bash
pip install numpy pandas scikit-learn scipy matplotlib     # versions: code/ENVIRONMENT.txt
cd code
python test_core.py                                         # unit tests: critical level, coverage, censoring bound, causality, Tango CI
for d in PHM08 FD001 FD002 FD003 FD004 NCMAPSS BATTERY; do OMP_NUM_THREADS=1 python run.py $d 3; done   # main (~1.5 CPU-h)
sh run_v2_all.sh                                            # amendment experiments (~4 CPU-h)
python analyze.py && python analyze2.py && python figures.py && python figures2.py && python make_tables.py
cd ../paper_tex && python build_tex.py                      # needs pdflatex
```

All seeds are fixed (outer folds 2026; proper-training/calibration splits 1000 + 100·rep + split; CV+ folds
3000 + 100·rep + split; censoring draws 7000 + 100·rep + split; bootstrap 2026/2027; pilot draws 5000 + split).
Gradient-boosted trees can differ slightly between scikit-learn versions.
