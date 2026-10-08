# Conformal notice-calibrated end-of-life alarms (RESS manuscript)

Rewrite of the evaluation study *"What does an end-of-life alarm score measure?"* (original manuscript and its LaTeX source
in `original_reference/`) around one contribution: an end-of-life alarm whose warning level is calibrated by conformal
prediction on the **notice at the alarm**, with a finite-sample guarantee P(remaining life at the alarm > lead time) ≥ 1 − α,
its training-conditional form, a decision-aware variant, a pathwise result that keeps the guarantee under **any censoring**
of the calibration units, a cross-fitted variant (CV+), an online update with a long-run bound for any sequence of units,
and a supporting result on what alarm-window scores measure.

| Path | Content |
|---|---|
| `paper_tex/manuscript.pdf`, `paper_tex/*.tex` | **Submission version**: LaTeX in the Elsevier `cas-dc` class (RESS template), compiled PDF, `highlights.txt`; build with `python paper_tex/build_tex.py`; `sh paper_tex/package.sh` writes the self-contained `RESS_submission.zip` |
| `paper_tex/RESPONSE_TO_REVIEW.md` | How the rewrite answers the weaknesses of the 6/10 original |
| `paper/manuscript.{md,docx,pdf}` | First-round version of the manuscript (superseded) |
| `protocol.md`, `protocol_v2.md` … `protocol_v5.md` (+ `.sha256`) | Main protocol and four amendments, each written and hashed before its experiments (not registry preregistrations) |
| | Result numbers in the hashed protocols follow the drafts of their time and were not edited (that would break the hashes): in `protocol.md` Theorem 2 is the article's Theorem 1, Corollary 1 is Corollary 2 and Theorem 1 is Proposition 2; in `protocol_v2.md` Theorem 3 is Theorem 2 and Theorem 4 is Theorem 3; Theorem 5 of `protocol_v4.md` is Theorem 4; `protocol_v5.md` uses the article's numbering |
| `DECISIONS.md` | Choice of contribution, what worked, what failed, remaining weaknesses, self-assessed rating |
| `code/` | see below |
| `results/` | CSV tables, `hypotheses*.json`; per-unit outputs in `raw/` (main), `raw2/` (censoring, CV+, decision grid), `raw3/` (equal information), `raw4/` (online), `raw5/` (robustness); logs |
| `figures/` | Figures (PNG and PDF); `paper_tex/package.sh` renames the ones used in the article to `fig1`–`fig7` |
| `data/` | Input data (C-MAPSS/PHM08 training sets, N-CMAPSS per-flight summary, Severson et al. cell summaries) with `SHA256SUMS` |

Code: `core.py` (data, causal features, trigger, critical level), `policies.py` (all policies), `run.py` (main protocol),
`run2.py` (v2: censoring, CV+, decision grid), `run3.py` (v3: equal-information comparators), `run4.py` (v4: online
calibration), `run5.py` (v5: other learners and alarm settings), `analyze.py`, `analyze2.py` (v2–v4), `analyze5.py` (v5),
`analyze_conditional.py` (exploratory check of the training-conditional corollary), `stats.py`, `figures.py`,
`figures2.py`, `make_tables.py`, `make_tables_tex.py`, `test_core.py`.

## Reproduce

```bash
pip install numpy pandas scikit-learn scipy matplotlib     # versions: code/ENVIRONMENT.txt
cd code
python test_core.py                                         # unit tests: critical level, coverage, censoring bound, causality, Tango CI
for d in PHM08 FD001 FD002 FD003 FD004 NCMAPSS BATTERY; do OMP_NUM_THREADS=1 python run.py $d 3; done   # main (~1.5 CPU-h)
sh run_v2_all.sh                                            # amendment v2 (~4 CPU-h)
sh run_v3_all.sh                                            # amendment v3 (~3 CPU-h)
for d in BATTERY NCMAPSS; do python run4.py $d; done        # amendment v4
sh run_v5_all.sh                                            # amendment v5 (~2 CPU-h)
python analyze.py && python analyze2.py && python analyze5.py && python analyze_conditional.py
python figures.py && python figures2.py && python make_tables.py && python make_tables_tex.py
cd ../paper_tex && python build_tex.py && sh package.sh     # needs pdflatex
```

All seeds are fixed (outer folds 2026; proper-training/calibration splits 1000 + 100·rep + split; CV+ folds
3000 + 100·rep + split; censoring draws 7000 + 100·rep + split; online orders 9000 + 100·rep + split; bootstrap
2026/2027 and 6026 + split + 100·rep; pilot draws 5000 + split; learners of v5 seed 0). Gradient-boosted trees can differ
slightly between scikit-learn versions.
