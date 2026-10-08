# Protocol amendment v2: censored calibration data, cross-fitted calibration, decision grid

Written 2026-10-08 after the results of `protocol.md` were known and before the experiments below were run. Hashed with
SHA-256 (`protocol_v2.sha256`, and in the commit that adds it). **Not a registry preregistration.** The code (`code/run2.py`)
was smoke-tested on one FD001 fold before hashing; those outputs are not reported. Everything here is exploratory relative to
`protocol.md`; the decision rules below were fixed before the runs.

## Motivation

Three weaknesses of the first round: (1) field data are censored (units replaced before failure), and the guarantee was
shown only with complete run-to-failure calibration units; (2) the split design spends 40 % of the training units on
calibration, and the 1/(n+1) floor drove the cost gap; (3) the comparison with Kamariotis et al. used an adapted setting.

## Analyses (same datasets, outer splits, proper-training/calibration splits and seeds as protocol.md; 3 repetitions)

**A. Censored calibration units (split CNA, α ∈ {0.05, 0.10, 0.20}, all lead times).** Lemma 3: a unit alive at cycle c has
critical level V ≤ U(c), the critical level computed as if it had failed at c. Theorem 3: replacing each censored V by U keeps
the guarantee for any censoring mechanism. Mechanisms applied to the calibration units only (test units unchanged), 50 draws
per split: *random* (each unit censored with probability π ∈ {0.25, 0.5, 0.75} at a uniform time in [0.3L, L)),
*administrative* (staggered entry, common end of observation chosen so that a fraction π is censored; long lives censored
preferentially), *policy* (history under an incumbent alarm at the median critical level, jittered; units replaced at alarm
+ ℓ are censored there). Methods: censored-as-failure (U), complete-case (drop censored units), and the uncensored oracle.

**B. Cross-fitted calibration (CV+), CNA+.** Five folds of all outer training units (seed 3000 + 100·rep + split); the critical
level of each training unit under the model fitted without its fold; for a test unit, alarm at the first row with
#{i : ρ_t^(k(i)) ≤ V_i} ≥ n + 1 − ⌈(1 − α)(n + 1)⌉. Theorem 4 (from Barber et al. 2021, CV+): failure probability ≤ 2α + a
finite-K term. CNA+ at α ∈ {0.05, 0.10, 0.20} and a decision-aware CNA+-DA (α_max = 0.10; cycles of use approximated with each
training unit's own fold model).

**C. Setting of Kamariotis et al.** C-MAPSS FD001–FD004 and PHM08; decisions only at cycles 10k; no lead time; preventive cost
c_p = 1, corrective c_c = 1/r, r = c_p/c_c ∈ {0.1 (primary), 0.02}; metric M = R/R_perf − 1 with the grid-perfect policy.
Policies on the same proper-training model: CNA-DA and CNA (α = 0.05) on the grid, policy 1 with P-res (threshold r, and
optimised on calibration units), cost-tuned STW 1-SE on the grid, age replacement on the grid.

## Hypotheses and decision rules (repetition 0 unless stated)

* **H6 (censoring).** Censored-as-failure CNA at α = 0.10 and the primary lead time: for every dataset, mechanism and π, the
  mean test failure rate over draws and folds is ≤ 0.10 + 2·sqrt(0.09/n_test). Complete-case: descriptive (valid only under
  censoring independent of the unit).
* **H7 (CV+).** (a) CNA+ at α = 0.10, primary lead: failure ≤ 0.10 + 2·sqrt(0.09/n_test) on each of the 7 CV datasets
  (empirical claim; the theorem only gives 2α + O(K/n)). (b) Cost: CNA+-DA vs split CNA-DA and vs STW 1-SE, H4 rule
  ("cheaper" if paired bootstrap CI < 1 in ≥ 8 of 16 cells and > 1 in none). Prediction: CNA+-DA cheaper than split CNA-DA on
  at least 4 of 7 datasets.
* **H8 (metric M).** Paired bootstrap 95 % CI (2,000 resamples, grid-perfect reference recomputed in each) of
  M(CNA-DA) − M(X); "CNA-DA better / worse" only when the CI excludes 0. Descriptive, no directional prediction.
