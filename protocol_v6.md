# Protocol amendment v6: separation, censoring under a weaker assumption, covariate shift, field data

Written 2026-10-09, after all results of protocols v1–v5 were known and before any experiment below was run (no
simulation, re-analysis or field-data access has taken place). Theory in `theory/round3.md` (Theorems 5–7,
Proposition 3). Hashed with SHA-256 (`protocol_v6.sha256` and the commit that adds it). **Not a registry
preregistration.** Common tolerance used below (as for H6): a rate is "within its level α" if it is at most
α + 2√(α(1 − α)/n), n = number of test units of the dataset or group.

## Part 1. The per-cycle comparators and Theorem 5 on the existing data

**Reason.** The per-cycle lower-bound alarms of the main protocol (LB-all, LB-near) used debounce k = 2 and action level
ℓ + 1. By Theorem 5(e) even an exact bound then fires late. This part checks whether that choice drove the reported
failure rates.

**Design.** Seven datasets, five-fold cross-validation, repetitions 0–2, same splits, seeds and GBM signal as `run.py`.
Row spacing Δ = 1 cycle (PHM08, C-MAPSS, N-CMAPSS) and Δ = 3 cycles (battery blocks). Variants of LB-all and LB-near at
α ∈ {0.05, 0.10, 0.20}, all lead times: ORIG (k = 2, action ℓ + 1; must reproduce `run.py`), K1 (k = 1, action
ℓ + Δ), K2C (k = 2, action ℓ + 2Δ). Repetition 0 also with the three learners of amendment v5 (ridge, random forest,
MLP on raw windows; K1 only). Code `code/run6_lb.py`.

* **H14a (implementation).** ORIG reproduces the `run.py` decisions exactly (all datasets, repetitions, policies).
* **H14b.** LB-all K1 at nominal 0.10, primary lead time, repetition 0: one-sided exact binomial test of rate > 0.10,
  Holm over the seven datasets; supported if significant on at least 4 of 7. If supported, the main protocol's H2
  conclusion stands with corrected numbers; if not, the paper withdraws H2 as stated.
* **H14c.** LB-near K1, same test. If no dataset is significant, the paper states that the near-failure per-cycle bound
  with the correct action level kept its level in these data, and the claim that it failed on N-CMAPSS is withdrawn.
* **H14d.** Learners (repetition 0): LB-all K1 significant (as in H14b) on at least 4 of 7 datasets for at least 2 of
  the 3 learners.
* Descriptive: K2C; other α and lead times; notice of K1 against CNA.

**Theorem 5 in data (H15).** Repetition 0, GBM, α = 0.10, primary lead time, LB-all and LB-near K1, test units:
* **H15a (implementation).** Every test unit whose LB alarm fails has a miss at its last admissible row (Theorem 5(b);
  pathwise, must hold for 100 % of units).
* **H15b.** For LB-all, the miss rate at the last admissible row is at least twice the pooled miss rate over all test
  rows on at least 5 of 7 datasets.
* **H15c.** The realised failure rate lies between the independence product (∏ over admissible rows of the miss rate
  at that remaining life, estimated on test rows in remaining-life bins) and the critical-row miss rate on all seven
  datasets (count reported).
* Descriptive: P(failure | critical-row miss), the lag m at which the residual autocorrelation falls below 0.2, and the
  m-dependence bound.

## Part 2. Simulation checks of Theorems 5–7 and Proposition 3 (`code/sim6.py`, seeds fixed in the code)

Synthetic data are used only here, to check the theorems; they are labelled as simulations wherever reported.
Pass criterion for a "valid" check: estimate ≤ nominal + 4 Monte Carlo standard errors.
* **S1 (Theorem 5(a)).** α = 0.10, ℓ = 10, action 11, k = 2, construction of the proof; 200,000 units: per-cycle miss
  rate within 4 SE of α at cycles 0, 10, 50, 100 and pooled; every unit's alarm fails.
* **S2 (Theorem 5(c)).** n = 100 rows, α = 0.005: P(F) within 4 SE of 0.5 and pooled miss rate ≤ α + 4 SE.
* **S3 (Theorem 5(d)).** 150 rows, ℓ = 10, action 11, k = 1, heteroscedastic Gaussian AR(1) errors (φ ∈ {0, 0.5, 0.9,
  0.99}) and MA(5) errors, q set for pooled coverage 0.90: estimated P(F) between the product and the minimum bound
  (and below the m-dependence bound for MA(5)), each within 4 SE.
* **S4 (Theorem 6).** Weibull lives depending on a binary covariate, AR(1) signal noise, horizon 100, ℓ = 10, k = 2,
  200 calibration units, 1,000 test units, 2,000 replications; (a) censoring at random: E2 with oracle p⁰ at Γ = 1
  valid; (b) informative censoring with odds ratio exactly 4: E2 valid at Γ = 4 and Γ = ∞; Γ = 1 reported
  (violation expected, not required).
* **S5 (Proposition 3).** Censoring depending on the covariate only, known G: IPCW valid; under (b) reported.
* **S6 (Theorem 7).** Covariate shift N(0, 1) → N(0.8, 1), lives depending on X: weighted CNA with known w valid;
  with logistic-regression ŵ fitted on independent samples, failure ≤ α + ½ E|ŵ − w| + 4 SE; unweighted CNA and a
  concept shift (lives depend on X differently under Q) reported.

## Part 3. Theorem 6 and Proposition 3 on the existing data (`code/run6_cens.py`)

**Design.** Repetition 0, five-fold cross-validation, GBM signal and calibration units of `run.py`; lead time primary;
α = 0.10 primary (0.05, 0.20 descriptive). Horizon c0 = median life of the split's proper-training units (rounded).
Target: failure within the horizon before the planned replacement (V°). Test units are not censored. Censoring of
calibration units, 50 draws per split (seed 7700 + split):
* **M1 random:** each unit is subject to censoring with probability 0.5, at C ~ Uniform(0, c0) (at random).
* **M2 staggered entry:** every unit enters at e ~ Uniform(0, c0) and observation ends at c0, so C = c0 − e (at random).
* **M3 policy:** an incumbent alarm at the median critical level (jittered as in amendment v2); units alive ℓ cycles
  after its alarm are replaced and censored then (at random given the observed history, not given baseline covariates).
* **M4 informative:** units with T ≤ c0 are subject to censoring with probability 0.8, others with probability 0.2,
  at C ~ Uniform(0, c0) (odds ratio 4 given survival to C; violates censoring at random).

**Estimators.** ORACLE (no censoring); T2 (Theorem 2); CC (complete case); IPCW (Proposition 3, G by Kaplan–Meier of
the censoring times of the calibration units, no covariates); E2-BASE(Γ) (Theorem 6, p⁰ = 1 − Ŝ(c0)/Ŝ(c) from the
lives of the proper-training units); E2-HIST(Γ) (Theorem 6, p⁰ from a logistic regression of 1{T ≤ c0} on
(t/c0, ρ_t/cap, s_t/cap) over the rows of the proper-training units (never censored) at which the unit is alive and
t < c0, using out-of-fold signals from a two-fold cross-fit of the GBM within the proper-training units (seed
7800 + split), evaluated at the censored unit's last observed row); Γ ∈ {1, 2, 4, ∞}; one imputation per censoring
draw. Known limitation stated in advance: under M2 no unit surviving the horizon is complete, so G(c0) = 0 and IPCW
gives an infinite level (Proposition 3 requires G(c0 | x) > 0).
**Metrics.** Failure within the horizon before replacement (test units), mean notice at planned replacements of test
units failing within the horizon, share of test units alarmed at their first eligible row, alarm rate among test units
not failing within the horizon.

* **H17a.** M1 and M2: E2-HIST(1) and E2-BASE(1) within the level on all 7 datasets (14 cells each).
* **H17b.** M1 and M2: E2-HIST(1) gives lower mean notice than T2 on at least 5 of 7 datasets for both mechanisms.
* **H17c.** M3: E2-HIST(1) within the level on all 7 datasets; E2-BASE(1) descriptive.
* **H17d.** M4: T2 within the level on all 7 datasets; E2-HIST(1) and E2-BASE(1) above the tolerance on at least one
  dataset (the violation the theory allows); smallest Γ restoring validity reported.
* IPCW and CC: descriptive.

## Part 4. Theorem 7 on the existing data (`code/run6_shift.py`)

**Design.** Leave-one-group-out splits of `run.py` (battery batches, N-CMAPSS subsets), repetitions 0–2, GBM signal and
calibration units of `run.py`, α = 0.10 (0.05, 0.20 descriptive), primary lead time. Covariates known before
deployment: battery — first-step C-rate, switch state of charge and second-step C-rate parsed from the charging policy
(the "new structure" flag of batch 3 has no overlap with batches 1–2 and is not used; this is a known violation of the
covariate-shift assumption for batch 3); N-CMAPSS — flight class (one-hot) and the mean altitude, Mach number and
throttle angle of the unit's first 10 flights. Weights: logistic regression (C = 1, standardised covariates) of
target versus source, source = proper-training units, target = held-out units of the other half of a two-fold split
of the held-out group (cross-fitting, so ŵ is independent of the calibration units and the test unit);
w = odds × n_source/n_target; no clipping. Weighted CNA (Theorem 7) against static CNA on the same calibration units.

* **H18.** On every held-out group where static CNA at α = 0.10 exceeded its tolerance in repetition 0, weighted CNA
  is within the tolerance (repetition 0). Effective sample size (Σw)²/Σw², share of infinite levels and notice reported.

## Part 5. Field data (pre-registered; to be run when the data are accessible)

Data access was refused by the computing environment's network policy on 2026-10-08 and 2026-10-09 (Backblaze and SND
hosts). This part fixes the analysis in advance; nothing is run or reported until the files are obtained, verified
(checksums, row counts, schema) and described to the co-author.

* **Backblaze Drive Stats**, quarterly files 2022 Q1 – 2024 Q4. Unit = serial number (+ model). Cohort Y = drives
  reporting on 1 January of year Y; daily rows to 31 December (horizon c0 = 365 days). Failure = Backblaze's failure
  flag (first occurrence); censoring = last report before 31 December without a failure record; drives that report
  again after a failure record are excluded (count reported). Drive models: those with at least 5,000 drives in the
  2022 cohort (outcome-blind). Features at day t from days ≤ t only: SMART raw values 5, 9, 187, 188, 194, 197, 198
  and those of 1, 7, 10, 12, 199 when present; their 7- and 30-day differences; EWMs (spans 7, 30). Truncation test as
  `test_core.py`. Lead times 7, 14, 30 days (14 primary); α ∈ {0.005, 0.01} (absolute, within the horizon) and the
  per-failure miss rate reported alongside.
* **SCANIA Component X**, training set (time-to-event). Unit = vehicle; failure = `in_study_repair` = 1 at
  `length_of_study_time_step`; censoring = 0 (in service at the end of its study). Features: readouts at times ≤ t
  (counters, their differences per time step, histogram bins); lead times 6, 12, 24 time steps (12 primary); horizon
  = median study length.
* **Signals:** the GBM of the main protocol on min(r, cap) (cap 60 days; Scania 48 steps), and a hazard learner (GBM
  classifier of failure within ℓ + 30 days; Scania ℓ + 24 steps) whose alarm is a threshold on −p̂.
* **Policies:** CNA, CNA-DA, CNA+, LB-all and LB-near (K1), with censoring handled by T2, CC, IPCW (G by Kaplan–Meier
  within model and age band), E2-BASE and E2-HIST (Γ = 1, 2, 4, ∞).
* **Splits:** within-cohort 60/40 by unit (exchangeable); cohort 2022 → 2023 and 2023 → 2024 (shift); manufacturer
  split (Seagate/HGST → Toshiba/WDC); Theorem 7 weights from model, capacity and age band; online updating (Theorem 4)
  with weekly updates from the failures observed that week (g = failures per week).
* **H19.** Within-cohort split: E2-HIST(1) within the level at α = 0.01 on both cohorts.
* **H20.** Within-cohort split: T2 raises the alarm at the first eligible row for more than half of the test drives
  (uninformative), E2-HIST(1) for less than a tenth.
* **H21.** Cohort → next cohort: static failure rate and Theorem 7-weighted failure rate reported; supported if the
  weighted rate is within the level whenever the static one is not.
* **H22.** LB-all K1 exceeds its level on both cohorts.
* **H23 (Scania).** As H19 and H22 on the vehicle split.
* Metrics: as in Part 3, with Clopper–Pearson intervals, number of calibration units, number censored and when.

## Reporting

All results are reported whatever they are, with the same prominence for failed hypotheses. Every number in the paper
for this amendment is written by `code/make_numbers.py` into `paper_tex/numbers.tex` from result files.
