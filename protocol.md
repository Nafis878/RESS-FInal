# Protocol: conformal notice-calibrated end-of-life alarms

Written 2026-10-08, before the final experiments. Its SHA-256 hash is recorded in `protocol.sha256` and in the commit that
adds it. **The hash is not a registry preregistration**: it shows only that this file has not changed since it was hashed;
the time of hashing rests on the git history of this repository. Before hashing, the code was smoke-tested on one
cross-validation fold of PHM08 (the development dataset of the original study) to remove programming errors; those outputs
are overwritten by the final run and are not reported.

## 1. Question and headline contribution

An end-of-life alarm is useful only if it leaves enough notice for the maintenance action. Prior work calibrates RUL
*intervals* at every cycle (conformal RUL prediction), which does not control what happens at the data-dependent alarm time.
We calibrate the *alarm itself*: for a threshold-crossing alarm and a lead time ℓ, each calibration unit has a critical
level V (the smallest warning level whose alarm leaves more than ℓ cycles); the split-conformal quantile of V gives a warning
level Ĥ with P(remaining life at the alarm > ℓ) ≥ 1 − α for an exchangeable new unit (Theorem 2). A decision-aware variant
chooses α ≤ α_max from the cost ratio and keeps the guarantee at α_max (Corollary 1). A supporting result (Theorem 1)
bounds the difference between the window score of the model's estimate and that of a constant estimate at the same alarm
times, and is checked with an equivalence test.

## 2. Data and splits

* PHM08 (218 engines; development dataset of the original study), C-MAPSS FD001–FD004 training sets, N-CMAPSS (99 engines,
  one row per flight, six sensors), Severson et al. battery cells (123 cells, 3 batches, blocks of κ = 3 cycles fixed a priori).
* Bearings (PRONOSTIA, 17 units) are **excluded**: with at most about 6 calibration bearings per split the smallest
  attainable α is 1/7, so a 1 − α guarantee at α ≤ 0.10 is impossible. The original result (0 of 17) is cited as a prior result.
* Outer splits: 5-fold cross-validation by unit (seed 2026) on every dataset ("CV", exchangeable). Transfer splits:
  leave-one-batch-out (battery, 3 splits) and leave-one-subset-out (N-CMAPSS, 9 subsets).
* Inside each outer training set: 60 % proper-training units (normalisation statistics and regressors), 40 % calibration
  units (every selection, tuning and conformal step). Seeds 1000 + 100·rep + split index; 3 repetitions (rep 0 primary).
  Capacity-based battery policies fit nothing and use all outer training cells.
* Leakage checks: disjoint proper-training, calibration and test units (assertion in code); causal features (row t uses rows
  ≤ t only); no test life enters any setting.

## 3. Instrument

Gradient-boosted regressor (absolute error, 500 iterations, learning rate 0.05, 31 leaves, ≥ 40 samples per leaf, seed 0)
on causal features (as in the original study), target min(r, c). Trigger: EWM-smoothed estimate, alarm at the first row at
which it has been ≤ H for k consecutive rows.

## 4. Policies (all selection on calibration units)

| Policy | Definition |
|---|---|
| CNA_a{α} | Conformal notice alarm, α ∈ {0.05, 0.10, 0.20}; fixed cap 60 (battery 180 cycles), span 3, debounce 2, chosen a priori. |
| CNA_DA | Decision-aware CNA: choose k ≥ ⌈(n+1)(1−α_max)⌉, α_max = 0.10, minimising (1 + (ρ−1)(n+1−k)/(n+1)) / mean calibration use. |
| LBall / LBnear_a{α} | Per-cycle split-conformal lower bound on the same signal: q = (1−α) conformal quantile of s_t − min(r_t, cap) over all calibration rows (all) or rows with r_t ≤ cap (near); alarm when s_t − q ≤ ℓ + 1 (k = 2). |
| STW_window | Window-tuned recipe: per H, cap/span/debounce/shift maximise correct calibration warnings; H ∈ {10,15,20,30,40} (battery ×3 cycles) by lowest calibration cost rate. |
| Constant control | STW_window alarm times at H_w = 20 (battery 60 cycles) with one constant estimate chosen on calibration alarms. |
| STW_cost_plain / _1se | Cap, span, debounce, threshold τ chosen by lowest calibration cost rate (plain), or one-standard-error rule (200 bootstrap resamples; largest mean notice within 1 SE). |
| KAM_P1_heur / _opt | Policy 1 of Kamariotis et al. adapted to per-row decisions with lead time: P̂(r ≤ ℓ + Δt | estimate bin) from calibration rows (P-res, ≥ 200 rows per bin, isotonic); alarm when P̂ ≥ 1/ρ (heur) or an optimised threshold (opt). |
| age | Optimal age replacement on outer training lives. |
| trend_window, trend_cost_1se, CNAtrend_a{α}, CNAtrend_DA | Battery only: capacity trend extrapolated to 0.88 Ah (windows 30/60/120 cycles), window-tuned / 1-SE cost-tuned / conformal (60-cycle window, span 3, debounce 2). |

Renewal–reward cost: alarm at row j with r_j > ℓ → planned replacement at t_j + ℓ, cost 1; otherwise failure at L, cost ρ.
Leads ℓ ∈ {0, 5, 10, 15} (battery {0, 15, 30, 45} cycles); ρ ∈ {5, 10, 20, 50}. Primary lead: 10 (battery 30).

## 5. Hypotheses and decision rules (rep 0, CV splits unless stated)

* **H1 (validity).** For CNA_a0.10 at the primary lead, the test failure-before-replacement rate is ≤ 0.10. Rejected on a
  dataset if the one-sided exact binomial test of rate > 0.10 gives Holm-adjusted p < 0.05 over the 7 datasets.
* **H2 (interval calibration does not calibrate the alarm).** LBall_a0.10 at the primary lead exceeds 0.10 failure with
  one-sided exact binomial p < 0.05 on at least 4 of 7 datasets. LBnear is descriptive.
* **H3 (equivalence of model and constant estimates).** For STW_window at H_w, the Tango score 90 % CI of the paired
  difference in window score (model − constant) lies within ±5 percentage points (TOST, α = 0.05) on each dataset.
  Theorem 1 check (descriptive): the calibration-based bound ≥ the realised |difference| on at least 6 of 7 datasets.
* **H4 (cost).** In the 16 (ℓ, ρ) cells, paired bootstrap 95 % CIs (2,000 resamples of test units, seed 2027) of the ratio
  of cost rates CNA_DA / comparator. "CNA_DA cheaper than X on a dataset" if CI < 1 in ≥ 8 cells and > 1 in none;
  "dearer" symmetric; otherwise "mixed". H4b: CNA_DA's failure rate ≤ α_max at every (ℓ > 0, ρ) on every CV dataset
  (exact binomial, one-sided p ≥ 0.05).
* **H5 (transfer).** (a) Prediction: CNA_a0.10 calibrated on the other batches exceeds 0.10 failure at the primary lead on
  at least one held-out battery batch (one-sided exact binomial p < 0.05). (b) Pilot recalibration with m = 10 cells of the
  held-out batch: mean failure over 300 random pilot draws ≤ 0.10 + 2 Monte-Carlo SE on every batch and lead.
  N-CMAPSS leave-one-subset-out: descriptive.

If a hypothesis fails, the paper reports the failure and is framed around what holds. Every other cell, α, lead and
repetition is reported as descriptive.

## 6. Statistics

Clopper–Pearson 95 % intervals for rates; exact binomial tests; Tango score interval for paired proportions (TOST);
paired bootstrap for cost ratios. Intervals cover sampling of test units given the fitted policies; repetitions show the
variability of the proper-training/calibration split.
