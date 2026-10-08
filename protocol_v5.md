# Protocol amendment v5: robustness of the notice guarantee to the learner and to the alarm settings

Written 2026-10-08, after the results of protocols v1–v4 were known and before the experiment below was run. Hashed with
SHA-256 (`protocol_v5.sha256` and the commit that adds it). **Not a registry preregistration.** `code/run5.py` was
smoke-tested on fold 1 of FD001 with the reference learner only (it reproduced the main run's decisions exactly), and the
fitting time of the three other learners was measured on fold 1 of FD002 without evaluating any alarm.

## Question

The main results use one learner (histogram gradient boosting, GBM) and one a-priori setting of the alarm (cap 60 cycles,
battery 180; smoothing span 3; debounce 2). Theorem 1 holds for any fixed signal map and setting; whether the failure of
per-cycle interval calibration (H2) is specific to GBM is an empirical question.

## Design

Repetition 0 of the five-fold cross-validation of the main protocol on all seven datasets, with the same outer folds and
the same proper-training / calibration split (seed 1000 + split index); everything fitted on proper-training units only.

* **Learners** (default alarm setting; target min(r, cap)):
  GBM (reference, as in the main protocol); RIDGE — ridge regression (penalty 1) on the standardised causal features;
  RF — random forest (200 trees, at least 20 rows per leaf, one third of the features per split, seed 0) on the causal
  features; WMLP — multilayer perceptron (hidden layers 64 and 32, ReLU, Adam, learning rate 0.001, batch 256, at most 60
  epochs, early stopping on 10 % of the rows, seed 0) on the normalised raw sensors of the last 30 rows (no hand-made
  features; target scaled by the cap).
* **Settings** of the GBM signal: cap {40, 60, 90} (battery {120, 180, 270}) × span {1, 3, 5} × debounce {1, 2, 3}
  (27 settings), primary lead time only.
* **Policies** for every learner and setting: CNA, LB-all and LB-near at α ∈ {0.05, 0.10, 0.20}; lead times as in the
  main protocol for the learners.

## Hypotheses and decision rules

* **Implementation check** (must hold before any hypothesis is evaluated): the GBM reference with the default setting
  reproduces the main run's repetition-0 decisions (failure and notice of every test unit) for CNA, LB-all and LB-near.
* **H11 (validity across learners).** For RIDGE, RF and WMLP, CNA at α = 0.10 and the primary lead time (10 cycles;
  battery 30): one-sided exact binomial test of failure rate > 0.10 per dataset, Holm over the 21 tests. Supported if no
  adjusted p < 0.05 (prediction implied by Theorem 1).
* **H12 (interval calibration across learners).** LB-all at nominal 0.10 on the same signal and trigger: one-sided exact
  binomial test of rate > 0.10 per dataset, Holm over the 21 tests. Supported if it is significant on at least 4 of the 7
  datasets for at least 2 of the 3 alternative learners. LB-near: descriptive.
* **H13 (alarm settings).** For each of the 27 settings, CNA at α = 0.10, primary lead time, failures pooled over the seven
  datasets: one-sided exact binomial test of pooled rate > 0.10, Holm over the 27 tests. Supported if none is significant.
  Mean notice across settings: descriptive.

Other α, lead times and per-dataset rates under the settings: descriptive. Results are reported whatever they are.
