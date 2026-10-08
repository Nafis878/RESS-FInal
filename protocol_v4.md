# Protocol amendment v4: online calibration of the warning level under batch shift

Written 2026-10-08, after the results of protocols v1–v3 were known (in particular that a calibrated level does not transfer
across battery batches) and before the experiment below was run. Hashed with SHA-256 (`protocol_v4.sha256` and the commit
that adds it). **Not a registry preregistration.** `code/run4.py` was smoke-tested on battery batch 2, repetition 0
(failure 12.7 % against 76.7 % for the static level at η = 0.1 cap, g = 1); those outputs are overwritten by the run.

## Method (Theorem 5)

Units of a new population arrive in groups of g. All units of a group use the current level q; once their outcomes
(failure before the planned replacement, or not; observable also for replaced units) are known,
q ← q + η Σ(fail − α). Start: q₁ = the split-conformal level of the old population (calibration units of run.py),
α = 0.10. For any sequence of units whose critical levels lie in a range of width B:
|mean failure − α| ≤ (B + gη)/(ηT), with no exchangeability assumption.

## Analysis

Battery cells (leave-one-batch-out, and cross-validation by cell as an exchangeable reference) and N-CMAPSS
(leave-one-subset-out, and cross-validation); same models, splits and seeds as run.py (repetitions 0–2). For each split,
lead time, step η ∈ {0.05, 0.10, 0.20} × cap (0.10 primary) and group size g ∈ {1, 5} (1 primary): 200 random orders of the
held-out units (seed 9000 + 100·rep + split). Metrics: mean failure over the sequence, failure in its second half, mean
notice at planned replacements; compared with the static level q₁ and with pilot recalibration (run.py).

## Hypothesis and decision rule

* **H10.** For the held-out battery batches at the primary lead time (30 cycles), η = 0.10 cap and g = 1, repetition 0:
  the mean failure over the sequence is ≤ 0.10 + its Theorem 5 bound (prediction implied by the theorem; a violation would
  indicate an implementation error), and is lower than the static level's failure on every batch where the static level
  exceeds 0.10. Second-half failure, N-CMAPSS subsets, other η and g: descriptive.
