# Plan for round 3: field data (Part A) and novelty (Part B)

Status 2026-10-09. Nothing in this file has been run; no new experiment will run before `protocol_v6.md` is written,
hashed with SHA-256 and committed. Existing results stay unchanged unless a bug is found (see flag F1).

## 0. Data access: both field datasets are blocked here

Every host was refused by this environment's network policy (proxy answers 403 to CONNECT), tested 2026-10-09:
`www.backblaze.com`, `f001.backblazeb2.com`, `f000.backblazeb2.com`, `s3.us-west-004.backblazeb2.com`, `doi.org`,
`snd.se`, `researchdata.se`, `datasets.snd.se`, `api.researchdata.se`, `arxiv.org`, `export.arxiv.org`, `zenodo.org`,
`figshare.com`, `www.kaggle.com`. No simulated or substitute data will be used for Part A. The Hugging Face hub has no
copy of either dataset (one derived "CAD-Scania" benchmark exists, without time-to-event labels; not suitable).

Two ways to unblock (your choice):

* **Allow the hosts** (preferred; nothing to upload): session title bar → cloud environment → Edit → Network access →
  add under Allowed domains (keep "Allow package managers" ticked): `f001.backblazeb2.com`, `www.backblaze.com`,
  `snd.se`, `researchdata.se`, `doi.org`, and `arxiv.org` (dataset paper). If the SND download button redirects to another
  host, add that one too. Docs: https://code.claude.com/docs/en/cloud-environments#network-access
* **Upload the files**:
  * Backblaze Drive Stats: the quarterly ZIPs `data_Q1_2022.zip` … `data_Q4_2024.zip` (12 files, roughly 1 GB each) from
    the "Hard Drive Data and Stats" page. If uploading 12 GB is impractical, I will write a DuckDB script that you run
    locally to cut them to the columns and drive models needed (a few hundred MB of Parquet).
  * SCANIA Component X (SND 2024-34, doi:10.5878/jvb5-d390): all files of the dataset (train/validation/test
    operational readouts, specifications, time-to-event and label files, README).

License note: you described Backblaze Drive Stats as MIT-licensed. The terms I know of are different (cite Backblaze as
the source; you are responsible for your use; do not sell the data). I could not open the page to check; please confirm
which terms apply before we write the data statement.

## 1. Stages and checkpoints

| Stage | Content | Needs data? | Checkpoint |
|---|---|---|---|
| S1 | Theory for B1, B2, B3 (statements, complete proofs) in `theory/round3.md`, then LaTeX | no | you review statements |
| S2 | `protocol_v6.md` (H14 onward): hypotheses, datasets, metrics, decision rules, simulation sanity checks; SHA-256, commit, push | no | hash committed before S3 |
| S3 | Runs on the existing datasets: LB debounce check (F1), B1 bound check, B2 under simulated censoring incl. violated assumptions, B3 on battery batches and N-CMAPSS subsets; theorem simulations | no | results table |
| S4 | Field data: download, verify (checksums, row counts, schema), build per-unit records, leakage truncation tests, descriptive table | **yes** | **show you the loading result before S5** |
| S5 | Field experiments (CNA, CNA-DA, CNA+, LB-all/near with k = 1 and k = 2; GBM and a hazard learner; within-period split, period-ahead shift, manufacturer split, online with delayed feedback; censoring estimators) | yes | results table |
| S6 | Paper: field-data subsection, one subsection per novelty item, revised contributions, related work, limitations; all new numbers emitted by `code/make_numbers.py` into `paper_tex/numbers.tex` (macros), so no number is typed by hand; hostile one-page self-review | — | final |

S1–S3 can proceed while data access is sorted out.

## 2. Part B: what I intend to prove (items 1, 2, 3; item 5 only if time allows; item 4 not planned)

Notation as in the paper: rows j with remaining life r_j, signal s_j, debounce k, running minimum ρ, critical level
V = ρ_{j_ℓ}, Lemma 2: N(H) > ℓ ⇔ H ≥ V.

### B1. Separation: per-cycle coverage does not imply alarm reliability (new theorem)

Per-cycle lower bound LB_j = s_j − q; the "LB alarm" acts when LB_j ≤ ℓ + 1 (debounce k = 1 for the theorem);
F = failure before the planned replacement; π_j = P(LB_j > ℓ + 1).

* (a) Impossibility. For every α ∈ (0,1), ℓ and q there is a joint law of life and signal (geometric lives, one row per
  cycle; the per-cycle bound equals the true remaining life except in the m rows with remaining life in (ℓ, ℓ+m],
  where it is too high) such that the per-cycle
  bound covers with probability exactly 1 − α at every cycle among units alive at that cycle, and exactly 1 − α for a
  row drawn from all rows, yet P(F) = 1. Proof by construction (memorylessness gives the same coverage at every cycle;
  m is chosen so that x^ℓ(1 − x^m) = α has a solution x = 1 − p, which exists for m large).
* (b) Universal bound. P(F) ≤ π_{j_ℓ}; with one row per cycle, π_{j_ℓ} is the miscoverage at remaining life ℓ + 1,
  which pooled per-cycle calibration does not control.
* (c) Sharpness in the number of rows. If every unit has n rows and the bound has row-averaged coverage 1 − α, then
  P(F) ≤ min(1, nα), and the bound is attained.
* (d) Dependence. P(F) ≤ min_j π_j (attained by comonotone errors); if the errors are associated,
  P(F) ≥ ∏_j π_j (Harris–FKG; attained by independent errors); if they are m-dependent,
  P(F) ≤ ∏_i π_{j_ℓ − i(m+1)}.
* (e) Remark (debounce): with k ≥ 2 and action at ℓ + 1, F can occur with a perfect bound (fires k − 1 rows late);
  the threshold must be ℓ + k. This is flag F1.
* Data check: estimate π(r) by remaining life on test units, the independence product, the m-dependence bound (m from
  the residual autocorrelation) and the upper bound; compare with the realised LB-alarm failure rate on all seven
  datasets and on the field data.

### B2. Censoring with a weaker, testable assumption (adaptation; recovers Theorem 2 at Γ = ∞)

Horizon c0 (a fixed period, e.g. one year for drives); horizon critical level V° = V if T ≤ c0, −∞ otherwise;
target P(V°_new > Ĥ) ≤ α (fails within the horizon before the planned replacement). A unit censored at c < min(T, c0)
has V° ∈ {−∞} ∪ (−∞, U(c)] (Lemma 3).

* Estimator E2, "probabilistic censored-as-failed": a censored unit counts as failed just after c (value U(c), as in
  Theorem 2) with probability p̃, otherwise as not failing within the horizon (−∞); conformal quantile as in Theorem 1.
  * Theorem: if p̃ ≥ the true conditional probability that the unit fails within the horizon given its observed data,
    then P(F) ≤ α (finite sample; proof by coordinatewise stochastic dominance and the rank argument of Theorem 1).
  * Γ-sensitivity: p̃ = Γp⁰/(1 + (Γ − 1)p⁰), where p⁰ is the horizon-failure probability of units with the same observed
    covariates/history (identified under censoring-at-random); valid whenever censored units' failure odds are at most
    Γ times p⁰'s. Γ = 1: censoring at random; Γ = ∞: p̃ = 1, exactly Theorem 2.
  * Estimated p⁰ (cross-fitted Kaplan–Meier or hazard model): P(F) ≤ α + (μ + √(2μ log 1/δ) + (2/3) log 1/δ)/(n + 1) + δ,
    where μ = Σ_censored (p − p̃)⁺ (proof by coupling and a Bernstein bound).
* Estimator E1, IPCW-weighted conformal (weighted conformal à la Tibshirani et al. 2019 / Candès–Lei–Ren 2023):
  complete units weighted by 1/G(min(T, c0) | X), test weight bounded by 1/G(c0 | X). Exact under censoring
  independent of the unit given baseline covariates X with known G; with estimated G the gap is
  ≤ ½ E|ŵ − w| (total-variation argument).
* Checks: simulations confirming each bound and showing the violation case; on the existing datasets with the
  amendment-v2 censoring mechanisms (random and administrative satisfy the assumption; censoring by an earlier alarm
  violates censoring-at-random given baseline covariates); on the field data against Theorem 2 and complete-case.

### B3. Per-unit guarantee under covariate shift without pilot failures (adaptation)

Weighted conformal on critical levels with likelihood-ratio weights w(x) = dQ_X/dP_X from batch/period covariates.
Theorem: exact P_Q(F) ≤ α with known w; with estimated ŵ, P_Q(F) ≤ α + ½ E_P|ŵ − w| (normalised). Combined with B2 by
multiplying weights. Tests: battery batches (covariates: charging-protocol parameters, known before cycling),
N-CMAPSS subsets (flight class), Backblaze year-to-year (model, capacity, age, data centre) and a manufacturer split
(expected to fail: no overlap and different SMART semantics), Scania specifications. Poor overlap shows up as an
infinite level for some units (alarm at once), which will be reported.

### B5 (only if time allows). Online update with delayed feedback

Theorem 4 with g replaced by the maximum number of outcomes pending at an update, plus a version in calendar time
for a fleet (weekly updates from the week's observed failures). No regret statement promised.

## 3. Part A design (draft; fixed in protocol_v6 before any field run)

* **Backblaze**: unit = serial number; daily rows; cohorts = drives alive on 1 January of a year, horizon to
  31 December; failure = Backblaze's failure flag; censoring = disappearance without a failure record (removal or
  migration) before 31 December. Years 2022–2024 (cohorts 2022, 2023; 2024 for period-ahead tests). Drive models chosen
  by drive count in the calibration year (outcome-blind). Features: SMART raw counters (5, 9, 187, 188, 194, 197, 198,
  and others present), 7- and 30-day changes, EWMs; all causal (truncation test added). Lead times 7, 14, 30 days
  (primary 14). Signals: the paper's GBM on min(r, cap) with cap 60 days, and a hazard learner (GBM classifier of
  failure within ℓ + 30 days). Splits: within-cohort 60/40 by drive (exchangeable), cohort k → k + 1 (shift),
  manufacturer split; online with weekly feedback.
* **Scania Component X**: unit = vehicle; irregular readouts in anonymised time steps; failure =
  `in_study_repair` = 1 at `length_of_study_time_step`; censoring = 0 (still in service at the end of its study);
  lead times 6, 12, 24 time steps (the dataset's own windows); shift by specification categories. Facts to be verified
  against the files.
* **Metrics**: failure within the horizon before the planned replacement with Clopper–Pearson intervals (complete test
  units; IPCW-weighted; and bounds counting unknown outcomes as success or failure), recall among observed failures,
  alarm rate and wasted life among drives that did not fail, notice, number of calibration units, number censored and
  when; Theorem 2 vs complete-case vs E1 vs E2(Γ = 1, 2, 4, ∞).

## 4. Decisions that could change the paper's claims

* **F1. Possible flaw in the published LB comparators.** They use debounce k = 2 with action at ℓ + 1, so even a
  perfect per-cycle bound fires one row late and fails. Their 18–70 % (LB-all) and 4–15 % (LB-near) failure rates
  may partly reflect this. Plan: re-run them with k = 1 and with threshold ℓ + k (pre-registered in protocol_v6);
  if the conclusion changes (likely for LB-near), the paper says so as a correction.
* **F2. Field estimand is horizon-limited.** For units that mostly never fail in the window, the paper's lifetime
  guarantee is replaced by "probability of failing within the horizon before the planned replacement", reported
  with recall among failures. This changes what the field guarantee means.
* **F3. Backblaze's failure label** (to be verified) appears to include drives removed proactively because of SMART
  readings; an alarm on SMART features partly re-learns that rule, which would flatter notice for those failures.
* **F4. Period-ahead tests are not exchangeable** (the same drives appear in consecutive cohorts; the fleet changes).
  Guarantees there rest on B3 (covariate shift) or Theorem 4 only.
* **F5. Theorem 2 will very likely be uninformative on field data** (most units censored long before they would fail),
  as already seen with simulated random censoring. The field value of the method then rests on B2's assumption.
* **F6. Novelty ceiling.** B1 is, to my knowledge, new in this setting but elementary. B2 and B3 adapt known weighted
  and survival-conformal results to the critical-level score. My honest expectation is a novelty score of about 7/10
  for a skeptical reviewer even if everything works; a solid 8 is not guaranteed.
* **F7. Scale.** Backblaze is about 90 million drive-days per year; training will subsample rows of drives that did not
  fail (training only, never calibration or test), and I may restrict to the most common models to fit memory (15 GB)
  and disk (29 GB free).
