# DECISIONS

Two rounds of work. Round 1 rewrote the 6/10 evaluation study around one contribution, a conformal notice-calibrated
end-of-life alarm (CNA), and was self-rated about 6.5/10. Round 2, prompted by a request to raise the paper to "a solid 9",
attacked the weaknesses listed at the end of round 1 and moved the manuscript into the RESS LaTeX template
(`paper_tex/`, Elsevier `cas-dc` class, compiled PDF, highlights, self-contained `RESS_submission.zip`). The uploaded
`LaTeX_source.zip` was byte-identical to the original 6/10 manuscript, so it served as the template, not as new content.

## 1. The contribution (unchanged headline, wider scope)

Calibrate the alarm, not the interval. For a debounced threshold trigger on any causal signal, the notice is monotone in the
warning level, so each calibration unit has one critical level V (Lemma 2), and the conformal quantile of V gives
P(remaining life at the alarm > lead time) ≥ 1 − α (Theorem 1). Round 2 added results that address what stands between this
guarantee and the field:

| Obstacle | Result | Evidence |
|---|---|---|
| Fleet data are censored | Lemma 3 / Theorem 2: a censored unit's critical level is bounded above by the level "as if it failed just after it was last seen alive"; using that bound keeps the guarantee under **any** censoring mechanism (pathwise argument) | H6 supported (49/49 cells). Censoring by an earlier alarm cost nothing (identical level in 100 % of 28,200 draws at α = 0.10); censoring early in life made the guarantee uninformative (alarm at the first cycle) — reported as a limitation, with the reason (no assumption-free method can do better) |
| Split calibration wastes data; 1/(n+1) floor drives cost | Theorem 3: cross-fitted (CV+) alarm, failure ≤ 2α + O(K/n) | H7a supported (3.8–10.1 % at α = 0.10, rep. 0; 4.3–10.1 % over three repetitions). H7b **not** supported (cheaper than split on 3 of 7, predicted ≥ 4). With **equal information** (H9) the certified alarm was never significantly dearer than the best uncertified rules and cheaper than the 1-SE rule on battery and FD004; mean excess 1–14 % vs 1–11 % (1-SE) and 2–14 % (Kamariotis P1 optimised), repetition 0; 1.9–14.6 / 0.9–11.7 / 1.9–11.8 % averaged over three repetitions |
| Populations change (batches) | Theorem 4: online update of the level from each unit's own outcome; deterministic bound on the long-run failure rate for **any** sequence, no exchangeability, no pilot failures | H10 supported: held-out battery batches 7.6 / 12.7 / 13.5 % failures over the sequence (≈10 % in the second half) vs 0 / 74 / 61 % static; the bound held in all 316,800 simulated sequences. N-CMAPSS subsets (9–15 units) too short to adapt fully (29–40 % → 18–21 %) |
| A guarantee averaged over calibration sets | Corollary 1: training-conditional form (Beta law of the conditional failure probability) and the order statistic needed for a (1 − δ)-confidence guarantee | Exploratory check (added after all runs, labelled as such): fold-level failure counts follow the predicted Beta-binomial law (KS p = 0.93 at α = 0.10; variance 0.0049 vs 0.0048 predicted; 103 of 105 folds inside the 90 % band) |
| Lead times are uncertain | Remark 1: V is monotone in the lead time; calibrate at an upper bound, or draw one lead time per calibration unit | Theory only |
| One learner, one alarm setting | — | Amendment v5 (hashed before running): three further learners and 27 settings — see section 1a |

Also added: the decision setting and metric M of Kamariotis et al. (H8): split CNA-DA was better than their optimised policy 1
on FD001 and indistinguishable on the other four datasets, better than age replacement everywhere, worse than a 1-SE
cost-tuned alarm on FD002 and FD004.

## 1a. Robustness amendment (protocol_v5.md)

Written and hashed (`protocol_v5.md`, sha256 a58226c1…) and committed before the runs; repetition 0, same units and
splits as the main protocol. The reference learner (GBM) reproduced the main run's decisions exactly on all seven datasets.

| Hypothesis | Result |
|---|---|
| H11: CNA keeps α = 0.10 with ridge regression, a random forest and an MLP on raw 30-row sensor windows | **Supported.** 4.0–17.0 % per dataset, 9.3–9.9 % pooled; smallest Holm-adjusted p = 0.43. Pooled over datasets and lead times all four learners gave 3.9–4.2 / 9.5–10.0 / 18.5–19.6 % at nominal 5 / 10 / 20 % |
| H12: the per-cycle bound (all rows) exceeds 10 % on ≥ 4 of 7 datasets for ≥ 2 of 3 learners | **Supported.** Significant on 7 (RF, 20–70 %), 7 (ridge, 26–71 %) and 5 (MLP, 6–38 %) of 7 datasets |
| H13: CNA keeps α = 0.10 for each of 27 settings (cap × span × debounce) of the GBM signal | **Supported.** Pooled 8.7–11.1 %, every Holm-adjusted p = 1; mean notice 17.4–18.1 engine cycles (battery 41–48) |

Descriptive findings: the near-failure restriction (LB-near), which nearly repaired the per-cycle bound for GBM, failed
with ridge regression (17–33 %); a weaker signal cost CNA notice, not reliability (ridge: median notice 18–21 vs 15.5–16
cycles on PHM08/C-MAPSS; MLP: 64 vs 42 battery cycles).

## 2. Process and safeguards

* Five hashed protocols: `protocol.md` (main, before any final run), `protocol_v2.md` (censoring, CV+, metric M),
  `protocol_v3.md` (equal-information comparators), `protocol_v4.md` (online calibration), `protocol_v5.md` (learners and
  alarm settings). Each amendment was written after earlier results were known and before its own experiment; none is a
  registry preregistration. One analysis was added after all runs and is labelled exploratory in the paper (the check of
  the training-conditional corollary).
* Implementation checks: unit tests for the critical level (brute force), the coverage guarantee (simulation), the
  censoring bound, Theorem-1 inequality, causality of features and the Tango interval; CNA+ recomputed independently in
  `run3.py` and matched `run2.py` exactly in all three repetitions; the Lemma-2 identity is asserted inside the censoring
  run; Theorem 4 checked on every simulated sequence; the reference learner of `run5.py` reproduced the main run's
  decisions exactly.
* Network and permission limits: zenodo.org, huggingface.co and data.matr.io are blocked in this environment, so an
  external cross-laboratory battery dataset (HUST, 77 cells of the same A123 type, in the BatteryLife corpus) could not be
  used; adding a public GitHub repository that redistributes processed HUST data was refused by the session's permission
  check and was not attempted another way.

## 3. What failed or came out against the hypotheses (all reported in the paper)

* H3 (estimate vs constant within ±5 points): not supported on 6 of 7 datasets; corrects the original paper's "within
  3.6 points" claim.
* H4 (split CNA-DA cheaper than comparators): not supported; explained by the 1/(n+1) floor (Proposition 1(ii)).
* H7b (CV+ cheaper than split on ≥ 4 datasets): not supported (3 of 7).
* Censored-as-failure calibration is useless under random or administrative censoring (valid but alarms at the first cycle).
* LB-near, the near-failure variant of the per-cycle bound that looked almost adequate with GBM, failed with ridge
  regression (17–33 % at nominal 10 %) — reported as evidence against interval calibration, not hidden.
* An error was caught and corrected before submission: an early draft claimed that a new batch could be calibrated from
  cells replaced under a conservative interim level; Theorem 2 shows the opposite (such cells can confirm a level but never
  relax it). The text now says so.

## 4. Remaining weaknesses

* No field data and no genuinely censored lives; five of seven datasets are from one simulator family; the battery cells
  are laboratory data. The external battery dataset could not be downloaded here.
* Per-unit guarantees need exchangeability; the online bound is long-run, weak for short sequences, and assumes prompt
  feedback (in a fleet, outcomes of concurrently operated units arrive late; group updates of 5 were tested).
* The conformal machinery itself is standard; the novelty is the stopping-time formulation, the closed-form critical level,
  the pathwise censoring result, the certification floor, and the evidence. A theory-minded reviewer may still call it an
  application.
* Cost: certification is roughly cost-neutral only with cross-fitting; with small calibration sets it is not.
* Several analyses are amendments made after earlier results were known.

## 5. Self-assessed RESS rating

**About 8/10** (likely "minor or major revision"; acceptance plausible). This is an estimate, not a promise; a reviewer's
score cannot be engineered.

Why higher than the 7.5 of the previous draft: the guarantee is now checked beyond its mean (training-conditional
corollary, with fold-level failure counts matching the predicted Beta-binomial spread), across four learners and 27 alarm
settings under a fifth hashed protocol, and the equal-information cost comparison is reported over all three repetitions;
uncertain lead times are covered; the theory section is complete for a RESS audience.

Why not 9: a "solid 9" at RESS normally needs either a real industrial case (field fleet, genuinely censored lives, real
cost data) or a method that is clearly superior across the board. Neither is true here: five of seven datasets are from one
simulator family and the sixth is laboratory data, censoring was simulated on complete records, and the certified alarm
matches but does not beat the best tuned heuristics on cost. I cannot honestly raise the rating further without data I do
not have. The most valuable next steps, in order: (1) a field fleet with genuinely censored lives; (2) the HUST cells as an
external cross-laboratory test (needs network access to zenodo.org or huggingface.co, or permission to add a repository
that redistributes them); (3) a per-unit guarantee under shift using covariate weighting; (4) an industrial co-author or
case study with real cost ratios.
