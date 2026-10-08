# DECISIONS

## 1. What was chosen, and why

**Headline contribution: a conformal notice-calibrated end-of-life alarm (CNA).** For a debounced threshold trigger the notice
at the alarm is monotone in the warning level, so each run-to-failure calibration unit has one *critical level*
V = (debounced running minimum of the signal at its last row with more than ℓ cycles left). The split-conformal quantile of
V gives a warning level with P(remaining life at the alarm > ℓ) ≥ 1 − α for an exchangeable new unit (Theorem 2, with
the matching upper bound). A decision-aware variant (CNA-DA) chooses the order statistic by a conformal cost bound and keeps
the guarantee at α_max (Corollary 1); Proposition 2 gives the floor 1/(n+1) that any order-statistic level carries.

Why this direction (suggested direction 1, narrowed):
* It answers the first engineering requirement of an alarm (failure before the planned replacement) with a guarantee,
  which none of the cited work gives. The literature search found conformal RUL *interval* work (Javanmardi & Hüllermeier
  2023; Robinson 2026; Wang et al. 2025, RESS), conformal survival lower bounds (Candès et al. 2023), conformal risk
  control (Angelopoulos et al. 2024) and nested conformal prediction (Gupta et al. 2022), but nothing that calibrates the
  stopping time of an alarm. The claimed novelty is therefore narrow and specific: the closed-form critical level of a
  debounced trigger, the guarantee at the alarm, the decision-aware selection that keeps it, and the evaluation. The
  general conformal machinery is not new and is cited as such.
* It turns the original paper's strongest observation (window scores measure timing) into theory (Theorem 1) that
  motivates calibrating notice directly, instead of leaving it as a descriptive finding.
* It is model-agnostic, so the "one narrow instrument" criticism is weakened: the same calibration was applied to the
  learned GBM signal and to a physically motivated capacity trend.
* It can be tested honestly against a natural competitor that differs only in what is calibrated (per-cycle conformal
  lower bound on the same signal and trigger).

Direction 2 (distribution-free recalibration for transfer) was folded in as the transfer experiment (pilot recalibration
with m units of the new batch) rather than made the headline, because without a model of the shift there is no
distribution-free statement to make across batches beyond "recalibrate on the new population", which the theory already
covers.

Bearings were dropped: with ≤ 6 calibration bearings α ≤ 0.10 is unattainable (α ≥ 1/7).

## 2. Process

1. Read the full manuscript (PDF and its LaTeX source in the reproduction package) and the original code.
2. Literature search (web) for conformal RUL, conformal survival, conformal risk control, alarm metrics, paired
   equivalence tests. Publisher pages were blocked by the network proxy; bibliographic details were checked through search
   results. Two RESS conformal-RUL papers were found; one (vol. 275, 2026, isotonic + split conformal) is not cited because
   its authors could not be verified.
3. Re-implemented the instrument (vectorised causal features verified identical to the original to 1e-12), the policies
   and the statistics in `code/`. Smoke-tested on one PHM08 fold.
4. Wrote `protocol.md` (hypotheses H1–H5, decision rules), hashed it (SHA-256 `4eddd0ed…fe281`, in `protocol.sha256` and in
   the first commit), then ran all experiments once (7 datasets, 3 repetitions, ~1.5 CPU-hours). Not a registry
   preregistration.
5. Analysis, figures and tables are generated from the per-unit outputs by scripts; the manuscript numbers were copied
   from `results/paper_tables.md` and checked against the CSVs.

## 3. What worked

* **H1 (validity) supported on all 7 datasets.** CNA at α = 0.10: 6.0–11.7 % failures (rep-averaged), pooled 9.3 %;
  α = 0.05 → 3.8 %, α = 0.20 → 19.4 % pooled over 13,788 unit–cell evaluations. Pilot recalibration matched the exact
  theoretical value 1/11 = 9.1 % (7.5–10.1 %).
* **H2 supported (7/7).** The per-cycle conformal lower-bound alarm at nominal 10 % failed on 18–70 % of units (pooled 46.6 %).
  This is the cleanest demonstration that interval calibration ≠ alarm calibration.
* **H4b supported.** CNA-DA never exceeded α_max = 0.10 (max 6.1 % in any cell).
* **H5 supported**: guarantee fails across battery batches (61–74 % failures on two held-out batches) as predicted, and
  10 pilot cells of the new batch restore it.
* Theorem 1 part (iii) held on 100 % of alarms (estimate pinned within kδ⁺ of the level).

## 4. What failed or came out against the hypothesis

* **H3 (equivalence of estimate and constant within ±5 points) failed** on 6 of 7 datasets. The constant was significantly
  *better* than the model's estimate on PHM08 (−6.4 points) and FD004 (−6.0), and significantly *worse* on the battery
  (+9.8). This contradicts the original manuscript's "within 3.6 points" summary (which used cross-fitting on all training
  units); the paper reports the correction. Theorem 1's plug-in bound held for |difference| on 6/7 datasets but for the
  discordance only on 4/7 (it uses estimated probabilities, so it is not a guarantee).
* **H4 (cost) not supported.** CNA-DA was not the cheapest policy. A one-standard-error cost-tuned alarm was cheaper on
  FD001, FD003 (8 and 11 of 16 cells) and partly FD002/FD004; the capacity trend was cheapest on the batteries. CNA-DA
  beat age replacement, the heuristic Kamariotis threshold, the LB-near alarm and plain cost tuning on several datasets.
  Proposition 2 (the 1/(n+1) floor) explains most of the gap; this explanation was derived after seeing the cost results
  (it is a theorem, but its prominence in the paper is post hoc).
* Under batch shift the uncertified, conservative rules (1-SE, capacity trend) failed on 0–2 % of held-out cells while
  CNA failed on 61–74 %. A calibrated alarm has no margin beyond what the calibration units require.
* With the 60/40 proper-training/calibration split, the window-tuned recipe scored lower across battery batches
  (54.5–62.6 %) than in the original fully nested cross-fitted analysis (79.7 %), so all learned policies here use less
  training data than in the original paper.

## 5. Remaining weaknesses

* Simulated (C-MAPSS family) and laboratory (battery) data only; no censoring, no field deployment. Real fleets censor
  units by replacing them, which the method does not yet handle (would need conformal survival methods).
* Exchangeability is required and failed under the transfer that matters most in practice; the remedy (pilot units run
  to failure) is expensive.
* The guarantee is marginal, not conditional on the calibration set or on unit covariates.
* Cost results are not favourable; the paper's value rests on the guarantee and on the negative result for interval
  calibration, not on savings.
* The comparison with Kamariotis et al. adapts their policy 1 to per-cycle decisions with lead time; it is not their
  original setting.
* Single regressor family (GBM) plus one physical signal; fixed CNA settings (cap, span, debounce) were not optimised.
* The hash is not a registry preregistration; Proposition 2's emphasis and the discussion of margins were written after
  the results.
* Some citations could only be verified through search-result metadata (volume of Gupta et al. 2022; the 2026 IJPHM paper
  has no volume yet; Wang et al. 2025 article number unknown).

## 6. Self-assessed RESS acceptance rating: about 6.5/10 (major revision likely)

Reasons for: a clear, correct, simple theoretical contribution directly tied to a reliability requirement (failure before
planned replacement); full proofs; a striking, reproducible negative result for the common practice of per-cycle conformal
lower bounds (up to 70 % failures at nominal 10 %); a pre-hashed protocol with honest reporting of two failed hypotheses;
leave-batch-out transfer with a practical recalibration recipe and a sizing rule (n ≥ 1/α − 1); much shorter than the
original (16 pages vs 27).

Reasons against: the conformal machinery is standard, so a reviewer may see the novelty as an application of split
conformal to a well-chosen score; the method does not save cost; the guarantee breaks under the shift that matters; data
are mostly simulated; no censoring. A reviewer from the decision-analysis side may ask for a comparison in the original
Kamariotis setting (decision grid, metric M) and for a censored-data extension. Raising the rating would most likely need
field data with censoring, a covariate-weighted conformal variant for batch shift, and a conditional (PAC) version of the
guarantee.
