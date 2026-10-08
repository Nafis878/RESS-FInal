# Protocol amendment v3: equal-information comparison for the cross-fitted alarm

Written 2026-10-08, after the repetition-0 results of `protocol_v2.md` were seen for PHM08, FD002, FD004 and N-CMAPSS,
and before the experiment below was run. Hashed with SHA-256 (`protocol_v3.sha256` and the commit that adds it).
**Not a registry preregistration.** `code/run3.py` was smoke-tested on one FD001 fold (it reproduced the CNA+ alarms of
`run2.py` exactly); those outputs are not reported.

## Reason

In `protocol_v2.md`, CNA+ (cross-fitted on all training units) was compared on cost with policies tuned on the 60/40
proper-training/calibration split, so CNA+ had more information. This amendment gives the comparators the same information.

## Analysis

Same outer splits and the same five fold models as CNA+ (seeds 3000 + 100·rep + split; cap 60, battery 180 cycles).
Comparators selected on the out-of-fold predictions of all training units and applied to a test unit through the average of
the five fold-model predictions: cost-tuned STW, plain and one-standard-error (spans × debounce × thresholds; 200 bootstrap
resamples, seed 6026 + split + 100·rep), and policy 1 of Kamariotis et al. with P-res (optimised threshold and threshold 1/ρ).
All lead times and cost ratios; repetitions 0–2 (0 primary).

## Hypothesis and decision rule

* **H9.** CNA+-DA (from `run2.py`) against STW_cost_1se_CF and KAM_P1_opt_CF, per dataset (cross-validation, repetition 0),
  H4 rule: "cheaper" if the paired bootstrap 95 % CI of the cost-rate ratio lies below 1 in ≥ 8 of 16 cells and above 1 in
  none; "dearer" symmetric; otherwise "mixed". No directional prediction. Failure rates of the comparators are reported
  (they carry no guarantee).
