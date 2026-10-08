# 7. Results

All numbers are on outer test units. Calibration sets had 32 units (FD001, FD003, N-CMAPSS), 39 (battery), 70 (PHM08), 80 (FD004) and 83 (FD002) under cross-validation. Unless stated, rates are averaged over the three repetitions of the proper-training/calibration split, and tests use repetition 0, in which every unit is tested once.

## 7.1 The notice guarantee holds where exchangeability holds; interval calibration does not deliver it (H1, H2)

Table 2 and Fig. 2 give the failure-before-replacement rate at the primary lead time. With $\alpha = 0.10$ the conformal notice alarm failed on 6.0–11.7 % of test units across the seven datasets, and with $\alpha = 0.05$ and $0.20$ on 1.3–5.1 % and 11.7–24.0 %. Pooled over the seven datasets, four lead times and three repetitions (13,788 unit–cell evaluations), the rates were 3.8 %, 9.3 % and 19.4 % for nominal 5 %, 10 % and 20 %. In repetition 0 the highest rates at $\alpha = 0.10$ were 14.0 % (FD003, 14 of 100) and 12.4 % (FD004, 31 of 249); no exact binomial test of rate $> 0.10$ was significant after Holm's correction (smallest adjusted $p = 0.84$), so **H1 is supported on all seven datasets**. Rates slightly above 10 % (FD003 11.7 %, N-CMAPSS 11.1 %, battery 10.6 %, FD004 10.4 %) are within sampling error; for distinct critical levels the expected rate is $(n + 1 - q)/(n+1)$, 9.1 % with 32 calibration units and 10.0 % with 39. The lowest rate, 6.0 % on FD001, lies slightly below the theoretical lower limit $\alpha - 1/(n+1) = 7.0 \%$, again within the sampling error of 100 test units.

The interval-calibrated alarm on the same signal and trigger did not deliver the nominal level. Calibrated with residuals over all rows (LB-all), the lower-bound alarm at nominal 10 % failed on 18.2 % (N-CMAPSS) to 69.6 % (battery) of units, 46.6 % pooled, with $p < 0.005$ on every dataset (**H2 supported**, 7 of 7). The reason is visible in Theorem 2's construction: rows far from failure, where the capped target makes residuals small, dominate the pooled residual quantile, so the bound is tight far from failure and loose at the stopping time. Restricting the residuals to rows with $r_t \le c$ (LB-near) brought the rate close to nominal on six datasets (4.0–8.6 %) but not on N-CMAPSS (15.2 %, and 17.8 % at the longest lead time), and it has no guarantee: its level depends on how the evaluator chooses the rows. The cost of the guarantee is notice. At $\alpha = 0.10$ CNA alarms left a median of 15–17 engine cycles at a lead time of 10, against 11–12 for LB-all; on the battery cells 43 cycles at a lead time of 30.

Table: **Table 2.** Failure before the planned replacement at the primary lead time (10 cycles; battery 30 cycles), five-fold cross-validation, mean of three repetitions; the second column gives repetition 0 for CNA at $\alpha$ = 0.10 with its Clopper–Pearson 95 % interval. Last column: median notice at the alarm (cycles), CNA 0.10 / LB-all 0.10.

| Dataset (test units) | CNA 0.10, rep. 0 | CNA 0.05 | CNA 0.10 | CNA 0.20 | CNA-DA | LB-all 0.10 | LB-near 0.10 | Notice |
|--------------|--------------------|-------|-------|-------|-------|-------|-------|-------|
| PHM08 (218) | 18 (0.083; 0.050–0.127) | 0.034 | 0.086 | 0.177 | 0.009 | 0.446 | 0.080 | 16 / 11 |
| FD001 (100) | 7 (0.070; 0.029–0.139) | 0.013 | 0.060 | 0.143 | 0.013 | 0.333 | 0.043 | 17 / 12 |
| FD002 (260) | 20 (0.077; 0.048–0.116) | 0.042 | 0.088 | 0.191 | 0.008 | 0.381 | 0.040 | 15 / 12 |
| FD003 (100) | 14 (0.140; 0.079–0.224) | 0.027 | 0.117 | 0.197 | 0.027 | 0.480 | 0.063 | 16 / 11 |
| FD004 (249) | 31 (0.124; 0.086–0.172) | 0.051 | 0.104 | 0.206 | 0.011 | 0.466 | 0.086 | 16 / 11 |
| N-CMAPSS (99) | 11 (0.111; 0.057–0.190) | 0.037 | 0.111 | 0.178 | 0.057 | 0.182 | 0.152 | 23 / 19 |
| Battery (123) | 15 (0.122; 0.070–0.193) | 0.033 | 0.106 | 0.198 | 0.011 | 0.696 | 0.070 | 43 / 28 |


![**Fig. 2.** Failure before the planned replacement at the primary lead time (10 cycles; battery 30) for the conformal notice alarm (CNA) and per-cycle split-conformal lower-bound alarms on the same signal and trigger, at three nominal levels (dashed). Dots: the three repetitions; bars: their mean. Five-fold cross-validation.](figures/Fig2_validity.png){width=100%}

## 7.2 What the window score measured (H3)

Theorem 1 says that the window scores of the estimate and of a constant can differ only through units whose notice falls near an edge of the constant's acceptance band. On every dataset the reported estimate at the alarm was pinned below the level as part (iii) states (100 % of alarms within $[H - k\delta^+, H]$) and varied much less than the notice: its standard deviation was 2.3–3.2 cycles on the engines against 4.9–18.0 cycles for the notice (battery 6.3 against 12.9). Discordant units, correct under one report and not the other, were 6.9–14.0 % on the C-MAPSS datasets and PHM08, 11.4 % on the battery cells and 30.3 % on N-CMAPSS (Table 3).

The protocol's equivalence test, however, was **not supported**. With a margin of ±5 points, model and constant were equivalent only on FD002. On FD001, FD003 and N-CMAPSS the intervals were too wide to decide. On PHM08 and FD004 the constant was significantly *better* than the model's estimate (−6.4 and −6.0 points; 90 % intervals −10.3 to −3.1 and −9.7 to −2.7), and on the battery cells the estimate was significantly better than the constant (+9.8 points, +5.4 to +15.3), the one dataset where the estimate's spread at the alarm was a sizeable fraction of the window width (6.3 of 45 cycles). This corrects the earlier study's summary that the estimate changed the score by at most 3.6 points [@prior]: in the present design, with 60 % of the training units used for fitting, the estimate moved the score by up to 10 points in either direction. What did hold is the qualitative reading of Theorem 1: the score is determined by the alarm time up to a band of discordant units that is small when the estimate's spread at the alarm is small. The plug-in bound of part (ii), computed on calibration units, exceeded the realised score difference on 6 of 7 datasets and the realised discordance on 4 of 7 (Fig. 3b); it is an estimate, not a guarantee, because $\pi_a$ and $\pi_b$ are estimated. For the purpose of this paper the conclusion is the one that motivates Section 5: a window score bounds neither the notice nor the failure probability, so neither should be inferred from it.

Table: **Table 3.** Window score of the window-tuned alarm's estimate and of one constant at the same alarm times ($H$ = 20 cycles; battery 60), repetition 0. Difference: estimate − constant with Tango 90 % interval; equivalence margin ±5 points (protocol). SD: standard deviation at the alarm (cycles). Bound: Theorem 1 (ii) estimated on calibration units; Disc.: realised share of discordant units.

| Dataset | n | Estimate (%) | Constant (%) | Discordant (est./const.) | Difference (90 % CI) | TOST p | SD estimate / notice | Bound / Disc. (points) |
|---|---|---|---|---|---|---|---|---|
| PHM08 | 218 | 87.6 | 94.0 | 4 / 18 | −6.4 (−10.3, −3.1) | 0.756 | 2.6 / 4.9 | 15.9 / 10.1 |
| FD001 | 100 | 86.0 | 85.0 | 6 / 5 | +1.0 (−4.9, +7.0) | 0.126 | 3.0 / 5.3 | 19.1 / 11.0 |
| FD002 | 260 | 90.4 | 91.9 | 7 / 11 | −1.5 (−4.5, +1.2) | 0.028 | 2.4 / 5.2 | 13.6 / 6.9 |
| FD003 | 100 | 84.0 | 86.0 | 6 / 8 | −2.0 (−8.6, +4.5) | 0.216 | 2.9 / 5.3 | 13.6 / 14.0 |
| FD004 | 249 | 79.1 | 85.1 | 6 / 21 | −6.0 (−9.7, −2.7) | 0.694 | 2.3 / 18.0 | 15.2 / 10.8 |
| N-CMAPSS | 99 | 52.5 | 56.6 | 13 / 17 | −4.0 (−13.3, +5.2) | 0.431 | 3.2 / 11.0 | 28.1 / 30.3 |
| Battery | 123 | 95.1 | 85.4 | 13 / 1 | +9.8 (+5.4, +15.3) | 0.964 | 6.3 / 12.9 | 8.9 / 11.4 |


![**Fig. 3.** (a) Difference in window score between the model's estimate and a constant at the same alarm times (window-tuned alarm at $H$ = 20 cycles, battery 60), with Tango 90 % intervals; shaded: equivalence margin of ±5 points fixed in the protocol. (b) Theorem 1 bound estimated on calibration units, against the realised discordance and score difference on test units.](figures/Fig3_equivalence.png){width=100%}

## 7.3 Cost: a certificate, not a saving (H4)

Table 4 and Fig. 4 compare the decision-aware CNA ($\alpha_{\max} = 0.10$) with every other policy in the 16 lead-time and cost-ratio cells. Its failure rate stayed at or below $\alpha_{\max}$ in every cell with a positive lead time on every dataset (largest 6.1 %; no binomial test significant; **H4b supported**), and over all cells and repetitions it failed on 1.8 % of units. By the protocol's rule it was cheaper than age replacement on PHM08, FD002, FD004 and the battery cells (12–16 of 16 cells) and with a battery batch held out (15 of 16), and never significantly dearer on any dataset; cheaper than the probability-threshold policy with the heuristic threshold $1/\rho$ on PHM08, FD002 and FD004 and with N-CMAPSS subsets held out; cheaper than the LB-near alarm on five of seven datasets; cheaper than plain cost tuning on PHM08 and across battery batches; and cheaper than the optimised probability-threshold policy of Kamariotis et al. on PHM08, FD001 and the battery cells (8 of 16 cells each, never dearer there).

It was **not the cheapest policy**. The one-standard-error cost-tuned alarm, which chooses the most conservative setting within one bootstrap standard error of the minimum calibration cost, was cheaper than CNA-DA in 8 of 16 cells on FD001, 11 on FD003, 5 on FD002 and 4 on FD004, and dearer only in 4 cells on PHM08 and 1 each on FD002 and N-CMAPSS; its mean excess over the cheapest policy in each cell was 0–6 % on the engine and battery datasets, against 9–84 % for CNA-DA. Against the window-tuned alarm the comparison was mixed on every dataset, and against the optimised probability-threshold policy on FD002–FD004 and N-CMAPSS. On the battery cells the capacity trend was cheaper than CNA-DA in 12 of 16 cells (window-tuned) and was the cheapest policy overall, as in the earlier study. By the protocol's rule, **H4 is not supported** for the claim that the guaranteed alarm saves cost.

Proposition 2 explains most of the gap. CNA-DA can only choose among calibration critical levels, so its failure probability is at least $1/(n+1)$, about 3 % with 32 calibration units. At $\rho = 50$ each such failure costs as much as 49 extra planned replacements, so the floor alone adds roughly 1.5 replacement costs per 33 renewals. The cost-tuned and window-tuned rules are free to choose levels beyond the most conservative calibration unit; when they do, they fail less on these test units, but nothing certifies that they will on others. The cost ratios against the one-standard-error rule grew with $\rho$ and were largest on the small calibration sets of FD001 and FD003 ($n = 32$; median ratio 1.37 and 1.45) and smallest on PHM08 and FD002 ($n = 70$–83; 1.00 and 1.04). N-CMAPSS, also with $n = 32$, was an exception (1.06) because every policy failed often there.

Table: **Table 4.** Cost of the decision-aware conformal alarm (CNA-DA, $\alpha_{\max}$ = 0.10) against each policy over the 16 lead-time and cost-ratio cells (repetition 0): cells in which the paired bootstrap 95 % interval of the cost-rate ratio CNA-DA / comparator lies below 1 / above 1, and the median ratio. LOGO: leave one group (N-CMAPSS subset, battery batch) out. Bottom rows: mean excess cost over the cheapest policy in each cell.

| CNA-DA vs | PHM08 | FD001 | FD002 | FD003 | FD004 | N-CMAPSS | Battery | N-CMAPSS LOGO | Battery LOGO |
|--------------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Age replacement | 15/0 (0.61) | 6/0 (0.86) | 12/0 (0.69) | 5/0 (0.82) | 12/0 (0.63) | 0/0 (1.17) | 16/0 (0.51) | 2/0 (1.03) | 15/0 (0.57) |
| STW, window-tuned | 4/0 (0.99) | 0/0 (1.12) | 4/0 (1.04) | 0/3 (1.15) | 0/1 (1.03) | 0/2 (1.10) | 0/4 (1.00) | 2/4 (1.09) | 0/16 (2.58) |
| STW, cost-tuned, plain | 12/0 (0.87) | 4/0 (0.88) | 0/0 (0.97) | 0/0 (0.87) | 7/0 (0.83) | 3/0 (0.91) | 0/0 (0.87) | 7/1 (0.91) | 8/0 (0.90) |
| STW, cost-tuned, 1-SE | 4/0 (1.00) | 0/8 (1.37) | 1/5 (1.04) | 0/11 (1.45) | 0/4 (1.13) | 1/0 (1.06) | 0/0 (1.10) | 0/5 (1.02) | 0/16 (4.17) |
| Kamariotis P1, $p = 1/\rho$ | 12/0 (0.62) | 7/0 (0.79) | 13/0 (0.69) | 9/2 (0.69) | 13/0 (0.64) | 12/1 (0.67) | 6/4 (0.80) | 14/0 (0.56) | 7/8 (1.12) |
| Kamariotis P1, optimised | 8/0 (0.99) | 8/0 (1.00) | 8/3 (1.00) | 0/4 (1.15) | 3/1 (1.03) | 0/3 (1.12) | 8/0 (0.99) | 1/0 (1.06) | 0/16 (2.41) |
| LB-near ($\alpha$ = 0.10) | 16/0 (0.47) | 4/0 (0.84) | 11/0 (0.81) | 0/0 (0.88) | 16/0 (0.58) | 13/0 (0.63) | 11/0 (0.71) | 13/1 (0.53) | 8/4 (0.92) |
| Capacity trend, window-tuned | – | – | – | – | – | – | 0/12 (1.17) | – | 0/16 (5.48) |
| Capacity trend, 1-SE | – | – | – | – | – | – | 0/4 (1.15) | – | 0/16 (5.41) |
| CNA-DA on capacity trend | – | – | – | – | – | – | 0/0 (1.03) | – | 0/16 (3.76) |
| *Mean excess: CNA-DA* | 9 % | 63 % | 12 % | 84 % | 16 % | 41 % | 30 % | 33 % | 708 % |
| *STW, cost-tuned, 1-SE* | 1 % | 0 % | 3 % | 5 % | 0 % | 41 % | 6 % | 35 % | 53 % |
| *Kamariotis P1, optimised* | 4 % | 51 % | 7 % | 42 % | 13 % | 8 % | 27 % | 22 % | 188 % |
| *STW, window-tuned* | 2 % | 35 % | 3 % | 38 % | 11 % | 22 % | 30 % | 13 % | 214 % |


![**Fig. 4.** Long-run cost rate relative to perfect foresight (log scale) at cost ratios 10 and 50 as a function of the lead time (cycles; battery charge–discharge cycles), five-fold cross-validation, repetition 0.](figures/Fig4_cost.png){width=100%}

## 7.4 Transfer: the guarantee does not survive a new batch unless it is recalibrated (H5)

With a whole battery batch held out, calibration units and test units are no longer exchangeable, and the guarantee failed as predicted (**H5a supported**). Calibrated on the other two batches, CNA at $\alpha = 0.10$ failed on 0 % of batch 1, but on 74 % of batch 2 and 61 % of batch 3 (Table 5, Fig. 5). The shortest-lived batch was the worst, which is the direction expected when a regressor trained mostly on longer-lived cells overestimates remaining life. CNA-DA failed on 45 % and 36 %, and in the worst cell its cost was 24 times that of the cheapest policy. The same loss of validity appeared on N-CMAPSS with a flight-data subset held out: 29–40 % failures on three of nine subsets and 0–2 % on the other six. Conservative uncertified rules did much better under this shift: the one-standard-error cost-tuned alarm failed on 0–2 % of the held-out battery cells and the capacity-trend policies on none. Their extra margin, which cost them little within batches, protected them across batches; a calibrated alarm has, by design, no margin beyond what the calibration units require.

Recalibration with pilot units of the new batch restored the guarantee (**H5b supported**). Using 10 randomly chosen cells of the held-out batch as the calibration set and testing on the rest, the mean failure rate over 300 draws was 7.5–10.1 % at $\alpha = 0.10$ across batches, lead times and repetitions (8.4–9.5 % at the primary lead time), around the exact value $1/11 = 9.1 \%$ of Proposition 2; with 20 pilot cells it was 8.3–10.4 %. The price is notice and pilot units. Mean notice at a planned replacement was 48–60 cycles with 10 pilot cells, against 37 cycles on batches 2 and 3 under cross-batch calibration, where most cells failed; five pilot cells cannot certify $\alpha = 0.10$ at all ($1/(m+1) > \alpha$), and certify $\alpha = 0.20$ with a failure rate of 16.5–16.7 %. The capacity-trend signal behaved the same way: calibrated across batches it failed on 56 % of batch 2, and with 10 pilot cells on 9 %.

Table: **Table 5.** Failure before replacement with a whole group held out (battery: lead time 30 cycles; N-CMAPSS: 10 cycles), mean of three repetitions. Pilot: CNA calibrated on $m$ randomly chosen units of the held-out group and tested on the rest (mean over 300 draws); with $m = 5$, $\alpha = 0.10$ is not attainable and $\alpha = 0.20$ is shown.

| Held-out group (units) | CNA 0.10, other groups | CNA-DA | Pilot $m$ = 10, $\alpha$ = 0.10 | Pilot $m$ = 5, $\alpha$ = 0.20 | STW window | STW 1-SE | Kamariotis P1 opt. | Trend 1-SE |
|---|---|---|---|---|---|---|---|---|
| Battery batch 1 (36) | 0.00 | 0.00 | 0.095 | 0.167 | 0.00 | 0.00 | 0.00 | 0.00 |
| Battery batch 2 (43) | 0.74 | 0.45 | 0.094 | 0.165 | 0.35 | 0.00 | 0.12 | 0.00 |
| Battery batch 3 (44) | 0.61 | 0.36 | 0.084 | 0.167 | 0.06 | 0.02 | 0.17 | 0.00 |
| N-CMAPSS, 6 subsets (9–15 each) | 0.00–0.02 | 0.00 | – | 0.165–0.177 | 0.00–0.02 | 0.00 | 0.00 | – |
| N-CMAPSS subset 4 (10) | 0.33 | 0.10 | – | 0.165 | 0.07 | 0.10 | 0.07 | – |
| N-CMAPSS subset 8 (15) | 0.29 | 0.18 | – | 0.169 | 0.11 | 0.18 | 0.11 | – |
| N-CMAPSS subset 9 (10) | 0.40 | 0.07 | – | 0.169 | 0.10 | 0.13 | 0.03 | – |


![**Fig. 5.** Battery cells with a whole batch held out ($\alpha$ = 0.10, lead time 30 cycles). (a) Failure before replacement when the alarm is calibrated on the other batches (dots: three repetitions) or on 10 pilot cells of the held-out batch (diamonds: mean over 300 random draws). (b) Failure against mean notice at planned replacements for the gradient-boosted signal.](figures/Fig5_transfer.png){width=100%}
