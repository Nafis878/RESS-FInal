---
title: "Calibrating the alarm, not the interval: distribution-free notice guarantees for end-of-life alarms"
author: "Anonymised for review"
---

**Abstract.** End-of-life alarms schedule replacements, so their first reliability requirement is that the unit does not fail before the replacement the alarm scheduled. Conformal prediction has been used to give remaining-useful-life intervals with distribution-free coverage, but an alarm acts at the data-dependent cycle at which an estimate first crosses a level, where per-cycle coverage does not apply. We calibrate the alarm itself. For a debounced threshold trigger on any causal signal, the notice left by the alarm is monotone in the warning level, so each run-to-failure calibration unit has a single critical level, and the split-conformal quantile of these levels gives a warning level with a finite-sample guarantee: a new exchangeable unit is replaced before it fails with probability at least 1 − α. A decision-aware choice of the level keeps the guarantee, and the calibration-set size n sets a floor of 1/(n+1) on the failure probability that can be certified. A supporting bound shows that the window score of such an alarm depends on its estimate only through units near the window edges. Under a protocol hashed before the experiments, on PHM08, C-MAPSS FD001–FD004, N-CMAPSS and 123 lithium-ion cells, failure rates were 3.8, 9.3 and 19.4 % at nominal 5, 10 and 20 %, whereas per-cycle conformal lower bounds on the same signal failed on 18–70 % of units at nominal 10 %. The guarantee was not free: a one-standard-error cost-tuned alarm and a capacity trend were often cheaper, without a guarantee. Calibrated on other battery batches, the alarm failed on 61–74 % of two held-out batches; ten pilot cells of the new batch restored the nominal rate. An equivalence test did not confirm that window scores of estimate and constant differ by less than five points.

**Keywords:** predictive maintenance; remaining useful life; end-of-life alarm; conformal prediction; lead time; distribution shift; lithium-ion battery

# 1. Introduction

A prognostic model creates value through the maintenance actions it triggers. For a component that is replaced once, near the end of its life, the action is usually triggered by an *end-of-life alarm*: a policy watches the unit cycle by cycle and raises one alarm, which schedules a replacement after a lead time for parts, crew and access. The first reliability requirement on such an alarm is therefore not the accuracy of a remaining useful life (RUL) estimate at some cycle chosen by the evaluator, but the probability that the unit fails before the replacement the alarm scheduled. That probability is the subject of this paper.

The common ways of evaluating and calibrating prognostics do not control it. Accuracy metrics and asymmetric acceptance windows score an estimate at a time; a window score at the alarm mixes the estimate with the alarm time [1, 2]. In an earlier evaluation study [3] we showed on eight run-to-failure datasets that the asymmetric window score of a threshold-crossing alarm was driven mainly by the alarm time, that optimising it was not the same as optimising maintenance cost, and that accuracy validated within familiar conditions did not hold on a held-out battery batch. That study proposed no method. Conformal prediction gives RUL intervals with distribution-free coverage [4–6], but the coverage holds at a cycle chosen independently of the data, whereas an alarm acts at the first cycle at which the estimate, or its lower bound, crosses a level: a stopping time at which the estimate is, by selection, at its most optimistic. Decision-oriented metrics, such as the metric $M$ of Kamariotis et al. [7], score and tune policies by long-run cost, but give no guarantee for a new unit.

This paper calibrates the alarm itself. For a threshold trigger on any causal signal, the notice left by the alarm is monotone in the warning level, so each run-to-failure calibration unit has a single *critical level*: the lowest warning level at which its alarm would still have left more than the lead time. The split-conformal quantile of these critical levels gives a warning level with a finite-sample, distribution-free guarantee that a new exchangeable unit is replaced before it fails with probability at least $1 - \alpha$. The contributions are:

1. **Theory.** A closed-form critical level for debounced threshold triggers (Lemma 2), the notice guarantee (Theorem 2) with a matching upper bound, a decision-aware choice of the level that keeps the guarantee (Corollary 1), and a cost bound and a floor that show what a calibration set of $n$ units can certify (Propositions 1 and 2). A supporting result (Theorem 1) bounds how far the window score of a threshold alarm's estimate can differ from that of a constant, which explains why such scores measure timing rather than the estimate and why they cannot certify notice.
2. **Method.** The conformal notice alarm (CNA) and its decision-aware variant (CNA-DA), which wrap any signal, including a learned regressor or a physically motivated capacity trend.
3. **Evidence.** A protocol fixed and hashed before the runs, seven datasets (PHM08, C-MAPSS FD001–FD004, N-CMAPSS, 123 lithium-ion cells in three batches), comparisons with interval-calibrated alarms, window- and cost-tuned alarms, the probability-threshold policy of Kamariotis et al., age replacement and a capacity-trend baseline, and transfer to held-out battery batches and N-CMAPSS subsets.

The main findings are four. (i) Under exchangeable splits the guarantee held on all seven datasets (pooled failure rates 3.8, 9.3 and 19.4 % at nominal 5, 10 and 20 %), whereas a per-cycle conformal lower bound on the same signal and trigger failed on 18–70 % of units at nominal 10 %. (ii) The guarantee is a certificate, not a saving: a decision-aware choice kept failures below the stated ceiling at every cost ratio, but a one-standard-error cost-tuned alarm and, on the batteries, a capacity trend were cheaper on several datasets; Proposition 2 explains why, through the $1/(n+1)$ floor. (iii) The guarantee did not transfer: calibrated on other battery batches, the alarm failed on 61–74 % of the cells of two held-out batches, and ten run-to-failure pilot cells of the new batch restored the nominal rate. (iv) The earlier finding that a constant estimate scores like the model's estimate was not confirmed by an equivalence test; the window score depended on the estimate by up to 10 points, in the direction and at the scale that Theorem 1 allows. Negative results are reported with the same weight as positive ones.

Section 2 reviews related work, Section 3 states the problem, Section 4 gives the theory, Section 5 the method and Section 6 the design. Section 7 reports the results, Section 8 discusses them and their limits, and Section 9 concludes.


# 2. Related work

**Evaluating prognostics through decisions.** Remaining useful life (RUL) estimates are usually scored at times chosen by the evaluator, with accuracy metrics or asymmetric scores that penalise late predictions [1, 2, 8]. A line of work in this journal evaluates prognostics through the maintenance decisions they trigger. de Pater et al. [9] schedule aircraft maintenance from alarms raised on RUL prognostics and protect against prognostic error with a safety factor; Kamariotis et al. [7] propose a metric $M$, the relative excess long-run cost of a policy over perfect prognostics, and tune probability-threshold policies with it; dynamic predictive-maintenance policies act on predicted failure probabilities [10, 11]. Classical replacement theory supplies the benchmark that any condition-based policy must beat, age replacement [12, 13]. None of these studies gives a finite-sample guarantee on the notice an alarm leaves.

**Conformal prediction for RUL.** Conformal prediction converts any point predictor into prediction sets with finite-sample, distribution-free coverage under exchangeability [14–16]. Javanmardi and Hüllermeier [4] conformalised gradient boosting and convolutional networks to obtain RUL intervals on C-MAPSS; Robinson [5] used asymmetric conformalised quantile regression [17] to make the lower RUL bound conservative; Wang et al. [6] combined conformal prediction with imputation of missing sensor data, and note that exchangeability is a concern for degradation data. In all of these the calibrated object is the interval at a given cycle, and coverage is averaged over cycles (or units at a fixed time). Candès et al. [18] give conformal lower predictive bounds on survival times under censoring, again at a fixed covariate value. An alarm, however, is a stopping rule: it is evaluated at the cycle at which the bound first crosses a level, a time that depends on the data. Per-cycle coverage does not imply coverage at a data-dependent time, and our experiments show that the difference is large (Section 7.1).

**Risk control and non-exchangeable data.** Conformal risk control [19] calibrates a scalar parameter so that the expected value of a monotone loss is bounded; nested conformal prediction [20] expresses conformal procedures through nested families of sets. Our construction is an instance of this nested view, with the warning level as the nesting parameter and failure before the planned replacement as the loss; what is specific is the closed-form critical level of a debounced threshold trigger (Lemma 2), which makes the calibration exact and cheap, and the decision-aware selection that keeps the guarantee (Corollary 1). When calibration and test units are not exchangeable, as across battery batches, the guarantee can fail; weighted and non-exchangeable conformal methods bound the resulting coverage gap [21, 22]. We do not use them, because they need a weight model for the shift; we instead measure the failure of exchangeability and the cost of restoring it with a few run-to-failure pilot units from the new population.

**What this paper adds.** (i) A conformal calibration of the alarm time itself, with a finite-sample guarantee on the probability that the remaining life at the alarm exceeds the lead time, a decision-aware variant that keeps the guarantee, and a cost bound. (ii) A bound on the difference between the window score of a threshold-crossing alarm's estimate and that of a constant, with an equivalence test, which explains why such scores measure timing. (iii) A controlled evaluation on seven datasets, against interval-calibrated alarms, window- and cost-tuned alarms, the probability-threshold policy of Kamariotis et al., age replacement and a capacity-trend baseline, with whole battery batches and N-CMAPSS subsets held out. We do not claim a new RUL model.

# 3. Problem formulation

**Units, alarms and notice.** A unit observed at rows $j = 1, \dots, n$ (cycles $t_1 < \dots < t_n$) has life $L$ and remaining life $r_j = L - t_j \ge 1$, decreasing in $j$. A causal signal $s_j$, here a smoothed RUL estimate, is computed from rows $1, \dots, j$. A *threshold trigger* with debounce $k$ and warning level $H$ raises an alarm at the first row at which the signal has been at or below $H$ for $k$ consecutive rows. Writing $m_j = \max(s_{j-k+1}, \dots, s_j)$ for $j \ge k$ ($m_j = +\infty$ otherwise) and $\rho_j = \min_{i \le j} m_i$ (the *debounced running minimum*), the alarm row is
$$\tau(H) = \min\{ j : \rho_j \le H \}, \qquad (\tau(H) = \infty \text{ if no such row}),$$
and the *notice* is $N(H) = r_{\tau(H)}$ ($N = 0$ if no alarm).

**Maintenance consequence.** An alarm schedules a replacement $\ell$ cycles later (the *lead time*). If $N(H) > \ell$ the planned replacement happens at $t_{\tau(H)} + \ell$ at cost 1; otherwise, or without an alarm, the unit fails at $L$ at cost $\rho > 1$. Units are renewed, so by the renewal–reward theorem the long-run cost per cycle is $E[\text{cost}]/E[\text{use}]$, estimated by total cost over total cycles of use; we report it relative to perfect foresight (replacement at $L-1$, cost 1) [12]. The engineering requirement we target is reliability of the alarm: the probability of failure before the planned replacement, $P(N(H) \le \ell)$, should not exceed a stated $\alpha$.

**The window score.** The original study scored the first alarm by the asymmetric window: with reported estimate $\hat Y$ and error $e = \hat Y - R$, $R = N(H)$, the alarm is correct if $a \le e \le b$, $[a, b] = [-13, +2]$ cycles for the engines and $[-39, +6]$ for the battery cells. The window score is the share of correct first alarms.

# 4. Theory

## 4.1 What the window score of a threshold alarm measures

**Theorem 1 (window score of an estimate versus a constant).** Let $R \in \mathbb{Z}$ be the notice at the alarm, $\hat Y$ the reported estimate, $c$ any constant and $D = \hat Y - c$. Define the half-open intervals
$$I_b = [\,c - b,\; c - b + D\,), \qquad I_a = (\,c - a,\; c - a + D\,] \qquad (D \ge 0),$$
and their mirror images $I_b = [c - b + D, c - b)$, $I_a = (c - a + D, c - a]$ for $D < 0$. Then:

(i) $\bigl|\,\mathbf 1\{a \le \hat Y - R \le b\} - \mathbf 1\{a \le c - R \le b\}\,\bigr| \le \mathbf 1\{R \in I_a\} + \mathbf 1\{R \in I_b\}$.

(ii) For the window scores $S_{\text{est}} = P(a \le \hat Y - R \le b)$ and $S_c = P(a \le c - R \le b)$ (or their empirical versions), $|S_{\text{est}} - S_c| \le P(R \in I_a) + P(R \in I_b)$. If, conditionally on $\hat Y$, the probability mass function of $R$ is at most $\pi_a$ on integers within $\Delta$ of $c - a$ and at most $\pi_b$ within $\Delta$ of $c - b$, and $|D| \le \Delta$, then
$$|S_{\text{est}} - S_c| \le (\pi_a + \pi_b)\, E\lceil |D| \rceil .$$

(iii) (Threshold-crossing alarms.) Let the reported estimate be the triggering signal at the alarm, $\hat Y = s_{\tau}$, and suppose the alarm is not raised at the first possible row ($\tau > k$). Then
$$H - k\,\delta^{+} \le \hat Y \le H, \qquad \delta^{+} = \max\bigl(0, \max_{\tau-k < i \le \tau} (s_{i-1} - s_i)\bigr),$$
so with $c = H - k\bar\delta/2$ and $\delta^+ \le \bar\delta$, $|D| \le k\bar\delta/2$ and $|S_{\text{est}} - S_c| \le (\pi_a + \pi_b)\lceil k\bar\delta/2 \rceil$.

*Proof.* (i) Take $D \ge 0$; the case $D < 0$ is symmetric. The estimate's acceptance set for $R$ is $[c - b + D, c - a + D]$ and the constant's is $[c - b, c - a]$. If $R$ lies in the first but not the second, then $R > c - a$ and $R \le c - a + D$, so $R \in I_a$. If $R$ lies in the second but not the first, then $R \ge c - b$ and $R < c - b + D$, so $R \in I_b$. Otherwise the indicators agree. This holds also when $D$ exceeds the window width $b - a$, in which case the two acceptance sets are disjoint and each is contained in $I_b$ or $I_a$. (ii) Take expectations of (i). A half-open interval of length $|D|$ contains at most $\lceil |D| \rceil$ integers, each of conditional probability at most $\pi_a$ (for $I_a$) or $\pi_b$ (for $I_b$); condition on $\hat Y$ and take expectations. (iii) $\hat Y = s_\tau \le m_\tau \le H$ because the alarm is raised at $\tau$. Because it was not raised at $\tau - 1 \ge k$, $\rho_{\tau-1} > H$, so $m_{\tau - 1} > H$: some row $i_0 \in [\tau - k, \tau - 1]$ has $s_{i_0} > H$. Then $s_\tau = s_{i_0} - \sum_{i = i_0 + 1}^{\tau} (s_{i-1} - s_i) > H - (\tau - i_0)\delta^+ \ge H - k\delta^+$. The final claim follows from (ii). $\square$

Theorem 1 makes the observation of the original study precise. For a threshold-crossing alarm the reported estimate is pinned to a band of width $k\delta^+$ below the level, so the window condition is, up to edge effects of size $(\pi_a + \pi_b)E\lceil|D|\rceil$, a condition on the notice: $R \in [c - b, c - a]$. A window score of such an alarm is therefore a *notice-band probability*. It can be high while the notice is too short for the lead time (the band admits $R$ as low as $c - b$), and it says nothing about the probability of failure before a replacement that needs $\ell$ cycles. Theorem 1 does not say that the score is uninformative; it says what it is informative about. The bound is computable from calibration data (Section 6.4) and is checked against held-out units in Section 7.2, together with an equivalence test.

## 4.2 A finite-sample guarantee on notice

The quantity an operator needs is $P(N(H) > \ell)$. Two elementary facts about the trigger make it calibratable.

**Lemma 1 (monotonicity).** For $H \le H'$, $\tau(H') \le \tau(H)$ and $N(H') \ge N(H)$.

*Proof.* $\rho$ is non-increasing, so $\{j : \rho_j \le H\} \subseteq \{j : \rho_j \le H'\}$; the first element of the larger set is not later, and $r$ is decreasing in $j$. $\square$

**Lemma 2 (critical level).** Let $j_\ell = \max\{ j : r_j > \ell \}$ and define the *critical level* $V(\ell) = \rho_{j_\ell}$ ($V = +\infty$ if no row has $r_j > \ell$). Then for every finite $H$, $N(H) > \ell \iff H \ge V(\ell)$.

*Proof.* Since $r$ is decreasing, $N(H) > \ell$ iff the alarm is raised at a row $\tau(H) \le j_\ell$, iff $\rho_j \le H$ for some $j \le j_\ell$, iff $\rho_{j_\ell} \le H$ because $\rho$ is non-increasing. $\square$

$V$ is the lowest warning level at which this unit's alarm would still have left more than $\ell$ cycles. It is a single number per unit, computed from its signal and its life in one pass, and it is all that calibration needs.

**Theorem 2 (notice guarantee).** Fix the signal map (for example, a regressor fitted on units disjoint from those below), the smoothing, the debounce $k$ and the lead time $\ell$, and assume that every unit has at least $k$ rows with $r_j > \ell$, so that $V < \infty$. Let units $1, \dots, n$ (calibration) and $n + 1$ (a new unit) be exchangeable, with critical levels $V_1, \dots, V_{n+1}$. For $\alpha \in [1/(n+1), 1)$ let $\hat H = V_{(\lceil (n+1)(1-\alpha) \rceil)}$, the order statistic of $V_1, \dots, V_n$. Then
$$P\bigl( N_{n+1}(\hat H) > \ell \bigr) \ge 1 - \alpha,$$
where the probability is over the calibration units and the new unit. If the $V_i$ are almost surely distinct, the probability is also at most $1 - \alpha + 1/(n+1)$.

*Proof.* By Lemma 2 the event is $\{V_{n+1} \le \hat H\}$. Let $q = \lceil (n+1)(1-\alpha) \rceil \le n$. If $V_{n+1} > V_{(q)}$, at least $q$ of the calibration values are strictly smaller than $V_{n+1}$, so the rank of $V_{n+1}$ among all $n+1$ values (ties broken uniformly at random) is at least $q + 1$. By exchangeability that rank is uniform on $\{1, \dots, n+1\}$, so $P(V_{n+1} > \hat H) \le (n + 1 - q)/(n+1) \le \alpha$. For the upper bound, with distinct values $V_{n+1} \le V_{(q)}$ iff the rank is at most $q$, which has probability $q/(n+1) < 1 - \alpha + 1/(n+1)$. $\square$ For $\alpha < 1/(n+1)$ the level is not attainable with $n$ calibration units, and the procedure reports this instead of an alarm level.

**Corollary 1 (decision-aware choice keeps the guarantee).** Let $q_0 = \lceil (n+1)(1 - \alpha_{\max}) \rceil$ and let $\hat q \ge q_0$ be any index chosen from the calibration data, for example to minimise an estimated cost. Then $P(N_{n+1}(V_{(\hat q)}) > \ell) \ge 1 - \alpha_{\max}$.

*Proof.* $V_{(\hat q)} \ge V_{(q_0)}$; by Lemma 1 the event for $V_{(\hat q)}$ contains the event for $V_{(q_0)}$, which has probability at least $1 - \alpha_{\max}$ by Theorem 2. $\square$

**Proposition 1 (cost bound).** Under the conditions of Theorem 2, the expected cost of one renewal cycle of a new unit is at most $1 + (\rho - 1)\alpha$, and the long-run cost rate is at most $(1 + (\rho - 1)\alpha)/E[\text{use}]$.

*Proof.* The cost is $1 + (\rho - 1)\mathbf 1\{N \le \ell\}$; take expectations and apply the renewal–reward theorem. $\square$

**Proposition 2 (what $n$ calibration units can certify).** If the critical levels are almost surely distinct, the alarm at $V_{(q)}$ fails before replacement with marginal probability exactly $(n + 1 - q)/(n + 1)$. Hence no level chosen among the order statistics has a failure probability below $1/(n+1)$, the expected cost of a renewal cycle of CNA-DA is at least $1 + (\rho - 1)/(n + 1)$, and certifying a failure probability $\alpha$ requires $n \ge 1/\alpha - 1$ calibration units.

*Proof.* With distinct values the rank of $V_{n+1}$ among the $n+1$ values is uniform, and the alarm fails iff $V_{n+1} > V_{(q)}$, i.e. iff the rank exceeds $q$; this has probability $(n + 1 - q)/(n+1)$, minimised at $q = n$. The cost statement follows as in Proposition 1. $\square$

Proposition 2 states the price of a distribution-free certificate. With 32 calibration units and a failure costing fifty planned replacements, any order-statistic level carries an expected excess of at least $49/33 \approx 1.5$ replacement costs per renewal over a policy that never fails. Heuristics that choose levels beyond the largest calibration critical level can do better on a given dataset, but nothing certifies them.

**Remarks.** (1) The guarantee is marginal: it averages over the draw of the calibration units. Conditional on a calibration set the realised failure probability is random, with a Beta$(q, n+1-q)$ distribution for continuous $V$ [23]; a larger $q$ gives a guarantee with high probability over the calibration draw. (2) The guarantee needs no model of the degradation or of the RUL error; it can be applied to any causal signal and any monotone trigger, including a physically motivated one (Section 7.3). (3) It does not hold when calibration and new units are not exchangeable, for example when the new unit comes from a different production batch; Section 7.4 measures what happens then. (4) Per-cycle conformal intervals do not deliver this guarantee: they control $P(r_t \ge \text{LB}_t)$ for a cycle drawn independently of the data, whereas the alarm is evaluated at the data-dependent time at which the bound first crosses a level, where the signal is, by selection, at its most optimistic.


# 5. Method: the conformal notice alarm

**Algorithm (CNA).** Inputs: a causal signal (here the smoothed estimate of a gradient-boosted regressor), debounce $k$, lead time $\ell$, level $\alpha$, and calibration units disjoint from the units used to fit the signal.

1. For each calibration unit $i$, compute the debounced running minimum $\rho^{(i)}$ of its signal and its critical level $V_i = \rho^{(i)}_{j_\ell}$, the value of $\rho$ at its last row with more than $\ell$ cycles left (Lemma 2).
2. Set $\hat H = V_{(\lceil (n+1)(1-\alpha) \rceil)}$.
3. In service, raise the alarm at the first row at which the signal has been at or below $\hat H$ for $k$ consecutive rows, and schedule the replacement $\ell$ cycles later.

Calibration costs one pass over the calibration series. Fig. 1 illustrates it on FD001.

![**Fig. 1.** The conformal notice alarm on FD001 (fold 1, lead time 10 cycles). (a) The smoothed estimate of one calibration engine and its debounced running minimum $\rho_t$; the critical level $V$ is the value of $\rho$ at the last cycle with more than $\ell$ cycles left (dot). (b) Critical levels of the 32 calibration engines and the conformal level $\hat H$ at $\alpha$ = 0.10.](figures/Fig1_method.png){width=100%}

**Decision-aware variant (CNA-DA).** When a cost ratio $\rho$ is available, the operator fixes a ceiling $\alpha_{\max}$ on the probability of failure before replacement and lets the data choose the warning level among the admissible order statistics $V_{(q)}$, $q \ge q_0 = \lceil (n+1)(1-\alpha_{\max}) \rceil$, by minimising the cost bound of Proposition 1 with the calibration estimate of the cycles of use:
$$\hat q = \arg\min_{q \ge q_0} \frac{1 + (\rho - 1)(n + 1 - q)/(n+1)}{\frac1n \sum_{i=1}^n \text{use}_i\bigl(V_{(q)}\bigr)} .$$
The numerator replaces the empirical failure rate by its conformal bound, which is never zero: with $q = n$ it is $1/(n+1)$, not 0. This is the difference from plain empirical cost minimisation, which sees no failure among calibration units at the edge of the feasible region and was fragile in the original study. By Corollary 1 the failure probability of CNA-DA is at most $\alpha_{\max}$ whatever $\rho$ is.

**Why calibrate notice rather than the interval.** A per-cycle conformal lower bound $\mathrm{LB}_t = s_t - q_\alpha$, with $q_\alpha$ the conformal quantile of $s_t - \min(r_t, c)$ over calibration rows, gives $P(r_t \ge \mathrm{LB}_t) \ge 1 - \alpha$ for a row drawn at random. The natural alarm "act when $\mathrm{LB}_t \le \ell + 1$" is a threshold trigger at level $H = \ell + 1 + q_\alpha$ on the same signal, so it belongs to the same family as CNA and differs only in how the level is calibrated. We use it as the direct comparator (LB-all, and LB-near with rows $r_t \le c$ only).

# 6. Experimental design

The protocol (hypotheses H1–H5, datasets, policies, metrics and decision rules) was written and hashed with SHA-256 before the final experiments (`protocol.md`, hash `4eddd0ed…fe281`). The hash is not a registry preregistration; it shows that the file has not changed since, and its timing rests on the repository history. The code had been smoke-tested on one PHM08 fold before hashing.

## 6.1 Data

We use the PHM08 training engines (218; the development dataset of the original study), the C-MAPSS FD001–FD004 training sets (100, 260, 100, 249 engines) [1], N-CMAPSS reduced to one row per flight (99 engines in nine subsets, six temperature and speed channels) [24], and 123 lithium-ion cells of Severson et al. [25] in three batches (36, 43, 44 cells; end of life at 80 % of nominal capacity), aggregated into blocks of 3 cycles fixed a priori. Engine datasets are simulations from one simulator family; the battery data are laboratory measurements. The 17 PRONOSTIA bearings [26] of the original study are excluded: with at most about six calibration bearings per split, $\alpha \le 0.10$ is not attainable ($\alpha \ge 1/7$). Their earlier result (0 of 17 correct alarms) is a prior result of the original study, not re-run.

## 6.2 Splits and information

Outer splits are five-fold cross-validation by unit (seed 2026), in which test units are exchangeable with training units, and, for transfer, leave-one-batch-out on the battery cells and leave-one-subset-out on N-CMAPSS. Inside each outer training set, 60 % of the units (proper training) fit the normalisation and the regressors, and the other 40 % (calibration) are used for every selection, tuning and calibration step of every policy; outer test units are scored once. This split is repeated three times with different seeds; repetition 0 is primary. The capacity-based battery policies fit nothing and use all outer training cells. Disjointness of proper-training, calibration and test units is asserted in the code; the features at row $t$ use rows $\le t$ only; no test life enters any setting.

## 6.3 Policies

The instrument is the regressor of the original study: histogram gradient boosting with absolute-error loss (500 iterations, learning rate 0.05, 31 leaves, ≥ 40 samples per leaf) on causal features of regime-normalised sensors (early-life baseline, exponentially weighted means, local slopes, age), target $\min(r, c)$ [27]. Table 1 lists the policies. CNA uses settings fixed a priori (cap 60 cycles, battery 180; span 3; debounce 2), so that no selection touches the calibration units before the conformal step.

Table: **Table 1.** Policies compared. All settings of learned policies are selected on calibration units; the regressors are fitted on proper-training units. LB: lower bound; P1: policy 1 of Kamariotis et al.; 1-SE: one-standard-error rule.

| Policy | Signal and trigger | Selected on calibration units | Guarantee |
|---|---|---|---|
| CNA ($\alpha$ = 0.05, 0.10, 0.20) | GBM estimate (cap 60; battery 180 cycles), EWM span 3, debounce 2 (fixed a priori) | Warning level $\hat H$ = conformal quantile of critical levels | $P(\text{fail}) \le \alpha$ (Thm 2) |
| CNA-DA | As CNA | Order statistic $q \ge q_0$ minimising the cost bound ($\alpha_{\max}$ = 0.10) | $P(\text{fail}) \le 0.10$ (Cor. 1) |
| LB-all / LB-near | As CNA; alarm when $s_t - q_\alpha \le \ell + 1$ | Conformal quantile of per-row residuals (all rows / rows with $r \le$ cap) | Per-row coverage only |
| STW, window-tuned | GBM estimate (cap 40/60/90), span, debounce, shift | Most correct first warnings per $H$; $H$ by calibration cost | None |
| STW, cost-tuned (plain, 1-SE) | GBM estimate, span, debounce, threshold (2,160 settings) | Lowest calibration cost rate; 1-SE: largest notice within one bootstrap SE | None |
| Kamariotis P1 ($1/\rho$, optimised) | $P(r \le \ell + \Delta t \mid$ estimate bin$)$ from calibration rows (P-res) | Threshold $1/\rho$, or optimised on calibration cost | None |
| Age replacement | Age | Replacement age on training lives | None |
| Capacity trend (window-tuned, 1-SE; CNA-DA on trend) | Discharge capacity extrapolated to 0.88 Ah (battery only) | As the STW and CNA counterparts, on all training cells | CNA version: Thm 2 |


Costs follow Section 3 with lead times $\ell \in \{0, 5, 10, 15\}$ cycles (battery $\{0, 15, 30, 45\}$) and cost ratios $\rho \in \{5, 10, 20, 50\}$; the primary lead time is 10 cycles (battery 30). The cost ratios are illustrative.

## 6.4 Metrics, tests and decision rules

The primary metric of the new method is the *failure-before-replacement rate* on test units, with Clopper–Pearson intervals [28] and a one-sided exact binomial test of rate $> \alpha$, Holm-adjusted over datasets [29]. Notice at the alarm and wasted life (notice minus lead time among planned replacements) measure what the guarantee costs. Window scores of the estimate and the constant are compared with the Tango score interval for paired proportions [30] and two one-sided tests (TOST) [31] with a margin of 5 percentage points fixed in the protocol; the Theorem 1 bound is evaluated with $\pi_a, \pi_b$ estimated on calibration alarms (mass of the notice within ±3 cycles of each edge, per cycle) and $E\lceil|D|\rceil$ on calibration units. Cost rates are compared by paired bootstrap ratios over test units (2,000 resamples). The decision rules of the protocol are: **H1** CNA at $\alpha = 0.10$ and the primary lead time does not exceed 0.10 failure (Holm-adjusted binomial $p \ge 0.05$ on each of seven datasets); **H2** the per-cycle lower-bound alarm LB-all at $\alpha = 0.10$ exceeds 0.10 ($p < 0.05$) on at least four of seven; **H3** model and constant estimates are equivalent within ±5 points on each dataset; **H4** CNA-DA is "cheaper" than a comparator on a dataset if cheaper in at least 8 of 16 cost cells and dearer in none (and conversely), and its failure rate stays at or below $\alpha_{\max} = 0.10$ for every positive lead time and ratio; **H5** calibrated on other batches, CNA exceeds 0.10 on at least one held-out battery batch, and recalibration with 10 pilot cells of the new batch restores a mean failure rate at or below 0.10. Intervals cover the sampling of test units given fitted policies; the three repetitions show the variability from the proper-training/calibration split.


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

The protocol's equivalence test, however, was **not supported**. With a margin of ±5 points, model and constant were equivalent only on FD002. On FD001, FD003 and N-CMAPSS the intervals were too wide to decide. On PHM08 and FD004 the constant was significantly *better* than the model's estimate (−6.4 and −6.0 points; 90 % intervals −10.3 to −3.1 and −9.7 to −2.7), and on the battery cells the estimate was significantly better than the constant (+9.8 points, +5.4 to +15.3), the one dataset where the estimate's spread at the alarm was a sizeable fraction of the window width (6.3 of 45 cycles). This corrects the earlier study's summary that the estimate changed the score by at most 3.6 points [3]: in the present design, with 60 % of the training units used for fitting, the estimate moved the score by up to 10 points in either direction. What did hold is the qualitative reading of Theorem 1: the score is determined by the alarm time up to a band of discordant units that is small when the estimate's spread at the alarm is small. The plug-in bound of part (ii), computed on calibration units, exceeded the realised score difference on 6 of 7 datasets and the realised discordance on 4 of 7 (Fig. 3b); it is an estimate, not a guarantee, because $\pi_a$ and $\pi_b$ are estimated. For the purpose of this paper the conclusion is the one that motivates Section 5: a window score bounds neither the notice nor the failure probability, so neither should be inferred from it.

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


# 8. Discussion

## 8.1 What the results establish

Calibrating the alarm rather than the interval did what Theorem 2 says it does. On seven datasets, with between 32 and 83 calibration units, the failure-before-replacement rate of the conformal notice alarm followed its nominal level (pooled 3.8, 9.3 and 19.4 % for 5, 10 and 20 %), while a per-cycle lower bound at the same nominal level on the same signal and trigger failed on up to 70 % of units. The difference is not a matter of tuning. Per-cycle coverage averages over cycles chosen independently of the data; an alarm is read at the first cycle at which the signal crosses a level, the cycle at which an optimistic error is most likely. Restricting the per-cycle residuals to the near-failure region repaired most of the gap on six datasets, but how close it came depended on that choice of rows, and on N-CMAPSS it exceeded its level by half. Calibrating the critical level removes the choice: the quantity calibrated is the event the operator cares about.

## 8.2 What the guarantee costs, and when it is worth having

The guarantee is a certificate, not a saving. A decision-aware choice of the level kept the failure rate below the stated ceiling in every cost setting, but well-regularised uncertified rules, notably a one-standard-error cost-tuned alarm and, on the batteries, a capacity trend, were cheaper on several datasets. Proposition 2 identifies the reason, and it is general: a distribution-free procedure that chooses among calibration critical levels cannot certify a failure probability below $1/(n+1)$, and with expensive failures that floor dominates the cost. Rules that extrapolate beyond the most conservative calibration unit can be cheaper on a given test set, and here they were, but the earlier study showed that plain empirical cost minimisation was fragile [3], and nothing bounds their failure probability for the next unit. For practice this gives a simple sizing rule. To certify a failure probability $\alpha$ one needs at least $1/\alpha - 1$ run-to-failure calibration units of the population in question; to keep the certified excess cost $(\rho - 1)/(n+1)$ below a fraction $\varepsilon$ of a planned replacement one needs $n + 1 \ge (\rho - 1)/\varepsilon$. Where such data exist, as for fleets of identical components or batteries from one production line, the guarantee is cheap; where they do not, a reported failure rate of an uncertified alarm on a few dozen test units is weak evidence of its reliability.

## 8.3 Transfer

The battery results show the limit of every exchangeability-based guarantee. Calibrated on two batches, the alarm failed on most cells of a third, shorter-lived batch, and similar failures occurred on three of nine N-CMAPSS subsets. This is not a defect of the conformal step, which did exactly what it promised for the population it was calibrated on, but it is a warning against reading a validated failure rate as a property of the alarm rather than of the alarm and a population. Two remedies were examined. Conservative uncertified rules carried enough margin to survive this shift, at the price of notice that is wasted when the population does not change. Recalibration on 10 pilot cells of the new batch restored the nominal rate on every batch, at the price of running those cells to failure. Between these, weighted or non-exchangeable conformal methods [21, 22] could use covariates of the new batch, such as its charging protocol, to reweight the calibration units; this needs a model of the shift and was not attempted here.

## 8.4 Relation to the earlier evaluation

The earlier study [3] argued that a window score at the alarm mainly measures alarm timing, that window tuning is not cost tuning, and that within-population validation overstated transfer. Theorem 1 states the first point precisely: the estimate can change the score only through units whose notice lies near an edge of the acceptance band, a set that is small when the estimate's spread at the alarm is small. The equivalence test, however, did not confirm the earlier claim that a constant scores within a few points of the estimate: in the present split, the difference reached 6–10 points in either direction on three datasets. The transfer finding was confirmed with a new method: the guarantee, like the window score, holds for the population it was calibrated on and not for a held-out batch. The earlier battery numbers (window score 95.1 % within batches and 79.7 % across batches under fully nested preprocessing) are prior results; in the present design, with 60 % of training cells for fitting and blocks of 3 cycles, the window-tuned alarm scored 88.6–95.1 % within batches and 54.5–62.6 % across batches over the three repetitions, and the capacity trend 97.6 % and 99.2 %.

## 8.5 Limitations

* **Data.** Five of the seven datasets come from one simulator family; the battery cells are laboratory data; no alarm was deployed and no data are censored. Real fleets replace units before failure, so the critical levels of censored units are only bounded; conformal methods for censored data [18] would be needed. Bearings were excluded because the 17 units cannot support the guarantee.
* **Exchangeability.** The guarantee is marginal and requires exchangeable calibration and test units. It failed across battery batches and N-CMAPSS subsets; it may fail under slower drifts within one population.
* **Scope of the method.** The theory covers threshold triggers with debounce on a fixed causal signal and a fixed lead time. The fixed settings of CNA (cap, smoothing, debounce) were chosen a priori and not optimised, and the proper-training/calibration split costs training data: the window-tuned alarm in this design scored below its cross-fitted version in the earlier study.
* **Cost.** The renewal–reward model assumes independent units, replacement as good as new, a fixed lead time and illustrative cost ratios; the policy of Kamariotis et al. was adapted to per-cycle decisions with a lead time and is not their original setting.
* **Statistics and pre-specification.** The protocol was hashed before the final runs but not registered. Theorem 1's plug-in bound uses estimated probabilities and is not a guarantee. Intervals are conditional on the fitted policies; the three repetitions show the variability from the split. The equivalence and cost hypotheses failed and are reported as such; Proposition 2 is a theorem, but the prominence given to it as the explanation of the cost results was decided after those results were seen.

# 9. Conclusion

An end-of-life alarm is useful when it leaves enough notice for the replacement it schedules. This paper calibrated that property directly. For a threshold trigger, the notice is monotone in the warning level, every run-to-failure calibration unit has a single critical level, and the split-conformal quantile of those levels gives a warning level whose probability of failure before the planned replacement is at most $\alpha$ for a new exchangeable unit. A decision-aware choice keeps this guarantee, and the size of the calibration set sets a floor, $1/(n+1)$, on the failure probability any such choice can certify.

On seven run-to-failure datasets the guarantee held: failure rates were 3.8, 9.3 and 19.4 % at nominal 5, 10 and 20 %, while per-cycle conformal lower bounds on the same signal failed on up to 70 % of units. It was not free and not robust to a change of population: uncertified but conservative rules were often cheaper, and calibration on other battery batches failed on 61–74 % of two held-out batches, until ten pilot cells of the new batch restored the nominal rate. A supporting bound showed that the window score of a threshold alarm depends on its estimate only through a band of units near the window edges, though an equivalence test did not confirm that the difference stays within five points. The practical recommendation is to state the failure probability an alarm is meant to achieve, to calibrate the alarm on whole run-to-failure units of the population in which it will be used, and to report how many such units the claim rests on.

# Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude (Anthropic), an AI assistant, to search the literature, write and run the analysis code, draft the theory and proofs, and draft and revise the manuscript text. The protocol, all experiments and all reported numbers were produced by the code in the accompanying repository and checked against its output files. After using this tool the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

# Data and code availability

PHM08, C-MAPSS and N-CMAPSS are available from the NASA Prognostics Center of Excellence data repository; the battery data of Severson et al. [25] from https://data.matr.io. Code, the hashed protocol, derived data summaries, per-unit results and the scripts that produce every table and figure are in a repository whose link is withheld for anonymised review.


# References {-}

1. Saxena A, Goebel K, Simon D, Eklund N. Damage propagation modeling for aircraft engine run-to-failure simulation. In: 2008 International Conference on Prognostics and Health Management. IEEE; 2008, p. 1–9. https://doi.org/10.1109/PHM.2008.4711414
2. Saxena A, Celaya J, Saha B, Saha S, Goebel K. Metrics for offline evaluation of prognostic performance. International Journal of Prognostics and Health Management 2010;1(1). https://doi.org/10.36001/ijphm.2010.v1i1.1336
3. Anonymous (authors of the present study). What does an end-of-life alarm score measure? Separating prediction accuracy, alarm timing, maintenance cost and transfer. Earlier manuscript version, 2026 (superseded by this article; numbers attributed to it are prior results).
4. Javanmardi A, Hüllermeier E. Conformal prediction intervals for remaining useful lifetime estimation. International Journal of Prognostics and Health Management 2023;14(2). https://doi.org/10.36001/ijphm.2023.v14i2.3417
5. Robinson CD. Remaining useful life estimation for aircraft engines with risk-aware prediction intervals via conformalized quantile regression. International Journal of Prognostics and Health Management 2026 (published online April 2026).
6. Wang W, Wang Z, Cai Z, Hu C, Si S. Robust uncertainty quantification for online remaining useful life prediction with randomly missing and partially faulty sensor data. Reliability Engineering & System Safety 2025;262.
7. Kamariotis A, Tatsis K, Chatzi E, Goebel K, Straub D. A metric for assessing and optimizing data-driven prognostic algorithms for predictive maintenance. Reliability Engineering & System Safety 2024;242:109723. https://doi.org/10.1016/j.ress.2023.109723
8. Ramasso E, Saxena A. Performance benchmarking and analysis of prognostic methods for CMAPSS datasets. International Journal of Prognostics and Health Management 2014;5(2). https://doi.org/10.36001/ijphm.2014.v5i2.2236
9. de Pater I, Reijns A, Mitici M. Alarm-based predictive maintenance scheduling for aircraft engines with imperfect remaining useful life prognostics. Reliability Engineering & System Safety 2022;221:108341. https://doi.org/10.1016/j.ress.2022.108341
10. Nguyen KTP, Medjaher K. A new dynamic predictive maintenance framework using deep learning for failure prognostics. Reliability Engineering & System Safety 2019;188:251–262. https://doi.org/10.1016/j.ress.2019.03.018
11. Mitici M, de Pater I, Barros A, Zeng Z. Dynamic predictive maintenance for multiple components using data-driven probabilistic RUL prognostics: The case of turbofan engines. Reliability Engineering & System Safety 2023;234:109199. https://doi.org/10.1016/j.ress.2023.109199
12. Barlow R, Hunter L. Optimum preventive maintenance policies. Operations Research 1960;8(1):90–100. https://doi.org/10.1287/opre.8.1.90
13. de Jonge B, Scarf PA. A review on maintenance optimization. European Journal of Operational Research 2020;285(3):805–824. https://doi.org/10.1016/j.ejor.2019.09.047
14. Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. New York: Springer; 2005. https://doi.org/10.1007/b106715
15. Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. Distribution-free predictive inference for regression. Journal of the American Statistical Association 2018;113(523):1094–1111. https://doi.org/10.1080/01621459.2017.1307116
16. Angelopoulos AN, Bates S. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning 2023;16(4):494–591. https://doi.org/10.1561/2200000101
17. Romano Y, Patterson E, Candès E. Conformalized quantile regression. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019); 2019.
18. Candès E, Lei L, Ren Z. Conformalized survival analysis. Journal of the Royal Statistical Society Series B 2023;85(1):24–45. https://doi.org/10.1093/jrsssb/qkac004
19. Angelopoulos AN, Bates S, Fisch A, Lei L, Schuster T. Conformal risk control. In: International Conference on Learning Representations (ICLR); 2024.
20. Gupta C, Kuchibhotla AK, Ramdas A. Nested conformal prediction and quantile out-of-bag ensemble methods. Pattern Recognition 2022;127:108496. https://doi.org/10.1016/j.patcog.2021.108496
21. Tibshirani RJ, Foygel Barber R, Candès E, Ramdas A. Conformal prediction under covariate shift. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019); 2019.
22. Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. Conformal prediction beyond exchangeability. Annals of Statistics 2023;51(2):816–845. https://doi.org/10.1214/23-AOS2276
23. Vovk V. Conditional validity of inductive conformal predictors. In: Proceedings of the Asian Conference on Machine Learning, PMLR 25; 2012, p. 475–490.
24. Arias Chao M, Kulkarni C, Goebel K, Fink O. Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. Data 2021;6(1):5. https://doi.org/10.3390/data6010005
25. Severson KA, Attia PM, Jin N, Perkins N, Jiang B, Yang Z, Chen MH, Aykol M, Herring PK, Fraggedakis D, Bazant MZ, Harris SJ, Chueh WC, Braatz RD. Data-driven prediction of battery cycle life before capacity degradation. Nature Energy 2019;4(5):383–391. https://doi.org/10.1038/s41560-019-0356-8
26. Nectoux P, Gouriveau R, Medjaher K, Ramasso E, Chebel-Morello B, Zerhouni N, Varnier C. PRONOSTIA: An experimental platform for bearings accelerated degradation tests. In: IEEE International Conference on Prognostics and Health Management (PHM'12), Denver, CO; 2012. https://hal.science/hal-00719503
27. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 2011;12:2825–2830.
28. Clopper CJ, Pearson ES. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika 1934;26(4):404–413. https://doi.org/10.1093/biomet/26.4.404
29. Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 1979;6(2):65–70.
30. Tango T. Equivalence test and confidence interval for the difference in proportions for the paired-sample design. Statistics in Medicine 1998;17(8):891–908.
31. Schuirmann DJ. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. Journal of Pharmacokinetics and Biopharmaceutics 1987;15(6):657–680. https://doi.org/10.1007/BF01068419
