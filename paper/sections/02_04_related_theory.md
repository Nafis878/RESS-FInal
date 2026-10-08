# 2. Related work

**Evaluating prognostics through decisions.** Remaining useful life (RUL) estimates are usually scored at times chosen by the evaluator, with accuracy metrics or asymmetric scores that penalise late predictions [@saxena2008; @saxena2010; @ramasso2014]. A line of work in this journal evaluates prognostics through the maintenance decisions they trigger. de Pater et al. [@depater2022] schedule aircraft maintenance from alarms raised on RUL prognostics and protect against prognostic error with a safety factor; Kamariotis et al. [@kamariotis2024] propose a metric $M$, the relative excess long-run cost of a policy over perfect prognostics, and tune probability-threshold policies with it; dynamic predictive-maintenance policies act on predicted failure probabilities [@nguyen2019; @mitici2023]. Classical replacement theory supplies the benchmark that any condition-based policy must beat, age replacement [@barlow1960; @dejonge2020]. None of these studies gives a finite-sample guarantee on the notice an alarm leaves.

**Conformal prediction for RUL.** Conformal prediction converts any point predictor into prediction sets with finite-sample, distribution-free coverage under exchangeability [@vovk2005; @lei2018; @angelopoulos2023]. Javanmardi and Hüllermeier [@javanmardi2023] conformalised gradient boosting and convolutional networks to obtain RUL intervals on C-MAPSS; Robinson [@robinson2026] used asymmetric conformalised quantile regression [@romano2019] to make the lower RUL bound conservative; Wang et al. [@wang2025robustuq] combined conformal prediction with imputation of missing sensor data, and note that exchangeability is a concern for degradation data. In all of these the calibrated object is the interval at a given cycle, and coverage is averaged over cycles (or units at a fixed time). Candès et al. [@candes2023] give conformal lower predictive bounds on survival times under censoring, again at a fixed covariate value. An alarm, however, is a stopping rule: it is evaluated at the cycle at which the bound first crosses a level, a time that depends on the data. Per-cycle coverage does not imply coverage at a data-dependent time, and our experiments show that the difference is large (Section 7.1).

**Risk control and non-exchangeable data.** Conformal risk control [@angelopoulos2024] calibrates a scalar parameter so that the expected value of a monotone loss is bounded; nested conformal prediction [@gupta2022] expresses conformal procedures through nested families of sets. Our construction is an instance of this nested view, with the warning level as the nesting parameter and failure before the planned replacement as the loss; what is specific is the closed-form critical level of a debounced threshold trigger (Lemma 2), which makes the calibration exact and cheap, and the decision-aware selection that keeps the guarantee (Corollary 1). When calibration and test units are not exchangeable, as across battery batches, the guarantee can fail; weighted and non-exchangeable conformal methods bound the resulting coverage gap [@tibshirani2019; @barber2023]. We do not use them, because they need a weight model for the shift; we instead measure the failure of exchangeability and the cost of restoring it with a few run-to-failure pilot units from the new population.

**What this paper adds.** (i) A conformal calibration of the alarm time itself, with a finite-sample guarantee on the probability that the remaining life at the alarm exceeds the lead time, a decision-aware variant that keeps the guarantee, and a cost bound. (ii) A bound on the difference between the window score of a threshold-crossing alarm's estimate and that of a constant, with an equivalence test, which explains why such scores measure timing. (iii) A controlled evaluation on seven datasets, against interval-calibrated alarms, window- and cost-tuned alarms, the probability-threshold policy of Kamariotis et al., age replacement and a capacity-trend baseline, with whole battery batches and N-CMAPSS subsets held out. We do not claim a new RUL model.

# 3. Problem formulation

**Units, alarms and notice.** A unit observed at rows $j = 1, \dots, n$ (cycles $t_1 < \dots < t_n$) has life $L$ and remaining life $r_j = L - t_j \ge 1$, decreasing in $j$. A causal signal $s_j$, here a smoothed RUL estimate, is computed from rows $1, \dots, j$. A *threshold trigger* with debounce $k$ and warning level $H$ raises an alarm at the first row at which the signal has been at or below $H$ for $k$ consecutive rows. Writing $m_j = \max(s_{j-k+1}, \dots, s_j)$ for $j \ge k$ ($m_j = +\infty$ otherwise) and $\rho_j = \min_{i \le j} m_i$ (the *debounced running minimum*), the alarm row is
$$\tau(H) = \min\{ j : \rho_j \le H \}, \qquad (\tau(H) = \infty \text{ if no such row}),$$
and the *notice* is $N(H) = r_{\tau(H)}$ ($N = 0$ if no alarm).

**Maintenance consequence.** An alarm schedules a replacement $\ell$ cycles later (the *lead time*). If $N(H) > \ell$ the planned replacement happens at $t_{\tau(H)} + \ell$ at cost 1; otherwise, or without an alarm, the unit fails at $L$ at cost $\rho > 1$. Units are renewed, so by the renewal–reward theorem the long-run cost per cycle is $E[\text{cost}]/E[\text{use}]$, estimated by total cost over total cycles of use; we report it relative to perfect foresight (replacement at $L-1$, cost 1) [@barlow1960]. The engineering requirement we target is reliability of the alarm: the probability of failure before the planned replacement, $P(N(H) \le \ell)$, should not exceed a stated $\alpha$.

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

**Remarks.** (1) The guarantee is marginal: it averages over the draw of the calibration units. Conditional on a calibration set the realised failure probability is random, with a Beta$(q, n+1-q)$ distribution for continuous $V$ [@vovk2012]; a larger $q$ gives a guarantee with high probability over the calibration draw. (2) The guarantee needs no model of the degradation or of the RUL error; it can be applied to any causal signal and any monotone trigger, including a physically motivated one (Section 7.3). (3) It does not hold when calibration and new units are not exchangeable, for example when the new unit comes from a different production batch; Section 7.4 measures what happens then. (4) Per-cycle conformal intervals do not deliver this guarantee: they control $P(r_t \ge \text{LB}_t)$ for a cycle drawn independently of the data, whereas the alarm is evaluated at the data-dependent time at which the bound first crosses a level, where the signal is, by selection, at its most optimistic.
