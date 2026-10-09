# Round-3 theory: separation, censoring under a weaker assumption, covariate shift

Written 2026-10-09 before any round-3 experiment. Numbering follows the manuscript after this round
(Theorems 1–4, Lemmas 1–3, Corollaries 1–2, Propositions 1–2 are the existing results).

## 0. Notation (as in the manuscript)

A unit has rows j = 1, …, n_u at cycles t_1 < … < t_{n_u}, life L (failure time T = L), remaining life
r_j = L − t_j ≥ 1 (decreasing in j), a causal signal s_j, debounce k, m_j = max(s_{j−k+1}, …, s_j) for j ≥ k
(m_j = +∞ otherwise), running minimum ρ_j = min_{i ≤ j} m_i, alarm row τ(H) = min{j : ρ_j ≤ H}, notice
N(H) = r_{τ(H)} (0 without alarm). For a lead time ℓ: admissible rows A = {j : r_j > ℓ} (a prefix of the rows),
j_ℓ = max A, critical level V = ρ_{j_ℓ}. Lemma 2: N(H) > ℓ ⇔ H ≥ V. "One row per cycle" means t_j = j − 1, so
r_j = L − j + 1, n_u = L and r_{j_ℓ} = ℓ + 1; "rows Δ cycles apart" means t_j = (j − 1)Δ, so r_{j_ℓ} ∈ (ℓ, ℓ + Δ].

---------------------------------------------------------------------------------------------------------------------

## B1. Per-cycle coverage does not imply alarm reliability

**Per-cycle lower bound and its alarm.** For a constant q (from any calibration, treated as fixed), LB_j = s_j − q.
The *LB alarm with action level a and debounce k* is raised at the first row at which LB_i ≤ a for the k most recent
rows i; it is the threshold trigger on s at level H = a + q. Its failure event is F = {N(a + q) ≤ ℓ}. The comparator
of the manuscript uses a = ℓ + 1 and k = 2. Write M_j = {LB_j > r_j} (the bound misses at row j).

**Coverage notions.**
(C1) *cycle-wise* (one row per cycle): P(M at cycle t | L > t) ≤ α for every cycle t ≥ 0;
(C2) *pooled over rows*: E[Σ_j 1(M_j)] ≤ α E[n_u] (a row drawn in proportion to rows is missed with probability ≤ α;
this is what split conformal calibration over pooled rows targets).

### Theorem 5 (separation)

(a) *Impossibility.* For every α ∈ (0, 1), integer ℓ ≥ 0, k ≥ 1, q ∈ ℝ and a ≥ ℓ + 1 there is a law of (L, s), one
row per cycle, under which (C1) holds with equality at every cycle, (C2) holds with equality, and P(F) = 1.

(b) *What the failure probability depends on.* Let k = 1 and a ≥ r_{j_ℓ} almost surely (for rows Δ cycles apart:
a ≥ ℓ + Δ). Then F ⊆ M_{j_ℓ} pathwise, so P(F) ≤ P(M_{j_ℓ}): the LB alarm can fail only if the bound misses at the last
admissible row.

(c) *Sharp bound in the number of rows.* Under (C2) and the conditions of (b), P(F) ≤ min{1, α E[n_u]}. For every
n ≥ ℓ + 1 and α ∈ (0, 1) there is a law with n_u = n, one row per cycle, a = ℓ + 1, satisfying (C2), with
P(F) = min{1, αn}.

(d) *Dependence across rows.* Let k = 1, a ≥ r_{j_ℓ}, condition on L (so that A is fixed), and write
π_j = P(LB_j > a | L) and ε_j = LB_j − r_j.
  (i) P(F | L) ≤ min_{j ∈ A} π_j, with equality when the events {LB_j > a}, j ∈ A, are nested (e.g. comonotone errors);
  (ii) if (ε_j)_{j ∈ A} are associated given L, then P(F | L) ≥ ∏_{j ∈ A} π_j, with equality when they are independent;
  (iii) if (ε_j)_{j ∈ A} are m-dependent given L, then P(F | L) ≤ ∏_{i = 0}^{I} π_{j_ℓ − i(m+1)}, I = ⌊(|A| − 1)/(m + 1)⌋.

(e) *Debounce.* With one row per cycle, k ≥ 2 and a = ℓ + 1, the exact bound LB_j = r_j (no miss at any row) gives
P(F) = 1. With a = ℓ + kΔ (rows Δ cycles apart), F ⊆ ∪_{i < k} M_{j_ℓ − i} and (c) holds unchanged.

(f) Under the law of (a), CNA calibrated as in Theorem 1 fails with probability at most α (Theorem 1 holds for every law).

**Proof.**
(a) Let ℓ ≥ 1 first. For x ∈ (0, 1) and an integer m ≥ a − ℓ, let L take values in {1, 2, …} with P(L > l) = x^l
(geometric). Rows are at cycles t = 0, …, L − 1 and r_t = L − t. Put M_0 = a − ℓ + 1 and
s_t = q + r_t + M_0 · 1{ℓ < r_t ≤ ℓ + m}, so that LB_t = r_t + M_0 · 1{ℓ < r_t ≤ ℓ + m}.
The bound misses exactly at rows in the zone ℓ < r_t ≤ ℓ + m. By the memoryless property, given L > t the remaining
life r_t = L − t has P(r_t > l | L > t) = x^l, hence P(M at t | L > t) = x^ℓ − x^{ℓ+m} =: g(x), the same for every t.
Also E[Σ_t 1(M_t)] = Σ_t P(L > t) g(x) = E[L] g(x), so (C1) and (C2) hold with equality when g(x) = α.
Existence: g is continuous on [0, 1] with g(0) = g(1) = 0 and maximum at x*^m = ℓ/(ℓ + m), where
g(x*) = (ℓ/(ℓ + m))^{ℓ/m} · m/(ℓ + m) → 1 as m → ∞ (because (ℓ/m) log(ℓ/(ℓ + m)) → 0). Take m with g(x*) ≥ α and
x ∈ (0, x*] with g(x) = α (intermediate value theorem). For ℓ = 0, g(x) = 1 − x^m decreases from 1 to 0 and any
α ∈ (0, 1) is attained.
Alarm: rows with r_t > ℓ + m have LB_t = r_t > ℓ + m ≥ a; zone rows have LB_t = r_t + M_0 ≥ ℓ + 1 + M_0 > a; so no row
with r_t > ℓ satisfies LB_t ≤ a. Whatever the debounce, the alarm is raised (if at all) at a row with r_t ≤ ℓ, so
N(a + q) ≤ ℓ and F occurs for every unit.

(b) With k = 1 the alarm is raised at the first row with LB_j ≤ a, and A is a prefix of the rows, so the alarm leaves
more than ℓ cycles iff some j ∈ A has LB_j ≤ a. Hence F = ∩_{j ∈ A} {LB_j > a} ⊆ {LB_{j_ℓ} > a} ⊆ {LB_{j_ℓ} > r_{j_ℓ}} =
M_{j_ℓ}, using a ≥ r_{j_ℓ}.

(c) By (b), P(F) ≤ P(M_{j_ℓ}) ≤ E[Σ_j 1(M_j)] ≤ α E[n_u], and P(F) ≤ 1. Sharpness: let L ≡ n (so r_j = n − j + 1),
B ~ Bernoulli(min{1, αn}), LB_j = r_j for j ≠ j_ℓ and LB_{j_ℓ} = ℓ + 1 + B. Rows j < j_ℓ have LB_j = r_j ≥ ℓ + 2 > a, so
the alarm is raised at j_ℓ (notice ℓ + 1 > ℓ) iff B = 0; P(F) = P(B = 1) = min{1, αn}. The only possible miss is at j_ℓ
when B = 1, so E[Σ_j 1(M_j)] = min{1, αn} ≤ αn = α E[n_u]: (C2) holds.

(d) (i) F = ∩_{j ∈ A} {LB_j > a} is contained in each of the intersected events; if they are nested, the intersection
is the smallest. (ii) {LB_j > a} = {ε_j > a − r_j} is a non-decreasing event in ε. For associated random variables the
indicators of non-decreasing events are non-negatively correlated, and an intersection of non-decreasing events is
non-decreasing, so by induction P(∩_j E_j) ≥ P(E_1) P(∩_{j ≥ 2} E_j) ≥ … ≥ ∏_j P(E_j) (Esary, Proschan and Walkup,
1967). Independence gives equality. (iii) F ⊆ ∩_{i=0}^{I} {LB_{j_ℓ − i(m+1)} > a}; the rows j_ℓ − i(m + 1) are pairwise
more than m apart, so under m-dependence these events are independent.

(e) With LB_j = r_j, the rows with LB_j ≤ ℓ + 1 are those with r_j ≤ ℓ + 1. The k-th consecutive such row has
r = ℓ + 2 − k ≤ ℓ for k ≥ 2, so the alarm leaves at most ℓ cycles (or is not raised). With a = ℓ + kΔ: if F occurs, the
alarm was not raised at j_ℓ, so some i < k has LB_{j_ℓ − i} > a ≥ r_{j_ℓ − i} (because r_{j_ℓ − i} ≤ ℓ + (i + 1)Δ),
i.e. M_{j_ℓ − i}. Then P(F) ≤ P(∪_{i < k} M_{j_ℓ − i}) ≤ E[Σ_j 1(M_j)] ≤ α E[n_u].

(f) Theorem 1. ∎

**Reading.** Pooled per-cycle calibration constrains the *average* miss rate over rows; the LB alarm's failure
probability is the miss rate at the critical row, which can be up to α E[n_u] (E[n_u] = 100–300 rows here, so the
bound is vacuous). Dependence between rows decides where between ∏ π_j and min π_j the failure probability falls.
CNA calibrates the critical-row event itself.

---------------------------------------------------------------------------------------------------------------------

## B2. Censored calibration units under a weaker, testable assumption

**Setting.** Units 1, …, n + 1 are independent and identically distributed. Each has baseline covariates X, a
signal process, a failure time T and a censoring time C ∈ (0, ∞]; it is observed up to min(T, C). Fix a horizon
c0 ∈ (0, ∞] and define the *horizon critical level* V° = V if T ≤ c0 and V° = −∞ if T > c0.
By Lemma 2, {V° > H} = {T ≤ c0 and N(H) ≤ ℓ}: the unit fails within the horizon before the replacement its alarm
scheduled. Target: P(V°_{n+1} > Ĥ) ≤ α.

A calibration unit is *complete* if C ≥ min(T, c0): then V° is observed (V if T ≤ c0, −∞ otherwise). It is
*censored* if C < min(T, c0): then c := C, T > c, the signal up to c and U := U(c) (Lemma 3) are observed, and
V° ≤ U (V ≤ U by Lemma 3, and −∞ ≤ U). Write 𝒪_i for the observed data of unit i and 𝒪 = (𝒪_1, …, 𝒪_n).

### Theorem 6 (probabilistic censored-as-failed)

For each censored unit let p̃_i ∈ [0, 1] be a function of 𝒪, let ξ_i ~ Bernoulli(p̃_i) be independent given 𝒪 (and
independent of unit n + 1), and set W_i = U_i if ξ_i = 1 and W_i = −∞ otherwise; for a complete unit W_i = V°_i. Let
q = ⌈(n + 1)(1 − α)⌉ ≤ n and Ĥ = W_(q).

(i) If p̃_i ≥ p_i := P(T_i ≤ c0 | 𝒪_i) for every censored unit, then P(V°_{n+1} > Ĥ) ≤ α.

(ii) (Γ-sensitivity.) Let p⁰_i be the probability that an at-risk unit with the same observed history fails within the
horizon, p⁰_i = P(T ≤ c0 | X, signal up to c_i, T > c_i) evaluated at unit i's history, and suppose censoring is
Γ-informative: p_i/(1 − p_i) ≤ Γ p⁰_i/(1 − p⁰_i) for some Γ ∈ [1, ∞]. Then
p̃_i = Γ p⁰_i / (1 + (Γ − 1) p⁰_i) (p̃_i = 1 for Γ = ∞) satisfies (i). Γ = 1 is censoring at random given the observed
history; Γ = ∞ gives W_i = U_i for every censored unit, i.e. exactly the level of Theorem 2.

(iii) (Estimated probabilities.) For arbitrary p̃_i, let D be the number of censored units with u_i ∈ (p̃_i, p_i],
where u_i are the uniforms generating ξ_i = 1{u_i ≤ p̃_i}; given 𝒪, D is a sum of independent Bernoulli variables with
means (p_i − p̃_i)^+. For every integer J ≥ 0,
P(V°_{n+1} > Ĥ) ≤ α + J/(n + 1) + P(D > J).
In particular, if μ := Σ_i (p_i − p̃_i)^+ ≤ μ̄ almost surely, then with J = ⌈μ̄ + √(2μ̄ log(1/δ)) + (2/3) log(1/δ)⌉,
P(V°_{n+1} > Ĥ) ≤ α + J/(n + 1) + δ.

**Proof.** (i) Given 𝒪, the units are independent, so the unobserved futures of the censored units are independent
and P(T_i ≤ c0 | 𝒪) = p_i. For a censored unit and every v: if v ≥ U_i, P(V°_i > v | 𝒪) = 0 = P(W_i > v | 𝒪) because
V°_i ≤ U_i; if v < U_i, P(V°_i > v | 𝒪) ≤ P(T_i ≤ c0 | 𝒪) = p_i ≤ p̃_i = P(W_i > v | 𝒪). So W_i is stochastically larger
than V°_i given 𝒪; for a complete unit they are equal. Because the coordinates are conditionally independent, there
is (by the quantile transform, coordinate by coordinate, given 𝒪) a vector W' with the same conditional law as W given
𝒪, independent of unit n + 1, and W'_i ≥ V°_i for all i. Order statistics are non-decreasing in each coordinate, so
W'_(q) ≥ V°_(q), and since (W, V°_{n+1}) and (W', V°_{n+1}) have the same law,
P(V°_{n+1} > W_(q)) = P(V°_{n+1} > W'_(q)) ≤ P(V°_{n+1} > V°_(q)) ≤ (n + 1 − q)/(n + 1) ≤ α,
the last two steps being the rank argument of Theorem 1 applied to the i.i.d. values V°_1, …, V°_{n+1} (ties, including
ties at −∞, broken uniformly at random).
(ii) x ↦ x/(1 − x) is increasing, and p̃_i/(1 − p̃_i) = Γ p⁰_i/(1 − p⁰_i) ≥ p_i/(1 − p_i), so p̃_i ≥ p_i. For Γ = ∞,
p̃_i = 1 and W_i = U_i for every censored unit, which is the vector of Theorem 2, so Ĥ is its level.
(iii) Let p*_i = max(p̃_i, p_i) and W*_i the imputation with ξ*_i = 1{u_i ≤ p*_i} (same uniforms). By (i), W* is valid,
and more precisely (from its proof) there is a coupling with W*_i ≥ V°_i for all i. W_i ≥ W*_i except for the D units with
u_i ∈ (p̃_i, p_i], and lowering D coordinates lowers the q-th order statistic by at most D positions, so
W_(q) ≥ W*_(q − D) ≥ V°_(q − D). Let R be the rank of V°_{n+1} among V°_1, …, V°_{n+1} (uniform on {1, …, n + 1} by
exchangeability). Then {V°_{n+1} > Ĥ} ⊆ {V°_{n+1} > V°_(q − D)} ⊆ {R > q − D} ⊆ {R > q − J} ∪ {D > J}, and
P(R > q − J) ≤ (n + 1 − q + J)/(n + 1) ≤ α + J/(n + 1). For the special case, Bernstein's inequality for sums of
independent Bernoulli variables gives P(D ≥ μ + t | 𝒪) ≤ exp(−t²/(2μ + 2t/3)) ≤ δ for t = √(2μ log(1/δ)) + (2/3) log(1/δ),
and μ ≤ μ̄. ∎

**Remarks.**
* Without a horizon (c0 = ∞) every unit fails eventually, p_i = 1, and Theorem 6 is Theorem 2: the gain over the
  pathwise bound comes entirely from the probability that a censored unit does *not* fail within the horizon.
* What is testable: p⁰ can be estimated and checked for calibration on at-risk units that remained under observation
  (complete units); what is not testable is whether censored units differ from them, which Γ quantifies, as in
  sensitivity analysis for observational studies.
* Validity does not require p̃ to be accurate, only not too small; a conservative p̃ costs notice, never reliability.

### Proposition 3 (inverse-probability-of-censoring weights)

Assume censoring at random given the baseline covariates: C is independent of (signal, T) given X, with
G(u | x) = P(C ≥ u | X = x) known and G(c0 | x) > 0. Let K be the set of complete calibration units,
w_i = 1/G(min(T_i, c0) | X_i) for i ∈ K, w̄ = 1/G(c0 | X_{n+1}), and
Ĥ = inf{v : Σ_{i ∈ K} w_i 1{V°_i ≤ v} ≥ (1 − α)(Σ_{i ∈ K} w_i + w̄)}.
Then P(V°_{n+1} > Ĥ) ≤ α. If G is replaced by an estimate Ĝ fitted independently of the calibration and test units,
P(V°_{n+1} > Ĥ) ≤ α + ½ E_Q |ŵ(Z)/E_Q ŵ − w(Z)/E_Q w|, where Q is the law of a complete unit.

**Proof.** Under the assumption, P(unit complete | X, signal, T) = G(min(T, c0) | X). Conditionally on K (and |K| = m),
the complete units are i.i.d. with law Q, dQ/dP(z) = G(min(t, c0) | x)/κ (κ = P(complete)), and the test unit has law
P, so dP/dQ(z) = κ w(z) with w(z) = 1/G(min(t, c0) | x).
*Weighted exchangeability lemma.* Let Z_1, …, Z_m be i.i.d. Q and Z_{m+1} ~ P with dP/dQ ∝ w. Given the unordered set
{Z_1, …, Z_{m+1}} = {z_1, …, z_{m+1}}, the joint density ∏_{i ≤ m} q(z_i) · w(z_{m+1}) q(z_{m+1}) shows that
P(Z_{m+1} = z_i | set) = w(z_i)/Σ_j w(z_j). Hence, given the set, the score V°_{m+1} has the discrete law
Σ_i p_i δ_{v_i} with p_i = w(z_i)/Σ_j w(z_j), and V°_{m+1} ≤ Quantile_{1−α}(Σ_i p_i δ_{v_i}) with conditional probability
≥ 1 − α (Tibshirani, Barber, Candès and Ramdas, 2019, Lemma 3).
Moving the test point's mass to +∞ and replacing its weight w(Z_{m+1}) by w̄ ≥ w(Z_{m+1}) (G is non-increasing, so
G(min(T, c0) | x) ≥ G(c0 | x)) can only raise the quantile; the result is Ĥ, which does not involve T_{n+1}.
With Ĝ: the procedure is exactly valid when the test unit has law P̂ with dP̂/dQ = ŵ/E_Q ŵ; the calibration data have
the same law in both cases, so the probability of the event changes by at most d_TV(P, P̂) = ½ E_Q|dP/dQ − dP̂/dQ|. ∎

---------------------------------------------------------------------------------------------------------------------

## B3. Per-unit guarantee under covariate shift, without pilot failures

**Setting.** Calibration units i.i.d. from P; a new unit from Q; *covariate shift*: the conditional law of
(signal, T) given the covariates X is the same under P and Q, and w(x) = dQ_X/dP_X(x) exists. Let V be the critical
level (or V° with a horizon). For a new unit with covariates x,

Ĥ(x) = inf{v : Σ_{i ≤ n} w(X_i) 1{V_i ≤ v} ≥ (1 − α)(Σ_{i ≤ n} w(X_i) + w(x))}   (+∞ if no such v).

### Theorem 7 (covariate-shift-weighted critical level)

(i) P_Q(V_{n+1} > Ĥ(X_{n+1})) ≤ α.
(ii) If ŵ (normalised so that E_P ŵ(X) = 1) is estimated independently of the calibration units and the new unit and
used in place of w, then P_Q(V_{n+1} > Ĥ(X_{n+1})) ≤ α + ½ E_P |ŵ(X) − w(X)|.
(iii) With censored calibration units and censoring at random given X (Proposition 3), the weights
w(X_i)/G(min(T_i, c0) | X_i) on complete units and test weight w(x)/G(c0 | x) give the same guarantee for V°.

**Proof.** (i) The joint density of (Z_1, …, Z_n, Z_{n+1}) is ∏_{i ≤ n} p(z_i) · w(x_{n+1}) p(z_{n+1}), which has the
form of the weighted exchangeability lemma above with weight w(x); apply it with the test point's mass at +∞ (which only
raises the quantile). (ii) The procedure with ŵ is exactly valid for a new unit drawn from Q̂ with
dQ̂_X/dP_X = ŵ and the same conditional law given X; d_TV(Q, Q̂) = d_TV(Q_X, Q̂_X) = ½ E_P|ŵ − w| because the
conditional laws coincide. (iii) Multiply the likelihood ratios: dQ/dP_complete(z) ∝ w(x)/G(min(t, c0) | x), and bound
the new unit's censoring factor by 1/G(c0 | x) as in Proposition 3. ∎

**Remarks.**
* Ĥ depends on the new unit's covariates: a per-unit level. If w(x)/(Σ_i w(X_i) + w(x)) > α the level is +∞ and the
  alarm is raised at once — the honest answer when the calibration units contain too little information about units
  like x (poor overlap).
* Covariate shift is an assumption: if the new population differs in a way the covariates do not capture (a new cell
  structure, a different failure mode) the guarantee does not apply. It is testable only partly (by checking the
  weighted level on a few units of the new population, if any fail).

## What is new and what is not

* Theorem 5 is, to our knowledge, new: it separates per-cycle coverage from alarm reliability, locates the alarm's
  failure probability at the critical-row miss, gives the sharp worst case min{1, α E[n_u]} and brackets the role of
  dependence between rows; part (e) shows that a debounced per-cycle rule needs the action level ℓ + kΔ.
* Theorem 6 combines standard ingredients (stochastic dominance, a sensitivity parameter in the style of Rosenbaum and
  Tan, conformal ranks) into an estimator that interpolates between censoring at random (Γ = 1) and the pathwise bound
  of Theorem 2 (Γ = ∞); the role of the horizon is specific to alarms.
* Proposition 3 and Theorem 7 are direct applications of weighted conformal prediction (Tibshirani et al., 2019;
  Lei and Candès, 2021; Candès, Lei and Ren, 2023) to the critical level, stated here for completeness and testing.
* Not attempted: item 4 (multi-parameter trigger families) and a regret statement for delayed feedback (item 5).
