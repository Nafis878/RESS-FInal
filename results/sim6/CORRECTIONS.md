# Corrections to the simulation checks (protocol_v6.md, Part 2)

* 2026-10-09, S1, pooled coverage. The first run computed the Monte Carlo standard error of the pooled miss rate as if
  all rows were independent (sqrt(alpha(1 - alpha)/total rows) = 0.000035), although each unit contributes a block of
  dependent rows. With that SE the check failed (0.10042 vs 0.10). The SE was corrected to the delta-method SE of a
  ratio of unit-level sums (0.00021); the check then passes. The estimate itself did not change. The pass criterion
  (estimate within 4 SE) is as pre-specified; only the SE computation was wrong. All other checks passed in the first run.
