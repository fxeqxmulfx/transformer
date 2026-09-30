/-
# AdaFisher: the unconditional regret-rate table needs additional assumptions

arXiv:2405.16397v3, §3.2–3.4, Table 1 and Proposition 3.3.
Even cumulative suboptimality for a single strongly convex objective fails
the displayed O(log(T) sqrt(T)) bound at a permitted fixed step. Thus Table 1
cannot be read as an unconditional guarantee for all allowed hyperparameters.
-/

import Transformer.AdaFisher.Section3_ConvexFalse
import Transformer.AdaFisher.Section3_NonconvexFalse

open scoped BigOperators
open Filter Topology

noncomputable section

namespace Transformer.AdaFisher

/-- Cumulative objective regret versus the true minimizer zero in the
quadratic counterexample, Table 1 and Proposition 3.3. -/
def quadraticRegret (T : ℕ) : ℝ :=
  ∑ t ∈ Finset.range T, (quadratic (quadraticRun 1 (1 / 1000) 1 t) - quadratic 0)

/-- Every iterate already has objective gap at least one half,
Table 1's fixed-step counterexample to its unconditional rate interpretation. -/
theorem quadraticRun_gap_lower (t : ℕ) :
    (1 / 2 : ℝ) ≤ quadratic (quadraticRun 1 (1 / 1000) 1 t) - quadratic 0 := by
  have hp : (1 : ℝ) ≤ (998001 : ℝ) ^ t := one_le_pow₀ (by norm_num)
  have heq : ((-999 : ℝ) ^ t) ^ 2 = (998001 : ℝ) ^ t := by
    rw [← pow_mul, Nat.mul_comm t 2, pow_mul]
    norm_num
  rw [quadraticRun_eq]
  norm_num [quadratic]
  rw [heq]
  linarith

/-- The regret grows at least linearly, Table 1, despite a smooth strongly
convex objective and the printed allowed α≤1/L step. -/
theorem quadraticRegret_linear_lower (T : ℕ) : (T : ℝ) / 2 ≤ quadraticRegret T := by
  calc
    _ = ∑ t ∈ Finset.range T, (1 / 2 : ℝ) := by simp; ring
    _ ≤ quadraticRegret T := by
      exact Finset.sum_le_sum fun t _ => quadraticRun_gap_lower t

/-- No finite constant gives the table's O(log(T) sqrt(T)) upper bound
for this allowed fixed-step example, Table 1 and Proposition 3.3. The
source's convergence conditions need correction before such a rate can
be asserted. β=0 reduces the adaptive first moment to the current gradient. -/
theorem quadraticRegret_rate_counterexample (C : ℝ) :
    ∃ T : ℕ, 0 < T ∧ C * (1 + Real.log T) * Real.sqrt T < quadraticRegret T := by
  have hlt := (nonconvexRate_tendsto 1 1 1 0 C 0 0 0).eventually_lt
    tendsto_const_nhds (by norm_num : (0 : ℝ) < 1 / 2)
  obtain ⟨T, hrate, hT⟩ := (hlt.and (eventually_gt_atTop 0)).exists
  have hs : Real.sqrt (T : ℝ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr (by exact_mod_cast hT))
  have heq : C * (1 + Real.log T) * Real.sqrt T =
      nonconvexRate 1 1 1 0 C 0 0 0 T * (T : ℝ) := by
    calc
      _ = (C * (1 + Real.log T) / Real.sqrt T) * Real.sqrt T ^ 2 := by field_simp
      _ = _ := by
        rw [Real.sq_sqrt (Nat.cast_nonneg T)]
        simp only [nonconvexRate, one_pow, mul_one, mul_zero, add_zero, Nat.cast_zero]
        ring
  have hb := mul_lt_mul_of_pos_right hrate (by exact_mod_cast hT : (0 : ℝ) < T)
  rw [← heq] at hb
  exact ⟨T, hT, by linarith [quadraticRegret_linear_lower T]⟩

end Transformer.AdaFisher
