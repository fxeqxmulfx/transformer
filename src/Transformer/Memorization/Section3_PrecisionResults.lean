import Transformer.Memorization.Section3_PrecisionData
import Mathlib.Tactic

/-!
# Certified arithmetic of the precision experiment

arXiv:2505.24832v3, Section 3.2, Table 1. These statements concern all
the printed observations. They do not turn empirical estimates into a
universal 3.6-bits-per-parameter capacity theorem.
-/

namespace Transformer.Memorization

open scoped BigOperators

/-- Section 3.2, Table 1: mean of the sixteen displayed full-precision bpp. -/
def meanFullBpp : ℚ := (∑ i, (precisionTable i).fullBpp) / 16

/-- Section 3.2, Table 1: mean of the sixteen displayed half-precision bpp. -/
def meanHalfBpp : ℚ := (∑ i, (precisionTable i).halfBpp) / 16

/-- Section 3.2, Table 1: the displayed capacity increases with precision
in each reported model configuration. -/
theorem table_precision_improves_capacity (i : Fin 16) :
    (precisionTable i).halfCapacity < (precisionTable i).fullCapacity := by
  fin_cases i <;> norm_num [precisionTable]

/-- Section 3.2, Table 1: exact means of the rounded bpp columns. -/
theorem table_precision_means :
    meanFullBpp = 3061 / 800 ∧ meanHalfBpp = 281 / 80 := by
  norm_num [meanFullBpp, meanHalfBpp, precisionTable, Fin.sum_univ_succ]

/-- Section 3.2, Table 1: the column means round to the source's 3.83
and 3.51, with each rounding error below half of the last printed digit. -/
theorem table_precision_mean_rounding :
    |meanFullBpp - 383 / 100| < 1 / 200 ∧
      |meanHalfBpp - 351 / 100| < 1 / 200 := by
  rw [table_precision_means.1, table_precision_means.2]
  norm_num

/-- Section 3.2, "How does precision affect capacity?": doubling storage
precision increases mean observed bpp by less than a factor of two. -/
theorem table_precision_gain_below_doubling :
    meanHalfBpp < meanFullBpp ∧ meanFullBpp < 2 * meanHalfBpp := by
  rw [table_precision_means.1, table_precision_means.2]
  norm_num

/-- Section 3.2, Table 1: 3.6 bpp is an approximate fitted summary, not
an exact per-configuration identity; the first half-precision entry is 3.93. -/
theorem table_bpp_is_not_exactly_36 :
    (precisionTable 0).halfBpp ≠ (18 / 5 : ℚ) := by
  norm_num [precisionTable]

end Transformer.Memorization
