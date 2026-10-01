import Transformer.Memorization.Section5_ValidationData
import Mathlib.Tactic

/-!
# Validation-table error bounds

arXiv:2505.24832v3, Section 5.2.2, Table 2. The prose says predictions
are "generally within 1.5 points". This is not a uniform error guarantee:
the largest displayed central-value discrepancy is 9.31 percentage points.
Uncertainty intervals are retained as separate observations.
-/

namespace Transformer.Memorization

/-- Section 5.2.2, Table 2: absolute prediction error in unit-interval F1. -/
def validationError (i : Fin 6) : ℚ :=
  |(membershipTable i).predicted - (membershipTable i).observed|

/-- Section 5.2.2, Table 2: every recorded central-value discrepancy is
at most 9.31 percentage points. -/
theorem validation_errors_le_0931 (i : Fin 6) : validationError i ≤ 931 / 10000 := by
  fin_cases i <;> norm_num [validationError, membershipTable]

/-- Section 5.2.2, Table 2: the 123,702,528-parameter model's 0.75 target
has a 9.31-point discrepancy, attaining the bound above. -/
theorem validation_largest_error : validationError 4 = 931 / 10000 := by
  norm_num [validationError, membershipTable]

/-- Section 5.2.2: counterexample to interpreting "generally within
1.5 points" as a universal bound for all reported observations. -/
theorem validation_not_uniformly_within_15_points :
    ¬∀ i : Fin 6, validationError i ≤ 3 / 200 := by
  intro h
  have hi := h 4
  rw [validation_largest_error] at hi
  norm_num at hi

/-- Section 5.2.2, Table 2: even the displayed ±0.6-point uncertainty
interval for that row lies more than 1.5 points below the prediction. -/
theorem validation_discrepancy_exceeds_uncertainty :
    (membershipTable 4).observed + (membershipTable 4).uncertainty + (3 / 200 : ℚ) <
      (membershipTable 4).predicted := by
  norm_num [membershipTable]

end Transformer.Memorization
