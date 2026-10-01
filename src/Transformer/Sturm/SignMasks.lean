/-
# Polynomial masks for finite real sign constraints

The masks are twice the indicator of the actual sign requirement.
Their quadratic formulas allow a finite linear combination of
Sturm--Tarski queries. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2.
-/

import Transformer.Sturm.TarskiQuery
import Transformer.Normalization.AnalyticSignStability

noncomputable section
open Polynomial Transformer.Normalization
open scoped Classical

namespace Transformer.Sturm

/-- Twice the indicator mask for equality, positivity, or
nonnegativity, written as a quadratic expression in the integer sign.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def scaledSignMask (requirement : AnalyticSignRequirement) (s : SignType) : ℤ :=
  match requirement with
  | .zero => 2 - 2 * (s : ℤ) ^ 2
  | .positive => (s : ℤ) ^ 2 + (s : ℤ)
  | .nonnegative => 2 + (s : ℤ) - (s : ℤ) ^ 2

/-- Coefficients of the quadratic sign mask, in degree order.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def signMaskCoefficients (requirement : AnalyticSignRequirement) : Fin 3 → ℤ :=
  match requirement with
  | .zero => ![2, 0, -2]
  | .positive => ![0, 1, 1]
  | .nonnegative => ![2, 1, -1]

/-- The three mask coefficients give their declared quadratic
expression for each actual sign. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem signMaskCoefficients_formula (requirement : AnalyticSignRequirement) (s : SignType) :
    signMaskCoefficients requirement 0 + signMaskCoefficients requirement 1 * (s : ℤ) +
      signMaskCoefficients requirement 2 * (s : ℤ) ^ 2 = scaledSignMask requirement s := by
  cases requirement <;> cases s <;> norm_num [signMaskCoefficients, scaledSignMask, SignType.cast]

/-- A mask evaluated on the sign of a real number is exactly twice the
indicator of the original requirement. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem scaledSignMask_value (requirement : AnalyticSignRequirement) (x : ℝ) :
    scaledSignMask requirement (SignType.sign x) = if requirement.Holds x then 2 else 0 := by
  rcases lt_trichotomy x 0 with hx | rfl | hx
  · cases requirement <;> simp [scaledSignMask, AnalyticSignRequirement.Holds,
      sign_neg hx, hx.ne, not_le_of_gt hx, not_lt_of_ge hx.le]
  · cases requirement <;> simp [scaledSignMask, AnalyticSignRequirement.Holds]
  · cases requirement <;> simp [scaledSignMask, AnalyticSignRequirement.Holds,
      sign_pos hx, hx, hx.ne', hx.le]

/-- The product of the finite sign masks at an actual real point.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
def signConstraintWeight : List (Polynomial ℝ × AnalyticSignRequirement) → ℝ → ℤ
  | [], _ => 1
  | (q, requirement) :: rest, x =>
      scaledSignMask requirement (SignType.sign (q.eval x)) * signConstraintWeight rest x

/-- The joint weight is `2^m` exactly at points satisfying every one of
the `m` actual sign conditions, and is zero elsewhere. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem signConstraintWeight_value (conditions : List (Polynomial ℝ × AnalyticSignRequirement))
    (x : ℝ) : signConstraintWeight conditions x =
      if ∀ c ∈ conditions, c.2.Holds (c.1.eval x) then (2 : ℤ) ^ conditions.length else 0 := by
  induction conditions with
  | nil => simp [signConstraintWeight]
  | cons c rest ih =>
      rcases c with ⟨q, requirement⟩
      rw [signConstraintWeight, scaledSignMask_value, ih]
      simp only [List.forall_mem_cons, List.length_cons]
      by_cases hc : requirement.Holds (q.eval x)
      · rw [ite_eq_left hc]
        by_cases hr : ∀ c ∈ rest, c.2.Holds (c.1.eval x)
        · rw [ite_eq_left hr, ite_eq_left ⟨hc, hr⟩, pow_succ]
          exact mul_comm _ _
        · have hn : ¬(requirement.Holds (q.eval x) ∧
              ∀ c ∈ rest, c.2.Holds (c.1.eval x)) := fun h => hr h.2
          rw [ite_eq_right hr, ite_eq_right hn, mul_zero]
      · have hn : ¬(requirement.Holds (q.eval x) ∧
            ∀ c ∈ rest, c.2.Holds (c.1.eval x)) := fun h => hc h.1
        rw [ite_eq_right hc, ite_eq_right hn, zero_mul]

end Transformer.Sturm
