import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic

/-!
# The printed membership scaling law

arXiv:2505.24832v3, Section 5.2.1. The functional form and all printed
fit coefficients are retained exactly. Their consequences disagree with
the claimed limit and monotonic direction; the following theorems record
those discrepancies rather than silently changing the fit.
-/

namespace Transformer.Memorization

open Filter
open scoped Topology

/-- Section 5.2.1: the proposed scaling curve as a function of the
capacity-to-dataset-size ratio. This is an empirical fit, not the actual
F1 score of an unspecified attack. -/
noncomputable def membershipFromRatio (ratio c₁ c₂ c₃ : ℝ) : ℝ :=
  (1 + c₁ * Real.sigmoid (c₂ * (ratio + c₃))) / 2

/-- Section 5.2.1: the printed capacity/dataset-size scaling formula.
Theorems requiring a meaningful dataset ratio assume positive size. -/
noncomputable def membershipPrediction (capacity size c₁ c₂ c₃ : ℝ) : ℝ :=
  membershipFromRatio (capacity / size) c₁ c₂ c₃

/-- Section 5.2.1, "Fitting": the coefficients 1.34, -0.034, -33.14.
The source omits the name before the third coefficient; it is `c₃` in the
displayed three-coefficient equation. -/
noncomputable def fittedMembership (capacity size : ℝ) : ℝ :=
  membershipPrediction capacity size (67 / 50) (-17 / 500) (-1657 / 50)

/-- Section 5.2.1: for fixed coefficients and finite capacity, the exact
large-dataset limit is the value at ratio zero. It equals 0.5 only if the
sigmoid correction vanishes, which it cannot for positive `c₁`. -/
theorem membership_limit (capacity c₁ c₂ c₃ : ℝ) :
    Tendsto (fun n : ℕ => membershipPrediction capacity n c₁ c₂ c₃) atTop
      (𝓝 (membershipFromRatio 0 c₁ c₂ c₃)) := by
  have hc : Continuous (fun r : ℝ => membershipFromRatio r c₁ c₂ c₃) := by
    unfold membershipFromRatio
    fun_prop
  exact hc.continuousAt.tendsto.comp (tendsto_const_div_atTop_nhds_zero_nat capacity)

/-- Section 5.2.1, "Limiting behavior", counterexample to the printed
limit 0.5: with the reported coefficients the true limit is above 0.835. -/
theorem fitted_limit_above_0835 :
    (167 / 200 : ℝ) < membershipFromRatio 0 (67 / 50) (-17 / 500) (-1657 / 50) := by
  have harg : (0 : ℝ) < (-17 / 500) * (0 + -1657 / 50) := by norm_num
  have hs := Real.sigmoid_strictMono harg
  rw [Real.sigmoid_zero] at hs
  unfold membershipFromRatio
  norm_num at hs ⊢
  linarith

/-- Section 5.2.1: the exact limit of the fitted finite-capacity curve. -/
theorem fitted_membership_limit (capacity : ℝ) :
    Tendsto (fun n : ℕ => fittedMembership capacity n) atTop
      (𝓝 (membershipFromRatio 0 (67 / 50) (-17 / 500) (-1657 / 50))) :=
  membership_limit _ _ _ _

/-- Section 5.2.1, counterexample: the fit with negative `c₂` increases
as a positive dataset size grows at fixed positive model capacity. The
text asserts the opposite direction. -/
theorem fitted_membership_increases_with_data (capacity n m : ℝ)
    (hcap : 0 < capacity) (hn : 0 < n) (hnm : n < m) :
    fittedMembership capacity n < fittedMembership capacity m := by
  have hratio : capacity / m < capacity / n := div_lt_div_of_pos_left hcap hn hnm
  have harg : (-17 / 500 : ℝ) * (capacity / n + -1657 / 50) <
      (-17 / 500) * (capacity / m + -1657 / 50) := by linarith
  have hs := Real.sigmoid_strictMono harg
  unfold fittedMembership membershipPrediction membershipFromRatio
  linarith

/-- Section 5.2.1: capacity 1 and dataset sizes 1 and 2 satisfy all
strict monotonicity hypotheses. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) < 2 := by norm_num

/-- Section 5.2.1: even the reported infinite-data limit is not 0.5. -/
theorem fitted_limit_ne_random :
    membershipFromRatio 0 (67 / 50) (-17 / 500) (-1657 / 50) ≠ (1 / 2 : ℝ) := by
  have h := fitted_limit_above_0835
  intro he
  rw [he] at h
  norm_num at h

end Transformer.Memorization
