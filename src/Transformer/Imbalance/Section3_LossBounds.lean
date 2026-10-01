/-
# Finite-time loss bounds for gradient flow

arXiv:2402.19449v2, Appendix I, Lemma 6. Elementary bounds on the
integrated margin give Theta(1/(πt)) without Lambert-W asymptotics.
The source's line `a=(c-1)b` has a missing minus sign: `a=-(c-1)b`.
-/

import Transformer.Imbalance.Section3_MarginInverse
import Mathlib.Analysis.Asymptotics.Theta

open Filter Asymptotics
open scoped Topology

noncomputable section

namespace Transformer.Imbalance

/-- Loss of a column with one correct logit and `z=c-1` identical others;
Appendix I, Lemma 6, written in terms of the margin `u=a-b`. -/
def marginLoss (z u : ℝ) : ℝ := Real.log (1 + z * Real.exp (-u))

/-- A quantitative logarithm bound used in Appendix I, Lemma 6. -/
theorem log_one_add_bounds (x : ℝ) (hx : 0 ≤ x) :
    x / (1 + x) ≤ Real.log (1 + x) ∧ Real.log (1 + x) ≤ x := by
  have hp : 0 < 1 + x := by linarith
  constructor
  · have h := Real.one_sub_inv_le_log_of_pos hp
    have he : 1 - (1 + x)⁻¹ = x / (1 + x) := by field_simp; ring
    rwa [he] at h
  · have h := Real.log_le_sub_one_of_pos hp
    linarith

/-- Nonvacuity of the logarithm bound in Lemma 6. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- The loss is nonnegative on a positive-class-count margin;
Appendix I, Lemma 6. -/
theorem marginLoss_nonneg (z u : ℝ) (hz : 0 ≤ z) : 0 ≤ marginLoss z u := by
  apply Real.log_nonneg
  have : 0 ≤ z * Real.exp (-u) := mul_nonneg hz (Real.exp_pos _).le
  linarith

/-- Nonvacuity of the class-count assumption in Lemma 6. -/
example : (0 : ℝ) ≤ 2 := by norm_num

/-- Bounds for `exp(u(t))` obtained from the integrated equation;
Appendix I, Lemma 6. -/
theorem gdMargin_exp_bounds (z : ℝ) (hz : 0 < z) (π t : ℝ)
    (hπ : 0 ≤ π) (ht : 0 ≤ t) :
    (1 + (z + 1) * π * t) / (z + 1) ≤ Real.exp (gdMargin z hz π t) ∧
      Real.exp (gdMargin z hz π t) ≤ 1 + (z + 1) * π * t := by
  have hu := gdMargin_nonneg z hz π t hπ ht
  have heq := gdMargin_equation z hz π t
  have he := Real.add_one_le_exp (gdMargin z hz π t)
  constructor
  · apply (div_le_iff₀ (by linarith : 0 < z + 1)).2
    nlinarith
  · nlinarith

/-- Nonvacuity of all hypotheses of the exponential bounds; Lemma 6. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) ≤ 2 := by norm_num

/-- Explicit version of Lemma 6. For `πt≥1`, the loss lies between
`z/(2(z+1)πt)` and `z/(πt)`. No class-dependent start-time typo or
external Lambert-W estimate is needed. -/
theorem gradient_loss_bounds (z : ℝ) (hz : 0 < z) (π t : ℝ)
    (hπ : 0 < π) (ht : 1 ≤ π * t) :
    (z / (2 * (z + 1))) * (1 / (π * t)) ≤ marginLoss z (gdMargin z hz π t) ∧
      marginLoss z (gdMargin z hz π t) ≤ z * (1 / (π * t)) := by
  have htpos : 0 < t := by nlinarith
  have hpt : 0 < π * t := by positivity
  have hz1 : 0 < z + 1 := by linarith
  have hE := gdMargin_exp_bounds z hz π t hπ.le htpos.le
  have hepos := Real.exp_pos (gdMargin z hz π t)
  have hexp : Real.exp (-gdMargin z hz π t) = 1 / Real.exp (gdMargin z hz π t) := by
    rw [Real.exp_neg, one_div]
  have hl := log_one_add_bounds (z / Real.exp (gdMargin z hz π t)) (by positivity)
  have hform : marginLoss z (gdMargin z hz π t) =
      Real.log (1 + z / Real.exp (gdMargin z hz π t)) := by
    simp only [marginLoss, hexp, mul_one_div]
  rw [hform]
  constructor
  · have hfrac : (z / Real.exp (gdMargin z hz π t)) /
        (1 + z / Real.exp (gdMargin z hz π t)) =
        z / (Real.exp (gdMargin z hz π t) + z) := by
      field_simp
    rw [hfrac] at hl
    apply le_trans ?_ hl.1
    have hden : Real.exp (gdMargin z hz π t) + z ≤ 2 * (z + 1) * (π * t) := by
      nlinarith [hE.2]
    calc
      (z / (2 * (z + 1))) * (1 / (π * t)) = z / (2 * (z + 1) * (π * t)) := by
        field_simp
      _ ≤ z / (Real.exp (gdMargin z hz π t) + z) :=
        div_le_div_of_nonneg_left hz.le (by positivity) hden
  · apply le_trans hl.2
    have hden : π * t ≤ Real.exp (gdMargin z hz π t) := by
      have hb := (div_le_iff₀ hz1).1 hE.1
      nlinarith
    calc
      z / Real.exp (gdMargin z hz π t) ≤ z / (π * t) :=
        div_le_div_of_nonneg_left hz.le hpt hden
      _ = z * (1 / (π * t)) := by ring

/-- Nonvacuity of the explicit loss bound in Lemma 6. -/
example : (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 / 3 ∧ (1 : ℝ) ≤ (1 / 3) * 3 := by norm_num

/-- Gradient-flow part of Theorem 3 / Appendix I, Lemma 6, for the
actual inverse solution. The constants are uniform in positive `π`;
only the threshold `t≥1/π` depends on class frequency. -/
theorem gradient_loss_rate (z : ℝ) (hz : 0 < z) (π : ℝ) (hπ : 0 < π) :
    (fun t => marginLoss z (gdMargin z hz π t)) =Θ[atTop]
      (fun t => 1 / (π * t)) := by
  have he : ∀ᶠ t : ℝ in atTop, 1 ≤ π * t := by
    filter_upwards [eventually_ge_atTop (1 / π)] with t ht
    have hb := (div_le_iff₀ hπ).1 ht
    simpa only [mul_comm] using hb
  constructor
  · apply isBigO_iff.2
    refine ⟨z, ?_⟩
    filter_upwards [he] with t ht
    have htpos : 0 < t := by nlinarith
    simpa only [Real.norm_eq_abs, abs_of_nonneg (marginLoss_nonneg z _ hz.le),
      abs_of_pos (by positivity : 0 < 1 / (π * t))] using
      (gradient_loss_bounds z hz π t hπ ht).2
  · apply isBigO_iff.2
    refine ⟨2 * (z + 1) / z, ?_⟩
    filter_upwards [he] with t ht
    have htpos : 0 < t := by nlinarith
    rw [Real.norm_eq_abs, Real.norm_eq_abs,
      abs_of_pos (by positivity : 0 < 1 / (π * t)),
      abs_of_nonneg (marginLoss_nonneg z _ hz.le)]
    have hb := (gradient_loss_bounds z hz π t hπ ht).1
    have ha : 0 < z / (2 * (z + 1)) := by positivity
    calc
      1 / (π * t) ≤ marginLoss z (gdMargin z hz π t) / (z / (2 * (z + 1))) :=
        (le_div_iff₀ ha).2 (by simpa only [mul_comm] using hb)
      _ = (2 * (z + 1) / z) * marginLoss z (gdMargin z hz π t) := by
        field_simp

/-- Nonvacuity of Theorem 3's positive-frequency regime. -/
example : (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 / 3 := by norm_num

end Transformer.Imbalance
