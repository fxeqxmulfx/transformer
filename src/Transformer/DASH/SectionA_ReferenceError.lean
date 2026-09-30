/-
# DASH — geometric accuracy of inverse-power reference polynomials

arXiv:2602.02016v2, Appendix A, regularized inverse-root approximation.
The positive interval stays inside a strict subdisk of the binomial
series, so its midpoint reference errors decrease geometrically.
-/

import Transformer.DASH.SectionA_BinomialReference

noncomputable section

namespace Transformer.DASH

/-- The degree-`d` midpoint binomial references have a uniform geometric
error bound on the whole regularized interval, for every real exponent.
The constants depend on `ε` and the exponent, and are independent of `d`.
Source: arXiv:2602.02016v2, Appendix A, approximation on `[ε,1+ε]`. -/
theorem inversePowerReference_geometric_error (ε exponent : ℝ) (hε : 0 < ε) :
    ∃ q ∈ Set.Ioo (0 : ℝ) 1, ∃ C > 0, ∀ d : ℕ, ∀ t : ℝ, -1 ≤ t → t ≤ 1 →
      |(inversePowerReference ε exponent d).eval t -
        (chebFromCoordinate ε (1 + ε) t) ^ exponent| ≤ C * q ^ (d + 1) := by
  have hden : (0 : ℝ) < 2 * ε + 1 := by linarith
  have hfrac : (0 : ℝ) < 1 / (2 * ε + 1) := by positivity
  have hfrac' : (1 : ℝ) / (2 * ε + 1) < 1 :=
    (div_lt_iff₀ hden).2 (by linarith)
  let R : NNReal := ⟨(1 + 1 / (2 * ε + 1)) / 2, by positivity⟩
  have hR : (R : ℝ) < 1 := by
    change (1 + 1 / (2 * ε + 1)) / 2 < 1
    linarith
  have hinside : (1 : ℝ) / (2 * ε + 1) < R := by
    change 1 / (2 * ε + 1) < (1 + 1 / (2 * ε + 1)) / 2
    linarith
  obtain ⟨q, hq, C, hC, hgeo⟩ :=
    (Real.one_add_rpow_hasFPowerSeriesOnBall_zero (a := exponent)).uniform_geometric_approx
      (r' := R) (by exact_mod_cast hR)
  have hm : (0 : ℝ) < (ε + 1 / 2) ^ exponent :=
    Real.rpow_pos_of_pos (by linarith) exponent
  refine ⟨q, hq, (ε + 1 / 2) ^ exponent * C, mul_pos hm hC, ?_⟩
  intro d t ht ht'
  have hball : t / (2 * ε + 1) ∈ Metric.ball (0 : ℝ) R := by
    rw [Metric.mem_ball, Real.dist_eq, sub_zero, abs_div, abs_of_pos hden]
    exact ((div_le_div_iff_of_pos_right hden).2 (abs_le.2 ⟨ht, ht'⟩)).trans_lt hinside
  have hseries : |(binomialSeries ℝ exponent).partialSum (d + 1) (t / (2 * ε + 1)) -
      (1 + t / (2 * ε + 1)) ^ exponent| ≤ C * q ^ (d + 1) := by
    simpa only [zero_add, Real.norm_eq_abs, abs_sub_comm] using
      hgeo (t / (2 * ε + 1)) hball (d + 1)
  rw [inversePowerReference_eval, inverse_power_midpoint_factor ε exponent t hε ht,
    ← mul_sub, abs_mul, abs_of_pos hm]
  exact (mul_le_mul_of_nonneg_left hseries hm.le).trans_eq (mul_assoc _ _ _).symm

/-- The positive regularization assumption has the source's intended
nondegenerate instance, arXiv:2602.02016v2, Appendix A. -/
example : (0 : ℝ) < 1 / 100 := by norm_num

end Transformer.DASH
