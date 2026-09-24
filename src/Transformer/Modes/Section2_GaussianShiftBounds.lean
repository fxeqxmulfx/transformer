/-
# The number of modes of a Gaussian KDE — a uniform shifted Gaussian moment bound

The second display of `lem:phi-t` reduces to bounding a one-sided Gaussian
moment when the squared shift in Gaussian units is bounded.  The bounds below
follow from comparison with the unshifted moments of `lem:gaussian-int`.

Source: arXiv:2412.09080v3, `lem:phi-t`, `lem:gaussian-int`.
-/

import Transformer.Modes.Section2_GaussianInt

open Real MeasureTheory

namespace Transformer
namespace Modes

/-- A shifted Gaussian first moment is integrable on the positive half-line. -/
theorem integrableOn_mul_exp_neg_shift_sq {a δ : ℝ} (ha : 0 < a) :
    IntegrableOn (fun y : ℝ => y * Real.exp (-(a / 2) * (y - δ) ^ 2)) (Set.Ioi 0) := by
  have hb : 0 < a / 4 := by positivity
  have hbase : IntegrableOn (fun y : ℝ => y * Real.exp (-(a / 4) * y ^ 2)) (Set.Ioi 0) := by
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := a / 4) hb (s := 1) (by norm_num))
  have hmajor : IntegrableOn
      (fun y : ℝ => (y * Real.exp (-(a / 4) * y ^ 2)) * Real.exp (a * δ ^ 2 / 2))
      (Set.Ioi 0) := hbase.mul_const _
  apply Integrable.mono' hmajor (by fun_prop)
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
  have hy0 : 0 ≤ y := le_of_lt hy
  have he : -(a / 2) * (y - δ) ^ 2 ≤ -(a / 4) * y ^ 2 + a * δ ^ 2 / 2 := by
    nlinarith [sq_nonneg (y - 2 * δ), mul_nonneg ha.le (sq_nonneg (y - 2 * δ))]
  rw [Real.norm_of_nonneg (mul_nonneg hy0 (Real.exp_pos _).le)]
  calc
    y * Real.exp (-(a / 2) * (y - δ) ^ 2)
        ≤ y * Real.exp (-(a / 4) * y ^ 2 + a * δ ^ 2 / 2) := by gcongr
    _ = y * Real.exp (-(a / 4) * y ^ 2) * Real.exp (a * δ ^ 2 / 2) := by
      rw [Real.exp_add]; ring

/-- The positivity hypothesis for the shifted moment is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

/-- If `a δ² ≤ M`, the shifted one-sided first moment remains between two
fixed multiples of `a⁻¹`. -/
theorem integral_mul_exp_neg_shift_sq_bounds {a δ M : ℝ} (ha : 0 < a)
    (hM : a * δ ^ 2 ≤ M) :
    Real.exp (-M) * (2 * a)⁻¹
      ≤ ∫ y in Set.Ioi (0 : ℝ), y * Real.exp (-(a / 2) * (y - δ) ^ 2) ∧
    (∫ y in Set.Ioi (0 : ℝ), y * Real.exp (-(a / 2) * (y - δ) ^ 2))
      ≤ Real.exp (M / 2) * (a / 2)⁻¹ := by
  have hshift := integrableOn_mul_exp_neg_shift_sq ha (δ := δ)
  have hcenter₁ : IntegrableOn (fun y : ℝ => y * Real.exp (-a * y ^ 2)) (Set.Ioi 0) := by
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := a) ha (s := 1) (by norm_num))
  have hcenter₂ : IntegrableOn (fun y : ℝ => y * Real.exp (-(a / 4) * y ^ 2)) (Set.Ioi 0) := by
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := a / 4) (by positivity) (s := 1) (by norm_num))
  constructor
  · have hcmp := setIntegral_mono_on (hcenter₁.const_mul _) hshift measurableSet_Ioi
      (fun y hy => show Real.exp (-M) * (y * Real.exp (-a * y ^ 2))
        ≤ y * Real.exp (-(a / 2) * (y - δ) ^ 2) from by
          have he : -M + -a * y ^ 2 ≤ -(a / 2) * (y - δ) ^ 2 := by
            nlinarith [sq_nonneg (y + δ), mul_nonneg ha.le (sq_nonneg (y + δ))]
          calc
            Real.exp (-M) * (y * Real.exp (-a * y ^ 2))
                = y * Real.exp (-M + -a * y ^ 2) := by rw [Real.exp_add]; ring
            _
                ≤ y * Real.exp (-(a / 2) * (y - δ) ^ 2) := by gcongr; exact hy.le
            )
    rw [integral_const_mul, integral_mul_exp_neg_mul_sq ha] at hcmp
    exact hcmp
  · have hcmp := setIntegral_mono_on hshift (hcenter₂.const_mul _) measurableSet_Ioi
      (fun y hy => show y * Real.exp (-(a / 2) * (y - δ) ^ 2)
        ≤ Real.exp (M / 2) * (y * Real.exp (-(a / 4) * y ^ 2)) from by
          have he : -(a / 2) * (y - δ) ^ 2 ≤ M / 2 + -(a / 4) * y ^ 2 := by
            nlinarith [sq_nonneg (y - 2 * δ), mul_nonneg ha.le (sq_nonneg (y - 2 * δ))]
          calc
            y * Real.exp (-(a / 2) * (y - δ) ^ 2)
                ≤ y * Real.exp (M / 2 + -(a / 4) * y ^ 2) := by gcongr; exact hy.le
            _ = Real.exp (M / 2) * (y * Real.exp (-(a / 4) * y ^ 2)) := by
              rw [Real.exp_add]; ring)
    rw [integral_const_mul,
      integral_mul_exp_neg_mul_sq (show 0 < a / 4 by positivity)] at hcmp
    convert hcmp using 1
    ring

/-- The bounded-shift hypotheses hold, for example, without a shift. -/
example : (0 : ℝ) < 1 ∧ (1 : ℝ) * 0 ^ 2 ≤ 0 := by norm_num

end Modes
end Transformer
