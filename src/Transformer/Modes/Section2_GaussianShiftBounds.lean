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

/-- **The shifted one-sided first moment grows linearly in the shift.**
`∫_0^∞ y e^{-a(y-δ)²/2} dy ≤ a⁻¹ (1 + 3 |δ| √a)`: on `y > 0`, `y` is at most the
positive part of `y - δ` plus `|δ|`, and the positive part integrates to `a⁻¹`
against the centred Gaussian, which is translation invariant.  Unlike
`integral_mul_exp_neg_shift_sq_bounds`, the bound is not exponential in `a δ²`.

Source: arXiv:2412.09080v3, `lem:phi-t`, second display (the shifted moment). -/
theorem integral_mul_exp_neg_shift_sq_le {a δ : ℝ} (ha : 0 < a) :
    ∫ y in Set.Ioi (0 : ℝ), y * Real.exp (-(a / 2) * (y - δ) ^ 2)
      ≤ a⁻¹ * (1 + 3 * (|δ| * Real.sqrt a)) := by
  have hb : 0 < a / 2 := by positivity
  -- the positive part of the centred first moment
  let f : ℝ → ℝ := (Set.Ioi (0 : ℝ)).indicator fun z => z * Real.exp (-(a / 2) * z ^ 2)
  have hf : Integrable f := by
    refine IntegrableOn.integrable_indicator ?_ measurableSet_Ioi
    simpa [Real.rpow_one] using
      (integrableOn_rpow_mul_exp_neg_mul_sq (b := a / 2) hb (s := 1) (by norm_num))
  let F : ℝ → ℝ := fun z => f z + |δ| * Real.exp (-(a / 2) * z ^ 2)
  have hFi : Integrable F := hf.add ((integrable_exp_neg_mul_sq hb).const_mul _)
  have hF0 : ∀ z, 0 ≤ F z := fun z => by
    refine add_nonneg (Set.indicator_nonneg (fun z hz => ?_) z) (by positivity)
    exact mul_nonneg (le_of_lt hz) (Real.exp_pos _).le
  have hpt : ∀ y ∈ Set.Ioi (0 : ℝ),
      y * Real.exp (-(a / 2) * (y - δ) ^ 2) ≤ F (y - δ) := by
    intro y hy
    have hy0 : 0 < y := hy
    have he := Real.exp_pos (-(a / 2) * (y - δ) ^ 2)
    by_cases hz : 0 < y - δ
    · have hfz : f (y - δ) = (y - δ) * Real.exp (-(a / 2) * (y - δ) ^ 2) :=
        Set.indicator_of_mem (show y - δ ∈ Set.Ioi (0 : ℝ) from hz) _
      change _ ≤ f (y - δ) + |δ| * _
      rw [hfz]
      nlinarith [mul_le_mul_of_nonneg_right (le_abs_self δ) he.le]
    · have hfz : f (y - δ) = 0 :=
        Set.indicator_of_notMem (show y - δ ∉ Set.Ioi (0 : ℝ) from hz) _
      change _ ≤ f (y - δ) + |δ| * _
      rw [hfz]
      nlinarith [mul_le_mul_of_nonneg_right (le_abs_self δ) he.le]
  have hshift := hFi.comp_sub_right δ
  calc ∫ y in Set.Ioi (0 : ℝ), y * Real.exp (-(a / 2) * (y - δ) ^ 2)
      ≤ ∫ y in Set.Ioi (0 : ℝ), F (y - δ) :=
        setIntegral_mono_on (integrableOn_mul_exp_neg_shift_sq ha) hshift.integrableOn
          measurableSet_Ioi hpt
    _ ≤ ∫ y, F (y - δ) := setIntegral_le_integral hshift (Filter.Eventually.of_forall fun y => hF0 _)
    _ = ∫ z, F z := integral_sub_right_eq_self F δ
    _ = (2 * (a / 2))⁻¹ + |δ| * Real.sqrt (π / (a / 2)) := by
        rw [show (∫ z, F z) = (∫ z, f z) + |δ| * ∫ z, Real.exp (-(a / 2) * z ^ 2) from by
          rw [← integral_const_mul, ← integral_add hf ((integrable_exp_neg_mul_sq hb).const_mul _)],
          integral_indicator measurableSet_Ioi, integral_mul_exp_neg_mul_sq hb, integral_gaussian]
    _ ≤ a⁻¹ * (1 + 3 * (|δ| * Real.sqrt a)) := by
        have hsa : 0 < Real.sqrt a := Real.sqrt_pos.2 ha
        have hsq : Real.sqrt (π / (a / 2)) ≤ 3 * Real.sqrt a / a := by
          rw [Real.sqrt_le_iff]
          refine ⟨by positivity, ?_⟩
          rw [div_pow, mul_pow, Real.sq_sqrt ha.le, div_le_div_iff₀ (by positivity) (by positivity)]
          nlinarith [Real.pi_le_four, ha]
        have h2 : (2 * (a / 2))⁻¹ = a⁻¹ := by field_simp
        rw [h2]
        calc a⁻¹ + |δ| * Real.sqrt (π / (a / 2))
            ≤ a⁻¹ + |δ| * (3 * Real.sqrt a / a) := by gcongr
          _ = a⁻¹ * (1 + 3 * (|δ| * Real.sqrt a)) := by ring

/-- The hypothesis of `integral_mul_exp_neg_shift_sq_le` is satisfiable, for
every shift, for example `a = 1`, `δ = 0`. -/
example : ∫ y in Set.Ioi (0 : ℝ), y * Real.exp (-((1 : ℝ) / 2) * (y - 0) ^ 2)
    ≤ (1 : ℝ)⁻¹ * (1 + 3 * (|(0 : ℝ)| * Real.sqrt 1)) :=
  integral_mul_exp_neg_shift_sq_le one_pos

end Modes
end Transformer
