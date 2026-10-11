/-
# Homogenized Transformers — a lower tail bound for a Gaussian product

The query/key assumption `ass:high_order_short` allows an entry of
`A = W W'ᵀ` to be the product of two independent standard Gaussians.
Their simultaneous occurrence in `[u,u+1]` has probability at least
`(2π)⁻¹ exp(-4u²)` for `u ≥ 1`. In particular this probability admits no
uniform upper bound `C exp(-6u²)`.

These are auxiliary estimates for checking the integrability used in the
proof of `prop:satisfying_MF`; they are not assertions made by the source.
Source: arXiv:2604.01978v1, `ass:high_order_short`, `eq:Lipschitz_measure`.
-/

import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.MeasureTheory.Measure.Prod

open scoped ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The normalizing constant of the standard Gaussian density. -/
noncomputable def gaussianUnitConstant : ℝ := (Real.sqrt (2 * Real.pi))⁻¹

/-- The actual joint law of two independent standard Gaussian weights. -/
noncomputable def gaussianProductLaw : Measure (ℝ × ℝ) :=
  (gaussianReal 0 1).prod (gaussianReal 0 1)

instance : IsProbabilityMeasure gaussianProductLaw := by
  unfold gaussianProductLaw
  infer_instance

/-- The Gaussian normalizer is strictly positive. -/
theorem gaussianUnitConstant_pos : 0 < gaussianUnitConstant := by
  unfold gaussianUnitConstant
  exact inv_pos.mpr (Real.sqrt_pos.mpr (mul_pos (by norm_num) Real.pi_pos))

/-- The unit Gaussian density, with its normalizer displayed explicitly. -/
theorem gaussianPDFReal_unit (x : ℝ) :
    gaussianPDFReal 0 1 x = gaussianUnitConstant * Real.exp (-(x ^ 2) / 2) := by
  simp [gaussianPDFReal, gaussianUnitConstant]

/-- On a positive unit interval the density is bounded below by its value
at the right endpoint. This retains the actual Gaussian density in the
estimate needed for `eq:Lipschitz_measure` of arXiv:2604.01978v1. -/
theorem gaussianPDFReal_unit_lower {u x : ℝ} (hu : 0 ≤ u)
    (hx : x ∈ Set.Icc u (u + 1)) :
    gaussianUnitConstant * Real.exp (-((u + 1) ^ 2) / 2) ≤ gaussianPDFReal 0 1 x := by
  rw [gaussianPDFReal_unit]
  apply mul_le_mul_of_nonneg_left _ gaussianUnitConstant_pos.le
  apply Real.exp_le_exp.mpr
  have hx0 : 0 ≤ x := hu.trans hx.1
  nlinarith [hx.2]

/-- The density lower bound has a nonempty, positive-interval witness. -/
example : (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ∈ Set.Icc 1 (1 + 1) := by norm_num

/-- Integrating the density lower bound over its interval gives an actual
probability lower bound. Source: arXiv:2604.01978v1,
`ass:high_order_short`, auxiliary to `eq:Lipschitz_measure`. -/
theorem gaussian_unit_interval_lower {u : ℝ} (hu : 0 ≤ u) :
    gaussianUnitConstant * Real.exp (-((u + 1) ^ 2) / 2) ≤
      (gaussianReal 0 1).real (Set.Icc u (u + 1)) := by
  let c := gaussianUnitConstant * Real.exp (-((u + 1) ^ 2) / 2)
  have hc : 0 ≤ c := mul_nonneg gaussianUnitConstant_pos.le (Real.exp_pos _).le
  have hbound : ENNReal.ofReal c ≤ (gaussianReal 0 1) (Set.Icc u (u + 1)) := by
    rw [gaussianReal_apply 0 (by norm_num)]
    calc ENNReal.ofReal c = ∫⁻ _x in Set.Icc u (u + 1), ENNReal.ofReal c := by
          simp
      _ ≤ ∫⁻ x in Set.Icc u (u + 1), gaussianPDF 0 1 x := by
          apply setLIntegral_mono' measurableSet_Icc
          intro x hx
          exact ENNReal.ofReal_le_ofReal (gaussianPDFReal_unit_lower hu hx)
  have hreal := ENNReal.toReal_mono (measure_ne_top _ _) hbound
  rw [ENNReal.toReal_ofReal hc] at hreal
  exact hreal

/-- The interval probability bound applies at `u = 1`. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- Independence squares the interval bound, rather than changing the
Gaussian product into a Gaussian. Source: arXiv:2604.01978v1,
`ass:high_order_short`, auxiliary to `eq:Lipschitz_measure`. -/
theorem gaussian_product_rectangle_lower {u : ℝ} (hu : 0 ≤ u) :
    gaussianUnitConstant ^ 2 * Real.exp (-((u + 1) ^ 2)) ≤
      gaussianProductLaw.real (Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)) := by
  rw [gaussianProductLaw, measureReal_prod_prod]
  have h := gaussian_unit_interval_lower hu
  have hc : 0 ≤ gaussianUnitConstant * Real.exp (-((u + 1) ^ 2) / 2) :=
    mul_nonneg gaussianUnitConstant_pos.le (Real.exp_pos _).le
  calc gaussianUnitConstant ^ 2 * Real.exp (-((u + 1) ^ 2))
      = (gaussianUnitConstant * Real.exp (-((u + 1) ^ 2) / 2)) *
          (gaussianUnitConstant * Real.exp (-((u + 1) ^ 2) / 2)) := by
        have he : Real.exp (-((u + 1) ^ 2)) =
            Real.exp (-((u + 1) ^ 2) / 2) * Real.exp (-((u + 1) ^ 2) / 2) := by
          rw [← Real.exp_add]
          congr 1
          ring
        rw [he]
        ring
    _ ≤ (gaussianReal 0 1).real (Set.Icc u (u + 1)) *
        (gaussianReal 0 1).real (Set.Icc u (u + 1)) := mul_le_mul h h hc (hc.trans h)

/-- Both coordinates of the rectangle can exceed a positive threshold. -/
example : (0 : ℝ) ≤ 1 ∧ ((1, 1) : ℝ × ℝ) ∈
    Set.Icc 1 (1 + 1) ×ˢ Set.Icc 1 (1 + 1) := by norm_num

/-- Every point of the rectangle gives a product of at least `u²`.
Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem gaussian_product_rectangle_subset {u : ℝ} (hu : 0 ≤ u) :
    Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1) ⊆ {p : ℝ × ℝ | u ^ 2 ≤ p.1 * p.2} := by
  intro p hp
  change u ^ 2 ≤ p.1 * p.2
  rw [pow_two]
  exact mul_le_mul hp.1.1 hp.2.1 hu (hu.trans hp.1.1)

/-- For `u ≥ 1` a simpler lower bound has a quadratic exponent with constant
`4`. Source: arXiv:2604.01978v1, `eq:Lipschitz_measure`. -/
theorem gaussian_product_rectangle_lower_quadratic {u : ℝ} (hu : 1 ≤ u) :
    gaussianUnitConstant ^ 2 * Real.exp (-4 * u ^ 2) ≤
      gaussianProductLaw.real (Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)) := by
  apply le_trans _ (gaussian_product_rectangle_lower (zero_le_one.trans hu))
  apply mul_le_mul_of_nonneg_left _ (sq_nonneg _)
  apply Real.exp_le_exp.mpr
  nlinarith [sq_nonneg (u - 1)]

/-- The quadratic bound's hypothesis holds at a nonzero threshold. -/
example : (1 : ℝ) ≤ 1 := le_rfl

/-- No constant can replace the actual rectangle probability by
`C exp(-6u²)`. This tests the exponential integrability needed by
arXiv:2604.01978v1, `eq:Lipschitz_measure`, without assuming it. -/
theorem gaussian_product_rectangle_not_upper (C : ℝ) :
    ∃ u : ℝ, 1 ≤ u ∧ C * Real.exp (-6 * u ^ 2) <
      gaussianProductLaw.real (Set.Icc u (u + 1) ×ˢ Set.Icc u (u + 1)) := by
  let u : ℝ := max 1 (C / gaussianUnitConstant ^ 2 + 1)
  have hu : 1 ≤ u := le_max_left _ _
  have hc : 0 < gaussianUnitConstant ^ 2 := sq_pos_of_pos gaussianUnitConstant_pos
  have hcu : C < gaussianUnitConstant ^ 2 * u := by
    have hlt : C / gaussianUnitConstant ^ 2 < u :=
      lt_of_lt_of_le (lt_add_one _) (le_max_right _ _)
    simpa only [mul_comm] using (div_lt_iff₀ hc).mp hlt
  have hexp : u ≤ Real.exp (2 * u ^ 2) := by
    have hpoly : u ≤ 2 * u ^ 2 + 1 := by nlinarith [sq_nonneg (u - 1)]
    exact hpoly.trans (Real.add_one_le_exp _)
  refine ⟨u, hu, lt_of_lt_of_le ?_ (gaussian_product_rectangle_lower_quadratic hu)⟩
  calc C * Real.exp (-6 * u ^ 2)
      < (gaussianUnitConstant ^ 2 * Real.exp (2 * u ^ 2)) *
          Real.exp (-6 * u ^ 2) :=
        mul_lt_mul_of_pos_right (hcu.trans_le (mul_le_mul_of_nonneg_left hexp hc.le))
          (Real.exp_pos _)
    _ = gaussianUnitConstant ^ 2 * Real.exp (-4 * u ^ 2) := by
      rw [mul_assoc, ← Real.exp_add]
      congr 2
      ring

end Transformer.Homogenized
