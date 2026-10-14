/-
# Uniform bounds for actual spherical kernel derivatives

ArXiv:2601.21366v2, §5.3, the localization argument for
`thm: any.d` (i), extending §5.2, `eq:Hess-bilinear-analytic`.
The actual transported positions have norm one and their velocities
keep their initial norms. Cauchy–Schwarz therefore bounds both
pointwise time derivatives uniformly over all real transport times.

The mixed derivative has a separate bound by the product of the
two initial speeds. After integration this gives the square of
the field's L¹ norm. Localizing near a non-atomic support point
then makes this bound smaller than the strict L² coercivity bound.
All constants retain the original positive temperature and kernel.
The first derivative bound is `e^β β (‖v‖+‖w‖)`; the second is
`e^β (β²+β) (‖v‖+‖w‖)²`. The mixed bound is instead
`e^β (β²+β) ‖v‖ ‖w‖`, which retains the localization factor
after integration against the original product measure.
-/

import Transformer.Perceptron.Section5_SphericalKernel

open Real

namespace Transformer.Perceptron

/-- The original kernel stays bounded on the actual geodesics.
Source: arXiv:2601.21366v2, §5.3, boundedness of the interaction kernel. -/
theorem abs_spherePairKernel_le {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |spherePairKernel β x v y w t| ≤ Real.exp β := by
  have hi := abs_real_inner_le_norm (spherePosition x v t) (spherePosition y w t)
  rw [norm_spherePosition hx hv, norm_spherePosition hy hw, one_mul] at hi
  rw [spherePairKernel, abs_of_pos (Real.exp_pos _)]
  apply Real.exp_le_exp.mpr
  simpa only [mul_one] using mul_le_mul_of_nonneg_left ((le_abs_self _).trans hi) hβ

/-- The inner-product derivative is bounded by the two actual speeds.
Source: arXiv:2601.21366v2, §5.3, the first transported kernel derivative. -/
theorem abs_spherePairSlope_le {d : ℕ} {x v y w : EucSpace d}
    (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |spherePairSlope x v y w t| ≤ ‖v‖ + ‖w‖ := by
  have h₁ := abs_real_inner_le_norm (sphereVelocity x v t) (spherePosition y w t)
  have h₂ := abs_real_inner_le_norm (spherePosition x v t) (sphereVelocity y w t)
  rw [norm_sphereVelocity hx hv, norm_spherePosition hy hw, mul_one] at h₁
  rw [norm_spherePosition hx hv, norm_sphereVelocity hy hw, one_mul] at h₂
  exact (abs_add_le _ _).trans (add_le_add h₁ h₂)

/-- Radial acceleration bounds the actual second inner-product derivative.
Source: arXiv:2601.21366v2, §5.3, the interaction second variation. -/
theorem abs_spherePairAcceleration_le {d : ℕ} {x v y w : EucSpace d}
    (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |spherePairAcceleration x v y w t| ≤ (‖v‖ + ‖w‖) ^ 2 := by
  have h₁ := abs_real_inner_le_norm (sphereVelocity x v t) (sphereVelocity y w t)
  have h₂ := abs_real_inner_le_norm (spherePosition x v t) (spherePosition y w t)
  rw [norm_sphereVelocity hx hv, norm_sphereVelocity hy hw] at h₁
  rw [norm_spherePosition hx hv, norm_spherePosition hy hw, one_mul] at h₂
  calc
    _ ≤ |2 * inner (𝕜 := ℝ) (sphereVelocity x v t) (sphereVelocity y w t)| +
        |(‖v‖ ^ 2 + ‖w‖ ^ 2) *
          inner (𝕜 := ℝ) (spherePosition x v t) (spherePosition y w t)| := by
      simpa only [spherePairAcceleration, sub_zero, zero_sub, abs_neg] using abs_sub_le
        (2 * inner (𝕜 := ℝ) (sphereVelocity x v t) (sphereVelocity y w t)) 0
        ((‖v‖ ^ 2 + ‖w‖ ^ 2) *
          inner (𝕜 := ℝ) (spherePosition x v t) (spherePosition y w t))
    _ ≤ 2 * (‖v‖ * ‖w‖) + (‖v‖ ^ 2 + ‖w‖ ^ 2) * 1 := by
      rw [abs_mul, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 2),
        abs_of_nonneg (by positivity : 0 ≤ ‖v‖ ^ 2 + ‖w‖ ^ 2)]
      gcongr
    _ = _ := by ring

/-- A global bound for the actual first kernel derivative.
Source: arXiv:2601.21366v2, §5.3, differentiating the interaction integral. -/
theorem abs_spherePairForce_le {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |spherePairForce β x v y w t| ≤ Real.exp β * (β * (‖v‖ + ‖w‖)) := by
  rw [spherePairForce, abs_mul, abs_mul, abs_of_nonneg hβ]
  exact mul_le_mul (abs_spherePairKernel_le β hβ hx hv hy hw t)
    (mul_le_mul_of_nonneg_left (abs_spherePairSlope_le hx hv hy hw t) hβ)
    (by positivity) (Real.exp_pos β).le

/-- The actual second derivative has a time-independent quadratic speed
bound. Source: arXiv:2601.21366v2, §5.3, the second interaction differentiation. -/
theorem abs_spherePairCurvature_le {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |spherePairCurvature β x v y w t| ≤
      Real.exp β * ((β ^ 2 + β) * (‖v‖ + ‖w‖) ^ 2) := by
  have hS := abs_spherePairSlope_le hx hv hy hw t
  have hS₂ : spherePairSlope x v y w t ^ 2 ≤ (‖v‖ + ‖w‖) ^ 2 := by
    nlinarith [abs_nonneg (spherePairSlope x v y w t),
      sq_abs (spherePairSlope x v y w t)]
  have hA := abs_spherePairAcceleration_le hx hv hy hw t
  have hbound : |β ^ 2 * spherePairSlope x v y w t ^ 2 +
      β * spherePairAcceleration x v y w t| ≤ (β ^ 2 + β) * (‖v‖ + ‖w‖) ^ 2 := by
    calc
      _ ≤ |β ^ 2 * spherePairSlope x v y w t ^ 2| +
          |β * spherePairAcceleration x v y w t| := abs_add_le _ _
      _ ≤ β ^ 2 * (‖v‖ + ‖w‖) ^ 2 + β * (‖v‖ + ‖w‖) ^ 2 := by
        rw [abs_of_nonneg (by positivity : 0 ≤ β ^ 2 * spherePairSlope x v y w t ^ 2),
          abs_mul, abs_of_nonneg hβ]
        gcongr
      _ = _ := by ring
  rw [spherePairCurvature, abs_mul]
  exact mul_le_mul (abs_spherePairKernel_le β hβ hx hv hy hw t) hbound
    (abs_nonneg _) (Real.exp_pos β).le

/-- The original mixed curvature is bounded by the product of the two
initial speeds. Source: arXiv:2601.21366v2, §5.3, the localized Hessian bound. -/
theorem abs_spherePairMixedCurvature_le {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    |spherePairMixedCurvature β x v y w| ≤
      Real.exp β * ((β ^ 2 + β) * ‖v‖ * ‖w‖) := by
  have h₁ := abs_real_inner_le_norm v y
  have h₂ := abs_real_inner_le_norm x w
  have h₃ := abs_real_inner_le_norm v w
  have h₄ := abs_real_inner_le_norm x y
  rw [hy, mul_one] at h₁
  rw [hx, one_mul] at h₂
  rw [hx, hy, one_mul] at h₄
  have hK : Real.exp (β * inner (𝕜 := ℝ) x y) ≤ Real.exp β := by
    apply Real.exp_le_exp.mpr
    simpa only [mul_one] using mul_le_mul_of_nonneg_left ((le_abs_self _).trans h₄) hβ
  have hbound : |β ^ 2 * inner (𝕜 := ℝ) v y * inner (𝕜 := ℝ) x w +
      β * inner (𝕜 := ℝ) v w| ≤ (β ^ 2 + β) * ‖v‖ * ‖w‖ := by
    calc
      _ ≤ |β ^ 2 * inner (𝕜 := ℝ) v y * inner (𝕜 := ℝ) x w| +
          |β * inner (𝕜 := ℝ) v w| := abs_add_le _ _
      _ ≤ β ^ 2 * ‖v‖ * ‖w‖ + β * (‖v‖ * ‖w‖) := by
        rw [abs_mul, abs_mul, abs_mul, abs_of_nonneg (sq_nonneg β), abs_of_nonneg hβ]
        gcongr
      _ = _ := by ring
  rw [spherePairMixedCurvature, abs_mul, abs_of_pos (Real.exp_pos _)]
  exact mul_le_mul hK hbound (abs_nonneg _) (Real.exp_pos β).le

/-- Positive temperature, unit positions and nonzero tangent velocities
jointly satisfy every input of these general-dimensional kernel bounds.
Source: arXiv:2601.21366v2, §2.2 and §5.2, the actual circle geodesic. -/
example : (0 : ℝ) ≤ 1 ∧ ‖(circlePoint 0 : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (circlePoint 0 : EucSpace 2)
      (circleTangent (circlePoint 0 : EucSpace 2)) = 0 ∧
    ‖circleTangent (circlePoint 0 : EucSpace 2)‖ = 1 :=
  ⟨zero_le_one, norm_eq_of_mem_sphere _, inner_circleTangent _,
    (norm_circleTangent _).trans (norm_eq_of_mem_sphere _)⟩

end Transformer.Perceptron
