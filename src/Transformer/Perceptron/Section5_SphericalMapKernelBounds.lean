module

public import Transformer.Perceptron.Section5_SphericalMapKernel

/-!
# Uniform bounds for the actual matrix kernel derivatives

ArXiv:2601.21366v2, §5.3, the localized-gradient argument
in `thm: any.d` (i), and §3.1, `rem: ext` (ii).
Write D for the operator norm of the actual attention matrix.
Cauchy–Schwarz and the constant original geodesic speeds give
uniform bounds at every real time, for arbitrary tangent inputs.

The actual first derivative is bounded by `exp D · D (‖v‖+‖w‖)`.
The actual second derivative is bounded by
`exp D · (D²+D) (‖v‖+‖w‖)²`. The mixed derivative instead retains
the product `‖v‖ ‖w‖`, which yields the square of the field's
L¹ norm after integration. This is the estimate required to compare
small localized mass with the strict L² lower bound.
The source's unspecified finite matrix constant is made explicit;
no positivity, symmetry or invertibility is needed for these bounds.
-/

@[expose] public section

open Real
namespace Transformer.Perceptron

/-- Cauchy–Schwarz and the actual matrix operator norm bound its bilinear form.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem abs_inner_map_le {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d) (x y : EucSpace d) :
    |inner (𝕜 := ℝ) x (B y)| ≤ ‖B.toContinuousLinearMap‖ * ‖x‖ * ‖y‖ := by
  calc
    _ ≤ ‖x‖ * ‖B y‖ := abs_real_inner_le_norm x (B y)
    _ ≤ ‖x‖ * (‖B.toContinuousLinearMap‖ * ‖y‖) :=
      mul_le_mul_of_nonneg_left (B.toContinuousLinearMap.le_opNorm y) (norm_nonneg x)
    _ = _ := by ring

/-- The original matrix kernel is bounded on the actual geodesics.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem abs_sphereMapPairKernel_le {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |sphereMapPairKernel B x v y w t| ≤ Real.exp ‖B.toContinuousLinearMap‖ := by
  have hi := abs_inner_map_le B (spherePosition x v t) (spherePosition y w t)
  rw [norm_spherePosition hx hv, norm_spherePosition hy hw, mul_one, mul_one] at hi
  rw [sphereMapPairKernel, abs_of_pos (Real.exp_pos _)]
  exact Real.exp_le_exp.mpr ((le_abs_self _).trans hi)

/-- Both actual speeds bound the original exponent's time derivative.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem abs_sphereMapPairSlope_le {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |sphereMapPairSlope B x v y w t| ≤ ‖B.toContinuousLinearMap‖ * (‖v‖ + ‖w‖) := by
  have h₁ := abs_inner_map_le B (sphereVelocity x v t) (spherePosition y w t)
  have h₂ := abs_inner_map_le B (spherePosition x v t) (sphereVelocity y w t)
  rw [norm_sphereVelocity hx hv, norm_spherePosition hy hw, mul_one] at h₁
  rw [norm_spherePosition hx hv, norm_sphereVelocity hy hw, mul_one] at h₂
  calc
    _ ≤ _ := (abs_add_le _ _).trans (add_le_add h₁ h₂)
    _ = _ := by ring

/-- The original radial accelerations give a quadratic speed bound.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem abs_sphereMapPairAcceleration_le {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |sphereMapPairAcceleration B x v y w t| ≤
      ‖B.toContinuousLinearMap‖ * (‖v‖ + ‖w‖) ^ 2 := by
  have h₁ := abs_inner_map_le B (sphereVelocity x v t) (sphereVelocity y w t)
  have h₂ := abs_inner_map_le B (spherePosition x v t) (spherePosition y w t)
  rw [norm_sphereVelocity hx hv, norm_sphereVelocity hy hw] at h₁
  rw [norm_spherePosition hx hv, norm_spherePosition hy hw, mul_one, mul_one] at h₂
  calc
    _ ≤ |2 * inner (𝕜 := ℝ) (sphereVelocity x v t) (B (sphereVelocity y w t))| +
        |(‖v‖ ^ 2 + ‖w‖ ^ 2) *
          inner (𝕜 := ℝ) (spherePosition x v t) (B (spherePosition y w t))| := by
      simpa only [sphereMapPairAcceleration, sub_zero, zero_sub, abs_neg] using abs_sub_le
        (2 * inner (𝕜 := ℝ) (sphereVelocity x v t) (B (sphereVelocity y w t))) 0
        ((‖v‖ ^ 2 + ‖w‖ ^ 2) *
          inner (𝕜 := ℝ) (spherePosition x v t) (B (spherePosition y w t)))
    _ ≤ 2 * (‖B.toContinuousLinearMap‖ * ‖v‖ * ‖w‖) +
        (‖v‖ ^ 2 + ‖w‖ ^ 2) * ‖B.toContinuousLinearMap‖ := by
      rw [abs_mul, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 2),
        abs_of_nonneg (by positivity : 0 ≤ ‖v‖ ^ 2 + ‖w‖ ^ 2)]
      gcongr
    _ = _ := by ring

/-- The actual first kernel derivative has a global bound in time.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem abs_sphereMapPairForce_le {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |sphereMapPairForce B x v y w t| ≤ Real.exp ‖B.toContinuousLinearMap‖ *
      (‖B.toContinuousLinearMap‖ * (‖v‖ + ‖w‖)) := by
  rw [sphereMapPairForce, abs_mul]
  exact mul_le_mul (abs_sphereMapPairKernel_le B hx hv hy hw t)
    (abs_sphereMapPairSlope_le B hx hv hy hw t) (abs_nonneg _) (Real.exp_pos _).le

/-- The actual second derivative is bounded by the two squared speeds.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem abs_sphereMapPairCurvature_le {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hv : inner (𝕜 := ℝ) x v = 0)
    (hy : ‖y‖ = 1) (hw : inner (𝕜 := ℝ) y w = 0) (t : ℝ) :
    |sphereMapPairCurvature B x v y w t| ≤ Real.exp ‖B.toContinuousLinearMap‖ *
      ((‖B.toContinuousLinearMap‖ ^ 2 + ‖B.toContinuousLinearMap‖) * (‖v‖ + ‖w‖) ^ 2) := by
  have hS := abs_sphereMapPairSlope_le B hx hv hy hw t
  have hS₂ : sphereMapPairSlope B x v y w t ^ 2 ≤
      (‖B.toContinuousLinearMap‖ * (‖v‖ + ‖w‖)) ^ 2 := by
    nlinarith [abs_nonneg (sphereMapPairSlope B x v y w t),
      sq_abs (sphereMapPairSlope B x v y w t)]
  have hA := abs_sphereMapPairAcceleration_le B hx hv hy hw t
  have hbound : |sphereMapPairSlope B x v y w t ^ 2 +
      sphereMapPairAcceleration B x v y w t| ≤
      (‖B.toContinuousLinearMap‖ ^ 2 + ‖B.toContinuousLinearMap‖) * (‖v‖ + ‖w‖) ^ 2 := by
    calc
      _ ≤ |sphereMapPairSlope B x v y w t ^ 2| +
          |sphereMapPairAcceleration B x v y w t| := abs_add_le _ _
      _ ≤ (‖B.toContinuousLinearMap‖ * (‖v‖ + ‖w‖)) ^ 2 +
          ‖B.toContinuousLinearMap‖ * (‖v‖ + ‖w‖) ^ 2 := by
        rw [abs_of_nonneg (sq_nonneg _)]
        exact add_le_add hS₂ hA
      _ = _ := by ring
  rw [sphereMapPairCurvature, abs_mul]
  exact mul_le_mul (abs_sphereMapPairKernel_le B hx hv hy hw t) hbound
    (abs_nonneg _) (Real.exp_pos _).le

/-- The mixed derivative retains the product of initial speeds,
needed for localization under the original product probability measure.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem abs_sphereMapPairMixedCurvature_le {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    {x v y w : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    |sphereMapPairMixedCurvature B x v y w| ≤ Real.exp ‖B.toContinuousLinearMap‖ *
      ((‖B.toContinuousLinearMap‖ ^ 2 + ‖B.toContinuousLinearMap‖) * ‖v‖ * ‖w‖) := by
  have h₁ := abs_inner_map_le B v y
  have h₂ := abs_inner_map_le B x w
  have h₃ := abs_inner_map_le B v w
  have h₄ := abs_inner_map_le B x y
  rw [hy, mul_one] at h₁
  rw [hx, mul_one] at h₂
  rw [hx, hy, mul_one, mul_one] at h₄
  have hK : Real.exp (inner (𝕜 := ℝ) x (B y)) ≤ Real.exp ‖B.toContinuousLinearMap‖ :=
    Real.exp_le_exp.mpr ((le_abs_self _).trans h₄)
  have hbound : |inner (𝕜 := ℝ) v (B y) * inner (𝕜 := ℝ) x (B w) +
      inner (𝕜 := ℝ) v (B w)| ≤
      (‖B.toContinuousLinearMap‖ ^ 2 + ‖B.toContinuousLinearMap‖) * ‖v‖ * ‖w‖ := by
    calc
      _ ≤ |inner (𝕜 := ℝ) v (B y) * inner (𝕜 := ℝ) x (B w)| +
          |inner (𝕜 := ℝ) v (B w)| := abs_add_le _ _
      _ ≤ (‖B.toContinuousLinearMap‖ * ‖v‖) * (‖B.toContinuousLinearMap‖ * ‖w‖) +
          ‖B.toContinuousLinearMap‖ * ‖v‖ * ‖w‖ := by
        rw [abs_mul]
        gcongr
      _ = _ := by ring
  rw [sphereMapPairMixedCurvature, abs_mul, abs_of_pos (Real.exp_pos _)]
  exact mul_le_mul hK hbound (abs_nonneg _) (Real.exp_pos _).le

/-- A genuine unit circle point and its nonzero unit tangent jointly
realize every geometric input of these arbitrary-matrix bounds.
Source: arXiv:2601.21366v2, §2.2 and §5.2. -/
example : ‖(circlePoint 0 : EucSpace 2)‖ = 1 ∧
    inner (𝕜 := ℝ) (circlePoint 0 : EucSpace 2)
      (circleTangent (circlePoint 0 : EucSpace 2)) = 0 ∧
    ‖circleTangent (circlePoint 0 : EucSpace 2)‖ = 1 :=
  ⟨norm_eq_of_mem_sphere _, inner_circleTangent _,
    (norm_circleTangent _).trans (norm_eq_of_mem_sphere _)⟩

end Transformer.Perceptron
