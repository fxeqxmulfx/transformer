import Transformer.Perceptron.Section5_MapFirstVariation
import Transformer.Perceptron.Section5_SphericalKernel

/-!
# Actual spherical derivatives for matrix interaction

ArXiv:2601.21366v2, §5.3, `thm: any.d` (i), and §3.1,
`rem: ext` (ii). The interaction is the original exponential
of `x · B y`, evaluated along the original spherical geodesics.
Both integration points move, with their own initial directions.
The derivatives retain the actual matrix and the radial acceleration
of each original geodesic, in every ambient dimension.

The first derivative is the kernel times the bilinear slope.
The second derivative is the kernel times the slope squared plus
its actual acceleration. No Hessian identity is assumed.
The diagonal cancellation removes the two one-point variations
and leaves twice the actual mixed matrix curvature. The factor two
cancels the original energy's factor one half after integration.

Only the swap identity needs symmetry. The differential identities
and their cancellation also hold for a nonsymmetric or singular
matrix. The source assumes symmetry and invertibility for its
stationary-measure conclusions. The example uses a genuine unit
position, a nonzero tangent direction, and identity attention.
-/

open Real
namespace Transformer.Perceptron

/-- The actual time derivative of the matrix bilinear exponent.
Both original integration points move along their own geodesics.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def sphereMapPairSlope {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  inner (𝕜 := ℝ) (sphereVelocity x v t) (B (spherePosition y w t)) +
    inner (𝕜 := ℝ) (spherePosition x v t) (B (sphereVelocity y w t))

/-- The actual exponent acceleration, including both radial terms.
Source: arXiv:2601.21366v2, §5.3, the second interaction variation,
and §3.1, `rem: ext` (ii). -/
noncomputable def sphereMapPairAcceleration {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  2 * inner (𝕜 := ℝ) (sphereVelocity x v t) (B (sphereVelocity y w t)) -
    (‖v‖ ^ 2 + ‖w‖ ^ 2) * inner (𝕜 := ℝ) (spherePosition x v t) (B (spherePosition y w t))

/-- The original matrix exponential kernel on two actual geodesics.
Source: arXiv:2601.21366v2, §2.2 and §3.1, `rem: ext` (ii). -/
noncomputable def sphereMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  Real.exp (inner (𝕜 := ℝ) (spherePosition x v t) (B (spherePosition y w t)))

/-- The actual first derivative of the transported matrix kernel.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def sphereMapPairForce {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  sphereMapPairKernel B x v y w t * sphereMapPairSlope B x v y w t

/-- The actual second derivative with the original matrix exponent.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def sphereMapPairCurvature {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  sphereMapPairKernel B x v y w t *
    (sphereMapPairSlope B x v y w t ^ 2 + sphereMapPairAcceleration B x v y w t)

/-- The genuine mixed derivative remaining after diagonal cancellation.
Source: arXiv:2601.21366v2, §5.2, `eq:Hess-bilinear-analytic`,
extended by §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def sphereMapPairMixedCurvature {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) : ℝ :=
  Real.exp (inner (𝕜 := ℝ) x (B y)) *
    (inner (𝕜 := ℝ) v (B y) * inner (𝕜 := ℝ) x (B w) + inner (𝕜 := ℝ) v (B w))

/-- Differentiate the original bilinear exponent along both curves.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem hasDerivAt_sphereMapPairInner {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (fun r => inner (𝕜 := ℝ) (spherePosition x v r) (B (spherePosition y w r)))
      (sphereMapPairSlope B x v y w t) t := by
  have hy : HasDerivAt (fun r => B (spherePosition y w r)) (B (sphereVelocity y w t)) t :=
    B.toContinuousLinearMap.hasFDerivAt.comp_hasDerivAt t (hasDerivAt_spherePosition y w t)
  exact (HasDerivAt.inner ℝ (hasDerivAt_spherePosition x v t) hy).congr_deriv
    (by unfold sphereMapPairSlope; ring)

/-- Its true second derivative includes both radial accelerations.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem hasDerivAt_sphereMapPairSlope {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (sphereMapPairSlope B x v y w) (sphereMapPairAcceleration B x v y w t) t := by
  have hy : HasDerivAt (fun r => B (spherePosition y w r)) (B (sphereVelocity y w t)) t :=
    B.toContinuousLinearMap.hasFDerivAt.comp_hasDerivAt t (hasDerivAt_spherePosition y w t)
  have hv : HasDerivAt (fun r => B (sphereVelocity y w r))
      (B (-‖w‖ ^ 2 • spherePosition y w t)) t :=
    B.toContinuousLinearMap.hasFDerivAt.comp_hasDerivAt t (hasDerivAt_sphereVelocity y w t)
  have h₁ := HasDerivAt.inner ℝ (hasDerivAt_sphereVelocity x v t) hy
  have h₂ := HasDerivAt.inner ℝ (hasDerivAt_spherePosition x v t) hv
  apply (h₁.fun_add h₂).congr_deriv
  simp only [sphereMapPairAcceleration, map_smul, real_inner_smul_left, real_inner_smul_right]
  ring

/-- The displayed force is the original kernel's actual derivative.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem hasDerivAt_sphereMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (sphereMapPairKernel B x v y w) (sphereMapPairForce B x v y w t) t :=
  (hasDerivAt_sphereMapPairInner B x v y w t).exp

/-- Differentiate the actual force, retaining the full chain rule.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem hasDerivAt_sphereMapPairForce {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (sphereMapPairForce B x v y w) (sphereMapPairCurvature B x v y w t) t := by
  apply ((hasDerivAt_sphereMapPairKernel B x v y w t).mul
    (hasDerivAt_sphereMapPairSlope B x v y w t)).congr_deriv
  unfold sphereMapPairForce sphereMapPairCurvature
  ring

/-- Symmetric attention permits swapping the two original points.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii). -/
theorem sphereMapPairKernel_swap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (hsymm : B.IsSymmetric) (x v y w : EucSpace d) (t : ℝ) :
    sphereMapPairKernel B x v y w t = sphereMapPairKernel B y w x v t := by
  unfold sphereMapPairKernel
  rw [← hsymm, real_inner_comm]

/-- Evaluate the actual second derivative at the original points.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem sphereMapPairCurvature_zero {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) :
    sphereMapPairCurvature B x v y w 0 = Real.exp (inner (𝕜 := ℝ) x (B y)) *
      ((inner (𝕜 := ℝ) v (B y) + inner (𝕜 := ℝ) x (B w)) ^ 2 +
        2 * inner (𝕜 := ℝ) v (B w) - (‖v‖ ^ 2 + ‖w‖ ^ 2) * inner (𝕜 := ℝ) x (B y)) := by
  simp only [sphereMapPairCurvature, sphereMapPairKernel, sphereMapPairSlope,
    sphereMapPairAcceleration, spherePosition_zero, sphereVelocity_zero]
  ring

/-- Subtracting both genuine one-point curvatures leaves twice
exactly the mixed curvature. Source: arXiv:2601.21366v2, §5.2,
`eq:Hess-bilinear-analytic`, §5.3, and §3.1, `rem: ext` (ii). -/
theorem sphereMapPairCurvature_diagonal_cancel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (x v y w : EucSpace d) :
    sphereMapPairCurvature B x v y w 0 - sphereMapPairCurvature B x v y 0 0 -
      sphereMapPairCurvature B x 0 y w 0 = 2 * sphereMapPairMixedCurvature B x v y w := by
  simp only [sphereMapPairCurvature_zero, sphereMapPairMixedCurvature, map_zero,
    inner_zero_left, inner_zero_right, norm_zero, zero_pow (by decide : 2 ≠ 0), zero_add, add_zero]
  ring

/-- Identity attention is symmetric; the two original unit positions
and nonzero unit directions are tangent. The mixed curvature is
strictly positive. Source: arXiv:2601.21366v2, §2.2 and §5.2. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    ‖(basePoint 1 : EucSpace 2)‖ = 1 ∧ ‖secondAxis‖ = 1 ∧
    inner (𝕜 := ℝ) (basePoint 1 : EucSpace 2) secondAxis = 0 ∧
    sphereMapPairMixedCurvature (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (basePoint 1 : EucSpace 2) secondAxis (basePoint 1 : EucSpace 2) secondAxis = Real.exp 1 := by
  refine ⟨fun _ _ => rfl, norm_basePoint_one, norm_secondAxis, inner_basePoint_secondAxis, ?_⟩
  have hz : inner (𝕜 := ℝ) secondAxis (basePoint 1 : EucSpace 2) = 0 := by
    rw [real_inner_comm]
    exact inner_basePoint_secondAxis
  simp only [sphereMapPairMixedCurvature, LinearMap.id_apply, hz, real_inner_self_eq_norm_sq,
    norm_basePoint_one, norm_secondAxis, one_pow, zero_mul, zero_add, mul_one]

end Transformer.Perceptron
