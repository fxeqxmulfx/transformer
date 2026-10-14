/-
# Actual pair-kernel calculus on spherical geodesics

ArXiv:2601.21366v2, §2.3, the interaction kernel `exp(β x·y)`,
and §5.3, the localization argument for `thm: any.d` (i). Both
arguments follow their original spherical exponential curves.
Their actual velocities and radial accelerations give two genuine
time derivatives of the original kernel in every dimension.

The second derivative has two diagonal terms and a mixed term.
Subtracting the two one-point second derivatives cancels precisely
the diagonal terms. This is the arbitrary-dimensional version of
the cancellation in §5.2, `eq:Hess-bilinear-analytic`, without
assuming that a Hessian exists or assigning a total derivative.
-/

import Transformer.Perceptron.Section5_SphereTransport

open Real

namespace Transformer.Perceptron

/-- The actual inner-product velocity of two transported points.
Source: arXiv:2601.21366v2, §5.2–§5.3, differentiating the interaction. -/
noncomputable def spherePairSlope {d : ℕ} (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  inner (𝕜 := ℝ) (sphereVelocity x v t) (spherePosition y w t) +
    inner (𝕜 := ℝ) (spherePosition x v t) (sphereVelocity y w t)

/-- The actual second derivative of the transported inner product.
Source: arXiv:2601.21366v2, §5.2–§5.3, the geodesic accelerations. -/
noncomputable def spherePairAcceleration {d : ℕ} (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  2 * inner (𝕜 := ℝ) (sphereVelocity x v t) (sphereVelocity y w t) -
    (‖v‖ ^ 2 + ‖w‖ ^ 2) * inner (𝕜 := ℝ) (spherePosition x v t) (spherePosition y w t)

/-- The original kernel evaluated at the actual exponential curves.
Source: arXiv:2601.21366v2, §2.3 and §5.3, `exp(β x·y)`. -/
noncomputable def spherePairKernel {d : ℕ} (β : ℝ) (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) (spherePosition x v t) (spherePosition y w t))

/-- The actual first-derivative integrand with both original velocities.
Source: arXiv:2601.21366v2, §5.2–§5.3, the first energy differentiation. -/
noncomputable def spherePairForce {d : ℕ} (β : ℝ) (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  spherePairKernel β x v y w t * (β * spherePairSlope x v y w t)

/-- The actual second-derivative integrand, including radial acceleration.
Source: arXiv:2601.21366v2, §5.2–§5.3, the interaction second variation. -/
noncomputable def spherePairCurvature {d : ℕ} (β : ℝ) (x v y w : EucSpace d) (t : ℝ) : ℝ :=
  spherePairKernel β x v y w t *
    (β ^ 2 * spherePairSlope x v y w t ^ 2 + β * spherePairAcceleration x v y w t)

/-- The original mixed second derivative in the two initial directions.
Source: arXiv:2601.21366v2, §5.3, the mixed Hessian after global cancellation. -/
noncomputable def spherePairMixedCurvature {d : ℕ} (β : ℝ) (x v y w : EucSpace d) : ℝ :=
  Real.exp (β * inner (𝕜 := ℝ) x y) *
    (β ^ 2 * inner (𝕜 := ℝ) v y * inner (𝕜 := ℝ) x w + β * inner (𝕜 := ℝ) v w)

/-- The actual velocity at zero is the original direction.
Source: arXiv:2601.21366v2, §2.2, the initial geodesic direction. -/
@[simp] theorem sphereVelocity_zero {d : ℕ} (x v : EucSpace d) : sphereVelocity x v 0 = v := by
  simp [sphereVelocity]

/-- A zero initial direction keeps its original point fixed.
Source: arXiv:2601.21366v2, §2.2, the exponential-map geodesic. -/
@[simp] theorem spherePosition_zero_direction {d : ℕ} (x : EucSpace d) (t : ℝ) :
    spherePosition x 0 t = x := by
  simp [spherePosition]

/-- The velocity of a fixed original point vanishes at every time.
Source: arXiv:2601.21366v2, §2.2, the zero geodesic direction. -/
@[simp] theorem sphereVelocity_zero_direction {d : ℕ} (x : EucSpace d) (t : ℝ) :
    sphereVelocity x 0 t = 0 := by
  simp [sphereVelocity]

/-- The actual first time derivative of the transported inner product.
Source: arXiv:2601.21366v2, §5.2–§5.3, the kernel differentiation. -/
theorem hasDerivAt_spherePairInner {d : ℕ} (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (fun r => inner (𝕜 := ℝ) (spherePosition x v r) (spherePosition y w r))
      (spherePairSlope x v y w t) t := by
  have h := HasDerivAt.inner ℝ (hasDerivAt_spherePosition x v t)
    (hasDerivAt_spherePosition y w t)
  exact h.congr_deriv (by unfold spherePairSlope; ring)

/-- Differentiating both actual velocities gives the original radial
acceleration formula. Source: arXiv:2601.21366v2, §5.2–§5.3. -/
theorem hasDerivAt_spherePairSlope {d : ℕ} (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (spherePairSlope x v y w) (spherePairAcceleration x v y w t) t := by
  have h₁ := HasDerivAt.inner ℝ (hasDerivAt_sphereVelocity x v t)
    (hasDerivAt_spherePosition y w t)
  have h₂ := HasDerivAt.inner ℝ (hasDerivAt_spherePosition x v t)
    (hasDerivAt_sphereVelocity y w t)
  apply (h₁.fun_add h₂).congr_deriv
  simp only [spherePairAcceleration, real_inner_smul_left, real_inner_smul_right]
  ring

/-- The actual kernel derivative along the original two-point geodesic.
Source: arXiv:2601.21366v2, §5.2–§5.3, the first interaction variation. -/
theorem hasDerivAt_spherePairKernel {d : ℕ} (β : ℝ) (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (spherePairKernel β x v y w) (spherePairForce β x v y w t) t :=
  ((hasDerivAt_spherePairInner x v y w t).const_mul β).exp

/-- A second genuine differentiation gives the stated curvature.
Source: arXiv:2601.21366v2, §5.2–§5.3, the actual interaction Hessian. -/
theorem hasDerivAt_spherePairForce {d : ℕ} (β : ℝ) (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (spherePairForce β x v y w) (spherePairCurvature β x v y w t) t := by
  have h := (hasDerivAt_spherePairKernel β x v y w t).mul
    ((hasDerivAt_spherePairSlope x v y w t).const_mul β)
  apply h.congr_deriv
  simp only [spherePairForce, spherePairCurvature]
  ring

/-- The actual derivative function itself is differentiable, with the
computed second derivative. Source: arXiv:2601.21366v2, §5.2–§5.3,
the existence input required by `eq:strict2order`. -/
theorem hasDerivAt_deriv_spherePairKernel {d : ℕ} (β : ℝ)
    (x v y w : EucSpace d) (t : ℝ) :
    HasDerivAt (deriv (spherePairKernel β x v y w)) (spherePairCurvature β x v y w t) t := by
  have heq : deriv (spherePairKernel β x v y w) = spherePairForce β x v y w :=
    funext fun r => (hasDerivAt_spherePairKernel β x v y w r).deriv
  rw [heq]
  exact hasDerivAt_spherePairForce β x v y w t

/-- The proved curvature is the actual iterated derivative's value.
Source: arXiv:2601.21366v2, §5.2–§5.3, the interaction Hessian. -/
theorem secondDeriv_spherePairKernel {d : ℕ} (β : ℝ) (x v y w : EucSpace d) (t : ℝ) :
    deriv^[2] (spherePairKernel β x v y w) t = spherePairCurvature β x v y w t :=
  (hasDerivAt_deriv_spherePairKernel β x v y w t).deriv

/-- The original pair kernel is symmetric along the actual transports.
Source: arXiv:2601.21366v2, §2.3, the symmetric interaction energy. -/
theorem spherePairKernel_swap {d : ℕ} (β : ℝ) (x v y w : EucSpace d) (t : ℝ) :
    spherePairKernel β x v y w t = spherePairKernel β y w x v t := by
  unfold spherePairKernel
  rw [real_inner_comm]

/-- The initial second derivative retains both diagonal and mixed terms.
Source: arXiv:2601.21366v2, §5.3, the full Hessian before cancellation. -/
theorem spherePairCurvature_zero {d : ℕ} (β : ℝ) (x v y w : EucSpace d) :
    spherePairCurvature β x v y w 0 = Real.exp (β * inner (𝕜 := ℝ) x y) *
      (β ^ 2 * (inner (𝕜 := ℝ) v y + inner (𝕜 := ℝ) x w) ^ 2 +
        β * (2 * inner (𝕜 := ℝ) v w - (‖v‖ ^ 2 + ‖w‖ ^ 2) * inner (𝕜 := ℝ) x y)) := by
  simp only [spherePairCurvature, spherePairKernel, spherePairSlope, spherePairAcceleration,
    spherePosition_zero, sphereVelocity_zero]

/-- Removing the original one-point second derivatives leaves twice the
actual mixed curvature. Source: arXiv:2601.21366v2, §5.3, the arbitrary-
dimensional counterpart of §5.2, `eq:Hess-bilinear-analytic`. -/
theorem spherePairCurvature_diagonal_cancel {d : ℕ} (β : ℝ) (x v y w : EucSpace d) :
    spherePairCurvature β x v y w 0 - spherePairCurvature β x v y 0 0 -
      spherePairCurvature β x 0 y w 0 = 2 * spherePairMixedCurvature β x v y w := by
  simp only [spherePairCurvature_zero, spherePairMixedCurvature, inner_zero_left,
    inner_zero_right, norm_zero, zero_pow (by decide : 2 ≠ 0), zero_add, add_zero]
  ring

end Transformer.Perceptron
