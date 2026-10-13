/-
# Actual matrix pair derivatives along circle transport

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending the
Monge-transport Hessian computation of §5.2 to `E_B` of §6,
`rem:unifiedlog-B`. Both transported points retain their initial
angular speeds. Their actual velocities and radial accelerations
are differentiated before the matrix kernel is differentiated.

For a general symmetric matrix the second derivative has two
diagonal curvature terms and a mixed term. The formula therefore
does not replace the two speeds by their difference. Symmetry
identifies the derivative in the second point with the reversed
first-point derivative and makes the mixed coefficients agree.
All pointwise derivative identities hold at every real transport time.
-/

import Transformer.Perceptron.Section5_MapKernel

open Real

namespace Transformer.Perceptron

/-- The actual transported point has its fixed initial angular speed.
Source: arXiv:2601.21366v2, §2.2, the Monge geodesic, and §5.2. -/
theorem hasDerivAt_circleRotation_coe (s : SSphere 2 → ℝ) (x : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => (circleRotation s r x : EucSpace 2))
      (s x • circleTangent (circleRotation s t x : EucSpace 2)) t := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  have ha : HasDerivAt (fun r : ℝ => θ + r * s (circlePoint θ)) (s (circlePoint θ)) t := by
    simpa only [id_eq, one_mul] using ((hasDerivAt_id t).mul_const (s (circlePoint θ))).const_add θ
  convert (hasDerivAt_circlePoint _).scomp t ha using 1
  · funext r
    rw [circleRotation_circlePoint]
    rfl
  · rw [circleRotation_circlePoint, circleTangent_circlePoint]

/-- The actual transported unit tangent has radial derivative.
Source: arXiv:2601.21366v2, §5.2, the unit circle acceleration. -/
theorem hasDerivAt_tangent_circleRotation (s : SSphere 2 → ℝ) (x : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => circleTangent (circleRotation s r x : EucSpace 2))
      (-s x • (circleRotation s t x : EucSpace 2)) t := by
  obtain ⟨θ, -, rfl⟩ := exists_circlePoint_eq x
  have ha : HasDerivAt (fun r : ℝ => θ + r * s (circlePoint θ)) (s (circlePoint θ)) t := by
    simpa only [id_eq, one_mul] using ((hasDerivAt_id t).mul_const (s (circlePoint θ))).const_add θ
  convert (hasDerivAt_circleVel _).scomp t ha using 1
  · funext r
    rw [circleRotation_circlePoint, circleTangent_circlePoint]
    rfl
  · rw [circleRotation_circlePoint, smul_neg, neg_smul]

/-- The velocity's actual derivative is the radial acceleration with
squared initial speed. Source: arXiv:2601.21366v2, §2.2 and §5.2,
the spherical geodesic used in the actual Hessian calculation. -/
theorem hasDerivAt_circleRotation_velocity (s : SSphere 2 → ℝ) (x : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => s x • circleTangent (circleRotation s r x : EucSpace 2))
      (-(s x) ^ 2 • (circleRotation s t x : EucSpace 2)) t := by
  have h := (hasDerivAt_tangent_circleRotation s x t).const_smul (s x)
  apply h.congr_deriv
  rw [smul_smul]
  congr 1
  ring

/-- The original matrix kernel at the actual transported pair.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §6, `E_B`. -/
noncomputable def rotatedMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (t : ℝ) (p : SSphere 2 × SSphere 2) : ℝ :=
  circleMapPairKernel B (circleRotation s t p.1) (circleRotation s t p.2)

/-- The first transported derivative, with both original speeds.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
noncomputable def rotatedMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (t : ℝ) (p : SSphere 2 × SSphere 2) : ℝ :=
  s p.1 * circleMapPairForce B (circleRotation s t p.1) (circleRotation s t p.2) +
    s p.2 * circleMapPairForce B (circleRotation s t p.2) (circleRotation s t p.1)

/-- The second transported derivative, including the mixed coefficient.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
noncomputable def rotatedMapPairCurvature (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (s : SSphere 2 → ℝ) (t : ℝ) (p : SSphere 2 × SSphere 2) : ℝ :=
  s p.1 ^ 2 * circleMapPairCurvature B (circleRotation s t p.1) (circleRotation s t p.2) +
    s p.2 ^ 2 * circleMapPairCurvature B (circleRotation s t p.2) (circleRotation s t p.1) +
      2 * s p.1 * s p.2 *
        circleMapPairMixedCurvature B (circleRotation s t p.1) (circleRotation s t p.2)

/-- The actual first derivative of the matrix kernel along both points.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
theorem hasDerivAt_rotatedMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (s : SSphere 2 → ℝ) (x y : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => rotatedMapPairKernel B s r (x, y))
      (rotatedMapPairForce B s t (x, y)) t := by
  have hx := hasDerivAt_circleRotation_coe s x t
  have hy := hasDerivAt_circleRotation_coe s y t
  have hBy := B.toContinuousLinearMap.hasFDerivAt.comp_hasDerivAt t hy
  change HasDerivAt (fun r => B (circleRotation s r y : EucSpace 2))
    (B (s y • circleTangent (circleRotation s t y : EucSpace 2))) t at hBy
  have h := (HasDerivAt.inner ℝ hx hBy).exp
  apply h.congr_deriv
  simp only [rotatedMapPairForce, circleMapPairForce, circleMapPairKernel,
    LinearMap.map_smul, real_inner_smul_left, real_inner_smul_right]
  rw [circleMapBilinear_symm B hsymm (circleTangent (circleRotation s t y : EucSpace 2))
    (circleRotation s t x : EucSpace 2),
    circleMapBilinear_symm B hsymm (circleRotation s t y : EucSpace 2)
      (circleRotation s t x : EucSpace 2)]
  ring

/-- The first-point force differentiates along both transported points.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2.
The second point contributes the actual mixed angular derivative. -/
theorem hasDerivAt_rotated_circleMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (s : SSphere 2 → ℝ) (x y : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => circleMapPairForce B (circleRotation s r x) (circleRotation s r y))
      (s x * circleMapPairCurvature B (circleRotation s t x) (circleRotation s t y) +
        s y * circleMapPairMixedCurvature B (circleRotation s t x) (circleRotation s t y)) t := by
  have hK := hasDerivAt_rotatedMapPairKernel B hsymm s x y t
  have hJ := hasDerivAt_tangent_circleRotation s x t
  have hy := hasDerivAt_circleRotation_coe s y t
  have hBy := B.toContinuousLinearMap.hasFDerivAt.comp_hasDerivAt t hy
  change HasDerivAt (fun r => B (circleRotation s r y : EucSpace 2))
    (B (s y • circleTangent (circleRotation s t y : EucSpace 2))) t at hBy
  have h := hK.mul (HasDerivAt.inner ℝ hJ hBy)
  apply h.congr_deriv
  simp only [rotatedMapPairKernel, rotatedMapPairForce, circleMapPairForce,
    circleMapPairCurvature, circleMapPairMixedCurvature, LinearMap.map_smul,
    real_inner_smul_left, real_inner_smul_right]
  rw [circleMapPairKernel_swap B hsymm (circleRotation s t x) (circleRotation s t y),
    circleMapBilinear_symm B hsymm (circleTangent (circleRotation s t y : EucSpace 2))
      (circleRotation s t x : EucSpace 2)]
  ring

/-- The actual derivative of the whole first-derivative integrand.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending the
second differentiation of `eq:Hess-bilinear.std` in §5.2. -/
theorem hasDerivAt_rotatedMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (s : SSphere 2 → ℝ) (x y : SSphere 2) (t : ℝ) :
    HasDerivAt (fun r => rotatedMapPairForce B s r (x, y))
      (rotatedMapPairCurvature B s t (x, y)) t := by
  have hx := (hasDerivAt_rotated_circleMapPairForce B hsymm s x y t).const_mul (s x)
  have hy := (hasDerivAt_rotated_circleMapPairForce B hsymm s y x t).const_mul (s y)
  apply (hx.fun_add hy).congr_deriv
  simp only [rotatedMapPairCurvature,
    circleMapPairMixedCurvature_swap B hsymm (circleRotation s t x) (circleRotation s t y)]
  ring

/-- All matrix-symmetry hypotheses hold simultaneously at `B=id`.
The speeds may be any function on the actual circle; continuity is
only needed later when these derivatives are integrated. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric := fun _ _ => rfl

end Transformer.Perceptron
