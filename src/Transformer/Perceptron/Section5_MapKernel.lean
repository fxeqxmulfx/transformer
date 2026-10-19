module

public import Transformer.Perceptron.GeneralAttention
public import Transformer.Perceptron.Section5_RotatedKernel

/-!
# Actual circle kernel derivatives for a general attention matrix

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), §5.1,
`rem: general-attention`, and §6, `rem:unifiedlog-B`.
The exponential kernel is evaluated at the original matrix `B`.
Its diagonal angular curvature and its mixed angular curvature
are distinct coefficients: a general symmetric matrix need not
be invariant under simultaneous rotations of both sphere points.

These formulas are actual derivatives of the kernel, proved in
the circle chart. Continuity and the mixed coefficient's symmetry
will allow their use under an arbitrary probability measure.
Invertibility is not needed for these calculus identities.
-/

@[expose] public section

open Real

namespace Transformer.Perceptron

/-- The original exponential pair kernel for `E_B`.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`. -/
noncomputable def circleMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (x y : SSphere 2) : ℝ :=
  Real.exp (inner (𝕜 := ℝ) (x : EucSpace 2) (B (y : EucSpace 2)))

/-- Its angular derivative in the first actual point.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §6, `E_B`. -/
noncomputable def circleMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (x y : SSphere 2) : ℝ :=
  circleMapPairKernel B x y *
    inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2)) (B (y : EucSpace 2))

/-- Its angular second derivative in the first point, including the
radial acceleration. Source: arXiv:2601.21366v2, §3.1,
`rem: ext` (ii), extending the Hessian computation of §5.2. -/
noncomputable def circleMapPairCurvature (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (x y : SSphere 2) : ℝ :=
  circleMapPairKernel B x y *
    (inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2)) (B (y : EucSpace 2)) ^ 2 -
      inner (𝕜 := ℝ) (x : EucSpace 2) (B (y : EucSpace 2)))

/-- The mixed angular coefficient in the actual transported kernel.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2.
The product term comes from differentiating the exponential twice. -/
noncomputable def circleMapPairMixedCurvature (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (x y : SSphere 2) : ℝ :=
  circleMapPairKernel B x y *
    (inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2))
        (B (circleTangent (y : EucSpace 2))) +
      inner (𝕜 := ℝ) (circleTangent (x : EucSpace 2)) (B (y : EucSpace 2)) *
        inner (𝕜 := ℝ) (x : EucSpace 2) (B (circleTangent (y : EucSpace 2))))

/-- A symmetric matrix gives a symmetric real bilinear form.
Source: arXiv:2601.21366v2, §5.1, `rem: general-attention`. -/
theorem circleMapBilinear_symm (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (x y : EucSpace 2) :
    inner (𝕜 := ℝ) x (B y) = inner (𝕜 := ℝ) y (B x) := by
  rw [← hsymm x y, real_inner_comm y (B x)]

/-- The actual pair kernel is symmetric for symmetric `B`.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`. -/
theorem circleMapPairKernel_swap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (x y : SSphere 2) :
    circleMapPairKernel B y x = circleMapPairKernel B x y := by
  rw [circleMapPairKernel, circleMapPairKernel, circleMapBilinear_symm B hsymm]

/-- The mixed coefficient inherits the same symmetry. Source:
arXiv:2601.21366v2, §3.1, `rem: ext` (ii), the symmetry used
when expanding the actual Hessian in §5.2. -/
theorem circleMapPairMixedCurvature_swap (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (hsymm : B.IsSymmetric) (x y : SSphere 2) :
    circleMapPairMixedCurvature B y x = circleMapPairMixedCurvature B x y := by
  unfold circleMapPairMixedCurvature
  rw [circleMapPairKernel_swap B hsymm,
    circleMapBilinear_symm B hsymm (circleTangent (y : EucSpace 2)) (circleTangent (x : EucSpace 2)),
    circleMapBilinear_symm B hsymm (circleTangent (y : EucSpace 2)) (x : EucSpace 2),
    circleMapBilinear_symm B hsymm (y : EucSpace 2) (circleTangent (x : EucSpace 2))]
  ring

/-- Joint continuity of the original matrix pair kernel. Source:
arXiv:2601.21366v2, §6, `rem:unifiedlog-B`. -/
theorem continuous_circleMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) :
    Continuous (fun p : SSphere 2 × SSphere 2 => circleMapPairKernel B p.1 p.2) := by
  have hB := B.continuous_of_finiteDimensional
  unfold circleMapPairKernel
  fun_prop

/-- Joint continuity of the actual first angular coefficient.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.2. -/
theorem continuous_circleMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) :
    Continuous (fun p : SSphere 2 × SSphere 2 => circleMapPairForce B p.1 p.2) := by
  have hB := B.continuous_of_finiteDimensional
  have hJ := continuous_circleTangent
  unfold circleMapPairForce
  exact (continuous_circleMapPairKernel B).mul (by fun_prop)

/-- Joint continuity of the diagonal angular curvature. Source:
arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
theorem continuous_circleMapPairCurvature (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) :
    Continuous (fun p : SSphere 2 × SSphere 2 => circleMapPairCurvature B p.1 p.2) := by
  have hB := B.continuous_of_finiteDimensional
  have hJ := continuous_circleTangent
  unfold circleMapPairCurvature
  exact (continuous_circleMapPairKernel B).mul (by fun_prop)

/-- Joint continuity of the mixed angular curvature. Source:
arXiv:2601.21366v2, §3.1, `rem: ext` (ii), extending §5.2. -/
theorem continuous_circleMapPairMixedCurvature (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) :
    Continuous (fun p : SSphere 2 × SSphere 2 => circleMapPairMixedCurvature B p.1 p.2) := by
  have hB := B.continuous_of_finiteDimensional
  have hJ := continuous_circleTangent
  unfold circleMapPairMixedCurvature
  exact (continuous_circleMapPairKernel B).mul (by fun_prop)

/-- The force formula is the actual chart derivative of the kernel.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §6, `E_B`. -/
theorem hasDerivAt_circleMapPairKernel (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (y : SSphere 2) (θ : ℝ) :
    HasDerivAt (fun r => circleMapPairKernel B (circlePoint r) y)
      (circleMapPairForce B (circlePoint θ) y) θ := by
  have hu := hasDerivAt_inner_circlePoint (B (y : EucSpace 2)) θ
  simpa only [circleMapPairKernel, circleMapPairForce, circleTangent_circlePoint,
    real_inner_comm] using hu.exp

/-- The curvature formula is the actual chart derivative of the force.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.2. -/
theorem hasDerivAt_circleMapPairForce (B : EucSpace 2 →ₗ[ℝ] EucSpace 2)
    (y : SSphere 2) (θ : ℝ) :
    HasDerivAt (fun r => circleMapPairForce B (circlePoint r) y)
      (circleMapPairCurvature B (circlePoint θ) y) θ := by
  have hu := hasDerivAt_inner_circlePoint (B (y : EucSpace 2)) θ
  have hv := hasDerivAt_inner_circleVel (B (y : EucSpace 2)) θ
  have h := hu.exp.mul hv
  have heq : Real.exp (inner (𝕜 := ℝ) (B (y : EucSpace 2)) (circlePoint θ : EucSpace 2)) *
      inner (𝕜 := ℝ) (B (y : EucSpace 2)) (circleVel θ) ^ 2 -
      Real.exp (inner (𝕜 := ℝ) (B (y : EucSpace 2)) (circlePoint θ : EucSpace 2)) *
        inner (𝕜 := ℝ) (B (y : EucSpace 2)) (circlePoint θ : EucSpace 2) =
      circleMapPairCurvature B (circlePoint θ) y := by
    simp only [circleMapPairCurvature, circleMapPairKernel, circleTangent_circlePoint,
      real_inner_comm]
    ring
  convert h using 1
  · funext r
    simp only [circleMapPairForce, circleMapPairKernel, circleTangent_circlePoint,
      real_inner_comm, Pi.mul_apply]
  · rw [← heq]
    ring

/-- Every symmetry hypothesis is realized by the actual identity map,
which also supplies a nondegenerate matrix example for these coefficients. -/
example : (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric := fun _ _ => rfl

end Transformer.Perceptron
