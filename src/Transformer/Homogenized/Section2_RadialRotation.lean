module

public import Mathlib.Analysis.Calculus.Deriv.Prod

public import Transformer.Homogenized.RotationHead

/-!
# A radial shift changes the discrete normalization step

Source: arXiv:2604.01978v1, §2.1, `eq:update_tokens`,
§2.2, `eq:SDE_ito_clean`, and §6.1, `lem:stability_generator`.
Take one planar token and the deterministic value matrix
`I+J`, where `J` generates unit-speed rotations. Its query-key
matrix is zero. This is a genuine high-order head law.

The attention field is `x+Jx`. On the sphere its projected
drift is exactly `Jx`, because the radial component vanishes.
The covariant corrector also vanishes along the actual
rotation curve. Thus the actual modified equation in the
source is still unit-speed rotation at every step size.

The radial component nevertheless remains inside the source's
discrete update `N(x+η*(x+Jx))`. That difference will produce
the one-step obstruction. This module verifies the actual
head law, projected field, covariant corrector, and modified
equation before making any approximation-error claim.

The computation retains the ambient drift off the sphere
when differentiating it; it does not replace that drift by
`Jx` away from unit configurations.
-/

@[expose] public section

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- A deterministic value matrix with both the radial identity and the actual planar rotation generator.
Source: arXiv:2604.01978v1, §2.1, the value matrix in `eq:update_tokens`. -/
def radialRotHead : HeadParam 2 := (1 + rotGen, 0)

/-- The actual value map of the shifted head is `x+Jx`.
Source: arXiv:2604.01978v1, §2.1, the value contribution in the attention field. -/
theorem valueMap_radialRotHead (x : EucSpace 2) :
    valueMap radialRotHead x = x + Matrix.toEuclideanLin rotGen x := by
  ext k
  fin_cases k <;> simp [valueMap, radialRotHead, rotGen, Matrix.toEuclideanLin,
    Matrix.mulVec, dotProduct, Fin.sum_univ_two]

/-- For one actual token the exact attention field of the shifted head is `x+Jx`, at every positive or negative temperature.
Source: arXiv:2604.01978v1, §2.1, `EQ:VELOCITY_FIELD_SELF_ATTENTION`. -/
theorem attnField_radialRotHead (β : ℝ) (x : Idx 1 → EucSpace 2) (z : EucSpace 2) :
    attnField β radialRotHead x z = x 0 + Matrix.toEuclideanLin rotGen (x 0) := by
  simp only [attnField, Fin.sum_univ_one, smul_smul]
  rw [inv_mul_cancel₀ (attnWeight_pos _ _ _ _).ne', one_smul]
  exact valueMap_radialRotHead (x 0)

/-- The actual ambient drift retains its radial projection correction away from the sphere.
Source: arXiv:2604.01978v1, §2.2.1, the exact projected mean field `b`. -/
theorem bField_radialRotHead (β : ℝ) (x : Idx 1 → EucSpace 2) (i : Idx 1) :
    bField β (Measure.dirac radialRotHead) x i =
      x 0 + Matrix.toEuclideanLin rotGen (x 0) -
        inner (𝕜 := ℝ) (x 0) (x 0 + Matrix.toEuclideanLin rotGen (x 0)) • x 0 := by
  rw [bField, meanField, integral_dirac, attnField_radialRotHead]
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  rfl

/-- At an actual unit token, projection removes the radial identity and leaves exactly the rotation generator.
Source: arXiv:2604.01978v1, §2.2.1, the spherical drift of the shifted value matrix. -/
theorem bField_radialRotHead_unit (β : ℝ) (x : Idx 1 → EucSpace 2)
    (hx : ‖x 0‖ = 1) (i : Idx 1) :
    bField β (Measure.dirac radialRotHead) x i = Matrix.toEuclideanLin rotGen (x 0) := by
  rw [bField_radialRotHead, inner_add_right, inner_rotGen, add_zero,
    real_inner_self_eq_norm_sq, hx, one_pow, one_smul]
  module

/-- The actual ambient shifted-head drift is smooth; its full ambient formula is used in the corrector calculation.
Source: arXiv:2604.01978v1, §2.2.3, the derivative entering `eq:SDE_ito_clean`. -/
theorem contDiff_bField_radialRotHead (β : ℝ) :
    ContDiff ℝ ⊤ (fun x : Idx 1 → EucSpace 2 => bField β (Measure.dirac radialRotHead) x 0) := by
  simp_rw [bField_radialRotHead]
  let P : (Idx 1 → EucSpace 2) →L[ℝ] EucSpace 2 := ContinuousLinearMap.proj 0
  let J : EucSpace 2 →L[ℝ] EucSpace 2 := LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin rotGen)
  exact (P.contDiff.add (J.contDiff.comp P.contDiff)).sub
    ((ContDiff.inner ℝ P.contDiff (P.contDiff.add (J.contDiff.comp P.contDiff))).smul P.contDiff)

/-- The actual unit-speed rotation solves the shifted head's projected drift equation.
Source: arXiv:2604.01978v1, §2.2, `eq: deterministic`. -/
theorem hasDerivAt_rotFlow_radialRotHead (β : ℝ) (t : ℝ) (i : Idx 1) :
    HasDerivAt (fun s => rotFlow s i) (bField β (Measure.dirac radialRotHead) (rotFlow t) i) t := by
  rw [bField_radialRotHead_unit β _ (norm_rotFlow t 0)]
  simpa only [bField_rotHead] using hasDerivAt_rotFlow β t i

/-- The actual covariant corrector vanishes along the full rotating solution, even though the ambient field contains the radial identity.
Source: arXiv:2604.01978v1, §2.2.3, the exact `∇_b b` in `eq:SDE_ito_clean`. -/
theorem covDerivB_radialRotHead_rotFlow (β t : ℝ) (i : Idx 1) :
    covDerivB β (Measure.dirac radialRotHead) (rotFlow t) i = 0 := by
  have hi : i = 0 := Subsingleton.elim _ _
  subst hi
  have hX := hasDerivAt_pi.mpr (hasDerivAt_rotFlow_radialRotHead β t)
  have hd := (contDiff_bField_radialRotHead β).differentiable (by simp)
  have hc := (hd (rotFlow t)).hasFDerivAt.comp_hasDerivAt t hX
  let J : EucSpace 2 →L[ℝ] EucSpace 2 := LinearMap.toContinuousLinearMap (Matrix.toEuclideanLin rotGen)
  have hj := J.hasFDerivAt.comp_hasDerivAt t (hasDerivAt_rotFlow_radialRotHead β t 0)
  have he : (fun s => bField β (Measure.dirac radialRotHead) (rotFlow s) 0) =
      fun s => Matrix.toEuclideanLin rotGen (rotFlow s 0) :=
    funext (fun s => bField_radialRotHead_unit β _ (norm_rotFlow s 0) 0)
  have hacc : HasDerivAt (fun s => bField β (Measure.dirac radialRotHead) (rotFlow s) 0) (-rotFlow t 0) t := by
    rw [he]
    apply hj.congr_deriv
    change Matrix.toEuclideanLin rotGen (bField β (Measure.dirac radialRotHead) (rotFlow t) 0) = -rotFlow t 0
    rw [bField_radialRotHead_unit β _ (norm_rotFlow t 0), rotGen_rotGen]
  have hder := hc.unique hacc
  rw [covDerivB, covDeriv, hder, proj, inner_neg_right, real_inner_self_eq_norm_sq, norm_rotFlow t 0]
  simp

/-- The actual shifted head satisfies every high-order law assumption with zero centered fluctuations and zero variance proxy.
Source: arXiv:2604.01978v1, §2.3, `ass:high_order_short`, and `eq: defining.alpha`. -/
theorem radialRotHead_law (β : ℝ) :
    HasHighOrderLaw 2 0 0 (Measure.dirac radialRotHead) ∧
      IsVarianceProxy 2 1 β (Measure.dirac radialRotHead) 0 := by
  refine ⟨hasHighOrderLaw_dirac _, le_rfl, ⟨?_, ?_⟩⟩
  · refine ⟨fun _ => EuclideanSpace.single 0 1, fun _ => by simp, 0, ?_⟩
    simp [Gfield, fluct, meanField, proj]
  · rintro v ⟨x, hx, i, rfl⟩
    simp [Gfield, fluct, meanField, proj]

/-- The actual rotation solves the exact modified SDE of the shifted head on an actual probability space, for every step size.
Source: arXiv:2604.01978v1, §2.2.3, `eq:SDE_ito_clean`, with its original covariant corrector. -/
theorem isModifiedSde_radialRotHead (η β : ℝ) :
    IsModifiedSde η β 0 0 (Measure.dirac radialRotHead) (Measure.dirac ())
      (fun t (_ : Unit) => rotFlow t) := by
  refine ⟨fun t _ => norm_rotFlow t, fun φ hφ t ht => ?_⟩
  have hG : noiseField (n := 1) β 0 0 (Measure.dirac radialRotHead) = fun _ _ => 0 := by
    funext θ x i
    simp [noiseField]
  simp only [integral_dirac, sphGenerator, hG, sphHess_zero, integral_zero, mul_zero, add_zero]
  have hB : (fun i => bField β (Measure.dirac radialRotHead) (rotFlow t) i -
      (η / 2) • covDerivB β (Measure.dirac radialRotHead) (rotFlow t) i) =
      bField β (Measure.dirac radialRotHead) (rotFlow t) := by
    funext i
    rw [covDerivB_radialRotHead_rotFlow, smul_zero, sub_zero]
  rw [hB]
  have hd : Differentiable ℝ φ := hφ.differentiable (by simp)
  exact ((hd _).hasFDerivAt.comp_hasDerivAt t
    (hasDerivAt_pi.mpr (hasDerivAt_rotFlow_radialRotHead β t))).hasDerivWithinAt

/-- A genuine unit initial configuration witnesses the unit-drift hypothesis and a nonzero actual rotational velocity.
Source: arXiv:2604.01978v1, §2.2, the nonstationary shifted-head construction. -/
example : ‖rotFlow 0 0‖ = 1 ∧ bField 1 (Measure.dirac radialRotHead) (rotFlow 0) 0 ≠ 0 := by
  refine ⟨norm_rotFlow 0 0, ?_⟩
  rw [bField_radialRotHead_unit 1 _ (norm_rotFlow 0 0), rotGen_apply, rotFlow_apply]
  intro h
  have he := congrArg (fun x : EucSpace 2 => x 1) h
  norm_num at he

end Transformer.Homogenized
