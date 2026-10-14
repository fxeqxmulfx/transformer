/-
# The original blended gradient and its actual Gaussian Hessian identity

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. With the scalar blending coefficients
fixed, the gradient of the original test at the actual blended vector
is a bounded `C¹` test on the original pair of vector arguments.
Its actual derivative in either original vector gives that vector's
blending coefficient times the actual original coordinate Hessian.

Vector integration by parts with the actual independent parameter
therefore turns the original coordinate-weighted gradient expectation
into the original covariance contraction with that Hessian. The scalar
coefficient is retained exactly, for its later cancellation against
the actual path velocity in the interpolation lemma.
Every derivative and bound is proved from the original bounded `C²`
test, rather than supplied as an integration-by-parts hypothesis.
-/

import Transformer.Homogenized.Section7_GaussianTestBounds

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual original linear blend of the two original vector
arguments. Source: arXiv:2604.01978v1, §7,
`eq:interpolation_gaussian`, with its scalar coefficients fixed. -/
noncomputable def gaussianBlendLinearMap {m : ℕ} (a b : ℝ) :
    EucSpace m × EucSpace m →L[ℝ] EucSpace m :=
  a • ContinuousLinearMap.fst ℝ (EucSpace m) (EucSpace m) +
    b • ContinuousLinearMap.snd ℝ (EucSpace m) (EucSpace m)

/-- The actual linear map is exactly the original blend.
Source: arXiv:2604.01978v1, §7, `eq:interpolation_gaussian`. -/
theorem gaussianBlendLinearMap_apply {m : ℕ} (a b : ℝ) (p : EucSpace m × EucSpace m) :
    gaussianBlendLinearMap a b p = a • p.1 + b • p.2 := rfl

/-- The actual original coordinate gradient at the blended vector.
Source: arXiv:2604.01978v1, §7, the original gradient in
the integration-by-parts proof of `lem:gaussian_interp`. -/
noncomputable def gaussianBlendGradient {m : ℕ} (φ : EucSpace m → ℝ)
    (a b : ℝ) (i : Fin m) : EucSpace m × EucSpace m → ℝ :=
  gaussianCoordinateGradient φ i ∘ gaussianBlendLinearMap a b

/-- The actual original blended gradient is `C¹`.
Source: arXiv:2604.01978v1, §7, the original Gaussian Stein
test in the proof of `lem:gaussian_interp`. -/
theorem contDiff_gaussianBlendGradient {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (a b : ℝ) (i : Fin m) : ContDiff ℝ 1 (gaussianBlendGradient φ a b i) :=
  (contDiff_gaussianCoordinateGradient φ hφ i).comp (gaussianBlendLinearMap a b).contDiff

/-- The original first derivative bound controls the actual blended
gradient. Source: arXiv:2604.01978v1, §7, the bounded Stein
test in the original Gaussian interpolation computation. -/
theorem norm_gaussianBlendGradient_le {m : ℕ} (φ : EucSpace m → ℝ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (a b : ℝ) (i : Fin m) (p : EucSpace m × EucSpace m) :
    ‖gaussianBlendGradient φ a b i p‖ ≤ C :=
  norm_gaussianCoordinateGradient_le φ C hbdd i (gaussianBlendLinearMap a b p)

/-- The actual derivative is the original gradient derivative
composed with the same original blending map.
Source: arXiv:2604.01978v1, §7, the actual chain rule
producing the Hessian in `lem:gaussian_interp`. -/
theorem fderiv_gaussianBlendGradient {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (a b : ℝ) (i : Fin m) (p : EucSpace m × EucSpace m) :
    fderiv ℝ (gaussianBlendGradient φ a b i) p =
      (fderiv ℝ (gaussianCoordinateGradient φ i) (gaussianBlendLinearMap a b p)).comp
        (gaussianBlendLinearMap a b) := by
  exact (((contDiff_gaussianCoordinateGradient φ hφ i).differentiable (by decide))
    (gaussianBlendLinearMap a b p)).hasFDerivAt.comp p
      (gaussianBlendLinearMap a b).hasFDerivAt |>.fderiv

/-- The original second derivative bound controls the actual full
derivative of the blended gradient. Source: arXiv:2604.01978v1,
§7, the derivative domination in the original vector Stein step. -/
theorem norm_fderiv_gaussianBlendGradient_le {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (a b : ℝ) (i : Fin m) (p : EucSpace m × EucSpace m) :
    ‖fderiv ℝ (gaussianBlendGradient φ a b i) p‖ ≤ C * ‖gaussianBlendLinearMap (m := m) a b‖ := by
  rw [fderiv_gaussianBlendGradient φ hφ a b i p]
  exact ((fderiv ℝ (gaussianCoordinateGradient φ i) (gaussianBlendLinearMap a b p)).opNorm_comp_le
    (gaussianBlendLinearMap a b)).trans
    (mul_le_mul_of_nonneg_right (norm_fderiv_gaussianCoordinateGradient_le φ hφ C hbdd i _)
      (norm_nonneg _))

/-- The derivative on the actual first vector axis gives precisely
its blending coefficient times the original Hessian entry.
Source: arXiv:2604.01978v1, §7, the original covariance
contraction in `lem:gaussian_interp`. -/
theorem fderiv_gaussianBlendGradient_first_axis {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (a b : ℝ) (i j : Fin m) (p : EucSpace m × EucSpace m) :
    fderiv ℝ (gaussianBlendGradient φ a b i) p (EuclideanSpace.single j 1, 0) =
      a * iteratedFDeriv ℝ 2 φ (a • p.1 + b • p.2)
        ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] := by
  rw [fderiv_gaussianBlendGradient φ hφ a b i p, ContinuousLinearMap.comp_apply,
    fderiv_gaussianCoordinateGradient φ hφ, ContinuousLinearMap.flip_apply]
  simp [gaussianBlendLinearMap_apply, iteratedFDeriv_two_apply, map_smul, smul_eq_mul]

/-- The actual second vector axis similarly retains its own blending
coefficient in the original Hessian. Source: arXiv:2604.01978v1,
§7, the original second Gaussian vector in `lem:gaussian_interp`. -/
theorem fderiv_gaussianBlendGradient_second_axis {m : ℕ} (φ : EucSpace m → ℝ)
    (hφ : ContDiff ℝ 2 φ) (a b : ℝ) (i j : Fin m) (p : EucSpace m × EucSpace m) :
    fderiv ℝ (gaussianBlendGradient φ a b i) p (0, EuclideanSpace.single j 1) =
      b * iteratedFDeriv ℝ 2 φ (a • p.1 + b • p.2)
        ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] := by
  rw [fderiv_gaussianBlendGradient φ hφ a b i p, ContinuousLinearMap.comp_apply,
    fderiv_gaussianCoordinateGradient φ hφ, ContinuousLinearMap.flip_apply]
  simp [gaussianBlendLinearMap_apply, iteratedFDeriv_two_apply, map_smul, smul_eq_mul]

/-- The original coordinate-weighted blended gradient expectation
equals its genuine original covariance–Hessian contraction.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
The actual vector Stein identity supplies this formula with its exact
scalar coefficient and original second-moment covariance entries. -/
theorem gaussian_stein_blend_gradient {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0)
    (hcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = S i j)
    (Y : Ω → EucSpace m) (hY : AEMeasurable Y P) (hindep : IndepFun X Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (a b : ℝ) (i : Fin m) :
    (∫ ω, X ω i * fderiv ℝ φ (a • X ω + b • Y ω) (EuclideanSpace.single i 1) ∂P) =
      a * ∫ ω, ∑ j : Fin m, S i j * iteratedFDeriv ℝ 2 φ (a • X ω + b • Y ω)
        ![EuclideanSpace.single j 1, EuclideanSpace.single i 1] ∂P := by
  have h := gaussian_stein_vector_independent_parameter P X hX S hcenter hcov Y hY hindep
    (gaussianBlendGradient φ a b i) (contDiff_gaussianBlendGradient φ hφ a b i) i C
    (C * ‖gaussianBlendLinearMap (m := m) a b‖) (norm_gaussianBlendGradient_le φ C hbdd a b i)
    (norm_fderiv_gaussianBlendGradient_le φ hφ C hbdd a b i)
  simp only [fderiv_gaussianBlendGradient_first_axis φ hφ] at h
  rw [← integral_const_mul]
  refine h.trans ?_
  apply integral_congr_ae
  apply ae_of_all
  intro ω
  simp only [Finset.mul_sum]
  exact Finset.sum_congr rfl fun j hj => by ring

/-- An actual nondegenerate Gaussian vector, a genuine independent
constant vector and a nonconstant bounded `C²` sine jointly witness
the original covariance, independence and test hypotheses. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    AEMeasurable (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧ ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  obtain ⟨φ, C, hφ, hbdd, hn⟩ := gaussian_interp_test_witness
  refine ⟨φ, C, IsGaussian.hasGaussianLaw_id, integral_stdGaussian_coordinate, ?_,
    measurable_const.aemeasurable, indepFun_const_right _ _, hφ, hbdd, hn⟩
  intro i j
  simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j

end Transformer.Homogenized
