module

public import Transformer.Homogenized.Section7_GaussianPolynomialGrowth

/-!
# Polynomial integration by parts on the original standard Gaussian

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and the polynomial
growth justification following `eq:interpolation_interm`. The original
coordinate is independent of its actual orthogonal residual. Fixing that
residual gives an affine scalar slice with polynomially growing test and
derivative. The density proof of scalar integration by parts applies to
every such slice. Actual Gaussian polynomial moments justify Fubini.

The reconstruction of the original vector then gives integration by parts
for its genuine coordinate and genuine Fréchet derivative. The bounds are
on the actual test and derivative; no integration identity is an input.
The residual law may be degenerate and is never assigned a density.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The actual coordinate-weighted polynomial test is integrable.
Source: arXiv:2604.01978v1, §7, domination of the original coordinate
factor in the integration-by-parts proof of `lem:gaussian_interp`. -/
theorem integrable_stdGaussian_coordinate_test_polynomial {m : ℕ}
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p) :
    Integrable (fun z : EucSpace m => z i * g z) (stdGaussian (EucSpace m)) := by
  have hC : 0 ≤ C := (norm_nonneg (g 0)).trans (by simpa using hbound 0)
  apply integrable_of_gaussian_polynomial_bound _ id IsGaussian.hasGaussianLaw_id
    _ ((EuclideanSpace.proj (𝕜 := ℝ) i).continuous.mul hg.continuous).aestronglyMeasurable C (p + 1)
  exact ae_of_all _ (norm_coordinate_mul_polynomial_le g C hC p hbound i)

/-- The actual derivative on the original unit axis obeys the same
polynomial bound as the actual Fréchet derivative.
Source: arXiv:2604.01978v1, §7, the polynomial derivative bound
used after `eq:interpolation_interm`. -/
theorem norm_stdGaussian_coordinate_derivative_polynomial_le {m : ℕ}
    (g : EucSpace m → ℝ) (i : Fin m) (C' : ℝ) (p : ℕ)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) (z : EucSpace m) :
    ‖fderiv ℝ g z (EuclideanSpace.single i 1)‖ ≤ C' * (1 + ‖z‖) ^ p := by
  have h := (fderiv ℝ g z).le_opNorm (EuclideanSpace.single i 1)
  simpa only [PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one] using h.trans
    (by simpa only [PiLp.norm_single, Real.norm_eq_abs, abs_one, mul_one] using hbound' z)

/-- The original coordinate derivative has a finite expectation.
Source: arXiv:2604.01978v1, §7, the actual derivative expectation
in `lem:gaussian_interp`, for polynomially growing derivatives. -/
theorem integrable_stdGaussian_coordinate_derivative_polynomial {m : ℕ}
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C' : ℝ) (p : ℕ)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    Integrable (fun z : EucSpace m => fderiv ℝ g z (EuclideanSpace.single i 1))
      (stdGaussian (EucSpace m)) := by
  apply integrable_of_gaussian_polynomial_bound _ id IsGaussian.hasGaussianLaw_id _
    ((hg.continuous_fderiv (by decide)).clm_apply continuous_const).aestronglyMeasurable C' p
  exact ae_of_all _ (norm_stdGaussian_coordinate_derivative_polynomial_le g i C' p hbound')

/-- Every actual scalar slice and its genuine derivative have
polynomial growth, with constants depending on the fixed residual.
Source: arXiv:2604.01978v1, §7, coordinate integration by parts
with the polynomial growth used after `eq:interpolation_interm`. -/
theorem gaussianCoordinateSlice_polynomial_bounds {m : ℕ}
    (g : EucSpace m → ℝ) (i : Fin m) (C C' : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) (y : EucSpace m) :
    ∃ A B : ℝ,
      (∀ r : ℝ, ‖gaussianCoordinateSlice g i (r, y)‖ ≤ A * (1 + |r|) ^ p) ∧
      (∀ r : ℝ, ‖gaussianCoordinateSliceDerivative g i (r, y)‖ ≤ B * (1 + |r|) ^ p) := by
  have hC : 0 ≤ C := (norm_nonneg (g 0)).trans (by simpa using hbound 0)
  have hC' : 0 ≤ C' := (norm_nonneg (fderiv ℝ g 0)).trans (by simpa using hbound' 0)
  let L : ℝ →L[ℝ] EucSpace m := (ContinuousLinearMap.id ℝ ℝ).smulRight (EuclideanSpace.single i 1)
  refine ⟨C * (1 + ‖L‖ + ‖y‖) ^ p, C' * (1 + ‖L‖ + ‖y‖) ^ p, ?_, ?_⟩
  · intro r
    simpa only [L, ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.id_apply,
      Real.norm_eq_abs, gaussianCoordinateSlice] using
      norm_polynomial_gaussian_affine_le L y g C hC p hbound r
  · intro r
    simpa only [L, ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.id_apply,
      Real.norm_eq_abs, gaussianCoordinateSliceDerivative] using
      norm_polynomial_gaussian_affine_le L y (fun z => fderiv ℝ g z (EuclideanSpace.single i 1))
        C' hC' p (norm_stdGaussian_coordinate_derivative_polynomial_le g i C' p hbound') r

/-- Original standard Gaussian integration by parts for polynomially
growing tests and actual derivatives. Source: arXiv:2604.01978v1,
§7, `lem:gaussian_interp`, with the growth condition used in its
bilinear softmax application. Actual scalar density integration by
parts, coordinate independence and finite original moments prove the identity. -/
theorem stdGaussian_stein_coordinate_polynomial {m : ℕ}
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C C' : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    (∫ z : EucSpace m, z i * g z ∂stdGaussian (EucSpace m)) =
      ∫ z : EucSpace m, fderiv ℝ g z (EuclideanSpace.single i 1) ∂stdGaussian (EucSpace m) := by
  let μ := stdGaussian (EucSpace m)
  let R := standardGaussianResidual i
  let ν := μ.map R
  have hR : AEMeasurable R μ := R.continuous.aemeasurable
  have : IsProbabilityMeasure ν := (Measure.isProbabilityMeasure_map_iff hR).mpr inferInstance
  let F := gaussianCoordinateSlice g i
  let D := gaussianCoordinateSliceDerivative g i
  have hF : Continuous F := continuous_gaussianCoordinateSlice g hg i
  have hD : Continuous D := continuous_gaussianCoordinateSliceDerivative g hg i
  have hpair : AEMeasurable (fun z : EucSpace m => (z i, R z)) μ :=
    (EuclideanSpace.proj (𝕜 := ℝ) i).continuous.aemeasurable.prodMk hR
  have hmap : μ.map (fun z : EucSpace m => (z i, R z)) = (gaussianReal 0 1).prod ν := by
    have h := (indepFun_stdGaussian_coordinate_residual i).map_prod_eq_prod_map_map
      (hasGaussianLaw_stdGaussian_coordinate i).aemeasurable hR
    simpa only [stdGaussian_map_coordinate] using h
  have hFint : Integrable (fun q : ℝ × EucSpace m => q.1 * F q) ((gaussianReal 0 1).prod ν) := by
    rw [← hmap]
    apply (integrable_map_measure (continuous_fst.mul hF).aestronglyMeasurable hpair).mpr
    simpa only [Function.comp_def, Pi.mul_apply, F, gaussianCoordinateSlice, R, μ,
      standardGaussianResidual_reconstruct] using
      integrable_stdGaussian_coordinate_test_polynomial g hg i C p hbound
  have hDint : Integrable D ((gaussianReal 0 1).prod ν) := by
    rw [← hmap]
    apply (integrable_map_measure hD.aestronglyMeasurable hpair).mpr
    simpa only [Function.comp_def, D, gaussianCoordinateSliceDerivative, R, μ,
      standardGaussianResidual_reconstruct] using
      integrable_stdGaussian_coordinate_derivative_polynomial g hg i C' p hbound'
  have hslice (y : EucSpace m) : (∫ r, r * F (r, y) ∂gaussianReal 0 1) =
      ∫ r, D (r, y) ∂gaussianReal 0 1 := by
    obtain ⟨A, B, hA, hB⟩ := gaussianCoordinateSlice_polynomial_bounds g i C C' p hbound hbound' y
    have h := gaussian_stein_random_variable_polynomial (gaussianReal 0 1) id
      IsGaussian.hasGaussianLaw_id integral_id_gaussianReal
      (fun r => F (r, y)) (fun r => D (r, y))
      (hF.comp (continuous_id.prodMk continuous_const))
      (hD.comp (continuous_id.prodMk continuous_const))
      (gaussianCoordinateSlice_hasDerivAt g hg i y) A B p hA hB
    simpa only [id_eq, variance_id_gaussianReal, NNReal.coe_one, one_mul] using h
  calc
    _ = ∫ q : ℝ × EucSpace m, q.1 * F q ∂μ.map (fun z => (z i, R z)) := by
      simpa only [Pi.mul_apply, F, gaussianCoordinateSlice, R, μ, standardGaussianResidual_reconstruct] using
        (integral_map hpair (continuous_fst.mul hF).aestronglyMeasurable).symm
    _ = ∫ q : ℝ × EucSpace m, q.1 * F q ∂(gaussianReal 0 1).prod ν := by rw [hmap]
    _ = ∫ q, D q ∂(gaussianReal 0 1).prod ν := by
      rw [integral_prod_symm _ hFint, integral_prod_symm _ hDint]
      simp_rw [hslice]
    _ = _ := by
      rw [← hmap, integral_map hpair hD.aestronglyMeasurable]
      simp only [D, gaussianCoordinateSliceDerivative, R, μ, standardGaussianResidual_reconstruct]

/-- A genuine unbounded linear coordinate and its actual constant
Fréchet derivative satisfy every polynomial Stein hypothesis. -/
example : ∃ g : EucSpace 2 → ℝ, ContDiff ℝ 1 g ∧
    (∀ z, ‖g z‖ ≤ 1 * (1 + ‖z‖) ^ 1) ∧
    (∀ z, ‖fderiv ℝ g z‖ ≤ ‖EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)‖ * (1 + ‖z‖) ^ 1) := by
  let L := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  refine ⟨L, L.contDiff, ?_, ?_⟩
  · intro z
    change ‖z (0 : Fin 2)‖ ≤ 1 * (1 + ‖z‖) ^ 1
    simpa only [one_mul, pow_one] using (PiLp.norm_apply_le z (0 : Fin 2)).trans
      (le_add_of_nonneg_left zero_le_one)
  · intro z
    rw [L.fderiv, pow_one]
    nlinarith [norm_nonneg L, norm_nonneg z]

end Transformer.Homogenized
