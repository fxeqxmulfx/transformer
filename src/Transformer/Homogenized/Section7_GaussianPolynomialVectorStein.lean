/-
# Integration by parts for the original Gaussian vector with polynomial growth

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and `eq:SIGMA_R`.
The actual covariance square root represents the original centered
Gaussian law as a genuine linear image of the standard law. The actual
linear-image integration-by-parts identity then recovers the original
second-moment covariance coefficients, even when the law is singular.
Every covariance coefficient remains a moment of the original Gaussian law.

The original coordinate-weighted test and every actual directional
derivative have finite expectations by genuine Gaussian moments.
The result is also transferred to the original random vector on its
original probability space, with its actual measurability retained.
The polynomial growth conditions are precisely the type of domination
used for the bilinear softmax test following `eq:interpolation_interm`.
The source states its interpolation lemma without growth control; these
explicit polynomial hypotheses justify the integration step it uses.
-/

import Transformer.Homogenized.Section7_GaussianPolynomialLinearImage

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators MatrixOrder

namespace Transformer.Homogenized

/-- The original Gaussian coordinate multiplied by the actual
polynomial test has a finite expectation. Source: arXiv:2604.01978v1,
§7, domination in the original vector Stein step of `lem:gaussian_interp`. -/
theorem integrable_gaussian_coordinate_test_polynomial {Ω : Type*} [MeasurableSpace Ω]
    {m : ℕ} (P : Measure Ω) (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p) (i : Fin m) :
    Integrable (fun ω => X ω i * g (X ω)) P := by
  have hC : 0 ≤ C := (norm_nonneg (g 0)).trans (by simpa using hbound 0)
  have ha := ((EuclideanSpace.proj (𝕜 := ℝ) i).continuous.mul
    hg.continuous).aestronglyMeasurable.comp_aemeasurable hX.aemeasurable
  apply integrable_of_gaussian_polynomial_bound P X hX _ ha C (p + 1)
  exact ae_of_all _ fun ω => norm_coordinate_mul_polynomial_le g C hC p hbound i (X ω)

/-- The actual derivative on a fixed original vector is integrable
under the original Gaussian law. Source: arXiv:2604.01978v1, §7,
the finite derivative expectations in `lem:gaussian_interp`. -/
theorem integrable_gaussian_directional_derivative_polynomial {Ω : Type*} [MeasurableSpace Ω]
    {m : ℕ} (P : Measure Ω) (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (v : EucSpace m) (C' : ℝ) (p : ℕ)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    Integrable (fun ω => fderiv ℝ g (X ω) v) P := by
  have ha : AEStronglyMeasurable (fun ω => fderiv ℝ g (X ω) v) P :=
    ((hg.continuous_fderiv (by decide)).clm_apply
    continuous_const).aestronglyMeasurable.comp_aemeasurable hX.aemeasurable
  apply integrable_of_gaussian_polynomial_bound P X hX _ ha (C' * ‖v‖) p
  apply ae_of_all
  intro ω
  calc
    _ ≤ ‖fderiv ℝ g (X ω)‖ * ‖v‖ := (fderiv ℝ g (X ω)).le_opNorm v
    _ ≤ (C' * (1 + ‖X ω‖) ^ p) * ‖v‖ :=
      mul_le_mul_of_nonneg_right (hbound' (X ω)) (norm_nonneg v)
    _ = _ := by ring

/-- The original centered Gaussian vector satisfies integration by
parts for the actual polynomial test and its actual derivative.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp` and the
covariance in `eq:SIGMA_R`. No density or invertible covariance is needed. -/
theorem gaussian_stein_coordinate_polynomial {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C C' : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    (∫ z : EucSpace m, z i * g z ∂μ) =
      ∫ z : EucSpace m, ∑ j : Fin m, S i j *
        fderiv ℝ g z (EuclideanSpace.single j 1) ∂μ := by
  let L := Matrix.toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt S)
  have hmap : μ = (stdGaussian (EucSpace m)).map L :=
    gaussian_eq_stdGaussian_map_sqrt μ hμ S hcenter hcov
  have hS (j : Fin m) : (∫ z : EucSpace m, (L z) i * (L z) j
      ∂stdGaussian (EucSpace m)) = S i j := by
    have h := hcov i j
    rw [hmap, integral_map L.continuous.aemeasurable (by fun_prop)] at h
    exact h
  have hD : Continuous (fderiv ℝ g) := hg.continuous_fderiv (by decide)
  have htest : Continuous (fun z : EucSpace m => z i * g z) :=
    (EuclideanSpace.proj (𝕜 := ℝ) i).continuous.mul hg.continuous
  have hcontract : Continuous (fun z : EucSpace m => ∑ j : Fin m,
      S i j * fderiv ℝ g z (EuclideanSpace.single j 1)) := by fun_prop
  rw [hmap, integral_map L.continuous.aemeasurable htest.aestronglyMeasurable,
    integral_map L.continuous.aemeasurable hcontract.aestronglyMeasurable]
  simpa only [hS] using gaussian_linearImage_stein_coordinate_polynomial L g hg i C C' p hbound hbound'

/-- The same polynomial integration-by-parts identity holds for the
actual Gaussian random vector on its original probability space.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
Actual measurability and original moments identify the original law and
covariance, including degenerate original Gaussian variables. -/
theorem gaussian_stein_random_vector_polynomial {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0)
    (hcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = S i j)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C C' : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    (∫ ω, X ω i * g (X ω) ∂P) =
      ∫ ω, ∑ j : Fin m, S i j * fderiv ℝ g (X ω) (EuclideanSpace.single j 1) ∂P := by
  have hc (j : Fin m) : (∫ z : EucSpace m, z j ∂P.map X) = 0 := by
    rw [integral_map (f := fun z : EucSpace m => z j) hX.aemeasurable (by fun_prop)]
    exact hcenter j
  have hv (j k : Fin m) : (∫ z : EucSpace m, z j * z k ∂P.map X) = S j k := by
    rw [integral_map hX.aemeasurable (by fun_prop)]
    exact hcov j k
  have hD : Continuous (fderiv ℝ g) := hg.continuous_fderiv (by decide)
  have htest : Continuous (fun z : EucSpace m => z i * g z) :=
    (EuclideanSpace.proj (𝕜 := ℝ) i).continuous.mul hg.continuous
  have hcontract : Continuous (fun z : EucSpace m => ∑ j : Fin m,
      S i j * fderiv ℝ g z (EuclideanSpace.single j 1)) := by fun_prop
  have h := gaussian_stein_coordinate_polynomial (P.map X) hX.isGaussian_map S hc hv
    g hg i C C' p hbound hbound'
  rw [integral_map hX.aemeasurable htest.aestronglyMeasurable,
    integral_map hX.aemeasurable hcontract.aestronglyMeasurable] at h
  exact h

/-- A genuine two-dimensional standard Gaussian, actual identity
covariance and an unbounded coordinate test witness all original
centering, covariance, measurability and polynomial derivative hypotheses. -/
example : ∃ g : EucSpace 2 → ℝ,
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    ContDiff ℝ 1 g ∧ (∀ z, ‖g z‖ ≤ 1 * (1 + ‖z‖) ^ 1) ∧
    (∀ z, ‖fderiv ℝ g z‖ ≤ ‖EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)‖ * (1 + ‖z‖) ^ 1) ∧
    fderiv ℝ g 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  let L := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  refine ⟨L, IsGaussian.hasGaussianLaw_id, integral_stdGaussian_coordinate, ?_, L.contDiff, ?_, ?_, ?_⟩
  · intro i j
    simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j
  · intro z
    change ‖z (0 : Fin 2)‖ ≤ 1 * (1 + ‖z‖) ^ 1
    simpa only [one_mul, pow_one] using (PiLp.norm_apply_le z (0 : Fin 2)).trans
      (le_add_of_nonneg_left zero_le_one)
  · intro z
    rw [L.fderiv, pow_one]
    nlinarith [norm_nonneg L, norm_nonneg z]
  · rw [L.fderiv]
    simp [L]

end Transformer.Homogenized
