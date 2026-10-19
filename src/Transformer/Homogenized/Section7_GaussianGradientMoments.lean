module

public import Transformer.Homogenized.Section7_GaussianInterpolationDerivative

/-!
# Actual coordinate and vector gradient expectations in Gaussian interpolation

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The actual derivative on an original
vector expands in its actual original coordinate components.
Each original coordinate-weighted gradient has a finite expectation,
from the genuine Gaussian first moment and original gradient bound.

Finite sums therefore commute with the actual original expectations.
The actual scalar path derivative splits into the two original vector
gradient terms, with their original square-root coefficients retained.
The original second-moment matrices are symmetric, directly from their
actual coordinate products. This supplies the original Hessian argument
order when the two original covariance contractions are combined.
No vector integration-by-parts identity or gradient moment is assumed.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The original real second-moment entries are symmetric.
Source: arXiv:2604.01978v1, §7, the original covariance
matrices in `lem:gaussian_interp` and `eq:SIGMA_R`. -/
theorem gaussian_secondMoment_symmetric {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = S i j) (i j : Fin m) : S i j = S j i := by
  rw [← hcov i j, ← hcov j i]
  exact integral_congr_ae (ae_of_all _ fun ω => mul_comm (X ω i) (X ω j))

/-- The actual original vector gradient is the sum of its genuine
original coordinate-weighted gradients.
Source: arXiv:2604.01978v1, §7, the coordinate expansion
in the integration-by-parts proof of `lem:gaussian_interp`. -/
theorem gaussianInterpolation_gradient_vector_expansion {m : ℕ} (φ : EucSpace m → ℝ)
    (t : ℝ) (x y : EucSpace m) :
    fderiv ℝ φ (gaussianInterpolationPoint t x y) x = ∑ i : Fin m,
      x i * fderiv ℝ φ (gaussianInterpolationPoint t x y) (EuclideanSpace.single i 1) := by
  simpa only [mul_comm] using gaussian_linearForm_expansion (fderiv ℝ φ (gaussianInterpolationPoint t x y)) x

/-- Each original Gaussian coordinate-weighted gradient has a finite
expectation on the original probability space.
Source: arXiv:2604.01978v1, §7, the finite original gradient
moments in the proof of `lem:gaussian_interp`. -/
theorem integrable_gaussianInterpolation_coordinate_gradient
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω)
    (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (Y : Ω → EucSpace m) (hY : AEMeasurable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) (t : ℝ) (i : Fin m) :
    Integrable (fun ω => X ω i * fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
      (EuclideanSpace.single i 1)) P := by
  have hXi : Integrable (fun ω => X ω i) P :=
    (hX.map (EuclideanSpace.proj (𝕜 := ℝ) i)).integrable
  have hg := aestronglyMeasurable_gaussianInterpolation_test P X Y hX.aemeasurable hY
    (gaussianCoordinateGradient φ i) (contDiff_gaussianCoordinateGradient φ hφ i).continuous t
  change Integrable (fun ω => X ω i * gaussianCoordinateGradient φ i
    (gaussianInterpolationPoint t (X ω) (Y ω))) P
  apply (hXi.norm.mul_const C).mono' (hXi.aestronglyMeasurable.mul hg)
  apply ae_of_all
  intro ω
  change ‖X ω i * gaussianCoordinateGradient φ i (gaussianInterpolationPoint t (X ω) (Y ω))‖ ≤ ‖X ω i‖ * C
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_left (norm_gaussianCoordinateGradient_le φ C hbdd i _) (norm_nonneg (X ω i))

/-- The actual original vector gradient also has a finite expectation.
Source: arXiv:2604.01978v1, §7, the original vector terms
in the differentiated interpolation expectation of `lem:gaussian_interp`. -/
theorem integrable_gaussianInterpolation_vector_gradient
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω)
    (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (Y : Ω → EucSpace m) (hY : AEMeasurable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) (t : ℝ) :
    Integrable (fun ω => fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (X ω)) P := by
  have hsum := integrable_finsetSum Finset.univ (fun i hi =>
    integrable_gaussianInterpolation_coordinate_gradient P X hX Y hY φ hφ C hbdd t i)
  exact hsum.congr (ae_of_all _ fun ω =>
    (gaussianInterpolation_gradient_vector_expansion φ t (X ω) (Y ω)).symm)

/-- The actual vector gradient expectation equals the actual sum of
its original coordinate-weighted gradient expectations.
Source: arXiv:2604.01978v1, §7, the finite coordinate summation
in the original Gaussian integration-by-parts computation. -/
theorem integral_gaussianInterpolation_vector_gradient
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω)
    (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (Y : Ω → EucSpace m) (hY : AEMeasurable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) (t : ℝ) :
    (∫ ω, fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (X ω) ∂P) =
      ∑ i : Fin m, ∫ ω, X ω i * fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω))
        (EuclideanSpace.single i 1) ∂P := by
  calc
    _ = ∫ ω, ∑ i : Fin m, X ω i *
        fderiv ℝ φ (gaussianInterpolationPoint t (X ω) (Y ω)) (EuclideanSpace.single i 1) ∂P :=
      integral_congr_ae (ae_of_all _ fun ω => gaussianInterpolation_gradient_vector_expansion φ t (X ω) (Y ω))
    _ = _ := integral_finsetSum _ (fun i hi =>
      integrable_gaussianInterpolation_coordinate_gradient P X hX Y hY φ hφ C hbdd t i)

/-- The actual scalar path derivative splits into its two actual
original vector gradient terms with its exact scalar coefficients.
Source: arXiv:2604.01978v1, §7, the original differentiated
path in the integration-by-parts proof of `lem:gaussian_interp`. -/
theorem gaussianInterpolation_velocity_gradient_expansion {m : ℕ} (φ : EucSpace m → ℝ)
    (t : ℝ) (x y : EucSpace m) :
    fderiv ℝ φ (gaussianInterpolationPoint t x y) (gaussianInterpolationVelocity t x y) =
      (1 / (2 * Real.sqrt t)) * fderiv ℝ φ (gaussianInterpolationPoint t x y) x -
        (1 / (2 * Real.sqrt (1 - t))) * fderiv ℝ φ (gaussianInterpolationPoint t x y) y := by
  simp only [gaussianInterpolationVelocity, map_sub, map_smul, smul_eq_mul]

/-- Each actual original Hessian coordinate has a finite expectation.
Source: arXiv:2604.01978v1, §7, the finite original covariance
trace in the corrected bounded-test `lem:gaussian_interp`. -/
theorem integrable_gaussianInterpolation_hessian_entry
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) [IsProbabilityMeasure P]
    (X Y : Ω → EucSpace m) (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (t : ℝ) (i j : Fin m) :
    Integrable (fun ω => iteratedFDeriv ℝ 2 φ (gaussianInterpolationPoint t (X ω) (Y ω))
      ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]) P := by
  let g : EucSpace m → ℝ := fun z => iteratedFDeriv ℝ 2 φ z
    ![EuclideanSpace.single i 1, EuclideanSpace.single j 1]
  have hD : ContDiff ℝ 1 (fderiv ℝ φ) := hφ.fderiv_right (by norm_num)
  have hD2 : Continuous (fderiv ℝ (fderiv ℝ φ)) := hD.continuous_fderiv (by decide)
  have hg : Continuous g := by
    dsimp only [g]
    simp only [iteratedFDeriv_two_apply, Matrix.cons_val_zero, Matrix.cons_val_one]
    fun_prop
  exact integrable_gaussianInterpolation_test P X Y hX hY g hg C
    (norm_gaussian_test_hessian_entry_le φ C hbdd i j) t

/-- The actual original covariance trace has a finite expectation.
Source: arXiv:2604.01978v1, §7, the original trace expectation
in the corrected bounded-test `lem:gaussian_interp`. -/
theorem integrable_gaussianInterpolation_covarianceTrace
    {Ω : Type*} [MeasurableSpace Ω] {m : ℕ} (P : Measure Ω) [IsProbabilityMeasure P]
    (X Y : Ω → EucSpace m) (hX : AEMeasurable X P) (hY : AEMeasurable Y P)
    (φ : EucSpace m → ℝ) (hφ : ContDiff ℝ 2 φ) (C : ℝ)
    (hbdd : ∀ k ≤ 2, ∀ z : EucSpace m, ‖iteratedFDeriv ℝ k φ z‖ ≤ C)
    (S : Matrix (Fin m) (Fin m) ℝ) (t : ℝ) :
    Integrable (fun ω => gaussianCovarianceTrace φ S (gaussianInterpolationPoint t (X ω) (Y ω))) P :=
  integrable_gaussianInterpolation_test P X Y hX hY (gaussianCovarianceTrace φ S)
    (continuous_gaussianCovarianceTrace φ hφ S) ((∑ i : Fin m, ∑ j : Fin m, ‖S i j‖) * C)
    (norm_gaussianCovarianceTrace_le φ C hbdd S) t

/-- The original difference of covariance matrices gives precisely
the difference of their original Hessian contractions.
Source: arXiv:2604.01978v1, §7, `Tr((Σˣ - Σʸ)ᵀ ∇²φ)`
in the original displayed identity of `lem:gaussian_interp`. -/
theorem gaussianCovarianceTrace_sub {m : ℕ} (φ : EucSpace m → ℝ)
    (S T : Matrix (Fin m) (Fin m) ℝ) (z : EucSpace m) :
    gaussianCovarianceTrace φ (S - T) z = gaussianCovarianceTrace φ S z - gaussianCovarianceTrace φ T z := by
  simp only [gaussianCovarianceTrace, Matrix.sub_apply, sub_mul, Finset.sum_sub_distrib]

/-- The original standard Gaussian second moments and the genuine
nonconstant bounded `C²` sine jointly witness all original covariance,
integrability, measurability and regularity hypotheses. -/
example : ∃ (φ : EucSpace 2 → ℝ) (C : ℝ),
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    AEMeasurable (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧ ContDiff ℝ 2 φ ∧
    (∀ k ≤ 2, ∀ z : EucSpace 2, ‖iteratedFDeriv ℝ k φ z‖ ≤ C) ∧
    fderiv ℝ φ 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  obtain ⟨φ, C, hφ, hbdd, hn⟩ := gaussian_interp_test_witness
  refine ⟨φ, C, IsGaussian.hasGaussianLaw_id, ?_, measurable_const.aemeasurable, hφ, hbdd, hn⟩
  intro i j
  simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j

end Transformer.Homogenized
