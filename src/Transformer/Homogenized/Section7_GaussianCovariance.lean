module

public import Transformer.Basic
public import Mathlib.Probability.Distributions.Gaussian.Fernique
public import Transformer.Homogenized.Section7_GaussianStein
public import Mathlib.Probability.Distributions.Gaussian.Multivariate
public import Mathlib.Probability.Distributions.Gaussian.CharFun

/-!
# The original Gaussian covariance and its actual square-root representation

ArXiv:2604.01978v1, §7, `lem:gaussian_interp`, `eq:SIGMA_R`,
and `eq:interpolation_gaussian`. The original covariance entries are
the original centered second moments of the actual Gaussian law.
They determine its full covariance bilinear form and a positive
semidefinite matrix, including every singular covariance matrix.

Gaussian uniqueness by mean and covariance then identifies the
original law with the actual multivariate Gaussian measure.
Its definition is the pushforward of the actual standard Gaussian
by the covariance matrix's genuine positive square root. Thus the
later integration-by-parts argument can work in independent standard
coordinates without assuming a density for a singular original law.
No covariance positivity or Gaussian representation is assumed.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal MatrixOrder

namespace Transformer.Homogenized

/-- Every actual Gaussian coordinate is square integrable.
Source: arXiv:2604.01978v1, §7, the original covariance entries
in `lem:gaussian_interp`, including degenerate coordinates. -/
theorem gaussian_coordinate_memLp_two {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (i : Fin m) : MemLp (fun z : EucSpace m => z i) 2 μ := by
  let : IsGaussian μ := hμ
  have h := (IsGaussian.hasGaussianLaw_id (μ := μ)).map (EuclideanSpace.proj (𝕜 := ℝ) i)
  exact h.memLp_two

/-- Every actual Gaussian coordinate product has a finite integral.
Source: arXiv:2604.01978v1, §7, the actual covariance matrix
in `lem:gaussian_interp`, without a nonsingularity assumption. -/
theorem integrable_gaussian_coordinate_mul {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (i j : Fin m) : Integrable (fun z : EucSpace m => z i * z j) μ :=
  (gaussian_coordinate_memLp_two μ hμ i).integrable_mul (gaussian_coordinate_memLp_two μ hμ j)

/-- The original coordinate centering implies the actual vector mean
is zero. Source: arXiv:2604.01978v1, §7, the original centered
Gaussian law in `lem:gaussian_interp`. -/
theorem gaussian_integral_eq_zero_of_coordinates {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0) :
    (∫ z, z ∂μ) = (0 : EucSpace m) := by
  let : IsGaussian μ := hμ
  ext i
  have h := integral_inner (𝕜 := ℝ) (IsGaussian.integrable_fun_id (μ := μ))
    (EuclideanSpace.single i (1 : ℝ))
  simp only [EuclideanSpace.inner_single_left, map_one, one_mul] at h
  exact h.symm.trans (hcenter i)

/-- The original second-moment matrix gives the full actual Gaussian
covariance bilinear form. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp` and the covariance trace in `eq:gaussian_interp_0`. -/
theorem gaussian_covarianceBilin_eq_matrix {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j) (x y : EucSpace m) :
    covarianceBilin μ x y = dotProduct x.ofLp (S.mulVec y.ofLp) := by
  let : IsGaussian μ := hμ
  have hm := gaussian_integral_eq_zero_of_coordinates μ hμ hcenter
  rw [covarianceBilin_apply (IsGaussian.memLp_two_id (μ := μ))]
  simp only [id_eq, hm, sub_zero, PiLp.inner_apply, Real.inner_apply]
  have he (z : EucSpace m) : (∑ i : Fin m, x i * z i) * (∑ j : Fin m, y j * z j) =
      ∑ i : Fin m, ∑ j : Fin m, (x i * y j) * (z i * z j) := by
    rw [Finset.sum_mul_sum]
    exact Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by ring
  simp_rw [he]
  have hint (i j : Fin m) : Integrable (fun z : EucSpace m => (x i * y j) * (z i * z j)) μ :=
    (integrable_gaussian_coordinate_mul μ hμ i j).const_mul _
  rw [integral_finsetSum _ (fun i hi => integrable_finsetSum _ (fun j hj => hint i j))]
  simp_rw [integral_finsetSum _ (fun j hj => hint _ j), integral_const_mul, hcov]
  simp only [dotProduct, Matrix.mulVec, Finset.mul_sum]
  exact Finset.sum_congr rfl fun i hi => Finset.sum_congr rfl fun j hj => by ring

/-- The actual original covariance matrix is positive semidefinite.
Source: arXiv:2604.01978v1, §7, the possibly singular covariance
`eq:SIGMA_R` and the original Gaussian interpolation lemma. -/
theorem gaussian_secondMoment_posSemidef {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j) : S.PosSemidef := by
  rw [Matrix.posSemidef_iff_dotProduct_mulVec]
  constructor
  · ext i j
    simp only [Matrix.conjTranspose_apply, star_trivial]
    rw [← hcov i j, ← hcov j i]
    apply integral_congr_ae
    filter_upwards with z
    ring
  · intro x
    have h := covarianceBilin_self_nonneg (μ := μ) (WithLp.toLp 2 x : EucSpace m)
    rw [gaussian_covarianceBilin_eq_matrix μ hμ S hcenter hcov] at h
    simpa only [WithLp.ofLp_toLp, star_trivial] using h

/-- The actual original centered Gaussian law equals the actual
multivariate Gaussian with its original covariance entries.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`, including
the singular Gram covariances discussed after `eq:SIGMA_R`. -/
theorem gaussian_eq_multivariate_of_secondMoments {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j) :
    μ = multivariateGaussian (0 : EucSpace m) S := by
  let : IsGaussian μ := hμ
  have hS := gaussian_secondMoment_posSemidef μ hμ S hcenter hcov
  apply IsGaussian.ext
  · simpa only [id_eq, integral_id_multivariateGaussian] using
      gaussian_integral_eq_zero_of_coordinates μ hμ hcenter
  · ext x y
    rw [gaussian_covarianceBilin_eq_matrix μ hμ S hcenter hcov,
      covarianceBilin_multivariateGaussian hS]

/-- The original Gaussian law is represented by the actual standard
Gaussian and the actual positive square root of its original covariance.
Source: arXiv:2604.01978v1, §7, the original Gaussian vectors in
`lem:gaussian_interp`. This construction includes degenerate covariances. -/
theorem gaussian_eq_stdGaussian_map_sqrt {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j) :
    μ = (stdGaussian (EucSpace m)).map (Matrix.toEuclideanCLM (𝕜 := ℝ) (CFC.sqrt S)) := by
  rw [gaussian_eq_multivariate_of_secondMoments μ hμ S hcenter hcov, multivariateGaussian]
  simp only [zero_add]

/-- The original standard Gaussian has coordinate means zero and the
identity as its actual second-moment matrix. This gives a nondegenerate
joint witness for the original covariance and representation hypotheses. -/
example : IsGaussian (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    ∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j := by
  refine ⟨inferInstance, fun i => ?_, fun i j => ?_⟩
  · have h := integral_inner (𝕜 := ℝ) (IsGaussian.integrable_fun_id (μ := stdGaussian (EucSpace 2)))
      (EuclideanSpace.single i (1 : ℝ))
    simpa only [EuclideanSpace.inner_single_left, map_one, one_mul, integral_id_stdGaussian,
      inner_zero_right] using h
  · have h := congrArg (fun B => B (EuclideanSpace.single i (1 : ℝ))
      (EuclideanSpace.single j (1 : ℝ))) (covarianceBilin_stdGaussian (E := EucSpace 2))
    rw [covarianceBilin_apply (IsGaussian.memLp_two_id (μ := stdGaussian (EucSpace 2)))] at h
    simpa only [id_eq, integral_id_stdGaussian, sub_zero, EuclideanSpace.inner_single_left,
      map_one, one_mul, innerSL_apply_apply ℝ, PiLp.single_apply, Matrix.one_apply] using h

end Transformer.Homogenized
