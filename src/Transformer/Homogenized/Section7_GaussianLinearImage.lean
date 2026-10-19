module

public import Transformer.Homogenized.Section7_StandardGaussianStein

/-!
# Actual linear images in the Gaussian integration-by-parts argument

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The original Gaussian vector is a genuine
linear image of a standard Gaussian, with no invertibility required.
Its coordinate expansion uses the actual coordinate axes, and the
chain rule differentiates the original test composed with that image.

The second-moment coefficients of the image are computed by integrating
the actual products of its actual coordinates. Standard-coordinate
cross moments identify them with the usual sum of products of linear
image coefficients. This will recover the original covariance matrix
from its given second moments, without assuming a density for the
original law or discarding singular covariance matrices.
All exchanges with finite sums use actual finite Gaussian moments.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- Every original vector is the sum of its actual coordinate
components. Source: arXiv:2604.01978v1, §7, the coordinate
expansion in the Gaussian integration-by-parts proof. -/
theorem eucSpace_eq_sum_coordinate {m : ℕ} (z : EucSpace m) :
    z = ∑ k : Fin m, z k • EuclideanSpace.single k 1 := by
  ext j
  change z j = (EuclideanSpace.proj (𝕜 := ℝ) j) (∑ k : Fin m, z k • EuclideanSpace.single k 1)
  simp [map_sum, map_smul]

/-- Each actual linear form expands in the original coordinate axes.
Source: arXiv:2604.01978v1, §7, the linear image reduction
in the integration-by-parts proof of `lem:gaussian_interp`. -/
theorem gaussian_linearForm_expansion {m : ℕ} (A : EucSpace m →L[ℝ] ℝ) (z : EucSpace m) :
    A z = ∑ k : Fin m, A (EuclideanSpace.single k 1) * z k := by
  calc
    A z = A (∑ k : Fin m, z k • EuclideanSpace.single k 1) := congrArg A (eucSpace_eq_sum_coordinate z)
    _ = _ := by simp only [map_sum, map_smul, smul_eq_mul, mul_comm]

/-- The original image coordinate is the actual linear combination of
the original standard coordinates. Source: arXiv:2604.01978v1,
§7, the original Gaussian covariance reduction in `lem:gaussian_interp`. -/
theorem gaussian_linearImage_coordinate_expansion {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (z : EucSpace n) (i : Fin m) :
    (L z) i = ∑ k : Fin n, (L (EuclideanSpace.single k 1)) i * z k :=
  gaussian_linearForm_expansion ((EuclideanSpace.proj (𝕜 := ℝ) i).comp L) z

/-- The derivative of the original test on the actual linear image is
its original derivative composed with that same map.
Source: arXiv:2604.01978v1, §7, the chain rule in the original
Gaussian integration-by-parts proof of `lem:gaussian_interp`. -/
theorem fderiv_gaussian_linearImage {n m : ℕ} (L : EucSpace n →L[ℝ] EucSpace m)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (z : EucSpace n) :
    fderiv ℝ (g ∘ L) z = (fderiv ℝ g (L z)).comp L := by
  rw [fderiv_comp z ((hg.differentiable (by decide)) (L z)) L.differentiableAt, L.fderiv]

/-- The actual composed derivative is bounded by the original bound
times the actual operator norm of the original linear image.
Source: arXiv:2604.01978v1, §7, domination in the standard
Gaussian reduction of `lem:gaussian_interp`. -/
theorem norm_fderiv_gaussian_linearImage_le {n m : ℕ} (L : EucSpace n →L[ℝ] EucSpace m)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g)
    (C' : ℝ) (hbound : ∀ z, ‖fderiv ℝ g z‖ ≤ C') (z : EucSpace n) :
    ‖fderiv ℝ (g ∘ L) z‖ ≤ C' * ‖L‖ := by
  rw [fderiv_gaussian_linearImage L g hg z]
  exact ((fderiv ℝ g (L z)).opNorm_comp_le L).trans
    (mul_le_mul_of_nonneg_right (hbound (L z)) (norm_nonneg L))

/-- The original image coordinate multiplied by the original bounded
test is genuinely integrable under the standard vector law.
Source: arXiv:2604.01978v1, §7, the original coordinate integral
after the Gaussian square-root reduction of `lem:gaussian_interp`. -/
theorem integrable_gaussian_linearImage_test {n m : ℕ} (L : EucSpace n →L[ℝ] EucSpace m)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m)
    (C : ℝ) (hbound : ∀ z, ‖g z‖ ≤ C) :
    Integrable (fun z : EucSpace n => (L z) i * g (L z)) (stdGaussian (EucSpace n)) := by
  have hcoord := ((IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace n))).map
    ((EuclideanSpace.proj (𝕜 := ℝ) i).comp L)).integrable
  apply (hcoord.norm.mul_const C).mono' (by fun_prop)
  apply ae_of_all
  intro z
  simp only [norm_mul]
  exact mul_le_mul_of_nonneg_left (hbound (L z)) (norm_nonneg ((L z) i))

/-- The actual image coordinate products have finite expectations.
Source: arXiv:2604.01978v1, §7, the original covariance entries
in `lem:gaussian_interp`, including every singular linear image. -/
theorem integrable_gaussian_linearImage_coordinate_mul {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (i j : Fin m) :
    Integrable (fun z : EucSpace n => (L z) i * (L z) j) (stdGaussian (EucSpace n)) := by
  have h (k : Fin m) : MemLp (fun z : EucSpace n => (L z) k) 2 (stdGaussian (EucSpace n)) :=
    ((IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace n))).map
      ((EuclideanSpace.proj (𝕜 := ℝ) k).comp L)).memLp_two
  exact (h i).integrable_mul (h j)

/-- The actual image second moment is the genuine sum of products
of its original image coefficients. Source: arXiv:2604.01978v1,
§7, the original covariance matrix in `lem:gaussian_interp`.
Each coefficient is identified by an actual standard cross moment. -/
theorem integral_gaussian_linearImage_coordinate_mul {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (i j : Fin m) :
    (∫ z : EucSpace n, (L z) i * (L z) j ∂stdGaussian (EucSpace n)) =
      ∑ k : Fin n, (L (EuclideanSpace.single k 1)) i * (L (EuclideanSpace.single k 1)) j := by
  let A := (EuclideanSpace.proj (𝕜 := ℝ) j).comp L
  have hint (k : Fin n) : Integrable (fun z : EucSpace n =>
      (L (EuclideanSpace.single k 1)) i * (z k * (L z) j)) (stdGaussian (EucSpace n)) :=
    (integrable_stdGaussian_coordinate_linearForm k A).const_mul _
  calc
    _ = ∫ z : EucSpace n, ∑ k : Fin n,
        (L (EuclideanSpace.single k 1)) i * (z k * (L z) j) ∂stdGaussian (EucSpace n) := by
      apply integral_congr_ae
      apply ae_of_all
      intro z
      change (L z) i * (L z) j = ∑ k : Fin n,
        (L (EuclideanSpace.single k 1)) i * (z k * (L z) j)
      rw [gaussian_linearImage_coordinate_expansion L z i, Finset.sum_mul]
      exact Finset.sum_congr rfl fun k hk => by ring
    _ = ∑ k : Fin n, (L (EuclideanSpace.single k 1)) i *
        ∫ z : EucSpace n, z k * (L z) j ∂stdGaussian (EucSpace n) := by
      rw [integral_finsetSum _ (fun k hk => hint k)]
      simp only [integral_const_mul]
    _ = _ := by
      exact Finset.sum_congr rfl fun k hk => congrArg
        (fun r => (L (EuclideanSpace.single k 1)) i * r)
        (integral_stdGaussian_coordinate_linearForm k A)

/-- A nonconstant actual coordinate sine jointly satisfies the
original test hypotheses with the identity image. Its original
derivative has a uniform bound and is nonzero on its actual axis. -/
example : ∃ g : EucSpace 2 → ℝ, ContDiff ℝ 1 g ∧ (∀ z, ‖g z‖ ≤ 1) ∧
    (∀ z, ‖fderiv ℝ g z‖ ≤ ‖EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)‖) ∧
    fderiv ℝ (g ∘ ContinuousLinearMap.id ℝ (EucSpace 2)) 0
      (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  let A := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  let g : EucSpace 2 → ℝ := fun z => Real.sin (A z)
  have hd (z : EucSpace 2) : fderiv ℝ g z = Real.cos (A z) • A := A.hasFDerivAt.sin.fderiv
  refine ⟨g, A.contDiff.sin, fun z => Real.abs_sin_le_one (A z), ?_, ?_⟩
  · intro z
    rw [hd, norm_smul]
    simpa only [Real.norm_eq_abs, one_mul, A] using
      mul_le_mul_of_nonneg_right (Real.abs_cos_le_one (A z)) (norm_nonneg A)
  · change fderiv ℝ g 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1
    rw [hd]
    simp [A]

end Transformer.Homogenized
