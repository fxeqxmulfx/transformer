module

public import Transformer.Homogenized.Section7_GaussianLinearImage

/-!
# Integration by parts for the original centered Gaussian law

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. Standard-coordinate integration by parts
is transferred through the actual linear Gaussian representation.
The chain rule gives the original derivative on each image axis.
Expanding those axes and integrating their actual coordinate products
identifies the resulting coefficients with the original covariance.

The original law is represented by its covariance square root, already
proved from its actual centering and second moments. No square root
is inverted, and no density or nondegeneracy is assumed. The resulting
identity therefore includes the singular Gram covariances discussed
immediately after `eq:SIGMA_R`. All derivatives and expectations refer
to the actual original test and actual original Gaussian measure.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators MatrixOrder

namespace Transformer.Homogenized

/-- The original derivative on any fixed actual vector is integrable
on the actual standard Gaussian image. Source: arXiv:2604.01978v1,
§7, domination in the integration-by-parts proof of `lem:gaussian_interp`. -/
theorem integrable_gaussian_linearImage_derivative {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g)
    (v : EucSpace m) (C' : ℝ) (hbound : ∀ z, ‖fderiv ℝ g z‖ ≤ C') :
    Integrable (fun z : EucSpace n => fderiv ℝ g (L z) v) (stdGaussian (EucSpace n)) := by
  apply Integrable.of_bound (((hg.continuous_fderiv (by decide)).comp L.continuous).clm_apply
    continuous_const).aestronglyMeasurable (C' * ‖v‖)
  exact ae_of_all _ fun z => ((fderiv ℝ g (L z)).le_opNorm v).trans
    (mul_le_mul_of_nonneg_right (hbound (L z)) (norm_nonneg v))

/-- The actual original linear-image coordinate integral is the sum
of standard-coordinate integration-by-parts identities with its genuine
image coefficients. Source: arXiv:2604.01978v1, §7, the original
Gaussian covariance reduction in `lem:gaussian_interp`. -/
theorem gaussian_linearImage_stein_sum {n m : ℕ} (L : EucSpace n →L[ℝ] EucSpace m)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C C' : ℝ)
    (hbound : ∀ z, ‖g z‖ ≤ C) (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C') :
    (∫ z : EucSpace n, (L z) i * g (L z) ∂stdGaussian (EucSpace n)) =
      ∑ k : Fin n, (L (EuclideanSpace.single k 1)) i *
        ∫ z : EucSpace n, fderiv ℝ g (L z) (L (EuclideanSpace.single k 1))
          ∂stdGaussian (EucSpace n) := by
  have hcomp : ContDiff ℝ 1 (g ∘ L) := hg.comp L.contDiff
  have hint (k : Fin n) : Integrable (fun z : EucSpace n =>
      (L (EuclideanSpace.single k 1)) i * (z k * g (L z))) (stdGaussian (EucSpace n)) :=
    (integrable_stdGaussian_coordinate_test (g ∘ L) hcomp k C (fun z => hbound (L z))).const_mul _
  calc
    _ = ∫ z : EucSpace n, ∑ k : Fin n,
        (L (EuclideanSpace.single k 1)) i * (z k * g (L z)) ∂stdGaussian (EucSpace n) := by
      apply integral_congr_ae
      apply ae_of_all
      intro z
      change (L z) i * g (L z) = ∑ k : Fin n,
        (L (EuclideanSpace.single k 1)) i * (z k * g (L z))
      rw [gaussian_linearImage_coordinate_expansion L z i, Finset.sum_mul]
      exact Finset.sum_congr rfl fun k hk => by ring
    _ = ∑ k : Fin n, (L (EuclideanSpace.single k 1)) i *
        ∫ z : EucSpace n, z k * g (L z) ∂stdGaussian (EucSpace n) := by
      rw [integral_finsetSum _ (fun k hk => hint k)]
      simp only [integral_const_mul]
    _ = _ := by
      apply Finset.sum_congr rfl
      intro k hk
      congr 1
      have h := stdGaussian_stein_coordinate (g ∘ L) hcomp k C (C' * ‖L‖)
        (fun z => hbound (L z)) (norm_fderiv_gaussian_linearImage_le L g hg C' hbound')
      simp only [fderiv_gaussian_linearImage L g hg, ContinuousLinearMap.comp_apply] at h
      simpa only [Function.comp_def] using h

/-- Expanding the actual image axes turns their derivative sum into
the original covariance contraction, computed from actual coordinate
second moments. Source: arXiv:2604.01978v1, §7, the genuine
covariance coefficients in the proof of `lem:gaussian_interp`. -/
theorem gaussian_linearImage_stein_coordinate {n m : ℕ} (L : EucSpace n →L[ℝ] EucSpace m)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C C' : ℝ)
    (hbound : ∀ z, ‖g z‖ ≤ C) (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C') :
    (∫ z : EucSpace n, (L z) i * g (L z) ∂stdGaussian (EucSpace n)) =
      ∫ z : EucSpace n, ∑ j : Fin m,
        (∫ w : EucSpace n, (L w) i * (L w) j ∂stdGaussian (EucSpace n)) *
          fderiv ℝ g (L z) (EuclideanSpace.single j 1) ∂stdGaussian (EucSpace n) := by
  rw [gaussian_linearImage_stein_sum L g hg i C C' hbound hbound']
  have hint (k : Fin n) : Integrable (fun z : EucSpace n =>
      (L (EuclideanSpace.single k 1)) i * fderiv ℝ g (L z) (L (EuclideanSpace.single k 1)))
      (stdGaussian (EucSpace n)) :=
    (integrable_gaussian_linearImage_derivative L g hg _ C' hbound').const_mul _
  simp_rw [← integral_const_mul]
  rw [← integral_finsetSum _ (fun k hk => hint k)]
  apply integral_congr_ae
  apply ae_of_all
  intro z
  simp only [integral_gaussian_linearImage_coordinate_mul]
  change (∑ k : Fin n, (L (EuclideanSpace.single k 1)) i *
      fderiv ℝ g (L z) (L (EuclideanSpace.single k 1))) =
    ∑ j : Fin m, (∑ k : Fin n, (L (EuclideanSpace.single k 1)) i *
      (L (EuclideanSpace.single k 1)) j) * fderiv ℝ g (L z) (EuclideanSpace.single j 1)
  calc
    _ = ∑ k : Fin n, ∑ j : Fin m, ((L (EuclideanSpace.single k 1)) i *
        (L (EuclideanSpace.single k 1)) j) * fderiv ℝ g (L z) (EuclideanSpace.single j 1) := by
      apply Finset.sum_congr rfl
      intro k hk
      rw [gaussian_linearForm_expansion (fderiv ℝ g (L z)) (L (EuclideanSpace.single k 1)),
        Finset.mul_sum]
      exact Finset.sum_congr rfl fun j hj => by ring
    _ = _ := by rw [Finset.sum_comm]; simp only [Finset.sum_mul]

/-- Integration by parts for the original centered Gaussian measure,
with its original second-moment covariance entries.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp` and the
covariance matrix in `eq:SIGMA_R`. The actual covariance square-root
representation proves this identity even for singular original laws. -/
theorem gaussian_stein_coordinate {m : ℕ} (μ : Measure (EucSpace m))
    (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j)
    (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g) (i : Fin m) (C C' : ℝ)
    (hbound : ∀ z, ‖g z‖ ≤ C) (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C') :
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
  simpa only [hS] using gaussian_linearImage_stein_coordinate L g hg i C C' hbound hbound'

/-- The genuine two-dimensional standard law and a nonconstant test
jointly witness centering, covariance, smoothness and the bounds in the
original vector identity. The test derivative on its own axis is one. -/
example : ∃ g : EucSpace 2 → ℝ,
    IsGaussian (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    ContDiff ℝ 1 g ∧ (∀ z, ‖g z‖ ≤ 1) ∧
    (∀ z, ‖fderiv ℝ g z‖ ≤ ‖EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)‖) ∧
    fderiv ℝ g 0 (EuclideanSpace.single (0 : Fin 2) 1) = 1 := by
  let A := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  let g : EucSpace 2 → ℝ := fun z => Real.sin (A z)
  have hd (z : EucSpace 2) : fderiv ℝ g z = Real.cos (A z) • A := A.hasFDerivAt.sin.fderiv
  refine ⟨g, inferInstance, integral_stdGaussian_coordinate, ?_, A.contDiff.sin,
    fun z => Real.abs_sin_le_one (A z), ?_, ?_⟩
  · intro i j
    simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j
  · intro z
    rw [hd, norm_smul]
    simpa only [Real.norm_eq_abs, one_mul, A] using
      mul_le_mul_of_nonneg_right (Real.abs_cos_le_one (A z)) (norm_nonneg A)
  · rw [hd]; simp [A]

end Transformer.Homogenized
