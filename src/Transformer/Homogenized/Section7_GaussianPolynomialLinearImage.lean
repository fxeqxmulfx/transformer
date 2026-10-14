/-
# Polynomial Gaussian integration by parts through an actual linear image

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and `eq:SIGMA_R`.
The original centered Gaussian vector is a linear image of a standard
Gaussian vector. Polynomial growth is preserved under that actual map.
The genuine chain rule and standard-coordinate Stein identities give
integration by parts for the image with its original covariance entries.

All coordinate-weighted tests and actual directional derivatives are
integrable by Gaussian polynomial moments. Finite sums can therefore be
exchanged with the original integrals. The image need not be invertible:
the covariance is computed from its actual second moments, including
singular images. The source's polynomial derivative growth, rather than
a uniform derivative bound, justifies this version of its argument.
-/

import Transformer.Homogenized.Section7_StandardGaussianPolynomialStein

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The actual test retains polynomial growth on the original linear
Gaussian image. Source: arXiv:2604.01978v1, §7, the covariance
square-root representation used in `lem:gaussian_interp`. -/
theorem norm_gaussian_linearImage_test_polynomial_le {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (g : EucSpace m → ℝ) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p) (z : EucSpace n) :
    ‖g (L z)‖ ≤ (C * (1 + ‖L‖) ^ p) * (1 + ‖z‖) ^ p := by
  have hC : 0 ≤ C := (norm_nonneg (g 0)).trans (by simpa using hbound 0)
  simpa only [norm_zero, add_zero] using norm_polynomial_gaussian_affine_le L 0 g C hC p hbound z

/-- The chain rule controls the actual composed derivative by a
polynomial with the original image's operator norm.
Source: arXiv:2604.01978v1, §7, domination in the linear-image
reduction of the original integration-by-parts argument. -/
theorem norm_fderiv_gaussian_linearImage_polynomial_le {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g)
    (C' : ℝ) (p : ℕ) (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p)
    (z : EucSpace n) :
    ‖fderiv ℝ (g ∘ L) z‖ ≤ (C' * (1 + ‖L‖) ^ p * ‖L‖) * (1 + ‖z‖) ^ p := by
  have hd := norm_gaussian_linearImage_test_polynomial_le L (fun z => ‖fderiv ℝ g z‖) C' p
    (fun z => by simpa only [norm_norm] using hbound' z) z
  rw [norm_norm] at hd
  rw [fderiv_gaussian_linearImage L g hg z]
  calc
    _ ≤ ‖fderiv ℝ g (L z)‖ * ‖L‖ := (fderiv ℝ g (L z)).opNorm_comp_le L
    _ ≤ (C' * (1 + ‖L‖) ^ p) * (1 + ‖z‖) ^ p * ‖L‖ :=
      mul_le_mul_of_nonneg_right hd (norm_nonneg L)
    _ = _ := by ring

/-- The actual original derivative on any fixed vector is integrable
on the actual standard Gaussian image. Source: arXiv:2604.01978v1,
§7, the finite derivative integrals in `lem:gaussian_interp`. -/
theorem integrable_gaussian_linearImage_derivative_polynomial {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g)
    (v : EucSpace m) (C' : ℝ) (p : ℕ)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    Integrable (fun z : EucSpace n => fderiv ℝ g (L z) v) (stdGaussian (EucSpace n)) := by
  apply integrable_of_gaussian_polynomial_bound _ id IsGaussian.hasGaussianLaw_id _
    ((((hg.continuous_fderiv (by decide)).comp L.continuous).clm_apply
      continuous_const).aestronglyMeasurable) (C' * (1 + ‖L‖) ^ p * ‖v‖) p
  apply ae_of_all
  intro z
  change ‖fderiv ℝ g (L z) v‖ ≤ (C' * (1 + ‖L‖) ^ p * ‖v‖) * (1 + ‖z‖) ^ p
  have hd := norm_gaussian_linearImage_test_polynomial_le L (fun z => ‖fderiv ℝ g z‖) C' p
    (fun z => by simpa only [norm_norm] using hbound' z) z
  rw [norm_norm] at hd
  calc
    _ ≤ ‖fderiv ℝ g (L z)‖ * ‖v‖ := (fderiv ℝ g (L z)).le_opNorm v
    _ ≤ (C' * (1 + ‖L‖) ^ p) * (1 + ‖z‖) ^ p * ‖v‖ :=
      mul_le_mul_of_nonneg_right hd (norm_nonneg v)
    _ = _ := by ring

/-- Integration by parts for the original image coordinate is the
actual sum of the standard-coordinate identities.
Source: arXiv:2604.01978v1, §7, the covariance square-root
reduction of `lem:gaussian_interp`, with polynomial test growth. -/
theorem gaussian_linearImage_stein_sum_polynomial {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g)
    (i : Fin m) (C C' : ℝ) (p : ℕ) (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    (∫ z : EucSpace n, (L z) i * g (L z) ∂stdGaussian (EucSpace n)) =
      ∑ k : Fin n, (L (EuclideanSpace.single k 1)) i *
        ∫ z : EucSpace n, fderiv ℝ g (L z) (L (EuclideanSpace.single k 1))
          ∂stdGaussian (EucSpace n) := by
  have hcomp : ContDiff ℝ 1 (g ∘ L) := hg.comp L.contDiff
  have hb := norm_gaussian_linearImage_test_polynomial_le L g C p hbound
  have hd := norm_fderiv_gaussian_linearImage_polynomial_le L g hg C' p hbound'
  have hint (k : Fin n) : Integrable (fun z : EucSpace n =>
      (L (EuclideanSpace.single k 1)) i * (z k * g (L z))) (stdGaussian (EucSpace n)) :=
    (integrable_stdGaussian_coordinate_test_polynomial (g ∘ L) hcomp k
      (C * (1 + ‖L‖) ^ p) p hb).const_mul _
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
      have h := stdGaussian_stein_coordinate_polynomial (g ∘ L) hcomp k
        (C * (1 + ‖L‖) ^ p) (C' * (1 + ‖L‖) ^ p * ‖L‖) p hb hd
      simp only [fderiv_gaussian_linearImage L g hg, ContinuousLinearMap.comp_apply] at h
      simpa only [Function.comp_def] using h

/-- Actual image-axis expansion identifies the original covariance
contraction for polynomially growing tests.
Source: arXiv:2604.01978v1, §7, the original covariance coefficients
in `lem:gaussian_interp` and `eq:SIGMA_R`, including singular images. -/
theorem gaussian_linearImage_stein_coordinate_polynomial {n m : ℕ}
    (L : EucSpace n →L[ℝ] EucSpace m) (g : EucSpace m → ℝ) (hg : ContDiff ℝ 1 g)
    (i : Fin m) (C C' : ℝ) (p : ℕ) (hbound : ∀ z, ‖g z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ g z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    (∫ z : EucSpace n, (L z) i * g (L z) ∂stdGaussian (EucSpace n)) =
      ∫ z : EucSpace n, ∑ j : Fin m,
        (∫ w : EucSpace n, (L w) i * (L w) j ∂stdGaussian (EucSpace n)) *
          fderiv ℝ g (L z) (EuclideanSpace.single j 1) ∂stdGaussian (EucSpace n) := by
  rw [gaussian_linearImage_stein_sum_polynomial L g hg i C C' p hbound hbound']
  have hint (k : Fin n) : Integrable (fun z : EucSpace n =>
      (L (EuclideanSpace.single k 1)) i * fderiv ℝ g (L z) (L (EuclideanSpace.single k 1)))
      (stdGaussian (EucSpace n)) :=
    (integrable_gaussian_linearImage_derivative_polynomial L g hg _ C' p hbound').const_mul _
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

/-- An unbounded actual coordinate test, its genuine derivative and
a singular zero image satisfy the growth and smoothness hypotheses. -/
example : ∃ (L : EucSpace 2 →L[ℝ] EucSpace 2) (g : EucSpace 2 → ℝ),
    L = 0 ∧ ContDiff ℝ 1 g ∧ (∀ z, ‖g z‖ ≤ 1 * (1 + ‖z‖) ^ 1) ∧
    (∀ z, ‖fderiv ℝ g z‖ ≤ ‖EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)‖ * (1 + ‖z‖) ^ 1) := by
  let A := EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)
  refine ⟨0, A, rfl, A.contDiff, ?_, ?_⟩
  · intro z
    change ‖z (0 : Fin 2)‖ ≤ 1 * (1 + ‖z‖) ^ 1
    simpa only [one_mul, pow_one] using (PiLp.norm_apply_le z (0 : Fin 2)).trans
      (le_add_of_nonneg_left zero_le_one)
  · intro z
    rw [A.fderiv, pow_one]
    nlinarith [norm_nonneg A, norm_nonneg z]

end Transformer.Homogenized
