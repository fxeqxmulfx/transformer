/-
# Polynomial vector integration by parts with the original independent Gaussian

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The other original Gaussian vector is kept
fixed while integration by parts is applied to an actual affine slice.
The derivative is the genuine full derivative on the Gaussian product axis.

Polynomial bounds on the original pair and its actual derivative give
finite expectations under its actual product Gaussian law. Fubini
integrates the proved vector Stein identity against the other original
Gaussian law. Independence then identifies that product with the original
joint law, so the final formula is on the original probability space.
Singular laws are included, without densities or inverse covariances.
-/

import Transformer.Homogenized.Section7_GaussianPolynomialVectorStein

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- A coordinate of the actual first vector raises the original
pair's polynomial degree by one. Source: arXiv:2604.01978v1, §7,
the original coordinate factor in the vector Stein step. -/
theorem norm_gaussian_product_coordinate_test_polynomial_le {m q : ℕ}
    (F : EucSpace m × EucSpace q → ℝ) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ p) (i : Fin m)
    (z : EucSpace m × EucSpace q) : ‖z.1 i * F z‖ ≤ C * (1 + ‖z‖) ^ (p + 1) := by
  have hC : 0 ≤ C := (norm_nonneg (F 0)).trans (by simpa using hbound 0)
  have hc : ‖z.1 i‖ ≤ ‖z‖ := (PiLp.norm_apply_le z.1 i).trans (norm_fst_le z)
  rw [norm_mul]
  calc
    _ ≤ ‖z‖ * (C * (1 + ‖z‖) ^ p) := mul_le_mul hc (hbound z) (norm_nonneg _) (norm_nonneg z)
    _ ≤ (1 + ‖z‖) * (C * (1 + ‖z‖) ^ p) :=
      mul_le_mul_of_nonneg_right (by linarith) (by positivity)
    _ = _ := by rw [pow_succ]; ring

/-- The actual coordinate-weighted pair test is integrable under the
original product Gaussian law. Source: arXiv:2604.01978v1, §7,
the finite Fubini input for `lem:gaussian_interp`. -/
theorem integrable_gaussian_vector_product_test_polynomial {m q : ℕ}
    (μ : Measure (EucSpace m)) (hμ : IsGaussian μ) (ν : Measure (EucSpace q)) (hν : IsGaussian ν)
    (F : EucSpace m × EucSpace q → ℝ) (hF : ContDiff ℝ 1 F) (i : Fin m) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ p) :
    Integrable (fun z : EucSpace m × EucSpace q => z.1 i * F z) (μ.prod ν) := by
  have : IsGaussian μ := hμ
  have : IsGaussian ν := hν
  have ht : Continuous (fun z : EucSpace m × EucSpace q => z.1 i * F z) :=
    ((EuclideanSpace.proj (𝕜 := ℝ) i).continuous.comp continuous_fst).mul hF.continuous
  apply integrable_of_gaussian_polynomial_bound _ id IsGaussian.hasGaussianLaw_id _
    ht.aestronglyMeasurable C (p + 1)
  exact ae_of_all _ (norm_gaussian_product_coordinate_test_polynomial_le F C p hbound i)

/-- The actual vector Stein identity commutes with integration over
the original independent Gaussian parameter, with polynomial growth.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
Every Fubini input comes from the actual product Gaussian moments. -/
theorem gaussian_stein_vector_product_polynomial {m q : ℕ}
    (μ : Measure (EucSpace m)) (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j)
    (ν : Measure (EucSpace q)) (hν : IsGaussian ν)
    (F : EucSpace m × EucSpace q → ℝ) (hF : ContDiff ℝ 1 F) (i : Fin m) (C C' : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ F z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    (∫ z : EucSpace m × EucSpace q, z.1 i * F z ∂μ.prod ν) =
      ∫ z : EucSpace m × EucSpace q, ∑ j : Fin m, S i j *
        fderiv ℝ F z (EuclideanSpace.single j 1, 0) ∂μ.prod ν := by
  have : IsGaussian μ := hμ
  have : IsGaussian ν := hν
  have hC : 0 ≤ C := (norm_nonneg (F 0)).trans (by simpa using hbound 0)
  have hC' : 0 ≤ C' := (norm_nonneg (fderiv ℝ F 0)).trans (by simpa using hbound' 0)
  let J := ContinuousLinearMap.inl ℝ (EucSpace m) (EucSpace q)
  have hslice (y : EucSpace q) : ContDiff ℝ 1 (fun x : EucSpace m => F (x, y)) :=
    hF.comp (contDiff_id.prodMk contDiff_const)
  have hchain (y : EucSpace q) (x : EucSpace m) :
      fderiv ℝ (fun z : EucSpace m => F (z, y)) x = (fderiv ℝ F (x, y)).comp J := by
    have haff : HasFDerivAt (fun z : EucSpace m => (z, y)) J x :=
      (hasFDerivAt_id x).prodMk (hasFDerivAt_const y x)
    exact (((hF.differentiable (by decide)) (x, y)).hasFDerivAt.comp x haff).fderiv
  have hidentity (y : EucSpace q) : (∫ x : EucSpace m, x i * F (x, y) ∂μ) =
      ∫ x : EucSpace m, ∑ j : Fin m, S i j *
        fderiv ℝ F (x, y) (EuclideanSpace.single j 1, 0) ∂μ := by
    have hb (x : EucSpace m) : ‖F (x, y)‖ ≤
        (C * (1 + ‖J‖ + ‖((0 : EucSpace m), y)‖) ^ p) * (1 + ‖x‖) ^ p := by
      simpa only [J, ContinuousLinearMap.inl_apply, Prod.mk_add_mk, zero_add, add_zero] using
        norm_polynomial_gaussian_affine_le J (0, y) F C hC p hbound x
    have hd (x : EucSpace m) : ‖fderiv ℝ (fun z : EucSpace m => F (z, y)) x‖ ≤
        (C' * (1 + ‖J‖ + ‖((0 : EucSpace m), y)‖) ^ p * ‖J‖) * (1 + ‖x‖) ^ p := by
      have h := norm_polynomial_gaussian_affine_le J (0, y) (fun z => ‖fderiv ℝ F z‖) C' hC' p
        (fun z => by simpa only [norm_norm] using hbound' z) x
      simp only [J, ContinuousLinearMap.inl_apply, Prod.mk_add_mk, zero_add, add_zero] at h
      rw [norm_norm] at h
      rw [hchain]
      calc
        _ ≤ ‖fderiv ℝ F (x, y)‖ * ‖J‖ := (fderiv ℝ F (x, y)).opNorm_comp_le J
        _ ≤ (C' * (1 + ‖J‖ + ‖((0 : EucSpace m), y)‖) ^ p) * (1 + ‖x‖) ^ p * ‖J‖ :=
          mul_le_mul_of_nonneg_right h (norm_nonneg J)
        _ = _ := by ring
    have h := gaussian_stein_coordinate_polynomial μ hμ S hcenter hcov (fun x => F (x, y))
      (hslice y) i (C * (1 + ‖J‖ + ‖((0 : EucSpace m), y)‖) ^ p)
      (C' * (1 + ‖J‖ + ‖((0 : EucSpace m), y)‖) ^ p * ‖J‖) p hb hd
    simpa only [hchain, ContinuousLinearMap.comp_apply, J, ContinuousLinearMap.inl_apply] using h
  have hint := integrable_gaussian_vector_product_test_polynomial μ hμ ν hν F hF i C p hbound
  have hD : Continuous (fderiv ℝ F) := hF.continuous_fderiv (by decide)
  have haxis (j : Fin m) : Integrable (fun z : EucSpace m × EucSpace q =>
      fderiv ℝ F z (EuclideanSpace.single j 1, 0)) (μ.prod ν) := by
    apply integrable_of_gaussian_polynomial_bound _ id IsGaussian.hasGaussianLaw_id _
      (hD.clm_apply continuous_const).aestronglyMeasurable C' p
    apply ae_of_all
    intro z
    have hn : ‖(EuclideanSpace.single j (1 : ℝ), (0 : EucSpace q))‖ = 1 := by simp
    exact ((fderiv ℝ F z).le_opNorm _).trans (by rw [hn, mul_one]; exact hbound' z)
  have hintD := integrable_finsetSum Finset.univ (fun j hj => (haxis j).const_mul (S i j))
  rw [integral_prod_symm _ hint, integral_prod_symm _ hintD]
  exact integral_congr_ae (ae_of_all _ hidentity)

/-- The actual pair of original independent Gaussian vectors has its
actual joint Gaussian law. Source: arXiv:2604.01978v1, §7,
the original independent vectors in `eq:interpolation_gaussian`. -/
theorem hasGaussianLaw_independent_gaussian_pair {Ω : Type*} [MeasurableSpace Ω] {m q : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (Y : Ω → EucSpace q)
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P) (hindep : IndepFun X Y P) :
    HasGaussianLaw (fun ω => (X ω, Y ω)) P := by
  have : IsProbabilityMeasure P := hX.isProbabilityMeasure
  have : IsGaussian (P.map X) := hX.isGaussian_map
  have : IsGaussian (P.map Y) := hY.isGaussian_map
  refine ⟨hX.aemeasurable.prodMk hY.aemeasurable, ?_⟩
  rw [hindep.map_prod_eq_prod_map_map hX.aemeasurable hY.aemeasurable]
  infer_instance

/-- An unbounded linear test depending on both vectors, its genuine
derivative, and two actual standard Gaussian laws witness the product
integration-by-parts hypotheses with nondegenerate identity covariance. -/
example : ∃ F : EucSpace 2 × EucSpace 2 → ℝ,
    IsGaussian (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧ ContDiff ℝ 1 F ∧
    ∃ C, (∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ 1) ∧
      (∀ z, ‖fderiv ℝ F z‖ ≤ C * (1 + ‖z‖) ^ 1) := by
  let A := (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)).comp
    (ContinuousLinearMap.fst ℝ (EucSpace 2) (EucSpace 2)) +
    (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)).comp
      (ContinuousLinearMap.snd ℝ (EucSpace 2) (EucSpace 2))
  refine ⟨A, inferInstance, integral_stdGaussian_coordinate, ?_, A.contDiff, ‖A‖, ?_, ?_⟩
  · intro i j
    simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j
  · intro z
    simp only [pow_one]
    exact (A.le_opNorm z).trans (mul_le_mul_of_nonneg_left (by linarith) (norm_nonneg A))
  · intro z
    rw [A.fderiv, pow_one]
    nlinarith [norm_nonneg A, norm_nonneg z]

/-- A nondegenerate original Gaussian and an actual independent
degenerate Gaussian jointly satisfy the original pair-law hypotheses. -/
example : HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    HasGaussianLaw (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) := by
  refine ⟨IsGaussian.hasGaussianLaw_id, ?_, indepFun_const_right _ _⟩
  simpa only [Function.comp_def, zero_apply] using
    (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace 2))).map
      (0 : EucSpace 2 →L[ℝ] EucSpace 2)

end Transformer.Homogenized
