module

public import Transformer.Homogenized.Section7_GaussianVectorStein

/-!
# Vector Gaussian integration by parts with the original independent parameter

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. In the original interpolation argument,
integration by parts in one original vector keeps the other original
independent vector fixed. The actual derivative in the Gaussian vector
is the full original Fréchet derivative on the corresponding product axis.

The genuine Gaussian first moment and the original bounded test and
derivative make both product expectations finite. Fubini integrates
the actual vector Stein identity over the other vector's original law.
Independence then identifies the original joint law with the product
of its actual marginal laws. The final identity is on the original
probability space with the original covariance entries. The independent
parameter need not be Gaussian or have a density.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The original Gaussian vector coordinate weighted by the actual
bounded product test is integrable. Source: arXiv:2604.01978v1,
§7, the Fubini input in the proof of `lem:gaussian_interp`. -/
theorem integrable_gaussian_vector_product_weighted_test {m q : ℕ}
    (μ : Measure (EucSpace m)) (hμ : IsGaussian μ)
    (ν : Measure (EucSpace q)) [IsProbabilityMeasure ν]
    (F : EucSpace m × EucSpace q → ℝ) (hF : ContDiff ℝ 1 F) (i : Fin m)
    (C : ℝ) (hbound : ∀ p, ‖F p‖ ≤ C) :
    Integrable (fun p : EucSpace m × EucSpace q => p.1 i * F p) (μ.prod ν) := by
  let : IsGaussian μ := hμ
  have hcoord := ((IsGaussian.hasGaussianLaw_id (μ := μ)).map
    (EuclideanSpace.proj (𝕜 := ℝ) i)).integrable
  have hcont : Continuous (fun p : EucSpace m × EucSpace q => p.1 i * F p) :=
    ((EuclideanSpace.proj (𝕜 := ℝ) i).continuous.comp continuous_fst).mul hF.continuous
  apply ((hcoord.norm.comp_fst ν).mul_const C).mono' hcont.aestronglyMeasurable
  exact ae_of_all _ fun p => by
    change ‖p.1 i * F p‖ ≤ ‖p.1 i‖ * C
    rw [norm_mul]
    exact mul_le_mul_of_nonneg_left (hbound p) (norm_nonneg (p.1 i))

/-- Vector integration by parts commutes with the actual independent
parameter law. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp`. The derivative is the original full derivative
on the actual Gaussian product axis; every Fubini input is proved finite. -/
theorem gaussian_stein_vector_product {m q : ℕ}
    (μ : Measure (EucSpace m)) (hμ : IsGaussian μ) (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ z, z i ∂μ = 0)
    (hcov : ∀ i j : Fin m, ∫ z, z i * z j ∂μ = S i j)
    (ν : Measure (EucSpace q)) [IsProbabilityMeasure ν]
    (F : EucSpace m × EucSpace q → ℝ) (hF : ContDiff ℝ 1 F) (i : Fin m) (C C' : ℝ)
    (hbound : ∀ p, ‖F p‖ ≤ C) (hbound' : ∀ p, ‖fderiv ℝ F p‖ ≤ C') :
    (∫ p : EucSpace m × EucSpace q, p.1 i * F p ∂μ.prod ν) =
      ∫ p : EucSpace m × EucSpace q, ∑ j : Fin m, S i j *
        fderiv ℝ F p (EuclideanSpace.single j 1, 0) ∂μ.prod ν := by
  let : IsGaussian μ := hμ
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
    have hderivbound (x : EucSpace m) :
        ‖fderiv ℝ (fun z : EucSpace m => F (z, y)) x‖ ≤ C' * ‖J‖ := by
      rw [hchain]
      exact ((fderiv ℝ F (x, y)).opNorm_comp_le J).trans
        (mul_le_mul_of_nonneg_right (hbound' (x, y)) (norm_nonneg J))
    have h := gaussian_stein_coordinate μ hμ S hcenter hcov (fun x => F (x, y))
      (hslice y) i C (C' * ‖J‖) (fun x => hbound (x, y)) hderivbound
    simpa only [hchain, ContinuousLinearMap.comp_apply, J, ContinuousLinearMap.inl_apply] using h
  have hint := integrable_gaussian_vector_product_weighted_test μ hμ ν F hF i C hbound
  have hD : Continuous (fderiv ℝ F) := hF.continuous_fderiv (by decide)
  have haxis (j : Fin m) : Integrable (fun p : EucSpace m × EucSpace q =>
      fderiv ℝ F p (EuclideanSpace.single j 1, 0)) (μ.prod ν) := by
    apply Integrable.of_bound (hD.clm_apply continuous_const).aestronglyMeasurable C'
    apply ae_of_all
    intro p
    have hnorm : ‖(EuclideanSpace.single j (1 : ℝ), (0 : EucSpace q))‖ = 1 := by simp
    exact ((fderiv ℝ F p).le_opNorm _).trans (by rw [hnorm, mul_one]; exact hbound' p)
  have hintD := integrable_finsetSum Finset.univ (fun j hj => (haxis j).const_mul (S i j))
  rw [integral_prod_symm _ hint, integral_prod_symm _ hintD]
  exact integral_congr_ae (ae_of_all _ hidentity)

/-- The vector identity holds for the original Gaussian vector and
its actual independent parameter on their original probability space.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
Actual measurability is part of `HasGaussianLaw`, as required by the
source's term "random vector"; the original covariance entries are retained. -/
theorem gaussian_stein_vector_independent_parameter {Ω : Type*} [MeasurableSpace Ω] {m q : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0)
    (hcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = S i j)
    (Y : Ω → EucSpace q) (hY : AEMeasurable Y P) (hindep : IndepFun X Y P)
    (F : EucSpace m × EucSpace q → ℝ) (hF : ContDiff ℝ 1 F) (i : Fin m) (C C' : ℝ)
    (hbound : ∀ p, ‖F p‖ ≤ C) (hbound' : ∀ p, ‖fderiv ℝ F p‖ ≤ C') :
    (∫ ω, X ω i * F (X ω, Y ω) ∂P) =
      ∫ ω, ∑ j : Fin m, S i j *
        fderiv ℝ F (X ω, Y ω) (EuclideanSpace.single j 1, 0) ∂P := by
  let : IsProbabilityMeasure P := hX.isProbabilityMeasure
  let : IsGaussian (P.map X) := hX.isGaussian_map
  let : IsProbabilityMeasure (P.map Y) := (Measure.isProbabilityMeasure_map_iff hY).mpr inferInstance
  have hc (j : Fin m) : (∫ z : EucSpace m, z j ∂P.map X) = 0 := by
    rw [integral_map (f := fun z : EucSpace m => z j) hX.aemeasurable
      (EuclideanSpace.proj (𝕜 := ℝ) j).continuous.aestronglyMeasurable]
    exact hcenter j
  have hS (j k : Fin m) : (∫ z : EucSpace m, z j * z k ∂P.map X) = S j k := by
    rw [integral_map hX.aemeasurable (by fun_prop)]
    exact hcov j k
  have hpair := hX.aemeasurable.prodMk hY
  have hmap := hindep.map_prod_eq_prod_map_map hX.aemeasurable hY
  have htest : Continuous (fun p : EucSpace m × EucSpace q => p.1 i * F p) :=
    ((EuclideanSpace.proj (𝕜 := ℝ) i).continuous.comp continuous_fst).mul hF.continuous
  have hD : Continuous (fderiv ℝ F) := hF.continuous_fderiv (by decide)
  have hcontract : Continuous (fun p : EucSpace m × EucSpace q => ∑ j : Fin m,
      S i j * fderiv ℝ F p (EuclideanSpace.single j 1, 0)) := by fun_prop
  calc
    _ = ∫ p : EucSpace m × EucSpace q, p.1 i * F p ∂P.map (fun ω => (X ω, Y ω)) :=
      (integral_map hpair htest.aestronglyMeasurable).symm
    _ = ∫ p : EucSpace m × EucSpace q, p.1 i * F p ∂(P.map X).prod (P.map Y) := by rw [hmap]
    _ = ∫ p : EucSpace m × EucSpace q, ∑ j : Fin m, S i j *
        fderiv ℝ F p (EuclideanSpace.single j 1, 0) ∂(P.map X).prod (P.map Y) :=
      gaussian_stein_vector_product _ hX.isGaussian_map S hc hS _ F hF i C C' hbound hbound'
    _ = _ := by rw [← hmap, integral_map hpair hcontract.aestronglyMeasurable]

/-- The original standard vector and a nonzero constant vector jointly
witness the original Gaussian, covariance and independence hypotheses.
A nonconstant sine depends on both actual vector arguments and has an
actual derivative equal to one on the first coordinate axis at zero. -/
example : ∃ F : EucSpace 2 × EucSpace 2 → ℝ,
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧
    AEMeasurable (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => EuclideanSpace.single (0 : Fin 2) (1 : ℝ))
      (stdGaussian (EucSpace 2)) ∧
    ContDiff ℝ 1 F ∧ (∀ p, ‖F p‖ ≤ 1) ∧
    ∃ C', (∀ p, ‖fderiv ℝ F p‖ ≤ C') ∧
      fderiv ℝ F 0 (EuclideanSpace.single (0 : Fin 2) 1, 0) = 1 := by
  let A := (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)).comp
    (ContinuousLinearMap.fst ℝ (EucSpace 2) (EucSpace 2)) +
    (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)).comp
      (ContinuousLinearMap.snd ℝ (EucSpace 2) (EucSpace 2))
  let F : EucSpace 2 × EucSpace 2 → ℝ := fun p => Real.sin (A p)
  have hd (p : EucSpace 2 × EucSpace 2) : fderiv ℝ F p = Real.cos (A p) • A := A.hasFDerivAt.sin.fderiv
  refine ⟨F, IsGaussian.hasGaussianLaw_id, integral_stdGaussian_coordinate, ?_,
    measurable_const.aemeasurable, indepFun_const_right _ _, A.contDiff.sin,
    fun p => Real.abs_sin_le_one (A p), ‖A‖, ?_, ?_⟩
  · intro i j
    simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j
  · intro p
    rw [hd, norm_smul]
    simpa only [Real.norm_eq_abs, one_mul] using
      mul_le_mul_of_nonneg_right (Real.abs_cos_le_one (A p)) (norm_nonneg A)
  · rw [hd]; simp [A]

end Transformer.Homogenized
