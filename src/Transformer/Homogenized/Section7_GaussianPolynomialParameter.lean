/-
# Polynomial integration by parts on the original independent Gaussian pair

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. The actual joint law of the two original
independent Gaussian vectors is their actual product Gaussian law.
The proved polynomial vector Stein identity therefore transfers back
to their original probability space with its original covariance entries.

Original pair tests, coordinate-weighted tests and actual directional
derivatives are integrable by genuine moments of that original joint law.
These facts justify both the original expectations and later finite sums.
Both vectors retain the actual measurability carried by `HasGaussianLaw`.
The other Gaussian vector may have a singular or even zero covariance.
All growth bounds concern the actual test or its actual derivative.
-/

import Transformer.Homogenized.Section7_GaussianPolynomialVectorConditioning

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators

namespace Transformer.Homogenized

/-- The original polynomial pair test has a finite expectation on
the original probability space. Source: arXiv:2604.01978v1, §7,
the original test expectation in `lem:gaussian_interp`. -/
theorem integrable_gaussian_pair_test_polynomial {Ω : Type*} [MeasurableSpace Ω] {m q : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (Y : Ω → EucSpace q)
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P) (hindep : IndepFun X Y P)
    (F : EucSpace m × EucSpace q → ℝ) (hF : Continuous F) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ p) :
    Integrable (fun ω => F (X ω, Y ω)) P := by
  have hXY := hasGaussianLaw_independent_gaussian_pair P X Y hX hY hindep
  apply integrable_of_gaussian_polynomial_bound P _ hXY _
    (hF.aestronglyMeasurable.comp_aemeasurable hXY.aemeasurable) C p
  exact ae_of_all _ fun ω => hbound (X ω, Y ω)

/-- The actual first-vector coordinate times the original pair test
is integrable. Source: arXiv:2604.01978v1, §7, the original
coordinate-weighted expectation in the Gaussian Stein computation. -/
theorem integrable_gaussian_pair_coordinate_test_polynomial {Ω : Type*} [MeasurableSpace Ω] {m q : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (Y : Ω → EucSpace q)
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P) (hindep : IndepFun X Y P)
    (F : EucSpace m × EucSpace q → ℝ) (hF : Continuous F) (i : Fin m) (C : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ p) :
    Integrable (fun ω => X ω i * F (X ω, Y ω)) P := by
  have hXY := hasGaussianLaw_independent_gaussian_pair P X Y hX hY hindep
  have ht : Continuous (fun z : EucSpace m × EucSpace q => z.1 i * F z) :=
    ((EuclideanSpace.proj (𝕜 := ℝ) i).continuous.comp continuous_fst).mul hF
  apply integrable_of_gaussian_polynomial_bound P _ hXY _
    (ht.aestronglyMeasurable.comp_aemeasurable hXY.aemeasurable) C (p + 1)
  exact ae_of_all _ fun ω => norm_gaussian_product_coordinate_test_polynomial_le F C p hbound i (X ω, Y ω)

/-- The genuine full derivative on any actual product vector has a
finite original expectation. Source: arXiv:2604.01978v1, §7,
the original derivative expectation in `lem:gaussian_interp`. -/
theorem integrable_gaussian_pair_directional_derivative_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m q : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (Y : Ω → EucSpace q)
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P) (hindep : IndepFun X Y P)
    (F : EucSpace m × EucSpace q → ℝ) (hF : ContDiff ℝ 1 F)
    (v : EucSpace m × EucSpace q) (C' : ℝ) (p : ℕ)
    (hbound' : ∀ z, ‖fderiv ℝ F z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    Integrable (fun ω => fderiv ℝ F (X ω, Y ω) v) P := by
  have hXY := hasGaussianLaw_independent_gaussian_pair P X Y hX hY hindep
  have ht : Continuous (fun z => fderiv ℝ F z v) :=
    (hF.continuous_fderiv (by decide)).clm_apply continuous_const
  apply integrable_of_gaussian_polynomial_bound P _ hXY _
    (ht.aestronglyMeasurable.comp_aemeasurable hXY.aemeasurable) (C' * ‖v‖) p
  apply ae_of_all
  intro ω
  calc
    _ ≤ ‖fderiv ℝ F (X ω, Y ω)‖ * ‖v‖ := (fderiv ℝ F (X ω, Y ω)).le_opNorm v
    _ ≤ (C' * (1 + ‖(X ω, Y ω)‖) ^ p) * ‖v‖ :=
      mul_le_mul_of_nonneg_right (hbound' (X ω, Y ω)) (norm_nonneg v)
    _ = _ := by ring

/-- The actual coordinate-weighted original test equals its original
covariance contraction with its genuine full derivative on the first
vector axis. Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`.
Polynomial growth replaces uniform boundedness, as needed for the
actual bilinear softmax application after `eq:interpolation_interm`. -/
theorem gaussian_stein_vector_independent_parameter_polynomial
    {Ω : Type*} [MeasurableSpace Ω] {m q : ℕ}
    (P : Measure Ω) (X : Ω → EucSpace m) (hX : HasGaussianLaw X P)
    (S : Matrix (Fin m) (Fin m) ℝ)
    (hcenter : ∀ i : Fin m, ∫ ω, X ω i ∂P = 0)
    (hcov : ∀ i j : Fin m, ∫ ω, X ω i * X ω j ∂P = S i j)
    (Y : Ω → EucSpace q) (hY : HasGaussianLaw Y P) (hindep : IndepFun X Y P)
    (F : EucSpace m × EucSpace q → ℝ) (hF : ContDiff ℝ 1 F) (i : Fin m) (C C' : ℝ) (p : ℕ)
    (hbound : ∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ p)
    (hbound' : ∀ z, ‖fderiv ℝ F z‖ ≤ C' * (1 + ‖z‖) ^ p) :
    (∫ ω, X ω i * F (X ω, Y ω) ∂P) =
      ∫ ω, ∑ j : Fin m, S i j *
        fderiv ℝ F (X ω, Y ω) (EuclideanSpace.single j 1, 0) ∂P := by
  have : IsProbabilityMeasure P := hX.isProbabilityMeasure
  have hc (j : Fin m) : (∫ z : EucSpace m, z j ∂P.map X) = 0 := by
    rw [integral_map (f := fun z : EucSpace m => z j) hX.aemeasurable (by fun_prop)]
    exact hcenter j
  have hS (j k : Fin m) : (∫ z : EucSpace m, z j * z k ∂P.map X) = S j k := by
    rw [integral_map hX.aemeasurable (by fun_prop)]
    exact hcov j k
  have hpair := hX.aemeasurable.prodMk hY.aemeasurable
  have hmap := hindep.map_prod_eq_prod_map_map hX.aemeasurable hY.aemeasurable
  have htest : Continuous (fun z : EucSpace m × EucSpace q => z.1 i * F z) :=
    ((EuclideanSpace.proj (𝕜 := ℝ) i).continuous.comp continuous_fst).mul hF.continuous
  have hD : Continuous (fderiv ℝ F) := hF.continuous_fderiv (by decide)
  have hcontract : Continuous (fun z : EucSpace m × EucSpace q => ∑ j : Fin m,
      S i j * fderiv ℝ F z (EuclideanSpace.single j 1, 0)) := by fun_prop
  calc
    _ = ∫ z : EucSpace m × EucSpace q, z.1 i * F z ∂P.map (fun ω => (X ω, Y ω)) :=
      (integral_map hpair htest.aestronglyMeasurable).symm
    _ = ∫ z : EucSpace m × EucSpace q, z.1 i * F z ∂(P.map X).prod (P.map Y) := by rw [hmap]
    _ = ∫ z : EucSpace m × EucSpace q, ∑ j : Fin m, S i j *
        fderiv ℝ F z (EuclideanSpace.single j 1, 0) ∂(P.map X).prod (P.map Y) :=
      gaussian_stein_vector_product_polynomial _ hX.isGaussian_map S hc hS _ hY.isGaussian_map
        F hF i C C' p hbound hbound'
    _ = _ := by rw [← hmap, integral_map hpair hcontract.aestronglyMeasurable]

/-- A nondegenerate original Gaussian, an independent zero Gaussian
and an unbounded actual pair test with its genuine derivative jointly
witness every original polynomial integration-by-parts hypothesis. -/
example : ∃ F : EucSpace 2 × EucSpace 2 → ℝ,
    HasGaussianLaw (id : EucSpace 2 → EucSpace 2) (stdGaussian (EucSpace 2)) ∧
    HasGaussianLaw (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    IndepFun id (fun _ : EucSpace 2 => (0 : EucSpace 2)) (stdGaussian (EucSpace 2)) ∧
    (∀ i : Fin 2, ∫ z : EucSpace 2, z i ∂stdGaussian (EucSpace 2) = 0) ∧
    (∀ i j : Fin 2, ∫ z : EucSpace 2, z i * z j ∂stdGaussian (EucSpace 2) =
      (1 : Matrix (Fin 2) (Fin 2) ℝ) i j) ∧ ContDiff ℝ 1 F ∧
    ∃ C, (∀ z, ‖F z‖ ≤ C * (1 + ‖z‖) ^ 1) ∧
      (∀ z, ‖fderiv ℝ F z‖ ≤ C * (1 + ‖z‖) ^ 1) := by
  let A := (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)).comp
    (ContinuousLinearMap.fst ℝ (EucSpace 2) (EucSpace 2)) +
    (EuclideanSpace.proj (𝕜 := ℝ) (0 : Fin 2)).comp
      (ContinuousLinearMap.snd ℝ (EucSpace 2) (EucSpace 2))
  refine ⟨A, IsGaussian.hasGaussianLaw_id, ?_, indepFun_const_right _ _,
    integral_stdGaussian_coordinate, ?_, A.contDiff, ‖A‖, ?_, ?_⟩
  · simpa only [Function.comp_def, zero_apply] using
      (IsGaussian.hasGaussianLaw_id (μ := stdGaussian (EucSpace 2))).map
        (0 : EucSpace 2 →L[ℝ] EucSpace 2)
  · intro i j
    simpa only [Matrix.one_apply] using integral_stdGaussian_coordinate_mul i j
  · intro z
    simp only [pow_one]
    exact (A.le_opNorm z).trans (mul_le_mul_of_nonneg_left (by linarith) (norm_nonneg A))
  · intro z
    rw [A.fderiv, pow_one]
    nlinarith [norm_nonneg A, norm_nonneg z]

end Transformer.Homogenized
