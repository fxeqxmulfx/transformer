module

public import Transformer.Homogenized.Section7_GaussianCovariance
public import Mathlib.MeasureTheory.Integral.Prod
public import Mathlib.Probability.Independence.Basic

/-!
# Gaussian integration by parts with the original independent parameter

ArXiv:2604.01978v1, §7, `lem:gaussian_interp` and
`eq:interpolation_gaussian`. Coordinate integration by parts keeps
the other original independent coordinates fixed, then integrates
the resulting scalar identity against their actual law.

Fubini is justified by the actual Gaussian first moment and the
original bounded test and derivative. Independence identifies the
original joint law with the actual product of the two marginals.
The final identity is therefore stated on the original probability
space, with its original scalar variance factor. No conditional
expectation formula or integration-by-parts input is assumed.
The independent parameter's law need not have a density or even
be Gaussian; in particular, degenerate coordinates are included.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped NNReal

namespace Transformer.Homogenized

/-- The original bounded test weighted by its original Gaussian
coordinate is integrable under the actual product law.
Source: arXiv:2604.01978v1, §7, the Fubini step in the original
coordinate integration-by-parts argument of `lem:gaussian_interp`. -/
theorem integrable_gaussian_product_weighted_test {m : ℕ}
    (μ : Measure ℝ) (hμ : IsGaussian μ) (ν : Measure (EucSpace m)) [IsProbabilityMeasure ν]
    (F : ℝ × EucSpace m → ℝ) (hF : Continuous F) (C : ℝ) (hbound : ∀ p, ‖F p‖ ≤ C) :
    Integrable (fun p : ℝ × EucSpace m => p.1 * F p) (μ.prod ν) := by
  let : IsGaussian μ := hμ
  have hdom := ((IsGaussian.integrable_fun_id (μ := μ)).norm.comp_fst ν).mul_const C
  apply hdom.mono' (continuous_fst.mul hF).aestronglyMeasurable
  apply ae_of_all
  intro p
  change ‖p.1 * F p‖ ≤ ‖p.1‖ * C
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_left (hbound p) (norm_nonneg p.1)

/-- Scalar Gaussian integration by parts commutes with integration
against the actual independent parameter law.
Source: arXiv:2604.01978v1, §7, `lem:gaussian_interp`, the original
coordinate argument followed by Fubini. The derivative is the actual
derivative of the original test in its Gaussian coordinate. -/
theorem gaussian_stein_product {m : ℕ}
    (μ : Measure ℝ) (hμ : IsGaussian μ) (hcenter : ∫ r, r ∂μ = 0)
    (ν : Measure (EucSpace m)) [IsProbabilityMeasure ν]
    (F D : ℝ × EucSpace m → ℝ) (hF : Continuous F) (hD : Continuous D)
    (hderiv : ∀ y : EucSpace m, ∀ r : ℝ, HasDerivAt (fun q => F (q, y)) (D (r, y)) r)
    (C C' : ℝ) (hbound : ∀ p, ‖F p‖ ≤ C) (hbound' : ∀ p, ‖D p‖ ≤ C') :
    (∫ p : ℝ × EucSpace m, p.1 * F p ∂μ.prod ν) =
      variance id μ * ∫ p, D p ∂μ.prod ν := by
  let : IsGaussian μ := hμ
  have hint := integrable_gaussian_product_weighted_test μ hμ ν F hF C hbound
  have hintD : Integrable D (μ.prod ν) :=
    Integrable.of_bound hD.aestronglyMeasurable C' (ae_of_all _ hbound')
  have hslice (y : EucSpace m) : (∫ r, r * F (r, y) ∂μ) =
      variance id μ * ∫ r, D (r, y) ∂μ := by
    have hFc : Continuous (fun r : ℝ => F (r, y)) := hF.comp (continuous_id.prodMk continuous_const)
    have hDc : Continuous (fun r : ℝ => D (r, y)) := hD.comp (continuous_id.prodMk continuous_const)
    exact gaussian_stein_random_variable μ id IsGaussian.hasGaussianLaw_id hcenter
      (fun r => F (r, y)) (fun r => D (r, y)) hFc hDc (hderiv y) C C'
      (fun r => hbound (r, y)) (fun r => hbound' (r, y))
  rw [integral_prod_symm _ hint]
  simp_rw [hslice]
  rw [integral_const_mul, integral_prod_symm _ hintD]

/-- The original bounded parameterized test is integrable on its
original probability space. Source: arXiv:2604.01978v1, §7,
the finite expectation differentiated in `lem:gaussian_interp`. -/
theorem integrable_gaussian_parameter_test {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X : Ω → ℝ) (hX : HasGaussianLaw X P)
    (Y : Ω → EucSpace m) (hY : AEMeasurable Y P)
    (F : ℝ × EucSpace m → ℝ) (hF : Continuous F) (C : ℝ) (hbound : ∀ p, ‖F p‖ ≤ C) :
    Integrable (fun ω => F (X ω, Y ω)) P := by
  let : IsProbabilityMeasure P := hX.isProbabilityMeasure
  have hFae := hF.aestronglyMeasurable.comp_aemeasurable (hX.aemeasurable.prodMk hY)
  exact Integrable.of_bound hFae C (ae_of_all _ fun ω => hbound (X ω, Y ω))

/-- The original coordinate-weighted parameterized test also has a
finite expectation. Source: arXiv:2604.01978v1, §7, the actual Gaussian
coordinate factor in the original integration-by-parts computation. -/
theorem integrable_gaussian_parameter_weighted_test {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X : Ω → ℝ) (hX : HasGaussianLaw X P)
    (Y : Ω → EucSpace m) (hY : AEMeasurable Y P)
    (F : ℝ × EucSpace m → ℝ) (hF : Continuous F) (C : ℝ) (hbound : ∀ p, ‖F p‖ ≤ C) :
    Integrable (fun ω => X ω * F (X ω, Y ω)) P := by
  have hFae := hF.aestronglyMeasurable.comp_aemeasurable (hX.aemeasurable.prodMk hY)
  apply (hX.integrable.norm.mul_const C).mono' (hX.aemeasurable.aestronglyMeasurable.mul hFae)
  apply ae_of_all
  intro ω
  change ‖X ω * F (X ω, Y ω)‖ ≤ ‖X ω‖ * C
  rw [norm_mul]
  exact mul_le_mul_of_nonneg_left (hbound (X ω, Y ω)) (norm_nonneg (X ω))

/-- The same original integration-by-parts identity holds on the
original probability space for a centered Gaussian coordinate and
an actual independent parameter. Source: arXiv:2604.01978v1, §7,
`lem:gaussian_interp`. Independence and the original marginal laws
supply every Fubini input; no Gaussian density is required of the parameter. -/
theorem gaussian_stein_independent_parameter {Ω : Type*} [MeasurableSpace Ω] {m : ℕ}
    (P : Measure Ω) (X : Ω → ℝ) (hX : HasGaussianLaw X P)
    (hcenter : ∫ ω, X ω ∂P = 0) (Y : Ω → EucSpace m) (hY : AEMeasurable Y P)
    (hindep : IndepFun X Y P)
    (F D : ℝ × EucSpace m → ℝ) (hF : Continuous F) (hD : Continuous D)
    (hderiv : ∀ y : EucSpace m, ∀ r : ℝ, HasDerivAt (fun q => F (q, y)) (D (r, y)) r)
    (C C' : ℝ) (hbound : ∀ p, ‖F p‖ ≤ C) (hbound' : ∀ p, ‖D p‖ ≤ C') :
    (∫ ω, X ω * F (X ω, Y ω) ∂P) = variance X P * ∫ ω, D (X ω, Y ω) ∂P := by
  let : IsProbabilityMeasure P := hX.isProbabilityMeasure
  let : IsGaussian (P.map X) := hX.isGaussian_map
  let : IsProbabilityMeasure (P.map Y) := (Measure.isProbabilityMeasure_map_iff hY).mpr inferInstance
  have hpair := hX.aemeasurable.prodMk hY
  have hmap := hindep.map_prod_eq_prod_map_map hX.aemeasurable hY
  have hmean : (∫ r : ℝ, r ∂P.map X) = 0 := by
    rw [integral_map (f := fun r : ℝ => r) hX.aemeasurable continuous_id.aestronglyMeasurable]
    exact hcenter
  have hvar : variance id (P.map X) = variance X P := by
    simpa only [Function.comp_def, id_eq] using variance_map measurable_id.aemeasurable hX.aemeasurable
  calc
    _ = ∫ p : ℝ × EucSpace m, p.1 * F p ∂P.map (fun ω => (X ω, Y ω)) :=
      (integral_map hpair (continuous_fst.mul hF).aestronglyMeasurable).symm
    _ = ∫ p : ℝ × EucSpace m, p.1 * F p ∂(P.map X).prod (P.map Y) := by rw [hmap]
    _ = variance id (P.map X) * ∫ p, D p ∂(P.map X).prod (P.map Y) :=
      gaussian_stein_product _ hX.isGaussian_map hmean _ F D hF hD hderiv C C' hbound hbound'
    _ = _ := by rw [hvar, ← hmap, integral_map hpair hD.aestronglyMeasurable]

/-- A nondegenerate actual Gaussian coordinate and an independent
actual vector parameter jointly satisfy the conditioning hypotheses
with a nonconstant original test depending on both arguments. -/
example : ∃ (P : Measure ℝ) (X : ℝ → ℝ) (Y : ℝ → EucSpace 2)
    (F D : ℝ × EucSpace 2 → ℝ),
    HasGaussianLaw X P ∧ (∫ ω, X ω ∂P) = 0 ∧ AEMeasurable Y P ∧ IndepFun X Y P ∧
    Continuous F ∧ Continuous D ∧
    (∀ y : EucSpace 2, ∀ r : ℝ, HasDerivAt (fun q => F (q, y)) (D (r, y)) r) ∧
    (∀ p, ‖F p‖ ≤ 1) ∧ (∀ p, ‖D p‖ ≤ 1) := by
  let a : EucSpace 2 := EuclideanSpace.single 0 1
  let F : ℝ × EucSpace 2 → ℝ := fun p => Real.sin (p.1 + inner (𝕜 := ℝ) a p.2)
  let D : ℝ × EucSpace 2 → ℝ := fun p => Real.cos (p.1 + inner (𝕜 := ℝ) a p.2)
  refine ⟨gaussianReal 0 1, id, fun _ => a, F, D, IsGaussian.hasGaussianLaw_id,
    integral_id_gaussianReal, measurable_const.aemeasurable, indepFun_const_right _ _,
    ?_, ?_, ?_, ?_, ?_⟩
  · dsimp only [F]; fun_prop
  · dsimp only [D]; fun_prop
  · intro y r
    simpa only [F, D, id_eq, mul_one] using ((hasDerivAt_id r).add_const (inner (𝕜 := ℝ) a y)).sin
  · intro p
    simpa only [F, Real.norm_eq_abs] using Real.abs_sin_le_one (x := p.1 + inner (𝕜 := ℝ) a p.2)
  · intro p
    simpa only [D, Real.norm_eq_abs] using Real.abs_cos_le_one (x := p.1 + inner (𝕜 := ℝ) a p.2)

end Transformer.Homogenized
