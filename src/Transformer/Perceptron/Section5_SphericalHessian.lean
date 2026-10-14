/-
# Actual mixed spherical Hessian after global cancellation

ArXiv:2601.21366v2, §5.3, the analytic alternative in
`thm: any.d` (i), extending §5.2, `eq:Hess-bilinear-analytic`.
The original energy along the actual exponential transport is
expressed using its actual constant first variation. Each of the
three remaining interaction integrals has two genuine derivatives.

Integrating the pointwise diagonal cancellation consequently gives
an actual second derivative of the original full energy, equal to
the original mixed kernel integral divided by the temperature.
The result also applies to each original smooth spherical gradient
field, whose continuity and tangency have already been proved.
Thus the conditional strict-SOPD inequality has a proved derivative
input before the localization argument uses it.
-/

import Transformer.Perceptron.Section5_SphericalCancellation

open Real MeasureTheory

namespace Transformer.Perceptron

/-- The actual mixed kernel coefficient is continuous in the original
pair for a continuous field. Source: arXiv:2601.21366v2, §5.3,
the mixed Hessian integrability after global cancellation. -/
theorem continuous_spherePairMixedCurvature_field {d : ℕ} (β : ℝ)
    (ξ : SSphere d → EucSpace d) (hcξ : Continuous ξ) :
    Continuous (fun p : SSphere d × SSphere d => spherePairMixedCurvature β
      (p.1 : EucSpace d) (ξ p.1) (p.2 : EucSpace d) (ξ p.2)) := by
  have hfst : Continuous (Prod.fst : SSphere d × SSphere d → SSphere d) := continuous_fst
  have hsnd : Continuous (Prod.snd : SSphere d × SSphere d → SSphere d) := continuous_snd
  have hX := continuous_subtype_val.comp hfst
  have hY := continuous_subtype_val.comp hsnd
  have hV := hcξ.comp hfst
  have hW := hcξ.comp hsnd
  have h₁ : Continuous (fun p : SSphere d × SSphere d =>
      inner (𝕜 := ℝ) (ξ p.1) (p.2 : EucSpace d)) := hV.inner hY
  have h₂ : Continuous (fun p : SSphere d × SSphere d =>
      inner (𝕜 := ℝ) (p.1 : EucSpace d) (ξ p.2)) := hX.inner hW
  exact (Perspective.continuous_expInner_prod β).mul
    (((h₁.const_mul (β ^ 2)).mul h₂).add ((hV.inner hW).const_mul β))

/-- The two original diagonal curvature integrals cancel, leaving twice
the actual mixed integral. Source: arXiv:2601.21366v2, §5.3, the
arbitrary-dimensional extension of `eq:Hess-bilinear-analytic`. -/
theorem integral_transportedPairCurvature_diagonal_cancel {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (ξ : SSphere d → EucSpace d) (hcξ : Continuous ξ) :
    (∫ p, transportedPairCurvature β ξ ξ 0 p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
      (∫ p, transportedPairCurvature β ξ 0 0 p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
      (∫ p, transportedPairCurvature β 0 ξ 0 p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) =
      2 * (∫ p, spherePairMixedCurvature β (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) := by
  have hI (η ζ : SSphere d → EucSpace d) (hη : Continuous η) (hζ : Continuous ζ) :
      Integrable (transportedPairCurvature β η ζ 0) ((μ : Measure (SSphere d)).prod μ) :=
    (continuous_transportedPairCurvature β η ζ hη hζ 0).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  rw [← integral_sub (hI ξ ξ hcξ hcξ) (hI ξ 0 hcξ continuous_const),
    ← integral_sub (f := fun p => transportedPairCurvature β ξ ξ 0 p -
      transportedPairCurvature β ξ 0 0 p)
      ((hI ξ ξ hcξ hcξ).sub (hI ξ 0 hcξ continuous_const))
      (hI 0 ξ continuous_const hcξ)]
  calc
    _ = ∫ p, 2 * spherePairMixedCurvature β (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
      integral_congr_ae (ae_of_all _ fun p => spherePairCurvature_diagonal_cancel β _ _ _ _)
    _ = _ := integral_const_mul _ _

/-- The actual full-energy second derivative is the mixed Hessian after
global first-variation cancellation. Source: arXiv:2601.21366v2,
§5.3, the Hessian used to contradict `eq:strict2order`.
Existence is proved by actual dominated differentiation of the original
energy; the curvature is not an assumed second-variation formula. -/
theorem hasDerivAt_deriv_energy_sphereTransport_of_constant_firstVariation {d : ℕ}
    (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) :
    HasDerivAt (deriv (fun t => energy β φ ω a (μ.map (sphereTransport ξ hξ t))))
      (β⁻¹ * (∫ p, spherePairMixedCurvature β (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))) 0 := by
  let I (η ζ : SSphere d → EucSpace d) (t : ℝ) := ∫ p, transportedPairKernel β η ζ t p
    ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))
  have hzero : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) ((0 : SSphere d → EucSpace d) x) = 0 :=
    fun x => by simp
  have hI (η ζ : SSphere d → EucSpace d) (hη : Continuous η) (hζ : Continuous ζ)
      (htη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0)
      (htζ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ζ x) = 0) (t : ℝ) :=
    hasDerivAt_integral_transportedPairKernel β hβ.le μ η ζ hη hζ htη htζ t
  have hE : (fun t => energy β φ ω a (μ.map (sphereTransport ξ hξ t))) =
      fun t => C + (2 * β)⁻¹ * (I ξ ξ t - I ξ 0 t - I 0 ξ t) :=
    funext fun t => energy_sphereTransport_of_constant_firstVariation
      β hβ.ne' φ hφ ω a μ C hC ξ hξ hcξ t
  have hfirst : deriv (fun t => energy β φ ω a (μ.map (sphereTransport ξ hξ t))) =
      fun t => (2 * β)⁻¹ * (deriv (I ξ ξ) t - deriv (I ξ 0) t - deriv (I 0 ξ) t) := by
    rw [hE]
    funext t
    exact (((((hI ξ ξ hcξ hcξ hξ hξ t).differentiableAt.hasDerivAt.sub
      (hI ξ 0 hcξ continuous_const hξ hzero t).differentiableAt.hasDerivAt).sub
      (hI 0 ξ continuous_const hcξ hzero hξ t).differentiableAt.hasDerivAt).const_mul
      ((2 * β)⁻¹)).const_add C).deriv
  rw [hfirst]
  have hD₁ := hasDerivAt_deriv_integral_transportedPairKernel β hβ.le μ ξ ξ hcξ hcξ hξ hξ 0
  have hD₂ := hasDerivAt_deriv_integral_transportedPairKernel β hβ.le μ ξ 0 hcξ continuous_const hξ hzero 0
  have hD₃ := hasDerivAt_deriv_integral_transportedPairKernel β hβ.le μ 0 ξ continuous_const hcξ hzero hξ 0
  apply (((hD₁.sub hD₂).sub hD₃).const_mul ((2 * β)⁻¹)).congr_deriv
  rw [integral_transportedPairCurvature_diagonal_cancel β μ ξ hcξ]
  field_simp

/-- Every original smooth-gradient direction has the actual mixed
second derivative along its original Monge pushforward.
Source: arXiv:2601.21366v2, §2.2, `eq:otto.tangent` and `eq:strict2order`,
and §5.3, the global-cancellation case of `thm: any.d` (i). -/
theorem hasDerivAt_deriv_energy_gradientTransport_of_constant_firstVariation {d : ℕ}
    (β : ℝ) (hβ : 0 < β) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d) (hξ : IsGradientField ξ) :
    HasDerivAt (deriv (fun t => energy β φ ω a
      (μ.map (sphereTransport ξ (inner_isGradientField hξ) t))))
      (β⁻¹ * (∫ p, spherePairMixedCurvature β (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))) 0 :=
  hasDerivAt_deriv_energy_sphereTransport_of_constant_firstVariation β hβ φ hφ ω a μ C hC
    ξ (inner_isGradientField hξ) (continuous_of_isGradientField ξ hξ)

/-- The original uniform probability with zero weights gives an actual
constant first variation. A nonzero continuous tangent realizes the
general formula, and the zero smooth gradient realizes the gradient
specialization. Source: arXiv:2601.21366v2, §5.3, global cancellation. -/
example : ∃ μ : Perspective.ProbSphere 2, ∃ C : ℝ, (0 : ℝ) < 1 ∧
    Continuous (fun _ : ℝ => (0 : ℝ)) ∧
    (∀ x : SSphere 2, firstVariation 1 0 (0 : Idx 2 → ℝ)
      (0 : Idx 2 → EucSpace 2) μ (x : EucSpace 2) = C) ∧
    Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) ∧ IsGradientField (0 : SSphere 2 → EucSpace 2) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hprob : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  have hpos : 0 < (MeanField.uniformLaw 2) (μ : Measure (SSphere 2)).support := by
    rw [support_eq_univ_of_uniform μ hν]
    simp
  obtain ⟨C, hC⟩ := exists_constant_firstVariation_of_positive_uniform_support le_rfl
    1 one_ne_zero 0 0 (fun s => by simp)
    (fun _ _ => analyticAt_const) 0 0 μ
    (isStationary_zero_weights_of_uniform 1 one_ne_zero 0 0 μ hν) _ hν hpos
  exact ⟨μ, C, one_pos, continuous_const, hC,
    continuous_circleTangent.comp continuous_subtype_val,
    (fun x => inner_circleTangent (x : EucSpace 2)), isGradientField_zero⟩

end Transformer.Perceptron
