import Transformer.Perceptron.Section5_SphericalMapCancellation

/-!
# Actual mixed spherical Hessian for matrix attention

ArXiv:2601.21366v2, §5.3, the analytic alternative of
`thm: any.d` (i), and §3.1, `rem: ext` (ii).
The full original energy along the actual Monge transport has
an exact constant-first-variation identity. Each of its remaining
three interaction integrals has two proved genuine derivatives.
Their original diagonal terms cancel after integration.

The resulting actual second derivative is exactly the integrated
mixed matrix curvature. Its factor two cancels the original
energy's factor one half. This supplies a genuine derivative input
to the original conditional strict-SOPD coercivity hypothesis.
Matrix symmetry is retained in the exact energy cancellation.
No density, atomic presentation or assumed Hessian identity is used.
The original primitive needs only continuity because the actual
constant first variation determines its transport integral exactly.
-/

open Real MeasureTheory
namespace Transformer.Perceptron

/-- The two original diagonal curvature integrals cancel, leaving twice
the actual mixed integral. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the
arbitrary-dimensional extension of `eq:Hess-bilinear-analytic`. -/
theorem integral_transportedMapPairCurvature_diagonal_cancel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (ξ : SSphere d → EucSpace d) (hcξ : Continuous ξ) :
    (∫ p, transportedMapPairCurvature B ξ ξ 0 p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
      (∫ p, transportedMapPairCurvature B ξ 0 0 p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
      (∫ p, transportedMapPairCurvature B 0 ξ 0 p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) =
      2 * (∫ p, sphereMapPairMixedCurvature B (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) := by
  have hI (η ζ : SSphere d → EucSpace d) (hη : Continuous η) (hζ : Continuous ζ) :
      Integrable (transportedMapPairCurvature B η ζ 0) ((μ : Measure (SSphere d)).prod μ) :=
    (continuous_transportedMapPairCurvature B η ζ hη hζ 0).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  rw [← integral_sub (hI ξ ξ hcξ hcξ) (hI ξ 0 hcξ continuous_const),
    ← integral_sub (f := fun p => transportedMapPairCurvature B ξ ξ 0 p -
      transportedMapPairCurvature B ξ 0 0 p)
      ((hI ξ ξ hcξ hcξ).sub (hI ξ 0 hcξ continuous_const))
      (hI 0 ξ continuous_const hcξ)]
  calc
    _ = ∫ p, 2 * sphereMapPairMixedCurvature B (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
      integral_congr_ae (ae_of_all _ fun p => sphereMapPairCurvature_diagonal_cancel B _ _ _ _)
    _ = _ := integral_const_mul _ _

/-- The actual full-energy second derivative is the mixed Hessian after
global first-variation cancellation. Source: arXiv:2601.21366v2,
§5.3, the Hessian used to contradict `eq:strict2order`.
Existence is proved by actual dominated differentiation of the original
energy; the curvature is not an assumed second-variation formula. -/
theorem hasDerivAt_deriv_energyMap_sphereTransport_of_constant_firstVariationMap {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) :
    HasDerivAt (deriv (fun t => energyMap B φ ω a (μ.map (sphereTransport ξ hξ t))))
      ((∫ p, sphereMapPairMixedCurvature B (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))) 0 := by
  let I (η ζ : SSphere d → EucSpace d) (t : ℝ) := ∫ p, transportedMapPairKernel B η ζ t p
    ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))
  have hzero : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) ((0 : SSphere d → EucSpace d) x) = 0 :=
    fun x => by simp
  have hI (η ζ : SSphere d → EucSpace d) (hη : Continuous η) (hζ : Continuous ζ)
      (htη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0)
      (htζ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ζ x) = 0) (t : ℝ) :=
    hasDerivAt_integral_transportedMapPairKernel B μ η ζ hη hζ htη htζ t
  have hE : (fun t => energyMap B φ ω a (μ.map (sphereTransport ξ hξ t))) =
      fun t => C + (2 : ℝ)⁻¹ * (I ξ ξ t - I ξ 0 t - I 0 ξ t) :=
    funext fun t => energyMap_sphereTransport_of_constant_firstVariationMap
      B hsymm φ hφ ω a μ C hC ξ hξ hcξ t
  have hfirst : deriv (fun t => energyMap B φ ω a (μ.map (sphereTransport ξ hξ t))) =
      fun t => (2 : ℝ)⁻¹ * (deriv (I ξ ξ) t - deriv (I ξ 0) t - deriv (I 0 ξ) t) := by
    rw [hE]
    funext t
    exact (((((hI ξ ξ hcξ hcξ hξ hξ t).differentiableAt.hasDerivAt.sub
      (hI ξ 0 hcξ continuous_const hξ hzero t).differentiableAt.hasDerivAt).sub
      (hI 0 ξ continuous_const hcξ hzero hξ t).differentiableAt.hasDerivAt).const_mul
      ((2 : ℝ)⁻¹)).const_add C).deriv
  rw [hfirst]
  have hD₁ := hasDerivAt_deriv_integral_transportedMapPairKernel B μ ξ ξ hcξ hcξ hξ hξ 0
  have hD₂ := hasDerivAt_deriv_integral_transportedMapPairKernel B μ ξ 0 hcξ continuous_const hξ hzero 0
  have hD₃ := hasDerivAt_deriv_integral_transportedMapPairKernel B μ 0 ξ continuous_const hcξ hzero hξ 0
  apply (((hD₁.sub hD₂).sub hD₃).const_mul ((2 : ℝ)⁻¹)).congr_deriv
  rw [integral_transportedMapPairCurvature_diagonal_cancel B μ ξ hcξ]
  ring

/-- Every original smooth-gradient direction has the actual mixed
second derivative along its original Monge pushforward.
Source: arXiv:2601.21366v2, §2.2, `eq:otto.tangent` and `eq:strict2order`,
and §5.3, the global-cancellation case of `thm: any.d` (i). -/
theorem hasDerivAt_deriv_energyMap_gradientTransport_of_constant_firstVariationMap {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d) (hξ : IsGradientField ξ) :
    HasDerivAt (deriv (fun t => energyMap B φ ω a
      (μ.map (sphereTransport ξ (inner_isGradientField hξ) t))))
      ((∫ p, sphereMapPairMixedCurvature B (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2)
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))) 0 :=
  hasDerivAt_deriv_energyMap_sphereTransport_of_constant_firstVariationMap B hsymm φ hφ ω a μ C hC
    ξ (inner_isGradientField hξ) (continuous_of_isGradientField ξ hξ)

/-- The same proved derivative gives the value of the actual iterated
second derivative along each original smooth-gradient transport.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3. -/
theorem secondDeriv_energyMap_gradientTransport_of_constant_firstVariationMap {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric)
    (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d) (hξ : IsGradientField ξ) :
    deriv^[2] (fun t => energyMap B φ ω a
      (μ.map (sphereTransport ξ (inner_isGradientField hξ) t))) 0 =
      ∫ p, sphereMapPairMixedCurvature B (p.1 : EucSpace d) (ξ p.1)
        (p.2 : EucSpace d) (ξ p.2) ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
  (hasDerivAt_deriv_energyMap_gradientTransport_of_constant_firstVariationMap
    B hsymm φ hφ ω a μ C hC ξ hξ).deriv

/-- The actual uniform probability and identity matrix supply the
constant first variation jointly with every primitive, symmetry,
continuity and tangency input. Source: arXiv:2601.21366v2,
§5.3, the global cancellation case, and §3.1, `rem: ext` (ii). -/
example : ∃ μ : Perspective.ProbSphere 2, ∃ C : ℝ,
    (LinearMap.id (R := ℝ) (M := EucSpace 2)).IsSymmetric ∧
    Continuous (fun _ : ℝ => (0 : ℝ)) ∧
    (∀ x : SSphere 2, firstVariationMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      0 (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2) μ (x : EucSpace 2) = C) ∧
    Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) ∧
    IsGradientField (0 : SSphere 2 → EucSpace 2) := by
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
  refine ⟨μ, C, (fun _ _ => rfl), continuous_const, ?_,
    continuous_circleTangent.comp continuous_subtype_val,
    (fun x => inner_circleTangent (x : EucSpace 2)), isGradientField_zero⟩
  intro x
  have hm := firstVariationMap_smul_id (d := 2) 1 one_ne_zero 0 0 0 μ (x : EucSpace 2)
  simp only [one_smul, inv_one, one_mul] at hm
  rw [hm]
  exact hC x

end Transformer.Perceptron
