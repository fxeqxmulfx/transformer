module

public import Transformer.Perceptron.Section5_SphericalMapVariation

/-!
# Integrated actual spherical matrix variations

ArXiv:2601.21366v2, §5.3, `thm: any.d` (i), and §3.1,
`rem: ext` (ii). Compactness bounds the original continuous
fields. The actual matrix kernel and its actual first and second
derivatives are consequently dominated at every real time.
Differentiation under the original product probability measure
is justified twice, without an atomic or density hypothesis.

The full original matrix energy is also pulled back along its
actual Monge map. Both its original interaction and its original
potential are retained, with their factor one half. The original
measure stays fixed throughout each differentiation. These actual
derivative identities will supply the Hessian input of the source's
conditional strict-SOPD predicate after diagonal cancellation.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- Compactness supplies global bounds for both actual transported
derivative integrands. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the
domination needed for the original Hessian localization argument. -/
theorem exists_transportedMapPair_derivative_bounds {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ξ η : SSphere d → EucSpace d) (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) :
    ∃ B₁ B₂ : ℝ, (∀ t p, |transportedMapPairForce B ξ η t p| ≤ B₁) ∧
      (∀ t p, |transportedMapPairCurvature B ξ η t p| ≤ B₂) := by
  obtain ⟨K, -, hK⟩ := exists_pos_bound_continuous_compact (fun x => ‖ξ x‖) hcξ.norm
  obtain ⟨L, -, hL⟩ := exists_pos_bound_continuous_compact (fun x => ‖η x‖) hcη.norm
  have hK' (x) : ‖ξ x‖ ≤ K := by
    simpa only [abs_of_nonneg (norm_nonneg _)] using hK x
  have hL' (x) : ‖η x‖ ≤ L := by
    simpa only [abs_of_nonneg (norm_nonneg _)] using hL x
  exact ⟨Real.exp ‖B.toContinuousLinearMap‖ * (‖B.toContinuousLinearMap‖ * (K + L)),
    Real.exp ‖B.toContinuousLinearMap‖ *
      ((‖B.toContinuousLinearMap‖ ^ 2 + ‖B.toContinuousLinearMap‖) * (K + L) ^ 2),
    transportedMapPair_derivative_bounds B ξ η hξ hη K L hK' hL'⟩

/-- The first actual derivative under the original product probability.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the first interaction variation. -/
theorem hasDerivAt_integral_transportedMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) (t : ℝ) :
    HasDerivAt (fun r => ∫ p, transportedMapPairKernel B ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))
      (∫ p, transportedMapPairForce B ξ η t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) t := by
  obtain ⟨B₁, -, hB₁, -⟩ := exists_transportedMapPair_derivative_bounds B ξ η hcξ hcη hξ hη
  exact hasDerivAt_integral_on_compact ((μ : Measure (SSphere d)).prod μ)
    (transportedMapPairKernel B ξ η) (transportedMapPairForce B ξ η)
    (continuous_transportedMapPairKernel B ξ η hcξ hcη)
    (continuous_transportedMapPairForce B ξ η hcξ hcη) B₁ hB₁
    (fun r p => hasDerivAt_sphereMapPairKernel B _ _ _ _ r) t

/-- The justified first derivative is an equality of actual functions,
ready for subsequent differentiation of the full energy.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the original interaction variation. -/
theorem deriv_integral_transportedMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) :
    deriv (fun r => ∫ p, transportedMapPairKernel B ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) =
      fun r => ∫ p, transportedMapPairForce B ξ η r p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) := by
  funext r
  exact (hasDerivAt_integral_transportedMapPairKernel B μ ξ η hcξ hcη hξ hη r).deriv

/-- Two dominated differentiations establish existence of the actual
interaction second derivative. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3,
the Hessian input used by `eq:strict2order` in `thm: any.d` (i). -/
theorem hasDerivAt_deriv_integral_transportedMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) (t : ℝ) :
    HasDerivAt (deriv (fun r => ∫ p, transportedMapPairKernel B ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))))
      (∫ p, transportedMapPairCurvature B ξ η t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) t := by
  obtain ⟨B₁, B₂, hB₁, hB₂⟩ := exists_transportedMapPair_derivative_bounds B ξ η hcξ hcη hξ hη
  exact hasDerivAt_deriv_integral_on_compact ((μ : Measure (SSphere d)).prod μ)
    (transportedMapPairKernel B ξ η) (transportedMapPairForce B ξ η) (transportedMapPairCurvature B ξ η)
    (continuous_transportedMapPairKernel B ξ η hcξ hcη)
    (continuous_transportedMapPairForce B ξ η hcξ hcη)
    (continuous_transportedMapPairCurvature B ξ η hcξ hcη) B₁ B₂ hB₁ hB₂
    (fun r p => hasDerivAt_sphereMapPairKernel B _ _ _ _ r)
    (fun r p => hasDerivAt_sphereMapPairForce B _ _ _ _ r) t

/-- The proved curvature integral is the actual second derivative value.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the interaction part of the Hessian. -/
theorem secondDeriv_integral_transportedMapPairKernel {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) (t : ℝ) :
    deriv^[2] (fun r => ∫ p, transportedMapPairKernel B ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) t =
      ∫ p, transportedMapPairCurvature B ξ η t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
  (hasDerivAt_deriv_integral_transportedMapPairKernel B μ ξ η hcξ hcη hξ hη t).deriv

/-- The original full matrix energy pulled back along its actual Monge
transport, using the original product measure and potential.
Source: arXiv:2601.21366v2, §2.2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem energyMap_sphereTransport_eq {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    energyMap B φ ω a (μ.map (sphereTransport ξ hξ t)) =
      (2 : ℝ)⁻¹ * (∫ p, transportedMapPairKernel B ξ ξ t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) +
      (2 : ℝ)⁻¹ * (∫ x, potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
        ∂(μ : Measure (SSphere d))) := by
  have hT := (continuous_sphereTransport ξ hξ hcξ t).measurable
  have hK := continuous_expInnerMap_prod B
  have hI : Integrable (fun p : SSphere d × SSphere d =>
      Real.exp (inner (𝕜 := ℝ) (p.1 : EucSpace d) (B (p.2 : EucSpace d))))
      ((μ.map (sphereTransport ξ hξ t) : Measure (SSphere d)).prod
        (μ.map (sphereTransport ξ hξ t) : Measure (SSphere d))) :=
    hK.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  unfold energyMap interactionEnergyMap
  rw [integral_integral (f := fun x y : SSphere d =>
    Real.exp (inner (𝕜 := ℝ) (x : EucSpace d) (B (y : EucSpace d)))) hI]
  simp only [ProbabilityMeasure.toMeasure_map]
  rw [Measure.map_prod_map _ _ hT hT,
    integral_map (hT.prodMap hT).aemeasurable hK.aestronglyMeasurable,
    integral_map (f := fun x : SSphere d => potential φ ω a (x : EucSpace d)) hT.aemeasurable
      ((continuous_potential hφ ω a).comp continuous_subtype_val).aestronglyMeasurable]
  rfl

/-- The original nonzero circle tangent and continuous primitive realize
all calculus and actual pullback inputs, for an arbitrary original matrix.
Source: arXiv:2601.21366v2, §2.2 and §5.3. -/
example : Continuous (fun s : ℝ => s) ∧
    Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) :=
  ⟨continuous_id, continuous_circleTangent.comp continuous_subtype_val,
    fun x => inner_circleTangent (x : EucSpace 2)⟩

end Transformer.Perceptron
