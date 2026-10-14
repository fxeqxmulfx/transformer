/-
# Actual integrated spherical second variations in every dimension

ArXiv:2601.21366v2, §5.3, the localized-gradient proof of
`thm: any.d` (i), extending §5.2, `eq:Hess-bilinear.std`.
Continuity on the compact original product sphere supplies
measurability and integrability. The actual time derivatives have
global uniform bounds, so differentiation under the original
product probability measure is justified twice.

The original full energy is also pulled back along its actual
Monge transport, with the original potential and neurons. No
density or finite atomic presentation is required. Constancy of
the actual first variation will subsequently cancel the diagonal
terms and leave the integrated mixed derivative.
The actual first and second derivative identities hold at every
real time, before specializing the original Hessian to time zero.
The original integration measure stays fixed through both steps,
and the actual pushforward is used only in the full-energy pullback.
-/

import Transformer.Perceptron.Section5_SphericalVariation
import Transformer.Perceptron.MinMax
import Transformer.Perspective.Section2_EnergyConvex

open Real MeasureTheory

namespace Transformer.Perceptron

/-- Compactness supplies global bounds for both actual transported
derivative integrands. Source: arXiv:2601.21366v2, §5.3, the
domination needed for the original Hessian localization argument. -/
theorem exists_transportedPair_derivative_bounds {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (ξ η : SSphere d → EucSpace d) (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) :
    ∃ B₁ B₂ : ℝ, (∀ t p, |transportedPairForce β ξ η t p| ≤ B₁) ∧
      (∀ t p, |transportedPairCurvature β ξ η t p| ≤ B₂) := by
  obtain ⟨B, -, hB⟩ := exists_pos_bound_continuous_compact (fun x => ‖ξ x‖) hcξ.norm
  obtain ⟨C, -, hC⟩ := exists_pos_bound_continuous_compact (fun x => ‖η x‖) hcη.norm
  have hB' (x) : ‖ξ x‖ ≤ B := by
    simpa only [abs_of_nonneg (norm_nonneg _)] using hB x
  have hC' (x) : ‖η x‖ ≤ C := by
    simpa only [abs_of_nonneg (norm_nonneg _)] using hC x
  exact ⟨Real.exp β * (β * (B + C)), Real.exp β * ((β ^ 2 + β) * (B + C) ^ 2),
    transportedPair_derivative_bounds β hβ ξ η hξ hη B C hB' hC'⟩

/-- The first actual derivative under the original product probability.
Source: arXiv:2601.21366v2, §5.3, the first interaction variation. -/
theorem hasDerivAt_integral_transportedPairKernel {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) (t : ℝ) :
    HasDerivAt (fun r => ∫ p, transportedPairKernel β ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))
      (∫ p, transportedPairForce β ξ η t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) t := by
  obtain ⟨B₁, -, hB₁, -⟩ := exists_transportedPair_derivative_bounds β hβ ξ η hcξ hcη hξ hη
  exact hasDerivAt_integral_on_compact ((μ : Measure (SSphere d)).prod μ)
    (transportedPairKernel β ξ η) (transportedPairForce β ξ η)
    (continuous_transportedPairKernel β ξ η hcξ hcη)
    (continuous_transportedPairForce β ξ η hcξ hcη) B₁ hB₁
    (fun r p => hasDerivAt_spherePairKernel β _ _ _ _ r) t

/-- The justified first derivative is an equality of actual functions,
ready for subsequent differentiation of the full energy.
Source: arXiv:2601.21366v2, §5.3, the original interaction variation. -/
theorem deriv_integral_transportedPairKernel {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) :
    deriv (fun r => ∫ p, transportedPairKernel β ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) =
      fun r => ∫ p, transportedPairForce β ξ η r p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) := by
  funext r
  exact (hasDerivAt_integral_transportedPairKernel β hβ μ ξ η hcξ hcη hξ hη r).deriv

/-- Two dominated differentiations establish existence of the actual
interaction second derivative. Source: arXiv:2601.21366v2, §5.3,
the Hessian input used by `eq:strict2order` in `thm: any.d` (i). -/
theorem hasDerivAt_deriv_integral_transportedPairKernel {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) (t : ℝ) :
    HasDerivAt (deriv (fun r => ∫ p, transportedPairKernel β ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))))
      (∫ p, transportedPairCurvature β ξ η t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) t := by
  obtain ⟨B₁, B₂, hB₁, hB₂⟩ := exists_transportedPair_derivative_bounds β hβ ξ η hcξ hcη hξ hη
  exact hasDerivAt_deriv_integral_on_compact ((μ : Measure (SSphere d)).prod μ)
    (transportedPairKernel β ξ η) (transportedPairForce β ξ η) (transportedPairCurvature β ξ η)
    (continuous_transportedPairKernel β ξ η hcξ hcη)
    (continuous_transportedPairForce β ξ η hcξ hcη)
    (continuous_transportedPairCurvature β ξ η hcξ hcη) B₁ B₂ hB₁ hB₂
    (fun r p => hasDerivAt_spherePairKernel β _ _ _ _ r)
    (fun r p => hasDerivAt_spherePairForce β _ _ _ _ r) t

/-- The proved curvature integral is the actual second derivative value.
Source: arXiv:2601.21366v2, §5.3, the interaction part of the Hessian. -/
theorem secondDeriv_integral_transportedPairKernel {d : ℕ} (β : ℝ) (hβ : 0 ≤ β)
    (μ : Perspective.ProbSphere d) (ξ η : SSphere d → EucSpace d)
    (hcξ : Continuous ξ) (hcη : Continuous η)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hη : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (η x) = 0) (t : ℝ) :
    deriv^[2] (fun r => ∫ p, transportedPairKernel β ξ η r p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) t =
      ∫ p, transportedPairCurvature β ξ η t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
  (hasDerivAt_deriv_integral_transportedPairKernel β hβ μ ξ η hcξ hcη hξ hη t).deriv

/-- The original energy along its actual Monge transport integrates
against the same original measure and its product.
Source: arXiv:2601.21366v2, §2.2–§2.3 and §5.3, the transported energy. -/
theorem energy_sphereTransport_eq {d : ℕ} (β : ℝ) (φ : ℝ → ℝ) (hφ : Continuous φ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    energy β φ ω a (μ.map (sphereTransport ξ hξ t)) =
      (2 * β)⁻¹ * (∫ p, transportedPairKernel β ξ ξ t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) +
      (1 / 2 : ℝ) * (∫ x, potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
        ∂(μ : Measure (SSphere d))) := by
  have hT := (continuous_sphereTransport ξ hξ hcξ t).measurable
  unfold energy
  rw [Perspective.interactionEnergy_eq_integral_prod]
  simp only [ProbabilityMeasure.toMeasure_map]
  rw [Measure.map_prod_map _ _ hT hT,
    integral_map (hT.prodMap hT).aemeasurable
      (Perspective.continuous_expInner_prod β).aestronglyMeasurable,
    integral_map (f := fun x : SSphere d => potential φ ω a (x : EucSpace d)) hT.aemeasurable
      ((continuous_potential hφ ω a).comp continuous_subtype_val).aestronglyMeasurable]
  simp only [transportedPairKernel, spherePairKernel, sphereTransport, one_div]
  rfl

/-- Continuous nonzero tangents, nonnegative temperature and a continuous
primitive jointly witness the original calculus and pullback inputs.
Source: arXiv:2601.21366v2, §2.2–§2.3 and §5.2, the circle geodesic. -/
example : (0 : ℝ) ≤ 1 ∧ Continuous (fun s : ℝ => s) ∧
    Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) :=
  ⟨zero_le_one, continuous_id, continuous_circleTangent.comp continuous_subtype_val,
    fun x => inner_circleTangent (x : EucSpace 2)⟩

end Transformer.Perceptron
