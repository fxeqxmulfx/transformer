module

public import Transformer.Perceptron.Section5_SphericalInteraction
public import Transformer.Perceptron.Section5_AnalyticStationarity

/-!
# Exact cancellation in the original transported spherical energy

ArXiv:2601.21366v2, §5.3, `H` constant in the analytic alternative
of `thm: any.d` (i), extending §5.2, `eq:global-cancel-analytic`.
Constancy of the actual first variation supplies an exact identity
for the potential along every actual Monge transport. Fubini
expresses its interaction term against the original product measure.

Substituting that identity into the original energy removes both
one-point interaction terms. The remaining two-point expression
has genuine second derivatives by `Section5_SphericalInteraction`.
This avoids requiring a separate second derivative of the original
primitive: the exact constant first-variation identity supplies it.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- Symmetry and swapping the original product measure identify the two
one-point transport terms. Source: arXiv:2601.21366v2, §5.3, the
diagonal cancellation, extending the symmetric expansion in §5.2. -/
theorem integral_transportedPairKernel_one_point_swap {d : ℕ} (β : ℝ)
    (μ : Perspective.ProbSphere d) (ξ : SSphere d → EucSpace d) (t : ℝ) :
    (∫ p, transportedPairKernel β ξ 0 t p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) =
      ∫ p, transportedPairKernel β 0 ξ t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) := by
  calc
    _ = ∫ p, transportedPairKernel β 0 ξ t p.swap
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
      integral_congr_ae (ae_of_all _ fun p => spherePairKernel_swap β _ _ _ _ t)
    _ = _ := integral_prod_swap _

/-- An actual constant first variation fixes the transported potential
integral exactly. Source: arXiv:2601.21366v2, §5.3, `H` constant
and the diagonal cancellation used in `thm: any.d` (i). -/
theorem potential_integral_of_constant_firstVariation {d : ℕ}
    (β : ℝ) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    (1 / 2 : ℝ) * (∫ x, potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
      ∂(μ : Measure (SSphere d))) = C - β⁻¹ *
        (∫ p, transportedPairKernel β ξ 0 t p
          ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) := by
  have hI : Integrable (transportedPairKernel β ξ 0 t)
      ((μ : Measure (SSphere d)).prod μ) :=
    (continuous_transportedPairKernel β ξ 0 hcξ continuous_const t).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hP : Integrable (fun x => potential φ ω a (sphereTransport ξ hξ t x : EucSpace d))
      (μ : Measure (SSphere d)) :=
    ((continuous_potential hφ ω a).comp (continuous_sphereTransport ξ hξ hcξ t).subtype_val)
      |>.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hpoint (x : SSphere d) : β⁻¹ * (∫ y, transportedPairKernel β ξ 0 t (x, y)
      ∂(μ : Measure (SSphere d))) +
      (1 / 2 : ℝ) * potential φ ω a (sphereTransport ξ hξ t x : EucSpace d) = C := by
    simpa only [firstVariation, Perspective.partitionMu, transportedPairKernel, spherePairKernel,
      spherePosition_zero_direction, sphereTransport, Pi.zero_apply, one_div] using
      hC (sphereTransport ξ hξ t x)
  have hsum : β⁻¹ * (∫ p, transportedPairKernel β ξ 0 t p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) +
      (1 / 2 : ℝ) * (∫ x, potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
        ∂(μ : Measure (SSphere d))) = C := by
    calc
      _ = ∫ x, β⁻¹ * (∫ y, transportedPairKernel β ξ 0 t (x, y)
          ∂(μ : Measure (SSphere d))) +
          (1 / 2 : ℝ) * potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
          ∂(μ : Measure (SSphere d)) := by
        rw [integral_add (hI.integral_prod_left.const_mul _) (hP.const_mul _),
          integral_const_mul, integral_const_mul,
          integral_integral (f := fun x y => transportedPairKernel β ξ 0 t (x, y)) hI]
      _ = C := by simp only [hpoint, integral_const]; simp
  linarith

/-- The original full energy equals its constant first-variation level
plus the original interaction with both one-point terms removed.
Source: arXiv:2601.21366v2, §5.3, the cancellation preceding the
localized Hessian contradiction in `thm: any.d` (i). -/
theorem energy_sphereTransport_of_constant_firstVariation {d : ℕ}
    (β : ℝ) (hβ : β ≠ 0) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    energy β φ ω a (μ.map (sphereTransport ξ hξ t)) = C + (2 * β)⁻¹ *
      ((∫ p, transportedPairKernel β ξ ξ t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
        (∫ p, transportedPairKernel β ξ 0 t p
          ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
        (∫ p, transportedPairKernel β 0 ξ t p
          ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))) := by
  rw [energy_sphereTransport_eq β φ hφ ω a μ ξ hξ hcξ t,
    potential_integral_of_constant_firstVariation β φ hφ ω a μ C hC ξ hξ hcξ t,
    ← integral_transportedPairKernel_one_point_swap β μ ξ t]
  have hcoeff : β⁻¹ = 2 * (2 * β)⁻¹ := by field_simp
  rw [hcoeff]
  ring

/-- Subtracting the actual initial energy leaves the original double
transport difference. Source: arXiv:2601.21366v2, §5.3, the global
first-variation cancellation used for the mixed Hessian.
The original energy and probability are retained at both times. -/
theorem energy_sphereTransport_sub_initial_of_constant_firstVariation {d : ℕ}
    (β : ℝ) (hβ : β ≠ 0) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariation β φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    energy β φ ω a (μ.map (sphereTransport ξ hξ t)) - energy β φ ω a μ = (2 * β)⁻¹ *
      ((∫ p, transportedPairKernel β ξ ξ t p ∂(μ : Measure (SSphere d)).prod μ) -
        (∫ p, transportedPairKernel β ξ 0 t p ∂(μ : Measure (SSphere d)).prod μ) -
        (∫ p, transportedPairKernel β 0 ξ t p ∂(μ : Measure (SSphere d)).prod μ) +
        (∫ p, transportedPairKernel β 0 0 0 p ∂(μ : Measure (SSphere d)).prod μ)) := by
  have hinit := energy_sphereTransport_of_constant_firstVariation β hβ φ hφ ω a μ C hC ξ hξ hcξ 0
  have hT : sphereTransport ξ hξ 0 = id := funext (sphereTransport_zero ξ hξ)
  have hmap : μ.map id = μ := by
    apply ProbabilityMeasure.toMeasure_injective
    simp only [ProbabilityMeasure.toMeasure_map, Measure.map_id]
  have hK (η ζ : SSphere d → EucSpace d) :
      transportedPairKernel β η ζ 0 = transportedPairKernel β 0 0 0 := by
    funext p
    simp only [transportedPairKernel, spherePairKernel, spherePosition_zero]
  rw [hT, hmap, hK ξ ξ, hK ξ 0, hK 0 ξ] at hinit
  rw [energy_sphereTransport_of_constant_firstVariation β hβ φ hφ ω a μ C hC ξ hξ hcξ t, hinit]
  ring

/-- The actual uniform probability with zero perceptron weights supplies
an actual constant first variation, jointly with every continuity,
tangency and temperature input of these exact energy identities.
Source: arXiv:2601.21366v2, §5.3, the global cancellation case. -/
example : ∃ μ : Perspective.ProbSphere 2, ∃ C : ℝ, (1 : ℝ) ≠ 0 ∧
    Continuous (fun _ : ℝ => (0 : ℝ)) ∧
    (∀ x : SSphere 2, firstVariation 1 0 (0 : Idx 2 → ℝ)
      (0 : Idx 2 → EucSpace 2) μ (x : EucSpace 2) = C) ∧
    Continuous (fun x : SSphere 2 => circleTangent (x : EucSpace 2)) ∧
    (∀ x : SSphere 2, inner (𝕜 := ℝ) (x : EucSpace 2)
      (circleTangent (x : EucSpace 2)) = 0) := by
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
  exact ⟨μ, C, one_ne_zero, continuous_const, hC,
    continuous_circleTangent.comp continuous_subtype_val,
    fun x => inner_circleTangent (x : EucSpace 2)⟩

end Transformer.Perceptron
