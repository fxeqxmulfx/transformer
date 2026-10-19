module

public import Transformer.Perceptron.Section5_SphericalMapInteraction

/-!
# Exact cancellation for the original transported matrix energy

ArXiv:2601.21366v2, §5.3, the constant first variation in
`thm: any.d` (i), and §3.1, `rem: ext` (ii).
Constancy of the actual first variation determines the potential
integral along each original Monge transport. The original matrix
is retained inside each original interaction integral.

Matrix symmetry and product-measure swap identify the two actual
one-point transport terms. Substituting into the original energy
leaves exactly their subtraction from the two-point transport,
with the original factor one half. This is an identity at every
real time; no second derivative of the primitive is assumed.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Perceptron

/-- Symmetry and swapping the original product measure identify the two
one-point transport terms. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the
diagonal cancellation, extending the symmetric expansion in §5.2. -/
theorem integral_transportedMapPairKernel_one_point_swap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (hsymm : B.IsSymmetric)
    (μ : Perspective.ProbSphere d) (ξ : SSphere d → EucSpace d) (t : ℝ) :
    (∫ p, transportedMapPairKernel B ξ 0 t p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) =
      ∫ p, transportedMapPairKernel B 0 ξ t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) := by
  calc
    _ = ∫ p, transportedMapPairKernel B 0 ξ t p.swap
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)) :=
      integral_congr_ae (ae_of_all _ fun p => sphereMapPairKernel_swap B hsymm _ _ _ _ t)
    _ = _ := integral_prod_swap _

/-- An actual constant first variation fixes the transported potential
integral exactly. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, `H` constant
and the diagonal cancellation used in `thm: any.d` (i). -/
theorem potential_integral_of_constant_firstVariationMap {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    (2 : ℝ)⁻¹ * (∫ x, potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
      ∂(μ : Measure (SSphere d))) = C - (∫ p, transportedMapPairKernel B ξ 0 t p
          ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) := by
  have hI : Integrable (transportedMapPairKernel B ξ 0 t)
      ((μ : Measure (SSphere d)).prod μ) :=
    (continuous_transportedMapPairKernel B ξ 0 hcξ continuous_const t).integrable_of_hasCompactSupport
      (HasCompactSupport.of_compactSpace _)
  have hP : Integrable (fun x => potential φ ω a (sphereTransport ξ hξ t x : EucSpace d))
      (μ : Measure (SSphere d)) :=
    ((continuous_potential hφ ω a).comp (continuous_sphereTransport ξ hξ hcξ t).subtype_val)
      |>.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hpoint (x : SSphere d) : (∫ y, transportedMapPairKernel B ξ 0 t (x, y)
      ∂(μ : Measure (SSphere d))) +
      (2 : ℝ)⁻¹ * potential φ ω a (sphereTransport ξ hξ t x : EucSpace d) = C := by
    simpa only [firstVariationMap, attentionWeightMap, transportedMapPairKernel, sphereMapPairKernel,
      spherePosition_zero_direction, sphereTransport, Pi.zero_apply, one_div] using
      hC (sphereTransport ξ hξ t x)
  have hsum : (∫ p, transportedMapPairKernel B ξ 0 t p
      ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) +
      (2 : ℝ)⁻¹ * (∫ x, potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
        ∂(μ : Measure (SSphere d))) = C := by
    calc
      _ = ∫ x, (∫ y, transportedMapPairKernel B ξ 0 t (x, y)
          ∂(μ : Measure (SSphere d))) +
          (2 : ℝ)⁻¹ * potential φ ω a (sphereTransport ξ hξ t x : EucSpace d)
          ∂(μ : Measure (SSphere d)) := by
        rw [integral_add hI.integral_prod_left (hP.const_mul _),
          integral_const_mul,
          integral_integral (f := fun x y => transportedMapPairKernel B ξ 0 t (x, y)) hI]
      _ = C := by simp only [hpoint, integral_const]; simp
  linarith

/-- The original full energy equals its constant first-variation level
plus the original interaction with both one-point terms removed.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the cancellation preceding the
localized Hessian contradiction in `thm: any.d` (i). -/
theorem energyMap_sphereTransport_of_constant_firstVariationMap {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    energyMap B φ ω a (μ.map (sphereTransport ξ hξ t)) = C + (2 : ℝ)⁻¹ *
      ((∫ p, transportedMapPairKernel B ξ ξ t p
        ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
        (∫ p, transportedMapPairKernel B ξ 0 t p
          ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d))) -
        (∫ p, transportedMapPairKernel B 0 ξ t p
          ∂(μ : Measure (SSphere d)).prod (μ : Measure (SSphere d)))) := by
  rw [energyMap_sphereTransport_eq B φ hφ ω a μ ξ hξ hcξ t,
    potential_integral_of_constant_firstVariationMap B φ hφ ω a μ C hC ξ hξ hcξ t,
    ← integral_transportedMapPairKernel_one_point_swap B hsymm μ ξ t]
  ring

/-- Subtracting the actual initial energy leaves the original double
transport difference. Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), and §5.3, the global
first-variation cancellation used for the mixed Hessian.
The original energy and probability are retained at both times. -/
theorem energyMap_sphereTransport_sub_initial_of_constant_firstVariationMap {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hsymm : B.IsSymmetric) (φ : ℝ → ℝ) (hφ : Continuous φ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (C : ℝ)
    (hC : ∀ x : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) = C)
    (ξ : SSphere d → EucSpace d)
    (hξ : ∀ x : SSphere d, inner (𝕜 := ℝ) (x : EucSpace d) (ξ x) = 0)
    (hcξ : Continuous ξ) (t : ℝ) :
    energyMap B φ ω a (μ.map (sphereTransport ξ hξ t)) - energyMap B φ ω a μ = (2 : ℝ)⁻¹ *
      ((∫ p, transportedMapPairKernel B ξ ξ t p ∂(μ : Measure (SSphere d)).prod μ) -
        (∫ p, transportedMapPairKernel B ξ 0 t p ∂(μ : Measure (SSphere d)).prod μ) -
        (∫ p, transportedMapPairKernel B 0 ξ t p ∂(μ : Measure (SSphere d)).prod μ) +
        (∫ p, transportedMapPairKernel B 0 0 0 p ∂(μ : Measure (SSphere d)).prod μ)) := by
  have hinit := energyMap_sphereTransport_of_constant_firstVariationMap B hsymm φ hφ ω a μ C hC ξ hξ hcξ 0
  have hT : sphereTransport ξ hξ 0 = id := funext (sphereTransport_zero ξ hξ)
  have hmap : μ.map id = μ := by
    apply ProbabilityMeasure.toMeasure_injective
    simp only [ProbabilityMeasure.toMeasure_map, Measure.map_id]
  have hK (η ζ : SSphere d → EucSpace d) :
      transportedMapPairKernel B η ζ 0 = transportedMapPairKernel B 0 0 0 := by
    funext p
    simp only [transportedMapPairKernel, sphereMapPairKernel, spherePosition_zero]
  rw [hT, hmap, hK ξ ξ, hK ξ 0, hK 0 ξ] at hinit
  rw [energyMap_sphereTransport_of_constant_firstVariationMap B hsymm φ hφ ω a μ C hC ξ hξ hcξ t, hinit]
  ring

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
  refine ⟨μ, C, (fun _ _ => rfl), continuous_const, ?_,
    continuous_circleTangent.comp continuous_subtype_val,
    fun x => inner_circleTangent (x : EucSpace 2)⟩
  intro x
  have hm := firstVariationMap_smul_id (d := 2) 1 one_ne_zero 0 0 0 μ (x : EucSpace 2)
  simp only [one_smul, inv_one, one_mul] at hm
  rw [hm]
  exact hC x

end Transformer.Perceptron
