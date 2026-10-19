module

public import Transformer.Perceptron.GeneralAttention
public import Transformer.Perceptron.Section5_AnalyticStationarity

/-!
# The actual first variation for general matrix attention

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), §5.3, and
`rem:unifiedlog-B`. For symmetric B the source's first variation is
its actual exponential weight plus half of the original potential.
Its actual derivative is the original ambient matrix attention
integral plus the original neuron sum. Tangent projection gives
precisely the original stationary field, with its correct constants.

Analytic activation makes this actual field analytic. Positive
uniform measure of its stationary support makes it zero everywhere,
and the actual first variation is then constant on the sphere.
The continuation and constancy conclusions are derived from the
original stationarity, rather than supplied as additional premises.

The differential identities hold for the displayed expression even
without symmetry or invertibility; the source assumes both for its
energy interpretation. The scalar specialization retains the source's
weight re-scaling, because the two energy prefactors differ by β.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped BigOperators
namespace Transformer.Perceptron

/-- The source first-variation expression for symmetric matrix attention.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
noncomputable def firstVariationMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) : ℝ :=
  attentionWeightMap B μ x + (2 : ℝ)⁻¹ * potential φ ω a x

/-- The actual derivative retains both the original interaction and potential terms.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem hasFDerivAt_firstVariationMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    HasFDerivAt (firstVariationMap B φ ω a μ)
      (innerSL ℝ (ambientAttentionGradMap B μ x +
        ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j)) x := by
  have hZ := hasFDerivAt_attentionWeightMap B μ x
  rw [attentionCovectorMap_eq_innerSL_ambientGradMap] at hZ
  have hv := hasGradientAt_iff_hasFDerivAt.mp (hasGradientAt_potential φ σ hφ ω a x)
  apply (hZ.fun_add (hv.const_mul (2 : ℝ)⁻¹)).congr_fderiv
  ext v
  simp only [add_apply, smul_apply, innerSL_apply_apply, InnerProductSpace.toDual_apply_apply,
    inner_add_left, real_inner_smul_left, smul_eq_mul]
  ring

/-- The actual matrix stationary field is the projection of the computed ambient derivative.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem energyGradMap_eq_proj {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    energyGradMap B σ ω a μ x = proj d x (ambientAttentionGradMap B μ x +
      ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j) := by
  rw [energyGradMap_eq_attentionGradMap_add_drift, attentionGradMap_eq_proj_ambientGradMap, drift]
  unfold proj
  rw [inner_add_right, add_smul]
  abel

/-- The actual first variation is continuous because its actual derivative exists.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem continuous_firstVariationMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (φ σ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    Continuous (firstVariationMap B φ ω a μ) :=
  continuous_iff_continuousAt.mpr (fun x => (hasFDerivAt_firstVariationMap B φ σ hφ ω a μ x).continuousAt)

/-- The scalar specialization uses exactly the original energy prefactor and weight re-scaling.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem firstVariationMap_smul_id {d : ℕ} (β : ℝ) (hβ : β ≠ 0)
    (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    firstVariationMap (β • LinearMap.id) φ ω a μ x =
      β * firstVariation β φ (β⁻¹ • ω) a μ x := by
  simp only [firstVariationMap, firstVariation, attentionWeightMap, Perspective.partitionMu,
    LinearMap.smul_apply, LinearMap.id_apply, real_inner_smul_right, potential_smul]
  field_simp

/-- The actual ambient stationary field is analytic for an analytic original activation.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem analyticOnNhd_energyGradMap {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (σ : ℝ → ℝ) (hσ : AnalyticOnNhd ℝ σ univ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : AnalyticOnNhd ℝ (energyGradMap B σ ω a μ) univ := by
  change AnalyticOnNhd ℝ (fun x => attentionGradMap B μ x + drift σ ω a x) univ
  intro x hx
  exact (analyticOnNhd_attentionGradMap B μ x hx).fun_add (analyticOnNhd_drift σ hσ ω a x hx)

/-- The actual first variation is constant when its actual spherical gradient vanishes.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem firstVariationMap_eq_of_energyGrad_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ σ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hz : ∀ x : SSphere d, energyGradMap B σ ω a μ (x : EucSpace d) = 0) :
    ∀ x y : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) =
      firstVariationMap B φ ω a μ (y : EucSpace d) := by
  apply eq_on_sphere_of_projected_derivative_zero hd (firstVariationMap B φ ω a μ)
    (fun z => ambientAttentionGradMap B μ z +
      ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) z)) • a j)
    (hasFDerivAt_firstVariationMap B φ σ hφ ω a μ)
  intro z hnorm
  rw [← energyGradMap_eq_proj]
  exact hz ⟨z, mem_sphere_zero_iff_norm.mpr hnorm⟩

/-- Positive original uniform support measure forces the actual analytic stationary field to vanish globally.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem energyGradMap_zero_of_stationary_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (σ : ℝ → ℝ) (hσ : AnalyticOnNhd ℝ σ univ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationaryMap B σ ω a μ) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∀ x : SSphere d, energyGradMap B σ ω a μ (x : EucSpace d) = 0 := by
  apply vector_eq_zero_on_sphere_of_positive_zero_measure hd (energyGradMap B σ ω a μ)
    (analyticOnNhd_energyGradMap B σ hσ ω a μ) ν hν
  exact hpos.trans_le (measure_mono (fun x hx => hμ x hx))

/-- The actual first variation has a constant level after the proved global continuation step.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem exists_constant_firstVariationMap_of_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ σ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * σ s) s) (hσ : AnalyticOnNhd ℝ σ univ)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationaryMap B σ ω a μ) (ν : Measure (SSphere d))
    (hν : Metastability.IsUniformOn d ν) (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∃ C : ℝ, ∀ x : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) = C := by
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  refine ⟨firstVariationMap B φ ω a μ (x₀ : EucSpace d), ?_⟩
  exact fun x => firstVariationMap_eq_of_energyGrad_zero hd B φ σ hφ ω a μ
    (energyGradMap_zero_of_stationary_positive_uniform_support hd B σ hσ ω a μ hμ ν hν hpos) x x₀

/-- The actual uniform circle law, identity attention, and original zero
weights realize all primitive, analytic, stationary, full-support, and
positive-uniform-support hypotheses jointly. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (1 : ℝ) ≠ 0 ∧
    (∀ s, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s) ∧
    AnalyticOnNhd ℝ (0 : ℝ → ℝ) univ ∧
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) 0 0 0 μ ∧
    (μ : Measure (SSphere 2)).support = univ ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    0 < (MeanField.uniformLaw 2) (μ : Measure (SSphere 2)).support ∧
    (∀ x : SSphere 2, energyGradMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) 0 0 0 μ (x : EucSpace 2) = 0) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hprob : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  have hfull := support_eq_univ_of_uniform μ hν
  have hstat : IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) 0 0 0 μ :=
    (isStationaryMap_id_iff 0 0 0 μ).mpr (isStationary_zero_weights_of_uniform 1 one_ne_zero 0 0 μ hν)
  refine ⟨μ, le_rfl, one_ne_zero, fun s => by simpa using hasDerivAt_const s (0 : ℝ),
    fun _ _ => analyticAt_const, hstat, hfull, hν, ?_, ?_⟩
  · rw [hfull]
    simp
  · intro x
    exact hstat x (by rw [hfull]; exact mem_univ x)

end Transformer.Perceptron
