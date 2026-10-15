import Transformer.Perceptron.Section5_MapFirstVariation
import Transformer.Perceptron.Section5_ReLUBranches
import Transformer.Perceptron.TransformMap

/-!
# Actual unnormalized matrix ReLU branch continuation and parity

ArXiv:2601.21366v2, §5.2–§5.3, `eq:K-equals-quad-global` and
`eq:global-Id`, together with §3.1, `rem: ext` (ii), and §6,
`rem:unifiedlog-B`. On an original activation pattern the actual
ReLU field equals the linear field with its inactive weights zero.
The continued field is analytic for the original attention matrix.
Positive original uniform measure of its stationary support pattern
therefore forces that actual branch to vanish on the whole sphere.

The actual matrix first variation makes the corresponding finite
quadratic expression constant. The original exponential transform
then has an explicit constant-minus-squares form and is even.
The proved matrix transform injectivity recovers antipodal invariance
of the original probability measure whenever the matrix is invertible.

These steps retain all original neurons, zero neurons and activation
boundaries. They use no scalar inversion for the matrix transform.
The source's symmetry assumption is unnecessary for these displayed
field identities and parity; invertibility enters only at injectivity.
-/

open Real MeasureTheory Set
open scoped BigOperators
namespace Transformer.Perceptron

/-- The actual original matrix ReLU field equals its masked linear branch.
Source: arXiv:2601.21366v2, §5.3, `eq:suppmu.zero.g`, and `rem: ext` (ii). -/
theorem energyGradMap_linear_masked_eq_relu_at_pattern {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) (x : EucSpace d)
    (hx : IsActivationPattern a J x) :
    energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ x =
      energyGradMap B (fun s => max s 0) ω a μ x := by
  rw [energyGradMap_eq_attentionGradMap_add_drift, energyGradMap_eq_attentionGradMap_add_drift,
    drift_linear_masked_eq_projected_branch, drift_relu_eq_projected_branchVector ω a J x hx]

/-- The actual continued matrix branch is analytic on the ambient space.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem analyticOnNhd_energyGradMap_linear_masked {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) :
    AnalyticOnNhd ℝ (energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ) univ :=
  analyticOnNhd_energyGradMap B (fun s => s) (fun _ _ => analyticAt_id) _ a μ

/-- Positive original uniform measure of one stationary pattern gives
an actual globally zero branch. Source: arXiv:2601.21366v2, §5.3,
the continuation preceding `eq:global-Id`, and `rem: ext` (ii). -/
theorem energyGradMap_linear_masked_zero_of_positive_pattern_support {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (J : Finset (Idx d)) (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν ((μ : Measure (SSphere d)).support ∩
      {x : SSphere d | IsActivationPattern a J (x : EucSpace d)})) :
    ∀ x : SSphere d, energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0 := by
  apply vector_eq_zero_on_sphere_of_positive_zero_measure hd _
    (analyticOnNhd_energyGradMap_linear_masked B ω a μ J) ν hν
  apply hpos.trans_le (measure_mono _)
  intro x hx
  change energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0
  rw [energyGradMap_linear_masked_eq_relu_at_pattern B ω a μ J _ hx.2]
  exact hμ x hx.1

/-- The finite original pattern cover supplies a globally zero branch
if the original stationary support has positive uniform measure.
Source: arXiv:2601.21366v2, §5.3 and §3.1, `rem: ext` (ii). -/
theorem exists_zero_linear_masked_of_stationaryMap_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∃ J : Finset (Idx d), ∀ x : SSphere d,
      energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0 := by
  obtain ⟨J, hJ⟩ := exists_positive_measure_activationPattern ν (μ : Measure (SSphere d)).support hpos a
  exact ⟨J, energyGradMap_linear_masked_zero_of_positive_pattern_support hd B ω a μ hμ J ν hν hJ⟩

/-- The masked square primitive is exactly the original finite branch sum.
Source: arXiv:2601.21366v2, §5.3, `eq:quad.pol.I`. -/
theorem potential_sq_masked_eq_sum {d : ℕ} (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (J : Finset (Idx d)) (x : EucSpace d) :
    potential (fun s : ℝ => s ^ 2) (branchMaskedWeights ω J) a x =
      ∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) x ^ 2 := by
  simp only [potential, branchMaskedWeights, ite_mul, zero_mul]
  exact Fintype.sum_ite_mem J _

/-- The actual constant first variation gives the source's explicit
matrix transform identity, with the original factor one half.
Source: arXiv:2601.21366v2, §5.3, `eq:global-Id`, and `rem: ext` (ii). -/
theorem exists_const_sub_squares_transformMap_of_zero_linear_masked {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d,
      energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0) :
    ∃ C : ℝ, ∀ x : SSphere d, attentionTransformMap B μ x =
      C - (2 : ℝ)⁻¹ * ∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2 := by
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  let C := firstVariationMap B (fun s => s ^ 2) (branchMaskedWeights ω J) a μ (x₀ : EucSpace d)
  refine ⟨C, fun x => ?_⟩
  have h := firstVariationMap_eq_of_energyGrad_zero hd B (fun s => s ^ 2) (fun s => s)
    (fun s => by simpa using hasDerivAt_pow 2 s) (branchMaskedWeights ω J) a μ hzero x x₀
  change attentionTransformMap B μ x + (2 : ℝ)⁻¹ *
    potential (fun s : ℝ => s ^ 2) (branchMaskedWeights ω J) a (x : EucSpace d) = C at h
  rw [potential_sq_masked_eq_sum] at h
  linarith

/-- The original transform is even after the proved branch integration.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`, unnormalized case. -/
theorem even_attentionTransformMap_of_zero_linear_masked {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d,
      energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0) :
    ∀ x : SSphere d, attentionTransformMap B μ (antipodeMap d x) = attentionTransformMap B μ x := by
  obtain ⟨C, hC⟩ := exists_const_sub_squares_transformMap_of_zero_linear_masked hd B ω a μ J hzero
  intro x
  simp only [hC, coe_antipodeMap, inner_neg_right, neg_sq]

/-- The original stationary ReLU probability is antipodal if its
original support has positive uniform measure. Source: arXiv:2601.21366v2,
§6, `rem:unifiedlog-B`, using §5.1, `rem: general-attention`, parity. -/
theorem antipodal_of_stationaryMap_positive_uniform_support {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hB : Function.Bijective B)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)) := by
  obtain ⟨J, hJ⟩ := exists_zero_linear_masked_of_stationaryMap_positive_uniform_support hd B ω a μ hμ ν hν hpos
  exact (even_attentionTransformMap_iff B hB μ).mp
    (even_attentionTransformMap_of_zero_linear_masked hd B ω a μ J hJ)

/-- Identity attention and the original uniform law with zero weights
jointly realize every pattern, global-zero, stationarity, invertibility
and positive-uniform-measure input. The zero neurons remain in the cover.
Source: arXiv:2601.21366v2, §5.3, the uniform stationary case. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => max s 0) (0 : Idx 2 → ℝ) (0 : Idx 2 → EucSpace 2) μ ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    0 < (MeanField.uniformLaw 2) ((μ : Measure (SSphere 2)).support ∩
      {x : SSphere 2 | IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (x : EucSpace 2)}) ∧
    (∀ x : SSphere 2, energyGradMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => s) (branchMaskedWeights (0 : Idx 2 → ℝ) ∅) 0 μ (x : EucSpace 2) = 0) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hprob : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  have hpat (x : SSphere 2) : IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (x : EucSpace 2) :=
    ⟨fun j hj => by simp at hj, fun _ _ => by simp⟩
  have hw : branchMaskedWeights (0 : Idx 2 → ℝ) ∅ = 0 := by ext j; simp [branchMaskedWeights]
  have hfull := support_eq_univ_of_uniform μ hν
  have hstat := (isStationaryMap_id_iff (fun s => max s 0) 0 0 μ).mpr
    (isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => max s 0) 0 μ hν)
  refine ⟨μ, le_rfl, Function.bijective_id, hstat, hν, ?_, ?_⟩
  · have heq : ((μ : Measure (SSphere 2)).support ∩
        {x : SSphere 2 | IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (x : EucSpace 2)}) = univ := by
      ext x
      simp only [hfull, mem_inter_iff, mem_univ, true_and, mem_ofPred_eq]
      exact iff_true_intro (hpat x)
    rw [heq]
    simp
  · intro x
    rw [hw]
    exact (isStationaryMap_id_iff (fun s => s) 0 0 μ).mpr
      (isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) 0 μ hν) x
      (by rw [hfull]; exact mem_univ x)

end Transformer.Perceptron
