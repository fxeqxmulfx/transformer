module

public import Transformer.Perceptron.Section5_MapReLUAtomicity

/-!
# The actual scalar signed Gram obstruction for matrix ReLU branches

ArXiv:2601.21366v2, §5.2–§5.3, `eq:global-Id`, and §6,
`eq:ratio-const-correct` and `rem:unifiedlog-B`.
Two globally zero complementary branches force their original signed
Gram operator to be a scalar identity, by their actual transform
identities and polarization. No non-degeneracy premise is used.

For positive original uniform support measure, or infinite original
circle support, the continued branch and its actual complement both
vanish globally. The same original activation pattern retains positive
measure or infinitely many original angles. Thus the remaining scalar
case belongs to the selected support pattern, rather than to an
unrelated subset of neurons. All original fields and measures remain.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Two globally zero complementary unnormalized branches force the
original signed Gram map to be scalar. Source: arXiv:2601.21366v2,
§6, `eq:ratio-const-correct` and `rem:unifiedlog-B`. -/
theorem exists_scalar_signedGram_of_complementary_matrix_branches_zero {d : ℕ}
    (hd : 2 ≤ d) (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hJ : ∀ x : SSphere d, energyGradMap B (fun s => s)
      (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0)
    (hC : ∀ x : SSphere d, energyGradMap B (fun s => s)
      (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace d) = 0) :
    ∃ c : ℝ, ∀ x : EucSpace d, signedGram ω a J x = c • x := by
  obtain ⟨CJ, hCJ⟩ := exists_const_sub_squares_transformMap_of_zero_linear_masked hd B ω a μ J hJ
  obtain ⟨CC, hCC⟩ := exists_const_sub_squares_transformMap_of_zero_linear_masked hd B ω a μ Jᶜ hC
  have hquad : ∀ x : EucSpace d, ‖x‖ = 1 →
      inner (𝕜 := ℝ) (signedGram ω a J x) x = 2 * (CJ - CC) := by
    intro x hx
    let xp : SSphere d := ⟨x, mem_sphere_zero_iff_norm.mpr hx⟩
    have h := (hCJ xp).symm.trans (hCC xp)
    rw [inner_signedGram_self]
    change CJ - (2 : ℝ)⁻¹ * ∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) x ^ 2 =
      CC - (2 : ℝ)⁻¹ * ∑ j ∈ Jᶜ, ω j * inner (𝕜 := ℝ) (a j) x ^ 2 at h
    linarith
  have hmap := eq_smul_id_of_constant_unit_quadratic (signedGramLinearMap ω a J)
    (isSymmetric_signedGramLinearMap ω a J) (2 * (CJ - CC))
    (by simpa only [signedGramLinearMap_apply] using hquad)
  refine ⟨2 * (CJ - CC), fun x => ?_⟩
  simpa only [signedGramLinearMap_apply, LinearMap.smul_apply, LinearMap.id_apply] using
    congrArg (fun L : EucSpace d →ₗ[ℝ] EucSpace d => L x) hmap

/-- The complementary branch is globally zero for the same original
positive-measure stationary pattern. Source: arXiv:2601.21366v2,
§6, the antipodal continuation in `rem:unifiedlog-B`.
The conclusion does not require the signed Gram condition. -/
theorem complementary_matrix_branches_zero_of_positive_support_pattern {d : ℕ}
    (hd : 2 ≤ d) (B : EucSpace d →ₗ[ℝ] EucSpace d) (hB : Function.Bijective B)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) (J : Finset (Idx d))
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν ((μ : Measure (SSphere d)).support ∩
      {x : SSphere d | IsActivationPattern a J (x : EucSpace d)})) :
    (∀ x : SSphere d, energyGradMap B (fun s => s)
      (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0) ∧
    ∀ x : SSphere d, energyGradMap B (fun s => s)
      (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace d) = 0 := by
  have hzero := energyGradMap_linear_masked_zero_of_positive_pattern_support hd B ω a μ hμ J ν hν hpos
  have hanti := (even_attentionTransformMap_iff B hB μ).mp
    (even_attentionTransformMap_of_zero_linear_masked hd B ω a μ J hzero)
  refine ⟨hzero, ?_⟩
  apply vector_eq_zero_on_sphere_of_positive_zero_measure hd _
    (analyticOnNhd_energyGradMap_linear_masked B ω a μ Jᶜ) ν hν
  apply hpos.trans_le (measure_mono _)
  exact fun x hx => complement_linear_masked_zero_on_support_pattern B ω a μ hμ hanti J x hx.1 hx.2

/-- Positive original uniform support measure selects a scalar signed
Gram operator for an actual positive-measure support pattern, together
with its two globally zero original branches.
Source: arXiv:2601.21366v2, §5.3, `eq:global-Id`, and §6,
`eq:ratio-const-correct` and `rem:unifiedlog-B`. -/
theorem exists_scalar_support_pattern_of_stationaryMap_positive_uniform {d : ℕ}
    (hd : 2 ≤ d) (B : EucSpace d →ₗ[ℝ] EucSpace d) (hB : Function.Bijective B)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν)
    (hpos : 0 < ν (μ : Measure (SSphere d)).support) :
    ∃ J : Finset (Idx d), ∃ c : ℝ,
      0 < ν ((μ : Measure (SSphere d)).support ∩
        {x : SSphere d | IsActivationPattern a J (x : EucSpace d)}) ∧
      (∀ x : SSphere d, energyGradMap B (fun s => s)
        (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0) ∧
      (∀ x : SSphere d, energyGradMap B (fun s => s)
        (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace d) = 0) ∧
      ∀ x : EucSpace d, signedGram ω a J x = c • x := by
  obtain ⟨J, hpat⟩ := exists_positive_measure_activationPattern ν (μ : Measure (SSphere d)).support hpos a
  obtain ⟨hJ, hC⟩ := complementary_matrix_branches_zero_of_positive_support_pattern hd B hB ω a μ hμ J ν hν hpat
  obtain ⟨c, hc⟩ := exists_scalar_signedGram_of_complementary_matrix_branches_zero hd B ω a μ J hJ hC
  exact ⟨J, c, hpat, hJ, hC, hc⟩

/-- Infinite actual circle support selects a scalar signed Gram
operator for the same original infinite angular pattern and its two
globally zero original branches. Source: arXiv:2601.21366v2,
§5.2, the continuation after `eq:K-equals-quad-global`, and §6,
`rem:unifiedlog-B`. No non-analyticity or non-degeneracy is assumed. -/
theorem exists_scalar_support_pattern_of_stationaryMap_infinite_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (hInf : (μ : Measure (SSphere 2)).support.Infinite) :
    ∃ J : Finset (Idx 2), ∃ c : ℝ,
      ((Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support) ∩
        {θ | IsActivationPattern a J (circlePoint θ : EucSpace 2)}).Infinite ∧
      (∀ x : SSphere 2, energyGradMap B (fun s => s)
        (branchMaskedWeights ω J) a μ (x : EucSpace 2) = 0) ∧
      (∀ x : SSphere 2, energyGradMap B (fun s => s)
        (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace 2) = 0) ∧
      ∀ x : EucSpace 2, signedGram ω a J x = c • x := by
  obtain ⟨J, hT, hJ⟩ := exists_zero_linear_masked_of_stationaryMap_infinite_circle B ω a μ hμ hInf
  have hanti := (even_attentionTransformMap_iff B hB μ).mp
    (even_attentionTransformMap_of_zero_linear_masked le_rfl B ω a μ J hJ)
  have hC : ∀ x : SSphere 2, energyGradMap B (fun s => s)
      (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace 2) = 0 := by
    apply vector_eq_zero_on_circle_of_infinite_zeros _
      (analyticOnNhd_energyGradMap_linear_masked B ω a μ Jᶜ) _ hT (fun _ hθ => hθ.1.1)
    intro θ hθ
    exact complement_linear_masked_zero_on_support_pattern B ω a μ hμ hanti J
      (circlePoint θ) hθ.1.2 hθ.2
  obtain ⟨c, hc⟩ := exists_scalar_signedGram_of_complementary_matrix_branches_zero le_rfl B ω a μ J hJ hC
  exact ⟨J, c, hT, hJ, hC, hc⟩

/-- All continuation, scalar, positive-measure and infinite-support
inputs hold jointly for the actual uniform circle law with zero weights.
The actual selected pattern may include every zero neuron boundary. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => max s 0) 0 0 μ ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    0 < (MeanField.uniformLaw 2) ((μ : Measure (SSphere 2)).support ∩
      {x : SSphere 2 | IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (x : EucSpace 2)}) ∧
    (μ : Measure (SSphere 2)).support.Infinite ∧
    ∀ J : Finset (Idx 2), ∀ x : SSphere 2, energyGradMap (LinearMap.id (R := ℝ) (M := EucSpace 2))
      (fun s => s) (branchMaskedWeights 0 J) 0 μ (x : EucSpace 2) = 0 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  let : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  have hfull := support_eq_univ_of_uniform μ hν
  have hstat (σ : ℝ → ℝ) := (isStationaryMap_id_iff σ 0 0 μ).mpr
    (isStationary_zero_weights_of_uniform 1 one_ne_zero σ 0 μ hν)
  refine ⟨μ, le_rfl, Function.bijective_id, hstat _, hν, ?_, ?_, ?_⟩
  · have hpat (x : SSphere 2) : IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (x : EucSpace 2) :=
      ⟨fun j hj => by simp at hj, fun j _ => by simp⟩
    have heq : ((μ : Measure (SSphere 2)).support ∩
        {x : SSphere 2 | IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (x : EucSpace 2)}) = univ := by
      ext x
      simp only [hfull, mem_inter_iff, mem_univ, true_and, mem_ofPred_eq]
      exact iff_true_intro (hpat x)
    rw [heq, measure_univ]
    exact zero_lt_one
  · intro hf
    have h := measure_finite_support_eq_zero_of_uniform le_rfl (μ : Measure (SSphere 2)) hf _ hν
    rw [hfull, measure_univ] at h
    exact one_ne_zero h
  · intro J x
    have hw : branchMaskedWeights (0 : Idx 2 → ℝ) J = 0 := by ext j; simp [branchMaskedWeights]
    rw [hw]
    exact hstat (fun s => s) x (hfull.symm ▸ mem_univ x)

end Transformer.Perceptron
