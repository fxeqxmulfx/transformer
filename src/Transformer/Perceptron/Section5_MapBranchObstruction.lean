module

public import Transformer.Perceptron.Section5_MapReLUBranches
public import Transformer.Perceptron.Section6_AntipodalFields
public import Transformer.Perceptron.Section6_SignedQuadratic

/-!
# Complementary unnormalized matrix branches under non-degeneracy

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), §5.3, and §6,
`rem:unifiedlog-B`. The source proposes extending the ReLU argument by
the parity mechanism used for normalized attention. That mechanism gives
a complete argument under the signed Gram non-degeneracy condition of
`eq: nondeg.normalized`. This is an explicit additional hypothesis here;
it is not inferred from non-analyticity of the original potential.

The actual unnormalized complementary branches have constant-minus-squares
transform identities. If both vanish globally, their difference is a
constant signed Gram quadratic form, contradicting non-degeneracy.
Antipodal invariance carries actual support points between the two patterns.
The original probability, matrix, neurons and stationary field are retained.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The continued original unnormalized branch is odd for an antipodal
probability law. Source: arXiv:2601.21366v2, §6,
`rem:unifiedlog-B`, the comparison of complementary patterns. -/
theorem energyGradMap_linear_masked_neg_of_antipodal {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d)
    (hanti : (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)))
    (J : Finset (Idx d)) (x : EucSpace d) :
    energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (-x) =
      -energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ x := by
  simp only [energyGradMap_eq_attentionGradMap_add_drift,
    drift_linear_masked_eq_projected_branch, attentionGradMap_neg_of_antipodal B μ hanti,
    branchVector_neg, proj_neg_neg, neg_add]

/-- Both actual complementary unnormalized branches cannot vanish
globally under the signed Gram condition. Source: arXiv:2601.21366v2,
§6, `rem:unifiedlog-B` and `eq:ratio-const-correct`, with the explicit
additional hypothesis `eq: nondeg.normalized`. -/
theorem not_both_energyGradMap_linear_masked_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (hnd : IsNonDegenerate ω a) (μ : Perspective.ProbSphere d) (J : Finset (Idx d)) :
    ¬ ((∀ x : SSphere d, energyGradMap B (fun s => s)
        (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0) ∧
      ∀ x : SSphere d, energyGradMap B (fun s => s)
        (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace d) = 0) := by
  rintro ⟨hJ, hC⟩
  obtain ⟨CJ, hCJ⟩ := exists_const_sub_squares_transformMap_of_zero_linear_masked hd B ω a μ J hJ
  obtain ⟨CC, hCC⟩ := exists_const_sub_squares_transformMap_of_zero_linear_masked hd B ω a μ Jᶜ hC
  apply not_constant_signedGram_of_nonDegenerate ω a hnd J
  refine ⟨2 * (CJ - CC), fun x hx => ?_⟩
  let xp : SSphere d := ⟨x, mem_sphere_zero_iff_norm.mpr hx⟩
  have h := (hCJ xp).symm.trans (hCC xp)
  rw [inner_signedGram_self]
  change CJ - (2 : ℝ)⁻¹ * ∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) x ^ 2 =
    CC - (2 : ℝ)⁻¹ * ∑ j ∈ Jᶜ, ω j * inner (𝕜 := ℝ) (a j) x ^ 2 at h
  linarith

/-- The complementary unnormalized branch has a uniform-null zero set
if one branch is globally zero and the signed Gram condition holds.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`, under the explicit
additional hypothesis `eq: nondeg.normalized`. -/
theorem measure_complement_linear_masked_zeros_eq_zero {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (hnd : IsNonDegenerate ω a) (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, energyGradMap B (fun s => s)
      (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν {x : SSphere d | energyGradMap B (fun s => s)
      (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace d) = 0} = 0 := by
  apply measure_sphere_vector_zeros_eq_zero_of_analytic_nonzero hd _
    (analyticOnNhd_energyGradMap_linear_masked B ω a μ Jᶜ) ν hν
  by_contra! hC
  exact not_both_energyGradMap_linear_masked_zero hd B ω a hnd μ J ⟨hzero, hC⟩

/-- Original stationarity gives a complementary-branch zero at each
support point of the chosen pattern when the original law is antipodal.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`, the antipodal step. -/
theorem complement_linear_masked_zero_on_support_pattern {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (hanti : (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)))
    (J : Finset (Idx d)) (x : SSphere d) (hx : x ∈ (μ : Measure (SSphere d)).support)
    (hpat : IsActivationPattern a J (x : EucSpace d)) :
    energyGradMap B (fun s => s) (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace d) = 0 := by
  have hxa := antipode_mem_support_of_invariant _ hanti x hx
  have hpa : IsActivationPattern a Jᶜ (antipodeMap d x : EucSpace d) := by
    simpa only [coe_antipodeMap] using isActivationPattern_compl_neg a J (x : EucSpace d) hpat
  have hz := hμ (antipodeMap d x) hxa
  rw [← energyGradMap_linear_masked_eq_relu_at_pattern B ω a μ Jᶜ _ hpa] at hz
  change energyGradMap B (fun s => s) (branchMaskedWeights ω Jᶜ) a μ (-(x : EucSpace d)) = 0 at hz
  rw [energyGradMap_linear_masked_neg_of_antipodal B ω a μ hanti] at hz
  exact neg_eq_zero.mp hz

/-- Under the explicit signed Gram condition, the actual unnormalized
matrix stationary support is uniform-null. Source: arXiv:2601.21366v2,
§3.1, `rem: ext` (ii), §5.3 and §6, `rem:unifiedlog-B`.
This intermediate result adds `eq: nondeg.normalized` to the ReLU case;
it does not assert that non-analyticity implies this extra condition.
Invertibility suffices for the field statement, without symmetry. -/
theorem measure_support_eq_zero_of_stationaryMap_nondegenerate {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hB : Function.Bijective B)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere d) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν (μ : Measure (SSphere d)).support = 0 := by
  by_contra hne
  obtain ⟨J, hpos⟩ := exists_positive_measure_activationPattern ν (μ : Measure (SSphere d)).support
    (pos_iff_ne_zero.mpr hne) a
  have hzero := energyGradMap_linear_masked_zero_of_positive_pattern_support hd B ω a μ hμ J ν hν hpos
  have hanti := (even_attentionTransformMap_iff B hB μ).mp
    (even_attentionTransformMap_of_zero_linear_masked hd B ω a μ J hzero)
  have hnull := measure_complement_linear_masked_zeros_eq_zero hd B ω a hnd μ J hzero ν hν
  have hpatnull : ν ((μ : Measure (SSphere d)).support ∩
      {x : SSphere d | IsActivationPattern a J (x : EucSpace d)}) = 0 := by
    apply measure_mono_null _ hnull
    exact fun x hx => complement_linear_masked_zero_on_support_pattern B ω a μ hμ hanti J x hx.1 hx.2
  exact hpos.ne' hpatnull

/-- The dimension, invertibility, signed Gram, stationarity and uniform
law inputs hold jointly for the original pinned circle Dirac. -/
example : (2 : ℕ) ≤ 2 ∧ Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsNonDegenerate (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => max s 0)
      (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2))) (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) :=
  ⟨le_rfl, Function.bijective_id, isNonDegenerate_pin,
    (isStationaryMap_id_iff _ _ _ _).mpr (isStationary_relu_pin 1 0 (basePoint 1)),
    Metastability.isUniformOn_uniformLaw (by norm_num)⟩

/-- A globally zero empty branch and non-degenerate original weights
coexist when the matrix is zero; the obstruction does not require its
invertibility. This supplies all inputs of the complementary zero-set
theorem with a genuine original probability and original uniform law. -/
example : (2 : ℕ) ≤ 2 ∧ IsNonDegenerate (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    (∀ x : SSphere 2, energyGradMap 0 (fun s => s)
      (branchMaskedWeights (Pi.single 0 1) ∅)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) (x : EucSpace 2) = 0) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) := by
  refine ⟨le_rfl, isNonDegenerate_pin, ?_, Metastability.isUniformOn_uniformLaw (by norm_num)⟩
  intro x
  simp [energyGradMap, branchMaskedWeights, drift, proj]

/-- Original zero-matrix stationarity, antipodal invariance, an actual
support point and its activation pattern hold jointly for the original
uniform circle law. This also witnesses the antipodal field theorem. -/
example : ∃ μ : Perspective.ProbSphere 2,
    IsStationaryMap 0 (fun s => max s 0) (0 : Idx 2 → ℝ) 0 μ ∧
    (μ : Measure (SSphere 2)).map (antipodeMap 2) = (μ : Measure (SSphere 2)) ∧
    basePoint 1 ∈ (μ : Measure (SSphere 2)).support ∧
    IsActivationPattern (0 : Idx 2 → EucSpace 2) ∅ (basePoint 1 : EucSpace 2) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  refine ⟨μ, ?_, hν.2 (LinearIsometryEquiv.neg ℝ), ?_, ?_⟩
  · intro x hx
    simp [energyGradMap, drift, proj]
  · rw [support_eq_univ_of_uniform μ hν]
    exact mem_univ _
  · exact ⟨fun j hj => by simp at hj, fun j _ => by simp⟩

end Transformer.Perceptron
