module

public import Transformer.Perceptron.Section5_MapBranchObstruction
public import Transformer.Perceptron.Section5_ReLUAtomicity
public import Transformer.Perceptron.Section6_CircleNullSupport

/-!
# Finite unnormalized matrix support under signed Gram non-degeneracy

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), §5.2, and §6,
`rem:unifiedlog-B`. Infinite actual circle support supplies infinitely
many actual angles in one activation pattern. The analytic unnormalized
matrix branch is consequently zero on the entire original circle.

The transform identity and matrix injectivity then make the original
law antipodal. Actual stationarity gives complementary-branch zeros
at the original infinite angle set. The same analytic identity theorem
makes the complementary branch globally zero. This contradicts the
signed Gram condition, giving finite original support and its actual
Dirac representation.

The signed Gram non-degeneracy condition is an explicit extra premise
of this intermediate result. The source's ReLU claims without that
condition are retained separately in `GeneralAtomicity`.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Infinite original support gives an infinite original angular
pattern and a globally zero actual continued matrix branch.
Source: arXiv:2601.21366v2, §5.2, `eq:K-equals-quad-global`,
and §3.1, `rem: ext` (ii). No non-degeneracy is needed for this step. -/
theorem exists_zero_linear_masked_of_stationaryMap_infinite_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (hInf : (μ : Measure (SSphere 2)).support.Infinite) :
    ∃ J : Finset (Idx 2),
      ((Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support) ∩
        {θ | IsActivationPattern a J (circlePoint θ : EucSpace 2)}).Infinite ∧
      ∀ x : SSphere 2, energyGradMap B (fun s => s)
        (branchMaskedWeights ω J) a μ (x : EucSpace 2) = 0 := by
  let S : Set ℝ := Icc (-π) π ∩ circlePoint ⁻¹' (μ : Measure (SSphere 2)).support
  have hS : S.Infinite := by
    intro hf
    apply hInf
    apply (hf.image circlePoint).subset
    intro x hx
    obtain ⟨θ, hθ, heq⟩ := exists_circlePoint_eq x
    refine ⟨θ, ⟨hθ, ?_⟩, heq⟩
    change circlePoint θ ∈ (μ : Measure (SSphere 2)).support
    rwa [heq]
  obtain ⟨J, hT⟩ := exists_infinite_activation_angles a S hS
  refine ⟨J, hT, ?_⟩
  apply vector_eq_zero_on_circle_of_infinite_zeros _
    (analyticOnNhd_energyGradMap_linear_masked B ω a μ J) _ hT (fun _ hθ => hθ.1.1)
  intro θ hθ
  rw [energyGradMap_linear_masked_eq_relu_at_pattern B ω a μ J _ hθ.2]
  exact hμ (circlePoint θ) hθ.1.2

/-- An original infinite stationary circle support makes the original
law antipodal through the actual invertible matrix transform.
Source: arXiv:2601.21366v2, §6, `rem:unifiedlog-B`, using
the parity implication of §5.1, `rem: general-attention`. -/
theorem antipodal_of_stationaryMap_infinite_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (hInf : (μ : Measure (SSphere 2)).support.Infinite) :
    (μ : Measure (SSphere 2)).map (antipodeMap 2) = (μ : Measure (SSphere 2)) := by
  obtain ⟨J, -, hzero⟩ := exists_zero_linear_masked_of_stationaryMap_infinite_circle B ω a μ hμ hInf
  exact (even_attentionTransformMap_iff B hB μ).mp
    (even_attentionTransformMap_of_zero_linear_masked le_rfl B ω a μ J hzero)

/-- Under the explicit signed Gram condition, the original unnormalized
stationary law has finite support on the actual circle.
Source: arXiv:2601.21366v2, §5.2 and §6, `rem:unifiedlog-B`.
This intermediate result adds `eq: nondeg.normalized` to the matrix
ReLU claim; symmetry and potential non-analyticity are unnecessary
under that explicit stronger hypothesis. -/
theorem finite_support_of_stationaryMap_nondegenerate_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite := by
  by_contra hInf
  obtain ⟨J, hT, hzero⟩ := exists_zero_linear_masked_of_stationaryMap_infinite_circle B ω a μ hμ hInf
  have hanti := (even_attentionTransformMap_iff B hB μ).mp
    (even_attentionTransformMap_of_zero_linear_masked le_rfl B ω a μ J hzero)
  have hzeroC : ∀ x : SSphere 2, energyGradMap B (fun s => s)
      (branchMaskedWeights ω Jᶜ) a μ (x : EucSpace 2) = 0 := by
    apply vector_eq_zero_on_circle_of_infinite_zeros _
      (analyticOnNhd_energyGradMap_linear_masked B ω a μ Jᶜ) _ hT (fun _ hθ => hθ.1.1)
    intro θ hθ
    exact complement_linear_masked_zero_on_support_pattern B ω a μ hμ hanti J
      (circlePoint θ) hθ.1.2 hθ.2
  exact not_both_energyGradMap_linear_masked_zero le_rfl B ω a hnd μ J ⟨hzero, hzeroC⟩

/-- The same original law has an actual finite Dirac presentation.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), with the
explicit additional signed Gram hypothesis `eq: nondeg.normalized`. -/
theorem isFinitelyAtomic_of_stationaryMap_nondegenerate_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hnd : IsNonDegenerate ω a)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_stationaryMap_nondegenerate_circle B hB ω a hnd μ hμ)

/-- Any remaining infinite-support case has an actual scalar signed
Gram operator for some original activation pattern.
Source: arXiv:2601.21366v2, §6, `eq:ratio-const-correct` and
`rem:unifiedlog-B`. The alternative is explicit; no claim that it
contradicts the source's non-analyticity hypothesis is made here. -/
theorem finite_support_or_scalar_signedGram_of_stationaryMap_circle
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite ∨
      ∃ J : Finset (Idx 2), ∃ c : ℝ, ∀ x : EucSpace 2, signedGram ω a J x = c • x := by
  by_cases hnd : IsNonDegenerate ω a
  · exact Or.inl (finite_support_of_stationaryMap_nondegenerate_circle B hB ω a hnd μ hμ)
  · right
    simpa only [IsNonDegenerate, not_forall, not_not] using hnd

/-- All finite-support and Dirac-presentation hypotheses hold jointly
for the original pinned circle Dirac and identity attention matrix. -/
example : Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsNonDegenerate (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => max s 0)
      (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) :=
  ⟨Function.bijective_id, isNonDegenerate_pin,
    (isStationaryMap_id_iff _ _ _ _).mpr (isStationary_relu_pin 1 0 (basePoint 1))⟩

/-- Infinite-support stationarity and invertibility hold jointly for
the actual uniform circle probability and zero original neuron weights.
This witnesses the continuation and parity steps without a finite-support
or non-degeneracy premise. -/
example : ∃ μ : Perspective.ProbSphere 2,
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    IsStationaryMap (LinearMap.id (R := ℝ) (M := EucSpace 2)) (fun s => max s 0)
      (0 : Idx 2 → ℝ) 0 μ ∧ (μ : Measure (SSphere 2)).support.Infinite := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  let : IsProbabilityMeasure (MeanField.uniformLaw 2) := hν.1
  refine ⟨μ, Function.bijective_id,
    (isStationaryMap_id_iff _ _ _ _).mpr
      (isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => max s 0) 0 μ hν), ?_⟩
  intro hf
  have h := measure_finite_support_eq_zero_of_uniform le_rfl (μ : Measure (SSphere 2)) hf
    (MeanField.uniformLaw 2) hν
  rw [support_eq_univ_of_uniform μ hν, measure_univ] at h
  exact one_ne_zero h

end Transformer.Perceptron
