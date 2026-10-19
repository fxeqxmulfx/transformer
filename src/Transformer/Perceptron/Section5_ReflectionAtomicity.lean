module

public import Transformer.Perceptron.Section5_ReflectionBranches

/-!
# Finite original support for a commuting neuron reflection

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), §5.2,
`thm: circle`, and §6, the symmetry mechanism of `rem:unifiedlog-B`.
This intermediate matrix case has an explicit extra premise: an actual
isometry commuting with the matrix negates one nonzero original neuron
and fixes the other nonzero original neuron. The first weight is nonzero.
No non-analyticity premise is needed in this case.

If the original support were infinite, analytic continuation would give
a globally zero masked branch. Its quadratic transform is invariant
under the reflection; matrix injectivity then makes the original law
and its support invariant. An actual pattern point with two nonzero
inputs exists because the neuron boundary angles form a finite set.
At the reflected support point both the continued branch and the
original ReLU field vanish. Their difference is a nonzero multiple
of a nonzero projected neuron, a contradiction.

The resulting finite support gives an actual finite Dirac presentation
and zero measure for every original uniform law. The unrestricted source
claim remains in `GeneralAtomicity`, with this case dispatched there.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- A commuting original neuron reflection forces finite original
stationary ReLU support. Source: arXiv:2601.21366v2, §5.2,
`thm: circle`, and §6, `rem:unifiedlog-B`, with the explicit
additional reflection and commutation hypotheses of this case.
Matrix symmetry and potential non-analyticity are unnecessary here. -/
theorem finite_support_of_stationaryMap_commuting_neuron_reflection
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (hcomm : ∀ y, B (U y) = U (B y))
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (h₀ : U (a 0) = -a 0) (h₁ : U (a 1) = a 1)
    (ha₀ : a 0 ≠ 0) (ha₁ : a 1 ≠ 0) (hω₀ : ω 0 ≠ 0)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite := by
  by_contra hInf
  obtain ⟨J, hT, hzero⟩ := exists_zero_linear_masked_of_stationaryMap_infinite_circle B ω a μ hμ hInf
  have hinv := map_neuron_reflection_eq_of_zero_branch B hB ω a μ J hzero U hcomm h₀ h₁
  obtain ⟨θ, hθ, ht₀, ht₁⟩ := exists_two_inner_ne_zero_in_infinite_angles
    (a 0) (a 1) ha₀ ha₁ _ hT (fun θ hθ => hθ.1.1)
  let p : SSphere 2 := circlePoint θ
  let q : SSphere 2 := Perspective.sphereMap 2 U p
  have hp : p ∈ (μ : Measure (SSphere 2)).support := hθ.1.2
  have hq : q ∈ (μ : Measure (SSphere 2)).support :=
    isometry_mem_support_of_invariant _ U hinv p hp
  have hi := inner_neuron_reflection_inputs U (a 0) (a 1) h₀ h₁ (p : EucSpace 2)
  have hq₀ : inner (𝕜 := ℝ) (a 0) (q : EucSpace 2) ≠ 0 := by
    change inner (𝕜 := ℝ) (a 0) (U (p : EucSpace 2)) ≠ 0
    rw [hi.1]
    exact neg_ne_zero.mpr ht₀
  have hq₁ : inner (𝕜 := ℝ) (a 1) (q : EucSpace 2) ≠ 0 := by
    change inner (𝕜 := ℝ) (a 1) (U (p : EucSpace 2)) ≠ 0
    rwa [hi.2]
  have hproj := proj_ne_zero_of_perpendicular_test (a 0) (a 1) (q : EucSpace 2) ha₀
    (inner_eq_zero_of_isometry_neg_fixed U (a 0) (a 1) h₀ h₁) hq₁
  have hcoef : signedBranchWeight ω J 0 ≠ 0 := by
    intro hz
    exact hω₀ ((signedBranchWeight_eq_zero_iff ω J 0).mp hz)
  have hdiff := energyGradMap_sub_relu_at_neuron_reflection B ω a μ U h₀ h₁ J
    (p : EucSpace 2) hθ.2
  change energyGradMap B (fun s => s) (branchMaskedWeights ω J) a μ (q : EucSpace 2) -
    energyGradMap B (fun s => max s 0) ω a μ (q : EucSpace 2) = _ at hdiff
  rw [hzero q, hμ q hq, sub_self] at hdiff
  exact hproj ((smul_eq_zero.mp hdiff.symm).resolve_left (mul_ne_zero hcoef hq₀))

/-- The same original law admits an actual finite Dirac presentation.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii),
with the explicit additional commuting-reflection premises. -/
theorem isFinitelyAtomic_of_stationaryMap_commuting_neuron_reflection
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (hcomm : ∀ y, B (U y) = U (B y))
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (h₀ : U (a 0) = -a 0) (h₁ : U (a 1) = a 1)
    (ha₀ : a 0 ≠ 0) (ha₁ : a 1 ≠ 0) (hω₀ : ω 0 ≠ 0)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_stationaryMap_commuting_neuron_reflection B hB U hcomm ω a
      h₀ h₁ ha₀ ha₁ hω₀ μ hμ)

/-- In the commuting-reflection circle case the original support is
null for every original uniform law. Source: arXiv:2601.21366v2,
§3.1, `thm: any.d` (i), `rem: ext` (ii), and §6,
`rem:unifiedlog-B`, restricted to this explicit matrix case and `d = 2`. -/
theorem measure_support_eq_zero_of_stationaryMap_commuting_neuron_reflection
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (hcomm : ∀ y, B (U y) = U (B y))
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (h₀ : U (a 0) = -a 0) (h₁ : U (a 1) = a 1)
    (ha₀ : a 0 ≠ 0) (ha₁ : a 1 ≠ 0) (hω₀ : ω 0 ≠ 0)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere 2)) (hν : Metastability.IsUniformOn 2 ν) :
    ν (μ : Measure (SSphere 2)).support = 0 :=
  measure_finite_support_eq_zero_of_uniform le_rfl _
    (finite_support_of_stationaryMap_commuting_neuron_reflection B hB U hcomm ω a
      h₀ h₁ ha₀ ha₁ hω₀ μ hμ) ν hν

/-- The original law in this circle case is singular with respect to
every original uniform law. Source: arXiv:2601.21366v2, §5.2–§5.3,
`thm: circle` and `thm: any.d` (i), under the explicit additional
commuting-reflection hypotheses. -/
theorem mutuallySingular_uniform_of_stationaryMap_commuting_neuron_reflection
    (B : EucSpace 2 →ₗ[ℝ] EucSpace 2) (hB : Function.Bijective B)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (hcomm : ∀ y, B (U y) = U (B y))
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (h₀ : U (a 0) = -a 0) (h₁ : U (a 1) = a 1)
    (ha₀ : a 0 ≠ 0) (ha₁ : a 1 ≠ 0) (hω₀ : ω 0 ≠ 0)
    (μ : Perspective.ProbSphere 2) (hμ : IsStationaryMap B (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere 2)) (hν : Metastability.IsUniformOn 2 ν) :
    (μ : Measure (SSphere 2)) ⟂ₘ ν :=
  mutuallySingular_of_measure_support_eq_zero
    (measure_support_eq_zero_of_stationaryMap_commuting_neuron_reflection B hB U hcomm ω a
      h₀ h₁ ha₀ ha₁ hω₀ μ hμ ν hν)

/-- Every finite-support, finite-atomicity and uniform-null hypothesis
holds jointly for identity attention, the actual coordinate reflection,
two nonzero original neurons, a nonzero first weight, the actual pinned
Dirac, and the original uniform law. The second weight may be zero. -/
example : Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    (∀ y : EucSpace 2, LinearMap.id (R := ℝ) (circleCoordinateReflection y) =
      circleCoordinateReflection (LinearMap.id (R := ℝ) y)) ∧
    circleCoordinateReflection (-(basePoint 1 : EucSpace 2)) = -(-(basePoint 1 : EucSpace 2)) ∧
    circleCoordinateReflection secondAxis = secondAxis ∧
    -(basePoint 1 : EucSpace 2) ≠ 0 ∧ secondAxis ≠ 0 ∧
    (Pi.single 0 (1 : ℝ) : Idx 2 → ℝ) 0 ≠ 0 ∧
    IsStationaryMap LinearMap.id (fun s => max s 0) (Pi.single 0 (1 : ℝ))
      ![-(basePoint 1 : EucSpace 2), secondAxis] (Perspective.diracProb 2 (basePoint 1)) ∧
    Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) := by
  refine ⟨Function.bijective_id, fun _ => rfl, ?_, circleCoordinateReflection_second,
    ?_, ?_, by simp, ?_, Metastability.isUniformOn_uniformLaw (by norm_num)⟩
  · rw [map_neg, circleCoordinateReflection_first]
  · intro hz
    have h := norm_basePoint_one
    rw [neg_eq_zero.mp hz, norm_zero] at h
    exact zero_ne_one h
  · intro hz
    have h := norm_secondAxis
    rw [hz, norm_zero] at h
    exact zero_ne_one h
  · intro x hx
    have hxe : x = basePoint 1 := Interpolation.eq_of_mem_support_dirac hx
    subst x
    simp [energyGradMap, Perspective.diracProb, drift, proj, basePoint, Fin.sum_univ_two,
      EuclideanSpace.inner_single_left]

end Transformer.Perceptron
