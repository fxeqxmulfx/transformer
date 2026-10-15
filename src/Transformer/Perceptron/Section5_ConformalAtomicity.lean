/-
# Original ReLU atomicity for conformal attention matrices

ArXiv:2601.21366v2, §3.1, `rem: ext` (ii), §5.2–§5.3,
and §6, `rem:unifiedlog-B`, in the explicit class `B=βU`,
where `β>0` and `U` is orthogonal. A continued zero branch has full
original support by the proved rotated-query inverse. Actual original
stationarity would then make the original potential analytic, contrary
to the original source hypothesis. This proves finite original circle
support and uniform-null original support in every dimension at least two.

Neither signed Gram non-degeneracy nor symmetry of `U` is required for
these actual field conclusions. For symmetric `U` they are precisely
the corresponding energy statements of the source. Negative scalar
matrices are included through `U=-I`, with the original weights and
primitive unchanged. The original claims for arbitrary symmetric
invertible matrices are retained separately in `GeneralAtomicity`.
The original dimension bound is retained for connectedness and the
scalar inverse. The circle conclusion concerns actual finite support
and its proved finite Dirac representation.
-/

import Transformer.Perceptron.Section5_ConformalSupport

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- A stationary non-analytic original potential rules out a globally
zero conformal matrix branch. Source: arXiv:2601.21366v2, §5.3,
the final contradiction after `eq:global-Id`, and `rem: ext` (ii),
specialized to `B=βU` with `β>0`. -/
theorem not_zero_conformal_matrix_branch_of_nonanalytic {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d)
    (hμ : IsStationaryMap (β • U.toLinearEquiv.toLinearMap) (fun s => max s 0) ω a μ)
    (J : Finset (Idx d)) :
    ¬ ∀ x : SSphere d, energyGradMap (β • U.toLinearEquiv.toLinearMap) (fun s => s)
      (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0 := by
  intro hzero
  have hfull := support_eq_univ_of_zero_conformal_matrix_branch hd β hβ U ω a μ J hzero
  exact hana (isAnalyticOnSphere_of_stationaryMap_full_support hd _ φ
    (fun s => max s 0) hφ ω a μ hμ hfull)

/-- Every original stationary conformal-matrix ReLU law with the source's
non-analytic potential has finite actual circle support.
Source: arXiv:2601.21366v2, §5.2, `thm: circle`, and §3.1,
`rem: ext` (ii), in the explicit class `B=βU` with `β>0`. -/
theorem finite_support_of_stationaryMap_conformal_circle (β : ℝ) (hβ : 0 < β)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hana : ¬ IsAnalyticOnSphere φ ω a) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap (β • U.toLinearEquiv.toLinearMap) (fun s => max s 0) ω a μ) :
    (μ : Measure (SSphere 2)).support.Finite := by
  by_contra hInf
  obtain ⟨J, -, hzero⟩ := exists_zero_linear_masked_of_stationaryMap_infinite_circle
    (β • U.toLinearEquiv.toLinearMap) ω a μ hμ hInf
  exact not_zero_conformal_matrix_branch_of_nonanalytic le_rfl β hβ U φ hφ ω a hana μ hμ J hzero

/-- The original conformal-matrix law has an actual finite Dirac
presentation. Source: arXiv:2601.21366v2, §3.1, `thm: circle` and
`rem: ext` (ii), in the explicit class `B=βU` with `β>0`. -/
theorem isFinitelyAtomic_of_stationaryMap_conformal_circle (β : ℝ) (hβ : 0 < β)
    (U : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2) (φ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s) (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2)
    (hana : ¬ IsAnalyticOnSphere φ ω a) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap (β • U.toLinearEquiv.toLinearMap) (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ :=
  isFinitelyAtomic_of_finite_topological_support μ
    (finite_support_of_stationaryMap_conformal_circle β hβ U φ hφ ω a hana μ hμ)

/-- The original conformal-matrix stationary support is null for every
original uniform law in dimensions at least two.
Source: arXiv:2601.21366v2, §5.3, `thm: any.d` (i), and §3.1,
`rem: ext` (ii), in the explicit class `B=βU` with `β>0`. -/
theorem measure_support_eq_zero_of_stationaryMap_conformal {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d)
    (hμ : IsStationaryMap (β • U.toLinearEquiv.toLinearMap) (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν (μ : Measure (SSphere d)).support = 0 := by
  by_contra hne
  obtain ⟨J, hzero⟩ := exists_zero_linear_masked_of_stationaryMap_positive_uniform_support
    hd (β • U.toLinearEquiv.toLinearMap) ω a μ hμ ν hν (pos_iff_ne_zero.mpr hne)
  exact not_zero_conformal_matrix_branch_of_nonanalytic hd β hβ U φ hφ ω a hana μ hμ J hzero

/-- A negative scalar matrix is a positive scalar times the antipodal
isometry. Source: arXiv:2601.21366v2, §3.1, the negative scalar
instances of symmetric invertible `B` in `rem: ext` (ii). -/
theorem scalar_map_eq_antipodal_conformal {d : ℕ} (β : ℝ) :
    β • (LinearMap.id : EucSpace d →ₗ[ℝ] EucSpace d) =
      (-β) • (LinearIsometryEquiv.neg ℝ : EucSpace d ≃ₗᵢ[ℝ] EucSpace d).toLinearEquiv.toLinearMap := by
  apply LinearMap.ext
  intro x
  change β • x = (-β) • (-x)
  rw [smul_neg, neg_smul, neg_neg]

/-- Negative scalar matrix attention retains actual finite atomicity
under the source's non-analyticity condition, without reweighting neurons.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), `B=βI`, `β<0`. -/
theorem isFinitelyAtomic_of_stationaryMap_negative_scalar_circle (β : ℝ) (hβ : β < 0)
    (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere 2)
    (hμ : IsStationaryMap (β • LinearMap.id) (fun s => max s 0) ω a μ) :
    IsFinitelyAtomic μ := by
  rw [scalar_map_eq_antipodal_conformal] at hμ
  exact isFinitelyAtomic_of_stationaryMap_conformal_circle (-β) (neg_pos.mpr hβ)
    (LinearIsometryEquiv.neg ℝ) φ hφ ω a hana μ hμ

/-- Negative scalar matrix stationarity has uniform-null original
support in dimensions at least two, for the same original neuron data.
Source: arXiv:2601.21366v2, §3.1, `rem: ext` (ii), `B=βI`, `β<0`. -/
theorem measure_support_eq_zero_of_stationaryMap_negative_scalar {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : β < 0) (φ : ℝ → ℝ) (hφ : ∀ s, HasDerivAt φ (2 * max s 0) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (hana : ¬ IsAnalyticOnSphere φ ω a)
    (μ : Perspective.ProbSphere d)
    (hμ : IsStationaryMap (β • LinearMap.id) (fun s => max s 0) ω a μ)
    (ν : Measure (SSphere d)) (hν : Metastability.IsUniformOn d ν) :
    ν (μ : Measure (SSphere d)).support = 0 := by
  rw [scalar_map_eq_antipodal_conformal] at hμ
  exact measure_support_eq_zero_of_stationaryMap_conformal hd (-β) (neg_pos.mpr hβ)
    (LinearIsometryEquiv.neg ℝ) φ hφ ω a hana μ hμ ν hν

/-- All non-analyticity and stationary inputs hold jointly at an actual
negative scalar matrix, equivalently a positive scalar antipodal matrix.
The original pinned Dirac, original neuron and primitive are unchanged. -/
example : (-1 : ℝ) < 0 ∧ (0 : ℝ) < 1 ∧ (2 : ℕ) ≤ 2 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    ¬ IsAnalyticOnSphere (fun s => max s 0 ^ 2) (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) ∧
    IsStationaryMap ((-1 : ℝ) • LinearMap.id) (fun s => max s 0) (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) (Perspective.diracProb 2 (basePoint 1)) ∧
    IsStationaryMap ((1 : ℝ) • (LinearIsometryEquiv.neg ℝ : EucSpace 2 ≃ₗᵢ[ℝ] EucSpace 2).toLinearEquiv.toLinearMap)
      (fun s => max s 0) (Pi.single 0 1) (Pi.single 0 (-(basePoint 1 : EucSpace 2)))
      (Perspective.diracProb 2 (basePoint 1)) ∧ Metastability.IsUniformOn 2 (MeanField.uniformLaw 2) := by
  have hstat : IsStationaryMap ((-1 : ℝ) • LinearMap.id) (fun s => max s 0) (Pi.single 0 1)
      (Pi.single 0 (-(basePoint 1 : EucSpace 2))) (Perspective.diracProb 2 (basePoint 1)) := by
    intro x hx
    have hxe : x = basePoint 1 := Interpolation.eq_of_mem_support_dirac hx
    subst x
    simp [energyGradMap, Perspective.diracProb, drift, proj, basePoint, Fin.sum_univ_two]
  refine ⟨by norm_num, one_pos, le_rfl, hasDerivAt_reluSq,
    not_isAnalyticOnSphere_relu 0 norm_basePoint_one norm_secondAxis inner_basePoint_secondAxis,
    hstat, ?_, Metastability.isUniformOn_uniformLaw (by norm_num)⟩
  simpa only [scalar_map_eq_antipodal_conformal, neg_neg] using hstat

end Transformer.Perceptron
