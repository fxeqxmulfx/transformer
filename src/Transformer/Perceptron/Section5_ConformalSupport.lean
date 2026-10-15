/-
# Full support and the original potential for conformal matrix attention

ArXiv:2601.21366v2, §5.3, the full-support contradiction after
`eq:global-Id`, and §3.1, `rem: ext` (ii).
For `B=βU`, with `β>0` and `U` orthogonal, rotating the original query
point gives the scalar transform of the same original probability.
The proved quadratic scalar inverse therefore applies after rotating
only the finitely many quadratic coefficients. No measure or support
is replaced in this transfer.

For every original matrix, a full-support stationary probability makes
the actual first variation constant on the sphere. The actual analytic
matrix interaction weight then gives an analytic extension of the
original perceptron potential. Thus a non-analytic original potential
excludes full support, with its original primitive and neurons retained.
-/

import Transformer.Perceptron.Section5_ZeroScalarCircle

open Real MeasureTheory Set
open scoped BigOperators

namespace Transformer.Perceptron

/-- Rotating the query gives the actual scalar transform of the same
original law. Source: arXiv:2601.21366v2, §5.1, `lem: quadpol`,
and §3.1, `rem: ext` (ii), specialized to `B=βU`. -/
theorem attentionTransformMap_isometry_query {d : ℕ} (β : ℝ)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (μ : Perspective.ProbSphere d) (x : SSphere d) :
    attentionTransformMap (β • U.toLinearEquiv.toLinearMap) μ (Perspective.sphereMap d U x) =
      attentionTransform β μ x := by
  apply integral_congr_ae
  exact ae_of_all _ fun y => by
    change exp (inner (𝕜 := ℝ) (U (x : EucSpace d)) (β • U (y : EucSpace d))) =
      exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
    rw [real_inner_smul_right, U.inner_map_map]

/-- A quadratic conformal-matrix transform gives full original support.
Source: arXiv:2601.21366v2, §5.3, the inverse after `eq:global-Id`,
and §3.1, `rem: ext` (ii), in the explicit class `B=βU`.
The original law is unchanged; only the quadratic vectors are rotated. -/
theorem support_eq_univ_of_quadratic_conformal_transform {d : ℕ} {ι : Type*}
    (hd : 2 ≤ d) (S : Finset ι) (β : ℝ) (hβ : 0 < β)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (μ : Perspective.ProbSphere d)
    (C : ℝ) (q : ι → ℝ) (a : ι → EucSpace d)
    (hquad : ∀ x : SSphere d, attentionTransformMap (β • U.toLinearEquiv.toLinearMap) μ x =
      C + ∑ i ∈ S, q i * inner (𝕜 := ℝ) (a i) (x : EucSpace d) ^ 2) :
    (μ : Measure (SSphere d)).support = univ := by
  apply support_eq_univ_of_const_add_squares_transform hd S β hβ μ C q (fun i => U.symm (a i))
  intro x
  have h := hquad (Perspective.sphereMap d U x)
  rw [attentionTransformMap_isometry_query] at h
  have hi (i : ι) : inner (𝕜 := ℝ) (a i) (U (x : EucSpace d)) =
      inner (𝕜 := ℝ) (U.symm (a i)) (x : EucSpace d) := by
    have hi := U.inner_map_map (U.symm (a i)) (x : EucSpace d)
    rwa [U.apply_symm_apply] at hi
  change attentionTransform β μ x = C + ∑ i ∈ S, q i *
    inner (𝕜 := ℝ) (a i) (U (x : EucSpace d)) ^ 2 at h
  simpa only [hi] using h

/-- A globally zero original conformal ReLU branch has full original
support. Source: arXiv:2601.21366v2, §5.3, `eq:global-Id`, and
§3.1, `rem: ext` (ii), specialized to `B=βU` with `β>0`. -/
theorem support_eq_univ_of_zero_conformal_matrix_branch {d : ℕ} (hd : 2 ≤ d)
    (β : ℝ) (hβ : 0 < β) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (J : Finset (Idx d)) (hzero : ∀ x : SSphere d,
      energyGradMap (β • U.toLinearEquiv.toLinearMap) (fun s => s)
        (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0) :
    (μ : Measure (SSphere d)).support = univ := by
  obtain ⟨C, hC⟩ := exists_const_sub_squares_transformMap_of_zero_linear_masked
    hd (β • U.toLinearEquiv.toLinearMap) ω a μ J hzero
  apply support_eq_univ_of_quadratic_conformal_transform hd J β hβ U μ C
    (fun j => -(2 : ℝ)⁻¹ * ω j) a
  intro x
  rw [hC]
  have hsum : ∑ j ∈ J, (-(2 : ℝ)⁻¹ * ω j) * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2 =
      -(2 : ℝ)⁻¹ * ∑ j ∈ J, ω j * inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2 := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun j _ => by ring
  rw [hsum]
  ring

/-- Original full-support matrix stationarity makes the actual first
variation constant. Source: arXiv:2601.21366v2, §5.3, the final step
after `eq:global-Id`, and §3.1, `rem: ext` (ii). -/
theorem firstVariationMap_eq_of_stationary_full_support {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ σ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationaryMap B σ ω a μ)
    (hfull : (μ : Measure (SSphere d)).support = univ) :
    ∀ x y : SSphere d, firstVariationMap B φ ω a μ (x : EucSpace d) =
      firstVariationMap B φ ω a μ (y : EucSpace d) := by
  apply firstVariationMap_eq_of_energyGrad_zero hd B φ σ hφ ω a μ
  exact fun x => hμ x (hfull.symm ▸ mem_univ x)

/-- Original full-support matrix stationarity forces analyticity of
the original potential. Source: arXiv:2601.21366v2, §5.3, the final
contradiction after `eq:global-Id`, and §3.1, `rem: ext` (ii).
This field argument needs neither symmetry nor invertibility. -/
theorem isAnalyticOnSphere_of_stationaryMap_full_support {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (φ σ : ℝ → ℝ)
    (hφ : ∀ s, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (hμ : IsStationaryMap B σ ω a μ)
    (hfull : (μ : Measure (SSphere d)).support = univ) : IsAnalyticOnSphere φ ω a := by
  have hconst := firstVariationMap_eq_of_stationary_full_support hd B φ σ hφ ω a μ hμ hfull
  let x₀ : SSphere d := Classical.choice (nonempty_of_isProbabilityMeasure (μ : Measure (SSphere d)))
  let C := firstVariationMap B φ ω a μ (x₀ : EucSpace d)
  refine ⟨fun z => 2 * (C - attentionWeightMap B μ z), ?_, fun z hz => ?_⟩
  · intro z _
    exact analyticAt_const.fun_mul (analyticAt_const.fun_sub
      (analyticOnNhd_attentionWeightMap B μ z (mem_univ z)))
  · have h := hconst (⟨z, mem_sphere_zero_iff_norm.mpr hz⟩ : SSphere d) x₀
    change attentionWeightMap B μ z + (2 : ℝ)⁻¹ * potential φ ω a z = C at h
    linarith

/-- All primitive, conformal inverse, globally zero branch and
full-support stationarity hypotheses hold jointly at the original
uniform circle law, identity isometry and zero original neuron weights. -/
example : ∃ μ : Perspective.ProbSphere 2, (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧
    (∀ s, HasDerivAt (fun t : ℝ => max t 0 ^ 2) (2 * max s 0) s) ∧
    IsStationaryMap ((1 : ℝ) • (LinearIsometryEquiv.refl ℝ (EucSpace 2)).toLinearEquiv.toLinearMap)
      (fun s => max s 0) 0 0 μ ∧ (μ : Measure (SSphere 2)).support = univ ∧
    (∀ x : SSphere 2, energyGradMap
      ((1 : ℝ) • (LinearIsometryEquiv.refl ℝ (EucSpace 2)).toLinearEquiv.toLinearMap)
      (fun s => s) (branchMaskedWeights 0 ∅) 0 μ (x : EucSpace 2) = 0) ∧
    ∀ x : SSphere 2, attentionTransformMap
      ((1 : ℝ) • (LinearIsometryEquiv.refl ℝ (EucSpace 2)).toLinearEquiv.toLinearMap) μ x =
        attentionTransform 1 μ (basePoint 1) +
          ∑ _ ∈ (Finset.univ : Finset (Idx 2)), (0 : ℝ) *
            inner (𝕜 := ℝ) (0 : EucSpace 2) (x : EucSpace 2) ^ 2 := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hfull := support_eq_univ_of_uniform μ hν
  have hstat (σ : ℝ → ℝ) := (isStationaryMap_id_iff σ 0 0 μ).mpr
    (isStationary_zero_weights_of_uniform 1 one_ne_zero σ 0 μ hν)
  refine ⟨μ, le_rfl, one_pos, hasDerivAt_reluSq, by simpa using hstat _, hfull, ?_, ?_⟩
  · intro x
    have hw : branchMaskedWeights (0 : Idx 2 → ℝ) ∅ = 0 := by ext j; simp [branchMaskedWeights]
    rw [hw]
    simpa using hstat (fun s => s) x (hfull.symm ▸ mem_univ x)
  · intro x
    have hi := attentionTransformMap_isometry_query 1
      (LinearIsometryEquiv.refl ℝ (EucSpace 2)) μ x
    have hp : Perspective.sphereMap 2 (LinearIsometryEquiv.refl ℝ (EucSpace 2)) x = x :=
      Subtype.ext rfl
    rw [hp] at hi
    rw [hi]
    simpa using attentionTransform_eq_of_uniform 1 μ hν x (basePoint 1)

end Transformer.Perceptron
