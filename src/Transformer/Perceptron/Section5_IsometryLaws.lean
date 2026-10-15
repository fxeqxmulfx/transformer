/-
# Isometric symmetries of the original matrix law

ArXiv:2601.21366v2, §5.1, `rem: general-attention`, and §6,
the parity step in `rem:unifiedlog-B`. The original exponential
transform is covariant under simultaneous isometric changes of query,
law and matrix. When the matrix commutes with the isometry, invariance
of the actual transform implies invariance of the original law, by the
proved matrix-transform injectivity.

A globally zero actual quadratic branch gives transform invariance
under any isometry that changes each original neuron by a scalar sign.
The support consequence transports actual original support points.
This includes the source's antipodal argument and the individual
neuron reflections used to examine the remaining circle case.
No law invariance or support inclusion is hidden in a definition.
-/

import Transformer.Perceptron.Section5_NonzeroScalarCircle

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- The original transform is covariant under actual isometric matrix
conjugation and actual pushforward of the original probability law.
Source: arXiv:2601.21366v2, §5.1, `rem: general-attention`. -/
theorem attentionTransformMap_push_isometry_query {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d) (x : SSphere d) :
    attentionTransformMap (U.toLinearEquiv.toLinearMap.comp
      (B.comp U.symm.toLinearEquiv.toLinearMap))
        (μ.map (Perspective.sphereMap d U)) (Perspective.sphereMap d U x) =
      attentionTransformMap B μ x := by
  let C := U.toLinearEquiv.toLinearMap.comp (B.comp U.symm.toLinearEquiv.toLinearMap)
  have hf : Continuous (fun y : SSphere d => exp (inner (𝕜 := ℝ)
      (U (x : EucSpace d)) (C (y : EucSpace d)))) :=
    Real.continuous_exp.comp (continuous_const.inner
      (C.continuous_of_finiteDimensional.comp continuous_subtype_val))
  change (∫ y, exp (inner (𝕜 := ℝ) (U (x : EucSpace d)) (C (y : EucSpace d)))
      ∂(μ : Measure (SSphere d)).map (Perspective.sphereMap d U)) = _
  rw [integral_map (Perspective.measurable_sphereMap d U).aemeasurable hf.aestronglyMeasurable]
  apply integral_congr_ae
  exact ae_of_all _ fun y => by
    change exp (inner (𝕜 := ℝ) (U (x : EucSpace d))
      (U (B (U.symm (U (y : EucSpace d)))))) = exp (inner (𝕜 := ℝ) (x : EucSpace d) (B (y : EucSpace d)))
    rw [U.symm_apply_apply, U.inner_map_map]

/-- A commuting original matrix needs no conjugation in the transform
identity. Source: arXiv:2601.21366v2, §5.1 and §6,
the symmetry mechanism in `rem:unifiedlog-B`. -/
theorem attentionTransformMap_push_commuting_isometry {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (hcomm : ∀ y, B (U y) = U (B y)) (μ : Perspective.ProbSphere d) (x : SSphere d) :
    attentionTransformMap B (μ.map (Perspective.sphereMap d U)) (Perspective.sphereMap d U x) =
      attentionTransformMap B μ x := by
  have hC : U.toLinearEquiv.toLinearMap.comp (B.comp U.symm.toLinearEquiv.toLinearMap) = B := by
    apply LinearMap.ext
    intro y
    change U (B (U.symm y)) = B y
    rw [← hcomm, U.apply_symm_apply]
  simpa only [hC] using attentionTransformMap_push_isometry_query B U μ x

/-- Original transform invariance gives actual law invariance when
the original matrix is invertible and commutes with the isometry.
Source: arXiv:2601.21366v2, §5.1, `rem: general-attention`, and
§6, the parity step of `rem:unifiedlog-B`, with a general isometry. -/
theorem map_isometry_eq_of_transform_invariant {d : ℕ}
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (hB : Function.Bijective B)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (hcomm : ∀ y, B (U y) = U (B y))
    (μ : Perspective.ProbSphere d)
    (hinv : ∀ x : SSphere d, attentionTransformMap B μ (Perspective.sphereMap d U x) =
      attentionTransformMap B μ x) :
    (μ : Measure (SSphere d)).map (Perspective.sphereMap d U) = (μ : Measure (SSphere d)) := by
  have hprob : μ.map (Perspective.sphereMap d U) = μ := by
    apply injective_attentionTransformMap B hB
    funext x
    let y := Perspective.sphereMap d U.symm x
    have hxy : Perspective.sphereMap d U y = x := Subtype.ext (U.apply_symm_apply (x : EucSpace d))
    rw [← hxy, attentionTransformMap_push_commuting_isometry B U hcomm, hinv]
  exact congrArg (fun ρ : Perspective.ProbSphere d => (ρ : Measure (SSphere d))) hprob

/-- A zero original quadratic branch has an invariant transform under
isometric sign changes of its original neurons.
Source: arXiv:2601.21366v2, §5.3, `eq:global-Id`, and §6,
the even quadratic identity used in `rem:unifiedlog-B`. -/
theorem transformMap_invariant_of_zero_branch_isometric_signs {d : ℕ} (hd : 2 ≤ d)
    (B : EucSpace d →ₗ[ℝ] EucSpace d) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (J : Finset (Idx d))
    (hzero : ∀ x : SSphere d, energyGradMap B (fun s => s)
      (branchMaskedWeights ω J) a μ (x : EucSpace d) = 0)
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d) (ε : Idx d → ℝ)
    (hε : ∀ j, ε j ^ 2 = 1) (ha : ∀ j, U (a j) = ε j • a j) :
    ∀ x : SSphere d, attentionTransformMap B μ (Perspective.sphereMap d U x) =
      attentionTransformMap B μ x := by
  obtain ⟨C, hC⟩ := exists_const_sub_squares_transformMap_of_zero_linear_masked hd B ω a μ J hzero
  intro x
  rw [hC, hC]
  congr 2
  apply Finset.sum_congr rfl
  intro j hj
  congr 1
  have hi := U.inner_map_map (a j) (x : EucSpace d)
  rw [ha, real_inner_smul_left] at hi
  change inner (𝕜 := ℝ) (a j) (U (x : EucSpace d)) ^ 2 = inner (𝕜 := ℝ) (a j) (x : EucSpace d) ^ 2
  calc
    _ = ε j ^ 2 * inner (𝕜 := ℝ) (a j) (U (x : EucSpace d)) ^ 2 := by rw [hε, one_mul]
    _ = (ε j * inner (𝕜 := ℝ) (a j) (U (x : EucSpace d))) ^ 2 := by ring
    _ = _ := by rw [hi]

/-- Original law invariance transports actual original support points
through the isometry. Source: arXiv:2601.21366v2, §6,
the support step after the parity implication in `rem:unifiedlog-B`. -/
theorem isometry_mem_support_of_invariant {d : ℕ} (μ : Measure (SSphere d))
    (U : EucSpace d ≃ₗᵢ[ℝ] EucSpace d)
    (hinv : μ.map (Perspective.sphereMap d U) = μ) (x : SSphere d) (hx : x ∈ μ.support) :
    Perspective.sphereMap d U x ∈ μ.support := by
  rw [← hinv]
  exact mem_support_map_of_continuous_sphere μ _ (Perspective.continuous_sphereMap d U)
    (Perspective.measurable_sphereMap d U) x hx

/-- Invertibility, commutation, original transform invariance, actual
law invariance and a support point hold jointly for the original uniform
circle probability and the genuine antipodal isometry. The zero-neuron
branch realizes the sign-change hypotheses without excluding boundaries. -/
example : ∃ μ : Perspective.ProbSphere 2,
    Function.Bijective (LinearMap.id (R := ℝ) (M := EucSpace 2)) ∧
    (∀ y : EucSpace 2, LinearMap.id (R := ℝ) ((LinearIsometryEquiv.neg ℝ) y) =
      (LinearIsometryEquiv.neg ℝ) (LinearMap.id (R := ℝ) y)) ∧
    (∀ x : SSphere 2, attentionTransformMap LinearMap.id μ (antipodeMap 2 x) =
      attentionTransformMap LinearMap.id μ x) ∧
    (μ : Measure (SSphere 2)).map (antipodeMap 2) = (μ : Measure (SSphere 2)) ∧
    basePoint 1 ∈ (μ : Measure (SSphere 2)).support ∧ (2 : ℕ) ≤ 2 ∧
    (∀ x : SSphere 2, energyGradMap LinearMap.id (fun s => s) (branchMaskedWeights 0 ∅) 0 μ
      (x : EucSpace 2) = 0) ∧
    (∀ _ : Idx 2, (-1 : ℝ) ^ 2 = 1) ∧
    ∀ j : Idx 2, (LinearIsometryEquiv.neg ℝ) ((0 : Idx 2 → EucSpace 2) j) =
      (-1 : ℝ) • ((0 : Idx 2 → EucSpace 2) j) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hfull := support_eq_univ_of_uniform μ hν
  have hinv : (μ : Measure (SSphere 2)).map (antipodeMap 2) = (μ : Measure (SSphere 2)) :=
    hν.2 (LinearIsometryEquiv.neg ℝ)
  refine ⟨μ, Function.bijective_id, fun _ => rfl,
    (even_attentionTransformMap_iff LinearMap.id Function.bijective_id μ).mpr hinv,
    hinv, hfull.symm ▸ mem_univ _, le_rfl, ?_, fun _ => by norm_num, fun _ => by simp⟩
  intro x
  have hw : branchMaskedWeights (0 : Idx 2 → ℝ) ∅ = 0 := by ext j; simp [branchMaskedWeights]
  rw [hw]
  exact (isStationaryMap_id_iff (fun s => s) 0 0 μ).mpr
    (isStationary_zero_weights_of_uniform 1 one_ne_zero (fun s => s) 0 μ hν) x (hfull.symm ▸ mem_univ x)

end Transformer.Perceptron
