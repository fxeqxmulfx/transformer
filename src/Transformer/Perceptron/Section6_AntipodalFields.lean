module

public import Transformer.Perceptron.Section6_AmbientBranches
public import Transformer.Metastability.Section4_UniformLaw

/-!
# Antipodal symmetry of the actual normalized fields and support

ArXiv:2601.21366v2, §6, proof of `prop: unified.log`, part (a),
passes from evenness of the original transform to antipodal invariance
of the original probability measure, then to the complementary branch.

The actual ambient interaction weight is even and the actual vector
integral is odd under that invariance. Both facts are proved by the
original pushforward integral, using its continuous actual integrand.
The original tangent projection has the same antipodal covariance,
so the complete original normalized branch field is odd as well.

The support lemma transports actual support points through a continuous
measurable sphere map, in every dimension. Thus antipodal invariance
carries an original support point to its actual antipode, without a
finite-support or density conclusion. The uniform spherical Lebesgue
law supplies a genuine nonempty invariant measure witness below.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Perceptron

/-- Antipodal invariance makes the original ambient weight even.
Source: arXiv:2601.21366v2, §6, the parity step after
`eq:log-identity-global`. The equality uses the original integral. -/
theorem attentionWeightMap_neg_of_antipodal {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d)
    (hanti : (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)))
    (x : EucSpace d) : attentionWeightMap B μ (-x) = attentionWeightMap B μ x := by
  let f : SSphere d → ℝ := fun y => Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d)))
  have hB : Continuous (fun y : SSphere d => B (y : EucSpace d)) :=
    B.continuous_of_finiteDimensional.comp continuous_subtype_val
  have hf : Continuous f := Real.continuous_exp.comp (continuous_const.inner hB)
  calc
    _ = ∫ y, f (antipodeMap d y) ∂(μ : Measure (SSphere d)) := by
      apply integral_congr_ae
      exact ae_of_all _ fun y => by
        simp only [f, coe_antipodeMap, map_neg, inner_neg_left, inner_neg_right]
    _ = ∫ y, f y ∂(μ : Measure (SSphere d)).map (antipodeMap d) :=
      (integral_map (measurable_antipodeMap d).aemeasurable hf.aestronglyMeasurable).symm
    _ = attentionWeightMap B μ x := by rw [hanti]; rfl

/-- Antipodal invariance makes the original ambient vector gradient odd.
Source: arXiv:2601.21366v2, §6, comparing the original stationary
fields at antipodal points. The vector integral is the actual one. -/
theorem ambientAttentionGradMap_neg_of_antipodal {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d)
    (hanti : (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)))
    (x : EucSpace d) : ambientAttentionGradMap B μ (-x) = -ambientAttentionGradMap B μ x := by
  let f : SSphere d → EucSpace d := fun y =>
    Real.exp (inner (𝕜 := ℝ) x (B (y : EucSpace d))) • B (y : EucSpace d)
  have hB : Continuous (fun y : SSphere d => B (y : EucSpace d)) :=
    B.continuous_of_finiteDimensional.comp continuous_subtype_val
  have hf : Continuous f := (Real.continuous_exp.comp (continuous_const.inner hB)).smul hB
  calc
    _ = -(∫ y, f (antipodeMap d y) ∂(μ : Measure (SSphere d))) := by
      rw [ambientAttentionGradMap, ← integral_neg]
      apply integral_congr_ae
      exact ae_of_all _ fun y => by
        simp only [f, coe_antipodeMap, map_neg, inner_neg_left, inner_neg_right, smul_neg, neg_neg]
    _ = -(∫ y, f y ∂(μ : Measure (SSphere d)).map (antipodeMap d)) :=
      congrArg (fun v : EucSpace d => -v)
        (integral_map (measurable_antipodeMap d).aemeasurable hf.aestronglyMeasurable).symm
    _ = -ambientAttentionGradMap B μ x := by rw [hanti]; rfl

/-- The original projection commutes with simultaneous antipodal signs.
Source: arXiv:2601.21366v2, §6, antipodal branch gradients. -/
theorem proj_neg_neg {d : ℕ} (x v : EucSpace d) : proj d (-x) (-v) = -proj d x v := by
  simp only [proj, inner_neg_left, inner_neg_right, neg_neg, smul_neg]
  abel

/-- The original projected attention field is odd under the original
measure's antipodal invariance. Source: arXiv:2601.21366v2,
§6, comparing the two actual stationary fields. -/
theorem attentionGradMap_neg_of_antipodal {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (μ : Perspective.ProbSphere d)
    (hanti : (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)))
    (x : EucSpace d) : attentionGradMap B μ (-x) = -attentionGradMap B μ x := by
  simp only [attentionGradMap_eq_proj_ambientGradMap,
    ambientAttentionGradMap_neg_of_antipodal B μ hanti, proj_neg_neg]

/-- The globally continued actual normalized branch field is odd under
antipodal invariance. Source: arXiv:2601.21366v2, §6,
the complementary branch argument after `eq:log-identity-global`. -/
theorem normalizedBranchField_neg_of_antipodal {d : ℕ} (B : EucSpace d →ₗ[ℝ] EucSpace d)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d)
    (hanti : (μ : Measure (SSphere d)).map (antipodeMap d) = (μ : Measure (SSphere d)))
    (J : Finset (Idx d)) (x : EucSpace d) :
    normalizedBranchField B ω a μ J (-x) = -normalizedBranchField B ω a μ J x := by
  simp only [normalizedBranchField, attentionGradMap_neg_of_antipodal B μ hanti,
    attentionWeightMap_neg_of_antipodal B μ hanti, branchVector_neg, proj_neg_neg,
    smul_neg, neg_add]

/-- A continuous measurable original sphere map transports actual
support points to its actual pushforward support in every dimension.
Source: arXiv:2601.21366v2, §6, the antipodal support step.
This generalizes `mem_support_map_of_continuous_circle` from
`Section6_NormalizedForce` without changing the neighborhood argument. -/
theorem mem_support_map_of_continuous_sphere {d : ℕ} (μ : Measure (SSphere d))
    (f : SSphere d → SSphere d) (hc : Continuous f) (hm : Measurable f)
    (x : SSphere d) (hx : x ∈ μ.support) : f x ∈ (μ.map f).support := by
  rw [Measure.support_eq_forall_isOpen]
  intro U hxU hU
  rw [Measure.map_apply hm hU.measurableSet]
  exact (Measure.mem_support_iff_forall x).mp hx _ ((hU.preimage hc).mem_nhds hxU)

/-- Antipodal invariance transports an actual original support point
to its antipode. Source: arXiv:2601.21366v2, §6, meeting the
complementary component after `eq:log-identity-global`. -/
theorem antipode_mem_support_of_invariant {d : ℕ} (μ : Measure (SSphere d))
    (hanti : μ.map (antipodeMap d) = μ) (x : SSphere d) (hx : x ∈ μ.support) :
    antipodeMap d x ∈ μ.support := by
  rw [← hanti]
  exact mem_support_map_of_continuous_sphere μ _
    (Perspective.continuous_sphereMap d (LinearIsometryEquiv.neg ℝ))
    (measurable_antipodeMap d) x hx

/-- The actual support itself is antipodally invariant. Source:
arXiv:2601.21366v2, §6, the two original complementary components.
Both directions use the actual measure invariance and the original
antipodal involution; no support description is assumed. -/
theorem antipode_mem_support_iff_of_invariant {d : ℕ} (μ : Measure (SSphere d))
    (hanti : μ.map (antipodeMap d) = μ) (x : SSphere d) :
    antipodeMap d x ∈ μ.support ↔ x ∈ μ.support := by
  constructor
  · intro hx
    have h := antipode_mem_support_of_invariant μ hanti (antipodeMap d x) hx
    have heq : antipodeMap d (antipodeMap d x) = x :=
      Subtype.ext (by simp only [coe_antipodeMap, neg_neg])
    rwa [heq] at h
  · exact antipode_mem_support_of_invariant μ hanti x

/-- The constructed actual uniform circle law realizes the complete
antipodal hypothesis and has an actual support point. The original antipodal
map is continuous and measurable, realizing the support-map inputs jointly.
Source: arXiv:2601.21366v2, §6, the original parity and support steps.
The witness is a probability law on the genuine nonempty circle. -/
example : ∃ μ : Perspective.ProbSphere 2,
    (μ : Measure (SSphere 2)).map (antipodeMap 2) = (μ : Measure (SSphere 2)) ∧
    ∃ x : SSphere 2, x ∈ (μ : Measure (SSphere 2)).support ∧
      Continuous (antipodeMap 2) ∧ Measurable (antipodeMap 2) := by
  have hν := Metastability.isUniformOn_uniformLaw (d := 2) (by norm_num)
  let μ : Perspective.ProbSphere 2 := ⟨MeanField.uniformLaw 2, hν.1⟩
  have hμ₀ : (μ : Measure (SSphere 2)) ≠ 0 := by
    intro h
    have hm := congrArg (fun m : Measure (SSphere 2) => m univ) h
    simp at hm
  obtain ⟨x, hx⟩ := Measure.nonempty_support hμ₀
  exact ⟨μ, hν.2 (LinearIsometryEquiv.neg ℝ), x, hx,
    Perspective.continuous_sphereMap 2 (LinearIsometryEquiv.neg ℝ), measurable_antipodeMap 2⟩

end Transformer.Perceptron
