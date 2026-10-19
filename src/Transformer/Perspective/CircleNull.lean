module

public import Transformer.Perspective.CircleUniform

/-!
# Null sets of the uniform law on the circle

The angle map `θ ↦ ((cos θ_i, sin θ_i))_i` has a measurable section, the
argument (`sphereAngles`), and its fibres are the orbits of `2πℤⁿ`
(`exists_int_of_circlePoint_eq`). Every `UniformTuple 2 n P` is its image of
the normalized Lebesgue measure on `[0, 2π)ⁿ` (`eq_map_angleCube`). So a set
of configurations whose preimage in angles is Lebesgue-null is `P`-null,
measurable or not (`measure_eq_zero_of_volume_preimage`): what holds for almost
every tuple of angles holds `P`-almost surely (`ae_of_ae_volume`), and a
`2π`-periodic Lebesgue-null set of angles maps to a `P`-null set
(`measure_image_eq_zero`).

Source: arXiv:2312.10794v5, §4 (the "almost every" of `p:beta0` and
`thm: beta.tiny`, for initial configurations drawn from `σ_d`) and §7.1
(particles on the circle as angles).
-/

@[expose] public section

open MeasureTheory Complex Set
open scoped Real ENNReal

namespace Transformer
namespace Perspective

theorem measurable_anglesToSphere {n : ℕ} : Measurable (anglesToSphere (n := n)) :=
  Measurable.of_eval fun i => measurable_circlePoint.comp (measurable_pi_apply i)

theorem measurable_sphereAngles {n : ℕ} : Measurable (sphereAngles (n := n)) :=
  Measurable.of_eval fun i => measurable_arg.comp
    (complexEuc.symm.continuous.measurable.comp (continuous_subtype_val.measurable.comp
      (measurable_pi_apply i)))

/-- **Null sets in angles are null for the uniform law.**  If the set of angles
`θ` with `((cos θ_i, sin θ_i))_i ∈ A` is Lebesgue-null, then `P A = 0` for every
`UniformTuple 2 n P`; `A` need not be measurable. The `2πℤⁿ`-saturation of a
measurable null hull of the preimage is null, and its preimage under
`sphereAngles` is a measurable set containing `A`. -/
theorem measure_eq_zero_of_volume_preimage {n : ℕ} {P : Measure (SphereTuple 2 n)}
    (hP : UniformTuple 2 n P) {A : Set (SphereTuple 2 n)}
    (hA : volume (anglesToSphere ⁻¹' A) = 0) : P A = 0 := by
  set T := toMeasurable volume (anglesToSphere ⁻¹' A)
  set T' : Set (Angles n) := ⋃ k : Idx n → ℤ, (fun θ => θ + fun i => 2 * π * k i) ⁻¹' T
  have hT' : MeasurableSet T' :=
    MeasurableSet.iUnion fun k => (measurableSet_toMeasurable _ _).preimage (measurable_add_const _)
  have hT'0 : volume T' = 0 := measure_iUnion_null fun k => by
    rw [measure_preimage_add_right, measure_toMeasurable, hA]
  have h0 : (fun i => 2 * π * ((0 : Idx n → ℤ) i : ℝ)) = 0 := funext fun i => by simp
  have hAB : A ⊆ sphereAngles ⁻¹' T' := fun X hX => mem_iUnion.2 ⟨0, by
    rw [mem_preimage, h0, add_zero]
    exact subset_toMeasurable (volume : Measure (Angles n)) (anglesToSphere ⁻¹' A)
      (show anglesToSphere (sphereAngles X) ∈ A by rwa [anglesToSphere_sphereAngles])⟩
  have hsub : anglesToSphere ⁻¹' (sphereAngles ⁻¹' T') ⊆ T' := by
    intro θ hθ
    obtain ⟨k, hk⟩ := mem_iUnion.1 hθ
    have hm : ∀ i, ∃ m : ℤ, θ i = sphereAngles (anglesToSphere θ) i + 2 * π * m := fun i =>
      exists_int_of_circlePoint_eq
        (congrFun (anglesToSphere_sphereAngles (anglesToSphere θ)) i).symm
    choose m hm using hm
    have hθk : (θ + fun i => 2 * π * ((k - m) i : ℤ)) =
        sphereAngles (anglesToSphere θ) + fun i => 2 * π * k i := funext fun i => by
      simp only [Pi.add_apply, Pi.sub_apply, Int.cast_sub]
      rw [hm i]
      ring
    exact mem_iUnion.2 ⟨k - m, by rw [mem_preimage, hθk]; exact hk⟩
  have hB := measurable_anglesToSphere (measurable_sphereAngles hT')
  refine measure_mono_null hAB ?_
  rw [eq_map_angleCube hP,
    Measure.map_apply measurable_anglesToSphere (measurable_sphereAngles hT'), Measure.smul_apply,
    Measure.restrict_apply hB, smul_eq_mul]
  exact mul_eq_zero_of_right _ (measure_mono_null (inter_subset_left.trans hsub) hT'0)

/-- **Almost every tuple of angles gives almost every configuration**: a
property of `((cos θ_i, sin θ_i))_i` that holds for Lebesgue-almost every `θ`
holds `P`-almost surely, for every `UniformTuple 2 n P`. -/
theorem ae_of_ae_volume {n : ℕ} {P : Measure (SphereTuple 2 n)} (hP : UniformTuple 2 n P)
    {p : SphereTuple 2 n → Prop} (h : ∀ᵐ θ ∂(volume : Measure (Angles n)), p (anglesToSphere θ)) :
    ∀ᵐ X ∂P, p X :=
  ae_iff.2 (measure_eq_zero_of_volume_preimage hP (ae_iff.1 h))

/-- **A `2π`-periodic Lebesgue-null set of angles maps to a null set**: if `S` is
invariant under the translations by `2πℤⁿ` and Lebesgue-null, its image in
`(𝕊¹)ⁿ` is `P`-null, for every `UniformTuple 2 n P`. -/
theorem measure_image_eq_zero {n : ℕ} {P : Measure (SphereTuple 2 n)} (hP : UniformTuple 2 n P)
    {S : Set (Angles n)} (hS : ∀ θ ∈ S, ∀ k : Idx n → ℤ, (θ + fun i => 2 * π * k i) ∈ S)
    (h0 : volume S = 0) : P (anglesToSphere '' S) = 0 := by
  refine measure_eq_zero_of_volume_preimage hP (measure_mono_null ?_ h0)
  rintro θ ⟨φ, hφ, hφθ⟩
  have hm : ∀ i, ∃ m : ℤ, θ i = φ i + 2 * π * m := fun i =>
    exists_int_of_circlePoint_eq (congrFun hφθ i).symm
  choose m hm using hm
  convert hS φ hφ m using 1
  funext i
  exact hm i

/-- The hypotheses of the theorems above are satisfiable: the product of circle
laws is a uniform tuple law, the empty set is null and periodic, and a property
that always holds holds almost everywhere. -/
example : UniformTuple 2 3 (Measure.pi fun _ : Idx 3 => circleLaw) ∧
    volume (anglesToSphere ⁻¹' (∅ : Set (SphereTuple 2 3))) = 0 ∧
    (∀ θ ∈ (∅ : Set (Angles 3)), ∀ k : Idx 3 → ℤ, (θ + fun i => 2 * π * k i) ∈ (∅ : Set _)) ∧
    volume (∅ : Set (Angles 3)) = 0 ∧
    ∀ᵐ θ ∂(volume : Measure (Angles 3)), (fun _ : SphereTuple 2 3 => True) (anglesToSphere θ) :=
  ⟨uniformTuple_circleLaw 3, by simp, fun _ h => h.elim, measure_empty,
    Filter.Eventually.of_forall fun _ => trivial⟩

end Perspective
end Transformer
