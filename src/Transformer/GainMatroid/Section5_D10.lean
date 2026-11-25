module

public import Transformer.GainMatroid.Section2_CycleDegrees
public import Transformer.GainMatroid.Section2_NablaBiased
public import Transformer.GainMatroid.Section3_Counts

/-!
# `D_{1,0}` and Lemma 5.10

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §3.2 and §5.2. `D_{1,0}` is `K₄` with exactly one
balanced cycle, a triangle; here `d10` is `K₄` with the balanced triangle `{3, 4, 5}` away from the
vertex `0`. Lemma 5.10 says that a `Y`-`Δ` exchange on a claw meeting the balanced triangle in two
edges gives, after deleting one edge of a balanced 2-cycle, the biased `2C₃ \ e` with no balanced
2-cycle and exactly one balanced triangle, `B₁'`. The claw at `1` of `K₄` meets `{3, 4, 5}` in
`{3, 4}`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph BiasedGraph

theorem K4_isCycle_345 : K4.IsCycle {3, 4, 5} := by
  have h := (K4_isCycle_iff {3, 4, 5}).2 (by decide)
  simpa using h

theorem encard_345 : ({3, 4, 5} : Set (Fin 6)).encard = 3 := by
  rw [show ({3, 4, 5} : Set (Fin 6)) = ↑({3, 4, 5} : Finset (Fin 6)) by simp,
    Set.encard_coe_eq_coe_finsetCard]
  rfl

/-- `D_{1,0}` (§3.1): `K₄` whose only balanced cycle is the triangle `{3, 4, 5}` on `1`, `2`,
`3`. -/
def d10 : BiasedGraph (Fin 4) (Fin 6) := K4.singleCycleBiased {3, 4, 5} K4_isCycle_345

theorem d10_isD10 : IsD10 d10 := by
  refine ⟨IsReorientationOf.refl _, ?_, ?_⟩
  · have h : {C | C ∈ d10.balanced ∧ C.encard = 3} = {{3, 4, 5}} := by
      ext C
      simp only [d10, singleCycleBiased, Set.mem_singleton_iff, Set.mem_ofPred_eq]
      constructor
      · exact fun h => h.1
      · rintro rfl
        exact ⟨rfl, encard_345⟩
    rw [h, Set.ncard_singleton]
  · rintro C rfl
    rw [encard_345]
    decide

/-- The claw at `1` of `K₄`: the edges `0`, `3`, `4` to `0`, `2`, `3`. -/
theorem K4_isLabelledClaw_one : K4.IsLabelledClaw 1 0 3 4 0 2 3 := by
  refine ⟨Or.inr ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, Or.inl ⟨rfl, rfl⟩, by decide, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, ?_⟩
  simp only [K4]
  decide

/-- **Lemma 5.10.** "Let `C` be the balanced triangle of `D_{1,0}` and let `Y` be a
`K_{1,3}`-subgraph of `D_{1,0}` meeting `C` in exactly two edges. Then the simplification of
`∇_Y D_{1,0}` is isomorphic to `B₁'`." The simplification deletes one edge `f` of the only balanced
cycle of at most two edges, a 2-cycle `{f, g}`, and is presented as the link minor deleting `f`;
`B₁'` is `2C₃ \ e` with no balanced 2-cycle and exactly one balanced triangle. -/
theorem IsD10.nabla_simplification {Ω Ω' : BiasedGraph (Fin 4) (Fin 6)} (hΩ : IsD10 Ω)
    {C : Set (Fin 6)} (hC : C ∈ Ω.balanced) (hC3 : C.encard = 3) {v : Fin 4} {a b c : Fin 6}
    {x y z : Fin 4} (hY : Ω.IsNablaExchange v a b c x y z Ω')
    (hYC : (({a, b, c} : Set (Fin 6)) ∩ C).encard = 2) :
    ∃ f g, f ≠ g ∧ ({f, g} : Set (Fin 6)) ∈ Ω'.balanced ∧
      (∀ D ∈ Ω'.balanced, D.encard ≤ 2 → D = {f, g}) ∧
      ∃ S : BiasedGraph (Fin 3) (Fin 5), IsBPrime S ∧ balancedCount S 3 = 1 ∧
        ∃ m : Ω'.LinkMinorMap S, m.contractSet = ∅ ∧ m.deleteSet = {f} := by
  sorry

/-- The hypotheses of Lemma 5.10 hold for `d10`, its triangle `{3, 4, 5}` and the claw at `1`. -/
example : IsD10 d10 ∧ ({3, 4, 5} : Set (Fin 6)) ∈ d10.balanced ∧
    ({3, 4, 5} : Set (Fin 6)).encard = 3 ∧ (∃ Ω', d10.IsNablaExchange 1 0 3 4 0 2 3 Ω') ∧
    (({0, 3, 4} : Set (Fin 6)) ∩ {3, 4, 5}).encard = 2 := by
  refine ⟨d10_isD10, rfl, encard_345, d10.exists_isNablaExchange K4_isLabelledClaw_one, ?_⟩
  rw [show ({0, 3, 4} : Set (Fin 6)) ∩ {3, 4, 5} = {3, 4} by ext e; fin_cases e <;> simp,
    Set.encard_pair (by decide)]

end Transformer.GainMatroid
