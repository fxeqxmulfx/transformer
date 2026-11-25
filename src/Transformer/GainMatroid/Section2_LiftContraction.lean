module

public import Transformer.GainMatroid.Section2_DeletionCycles
public import Transformer.GainMatroid.Section2_LiftMatroid

/-!
# The lift-type contraction of a joint

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2: "for any link `e`, `L(G, B)/e = L((G, B)/e)`,
and if `e` is a joint then `L(G, B)/e = M(G \ e)`", citing Zaslavsky. The second claim is proved
here (`BiasedGraph.IsLiftContraction.isLiftMatroid`): the lift-type contraction of a joint `e` has
graph `G \ e` and all its cycles balanced, so its lift matroid is the cycle matroid of `G \ e`.

The independent sets of a lift matroid are the sets containing no lift circuit
(`BiasedGraph.IsLiftMatroid.indep_iff`), and a joint is one of them
(`BiasedGraph.IsJoint.isLiftIndep_singleton`). A set `I` avoiding `e` is independent in
`L(G, B)/e` when `I ∪ {e}` has no balanced cycle and nullity at most one. As `e` is an unbalanced
loop, this says that `I` is a forest of `G`, that is, of `G \ e`
(`OrientedGraph.isForest_delete_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {M : Matroid E}

/-- The independent sets of a lift matroid are the sets containing no lift circuit. -/
theorem IsLiftMatroid.indep_iff (hM : Ω.IsLiftMatroid M) {I : Set E} :
    M.Indep I ↔ Ω.IsLiftIndep I := by
  rw [Matroid.indep_iff_forall_subset_not_isCircuit (by rw [hM.1]; exact Set.subset_univ I)]
  exact forall₂_congr fun X _ => not_congr (hM.2 X)

/-- A joint is independent in the lift matroid. -/
theorem IsJoint.isLiftIndep_singleton [Finite V] [Finite E] {e : E} (he : Ω.IsJoint e) :
    Ω.IsLiftIndep {e} := by
  refine isLiftIndep_iff.2 ⟨fun C hCe hC hCB => ?_, ?_⟩
  · obtain ⟨f, hf⟩ := hC.nonempty
    have hfe : f = e := hCe hf
    subst hfe
    exact he.2 (by rwa [← Set.Subset.antisymm hCe (Set.singleton_subset_iff.2 hf)])
  · have h₁ := nullity_insert_le (G := Ω.graph) ∅ e
    rwa [nullity_eq_zero_iff.2 isForest_empty, LawfulSingleton.insert_empty_eq] at h₁

/-- **§2.2** (Zaslavsky). `L(G, B) / e = M(G \ e)` for a joint `e`: the lift-type contraction has
graph `G \ e` and all its cycles balanced. -/
theorem IsLiftContraction.isLiftMatroid [Finite V] [Finite E] {e : E}
    {Ω₂ : BiasedGraph V ↥({e}ᶜ : Set E)} (h : Ω.IsLiftContraction e Ω₂)
    (hM : Ω.IsLiftMatroid M) :
    Ω₂.IsLiftMatroid ((M.contract {e}).comap (Subtype.val : ↥({e}ᶜ : Set E) → E)) := by
  obtain ⟨N, hN⟩ := Ω₂.exists_isLiftMatroid
  suffices hNM : (M.contract {e}).comap (Subtype.val : ↥({e}ᶜ : Set E) → E) = N by
    rw [hNM]
    exact hN
  have hne : M.IsNonloop e :=
    Matroid.indep_singleton.1 (hM.indep_iff.2 h.isJoint.isLiftIndep_singleton)
  refine Matroid.ext_indep ?_ fun I _ => ?_
  · rw [hN.1, Matroid.comap_ground_eq, Matroid.contract_ground, hM.1]
    ext f
    simp
  have hS : e ∉ Subtype.val '' I := fun ⟨f, _, hf⟩ => f.2 hf
  have hloop : Ω.graph.ReachableIn (Subtype.val '' I) (Ω.graph.tail e) (Ω.graph.head e) := by
    rw [h.isJoint.1]
    exact reachableIn_refl _ _
  rw [Matroid.comap_indep_iff, hne.contractElem_indep_iff, hM.indep_iff, hN.indep_iff,
    isLiftIndep_iff, isLiftIndep_iff, nullity_insert_of_reachableIn hS hloop, h.graph_eq]
  constructor
  · rintro ⟨⟨-, -, hn⟩, -⟩
    have hF : (Ω.graph.delete {e}).IsForest I :=
      isForest_delete_iff.2 (nullity_eq_zero_iff.1 (by omega))
    refine ⟨fun C hCI hC => absurd hCI (hF C hC), ?_⟩
    rw [nullity_eq_zero_iff.2 hF]
    exact zero_le_one
  · rintro ⟨hB, -⟩
    have hF : (Ω.graph.delete {e}).IsForest I := fun C hC hCI =>
      hB C hCI hC (h.balanced C (by rw [h.graph_eq]; exact hC))
    have h₀ := nullity_eq_zero_iff.2 (isForest_delete_iff.1 hF)
    refine ⟨⟨hS, fun C hCI hC hCB => ?_, by omega⟩, Subtype.val_injective.injOn⟩
    have hle : Ω.graph.nullity (insert e (Subtype.val '' I)) ≤ 1 := by
      have := nullity_insert_le (G := Ω.graph) (Subtype.val '' I) e
      omega
    obtain rfl := eq_of_nullity_le_one hle hC (isCycle_singleton_of_loop h.isJoint.1) hCI
      (Set.singleton_subset_iff.2 (Set.mem_insert e _))
    exact h.isJoint.2 hCB

end BiasedGraph

open BiasedGraph

/-- The loop `true` of `twoLoops` is a joint when no cycle is balanced. -/
theorem twoLoops_isJoint_true : twoLoops.contrabalanced.IsJoint true :=
  ⟨rfl, Set.notMem_empty _⟩

example := (Classical.choose_spec triangleBiased.exists_isLiftMatroid).indep_iff (I := ∅)

example := twoLoops_isJoint_true.isLiftIndep_singleton

example : ∃ M, ((twoLoops.delete {true}).gainBiased
    (1 : ↥({true}ᶜ : Set Bool) → ℚˣ)).IsLiftMatroid M := by
  obtain ⟨M, hM⟩ := twoLoops.contrabalanced.exists_isLiftMatroid
  exact ⟨_, IsLiftContraction.isLiftMatroid (Ω := twoLoops.contrabalanced) (e := true)
    ⟨twoLoops_isJoint_true, rfl,
      fun _ hC => mem_balancedCycles_iff.2 ⟨hC, 1, fun f _ => by simp [switch]⟩⟩ hM⟩

end Transformer.GainMatroid
