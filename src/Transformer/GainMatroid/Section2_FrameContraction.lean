module

public import Transformer.GainMatroid.Section2_FrameContractCore

/-!
# The frame-type contraction of a joint

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2: "for any edge `e` of a biased graph
`(G, B)`, ..., `F(G, B)/e = F((G, B)/e)`", citing Zaslavsky. For a joint `e` at `v` the contraction
is of frame type (§2.1): "`G'` is obtained from `G` by adding each loop `e' ≠ e` incident to `v` to
`B` and replacing each link `f` incident to `v` with a joint incident to its other endpoint. The
collection `B'` is `B` restricted to the subgraph `G − v` along with any new balanced loops incident
to `v`." This is proved here (`BiasedGraph.IsFrameContraction.isFrameMatroid`).

A joint is independent in `F(G, B)`, so a set `S` avoiding `e` is independent in `F(G, B)/e` when
`S ∪ {e}` is independent in `F(G, B)`, that is, when `S` is independent and none of its cycles is
linked to `v` along `S`. This says that `S` is independent in `F(G', B')`
(`BiasedGraph.isFrameIndep_and_not_reachesCycle_iff`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {M : Matroid E}

/-- A set is frame-independent with none of its cycles linked to `v` exactly when it contains no
cycle of the frame-type contraction at `v` that the source declares balanced, and no two distinct
cycles of it linked along the set. -/
theorem isFrameIndep_and_not_reachesCycle_iff {S : Set E} {v : V} :
    Ω.IsFrameIndep S ∧ ¬Ω.graph.ReachesCycle S v ↔
      (∀ C ⊆ S, (Ω.graph.frameContract v).IsCycle C →
        ¬((∃ f, Ω.graph.tail f = v ∧ Ω.graph.head f = v ∧ C = {f}) ∨
          (C ∈ Ω.balanced ∧ v ∉ Ω.graph.verts C))) ∧
      ∀ C D, (Ω.graph.frameContract v).IsCycle C → (Ω.graph.frameContract v).IsCycle D →
        C ⊆ S → D ⊆ S → ∀ u ∈ (Ω.graph.frameContract v).verts C,
          ∀ w ∈ (Ω.graph.frameContract v).verts D,
            (Ω.graph.frameContract v).ReachableIn S u w → C = D :=
  ⟨fun h => frameContract_conditions_of_isFrameIndep h.1 h.2,
    fun h => isFrameIndep_of_frameContract_conditions h.1 h.2⟩

/-- **§2.2** (Zaslavsky). `F(G, B) / e = F((G, B) / e)` for a joint `e`, by the frame-type
contraction. -/
theorem IsFrameContraction.isFrameMatroid [Finite V] [Finite E] {e : E} {v : V}
    {Ω₂ : BiasedGraph V ↥({e}ᶜ : Set E)} (h : Ω.IsFrameContraction e v Ω₂)
    (hM : Ω.IsFrameMatroid M) :
    Ω₂.IsFrameMatroid ((M.contract {e}).comap (Subtype.val : ↥({e}ᶜ : Set E) → E)) := by
  obtain ⟨N, hN⟩ := Ω₂.exists_isFrameMatroid
  suffices hNM : (M.contract {e}).comap (Subtype.val : ↥({e}ᶜ : Set E) → E) = N by
    rw [hNM]
    exact hN
  have hne : M.IsNonloop e :=
    Matroid.indep_singleton.1 (hM.indep_iff.2 h.isJoint.isFrameIndep_singleton)
  refine Matroid.ext_indep ?_ fun I _ => ?_
  · rw [hN.1, Matroid.comap_ground_eq, Matroid.contract_ground, hM.1]
    ext f
    simp
  have hS : e ∉ Subtype.val '' I := fun ⟨f, _, hf⟩ => f.2 hf
  rw [Matroid.comap_indep_iff, hne.contractElem_indep_iff, hM.indep_iff, hN.indep_iff,
    h.isJoint.isFrameIndep_insert_iff hS, h.tail_eq,
    isFrameIndep_iff_of_graph_eq_delete h.graph_eq, and_iff_left Subtype.val_injective.injOn,
    and_iff_right hS, isFrameIndep_and_not_reachesCycle_iff]
  refine and_congr (forall₂_congr fun C hCI => forall_congr' fun hC => ?_) Iff.rfl
  have hr : Subtype.val '' (Subtype.val ⁻¹' C) = C :=
    Set.image_preimage_eq_of_subset (hCI.trans (Set.image_subset_range _ _))
  rw [h.balanced_iff _ (by rw [h.graph_eq, isCycle_delete_iff, hr]; exact hC), hr]
  refine not_congr (or_congr ⟨fun ⟨f, ht, hh, hf⟩ => ?_, fun ⟨f, ht, hh, hf⟩ => ?_⟩ Iff.rfl)
  · obtain ⟨g, -, rfl⟩ := hCI (hf ▸ Set.mem_singleton f)
    refine ⟨g, ht, hh, ?_⟩
    rw [hf]
    ext k
    simp [Subtype.ext_iff]
  · exact ⟨f, ht, hh, by rw [← hr, hf, Set.image_singleton]⟩

end BiasedGraph

open BiasedGraph

example := isFrameIndep_and_not_reachesCycle_iff (Ω := triangleBiased) (S := ∅) (v := 0)

example : ∃ M, twoLoopsContract.IsFrameMatroid M := by
  obtain ⟨M, hM⟩ := twoLoops.contrabalanced.exists_isFrameMatroid
  exact ⟨_, twoLoops_isFrameContraction.isFrameMatroid hM⟩

end Transformer.GainMatroid
