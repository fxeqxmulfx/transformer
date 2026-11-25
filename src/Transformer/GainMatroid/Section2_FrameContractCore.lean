module

public import Transformer.GainMatroid.Section2_FrameContractIndep

/-!
# Frame independence after the frame-type contraction of a joint

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1, contract a joint `e` at `v` of frame type:
the links at `v` become joints at their other ends, the other loops at `v` become balanced, and the
other balanced cycles are those of `B` avoiding `v`. A set `S` is frame-independent with none of its
cycles linked to `v` exactly when it contains no cycle of the contraction in the new collection and
no two distinct cycles of the contraction linked along `S`
(`BiasedGraph.frameContract_conditions_of_isFrameIndep`,
`BiasedGraph.isFrameIndep_of_frameContract_conditions`). A cycle of the contraction is a loop at
`v`, the loop that a link at `v` becomes at its other end, or a cycle of `G − v`; and a cycle of `S`
through `v` yields two such loops, linked along the rest of the cycle.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

namespace BiasedGraph

variable {V E : Type*} {Ω : BiasedGraph V E} {M : Matroid E}

/-- A frame-independent set `S` none of whose cycles is linked to `v` contains no cycle of the
frame-type contraction at `v` that the source declares balanced, and no two distinct cycles of it
linked along `S`. -/
theorem frameContract_conditions_of_isFrameIndep {S : Set E} {v : V} (hS : Ω.IsFrameIndep S)
    (hv : ¬Ω.graph.ReachesCycle S v) :
    (∀ C ⊆ S, (Ω.graph.frameContract v).IsCycle C →
      ¬((∃ f, Ω.graph.tail f = v ∧ Ω.graph.head f = v ∧ C = {f}) ∨
        (C ∈ Ω.balanced ∧ v ∉ Ω.graph.verts C))) ∧
    ∀ C D, (Ω.graph.frameContract v).IsCycle C → (Ω.graph.frameContract v).IsCycle D →
      C ⊆ S → D ⊆ S → ∀ u ∈ (Ω.graph.frameContract v).verts C,
        ∀ w ∈ (Ω.graph.frameContract v).verts D,
          (Ω.graph.frameContract v).ReachableIn S u w → C = D := by
  obtain ⟨hSB, hSl⟩ := isFrameIndep_iff.1 hS
  refine ⟨?_, fun C D hC hD hCS hDS u hu w hw huw => ?_⟩
  · rintro C hCS - (⟨f, ht, hh, rfl⟩ | ⟨hCB, -⟩)
    · exact hv ((isCycle_singleton_of_loop (ht.trans hh.symm)).reachesCycle hCS
        (mem_verts.2 ⟨f, rfl, Or.inl ht⟩))
    · exact hSB C hCS (Ω.isCycle_of_mem _ hCB) hCB
  -- a cycle of the contraction is a cycle of `G` avoiding `v` or the loop of a link at `v`
  have hanchor : ∀ C ⊆ S, (Ω.graph.frameContract v).IsCycle C →
      ∀ u ∈ (Ω.graph.frameContract v).verts C, (Ω.graph.IsCycle C ∧ u ∈ Ω.graph.verts C) ∨
        ∃ f ∈ S, C = {f} ∧ Ω.graph.Joins f u v := by
    intro C hCS hC u hu
    rcases hC.of_frameContract with ⟨f, hloop, rfl⟩ | ⟨hCG, hvC⟩
    · have hfS : f ∈ S := hCS rfl
      have hu' : (Ω.graph.frameContract v).tail f = u := by
        obtain ⟨g, hg, h⟩ := mem_verts.1 hu
        rw [show g = f from hg] at h
        exact h.elim id hloop.trans
      by_cases hfv : Ω.graph.tail f = v ∨ Ω.graph.head f = v
      · by_cases hl : Ω.graph.tail f = v ∧ Ω.graph.head f = v
        · exact absurd ((isCycle_singleton_of_loop (hl.1.trans hl.2.symm)).reachesCycle hCS
            (mem_verts.2 ⟨f, rfl, Or.inl hl.1⟩)) hv
        · have hj := (joins_frameContract_tail hfv hl).1
          rw [hu'] at hj
          exact Or.inr ⟨f, hfS, rfl, hj⟩
      · obtain ⟨ht, hh⟩ := not_or.1 hfv
        obtain ⟨h₁, h₂⟩ := frameContract_ends ht hh
        exact Or.inl ⟨isCycle_singleton_of_loop (h₁.symm.trans (hloop.trans h₂)),
          mem_verts.2 ⟨f, rfl, Or.inl (h₁.symm.trans hu')⟩⟩
    · exact Or.inl ⟨hCG, by rwa [← verts_frameContract hvC]⟩
  have huw' := reachableIn_frameContract_iff.1 huw
  have huwS : Ω.graph.ReachableIn S u w := huw'.mono fun k hk => (mem_awayFrom.1 hk).1
  rcases hanchor C hCS hC u hu with ⟨hCG, huC⟩ | ⟨f, hfS, rfl, hfj⟩ <;>
    rcases hanchor D hDS hD w hw with ⟨hDG, hwD⟩ | ⟨g, hgS, rfl, hgj⟩
  · exact hSl C D hCG hDG hCS hDS u huC w hwD huwS
  · exact absurd ((hCG.reachesCycle hCS huC).of_reachableIn (huwS.step hgS hgj)) hv
  · exact absurd ((hDG.reachesCycle hDS hwD).of_reachableIn (huwS.symm.step hfS hfj)) hv
  · by_cases hfg : f = g
    · rw [hfg]
    -- the links `f` and `g` at `v` close a cycle through `v`
    refine absurd ?_ hv
    have hgv : Ω.graph.tail g = v ∨ Ω.graph.head g = v := hgj.symm.tail_or_head
    have hsub : Ω.graph.awayFrom S v ⊆ S \ {g} := fun k hk => Set.mem_sdiff_of_mem
      (mem_awayFrom.1 hk).1 fun (hkg : k = g) => by
        rw [hkg] at hk
        exact hgv.elim (mem_awayFrom.1 hk).2.1 (mem_awayFrom.1 hk).2.2
    have hr : Ω.graph.ReachableIn (S \ {g}) v w :=
      ((reachableIn_refl _ v).step (Set.mem_sdiff_of_mem hfS hfg) hfj.symm).trans (huw'.mono hsub)
    have hends : Ω.graph.ReachableIn (S \ {g}) (Ω.graph.tail g) (Ω.graph.head g) := by
      rcases hgj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · rw [h₁, h₂]
        exact hr.symm
      · rw [h₁, h₂]
        exact hr
    obtain ⟨C', hC'S, hC', hgC'⟩ :=
      exists_isCycle_insert (Set.notMem_sdiff_of_mem (Set.mem_singleton g)) hends
    rw [Set.insert_sdiff_singleton, Set.insert_eq_of_mem hgS] at hC'S
    exact hC'.reachesCycle hC'S (mem_verts.2 ⟨g, hgC', hgv⟩)

/-- Conversely, a set containing no cycle of the frame-type contraction at `v` that the source
declares balanced, and no two distinct cycles of it linked along the set, is frame-independent, and
none of its cycles is linked to `v`. -/
theorem isFrameIndep_of_frameContract_conditions {S : Set E} {v : V}
    (hA : ∀ C ⊆ S, (Ω.graph.frameContract v).IsCycle C →
      ¬((∃ f, Ω.graph.tail f = v ∧ Ω.graph.head f = v ∧ C = {f}) ∨
        (C ∈ Ω.balanced ∧ v ∉ Ω.graph.verts C)))
    (hB : ∀ C D, (Ω.graph.frameContract v).IsCycle C → (Ω.graph.frameContract v).IsCycle D →
      C ⊆ S → D ⊆ S → ∀ u ∈ (Ω.graph.frameContract v).verts C,
        ∀ w ∈ (Ω.graph.frameContract v).verts D,
          (Ω.graph.frameContract v).ReachableIn S u w → C = D) :
    Ω.IsFrameIndep S ∧ ¬Ω.graph.ReachesCycle S v := by
  -- a link `f` from `x` to `v` becomes a loop at `x`
  have hloop : ∀ f x, Ω.graph.Joins f x v → x ≠ v → (Ω.graph.frameContract v).IsCycle {f} ∧
      x ∈ (Ω.graph.frameContract v).verts {f} := fun f x hf hx =>
    ⟨isCycle_singleton_of_loop ((frameContract_ends_of_joins hf hx).1.trans
      (frameContract_ends_of_joins hf hx).2.symm),
      mem_verts.2 ⟨f, rfl, Or.inl (frameContract_ends_of_joins hf hx).1⟩⟩
  -- no cycle of `S` passes through `v`
  have hT : ∀ C ⊆ S, Ω.graph.IsCycle C → v ∉ Ω.graph.verts C := by
    intro C hCS hC hvC
    obtain ⟨W, hW, rfl⟩ := hC.exists_walk_at hvC
    cases W with
    | nil => exact hW.edges_ne_nil rfl
    | @cons _ x _ f hf W' =>
      have hfS : f ∈ S := hCS List.mem_cons_self
      by_cases hx : x = v
      · rw [hx] at hf
        have hl : Ω.graph.tail f = v ∧ Ω.graph.head f = v := hf.elim id id
        have hl' := frameContract_ends_of_loop hl.1 hl.2
        exact hA {f} (Set.singleton_subset_iff.2 hfS)
          (isCycle_singleton_of_loop (hl'.1.trans hl'.2.symm)) (Or.inl ⟨f, hl.1, hl.2, rfl⟩)
      · rcases W'.reachableIn_awayFrom_or (S := S) (fun g hg => hCS (List.mem_cons_of_mem _ hg)) hx
          with ⟨-, hvv⟩ | ⟨g, hgW, y, hy, hgj, hxy⟩
        · exact hvv rfl
        · have hfg := hB {f} {g} (hloop f x hf.symm hx).1 (hloop g y hgj hy).1
            (Set.singleton_subset_iff.2 hfS)
            (Set.singleton_subset_iff.2 (hCS (List.mem_cons_of_mem _ hgW))) x
            (hloop f x hf.symm hx).2 y (hloop g y hgj hy).2 (reachableIn_frameContract_iff.2 hxy)
          exact (List.nodup_cons.1 hW.edges_nodup).1 (Set.singleton_injective hfg ▸ hgW)
  have hRC : ¬Ω.graph.ReachesCycle S v := by
    rintro ⟨C, hCS, hC, u, hu, p, hp⟩
    have hvC := hT C hCS hC
    rcases p.reachableIn_awayFrom_or (v := v) hp (fun h => hvC (h ▸ hu)) with
      ⟨-, hvv⟩ | ⟨g, hgp, y, hy, hgj, huy⟩
    · exact hvv rfl
    · have hCg := hB C {g} (hC.frameContract hvC) (hloop g y hgj hy).1 hCS
        (Set.singleton_subset_iff.2 (hp g hgp)) u (by rwa [verts_frameContract hvC]) y
        (hloop g y hgj hy).2 (reachableIn_frameContract_iff.2 huy)
      exact hvC (mem_verts.2 ⟨g, hCg ▸ Set.mem_singleton g, hgj.symm.tail_or_head⟩)
  refine ⟨isFrameIndep_iff.2 ⟨fun C hCS hC hCB => hA C hCS (hC.frameContract (hT C hCS hC))
    (Or.inr ⟨hCB, hT C hCS hC⟩), fun C D hC hD hCS hDS u hu w hw huw => ?_⟩, hRC⟩
  have hvC := hT C hCS hC
  obtain ⟨p, hp⟩ := huw
  rcases p.reachableIn_awayFrom_or (v := v) hp (fun h => hvC (h ▸ hu)) with
    ⟨huw', -⟩ | ⟨g, hgp, y, -, hgj, huy⟩
  · exact hB C D (hC.frameContract hvC) (hD.frameContract (hT D hDS hD)) hCS hDS u
      (by rwa [verts_frameContract hvC]) w (by rwa [verts_frameContract (hT D hDS hD)])
      (reachableIn_frameContract_iff.2 huw')
  · exact absurd ((hC.reachesCycle hCS hu).of_reachableIn
      ((huy.mono fun k hk => (mem_awayFrom.1 hk).1).step (hp g hgp) hgj)) hRC

end BiasedGraph

open BiasedGraph

example := frameContract_conditions_of_isFrameIndep (v := 1) pumpkin_isFrameIndep_zero
  fun ⟨C, hCK, hC, _⟩ => pumpkin_isForest C hC hCK

example : (pumpkin.gainBiased (1 : Fin 3 → ℚˣ)).IsFrameIndep {0} ∧ ¬pumpkin.ReachesCycle {0} 1 := by
  have h := frameContract_conditions_of_isFrameIndep (v := 1) pumpkin_isFrameIndep_zero
    fun ⟨C, hCK, hC, _⟩ => pumpkin_isForest C hC hCK
  exact isFrameIndep_of_frameContract_conditions h.1 h.2

end Transformer.GainMatroid
