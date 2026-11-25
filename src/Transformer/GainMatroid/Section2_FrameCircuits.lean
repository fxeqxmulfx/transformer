module

public import Transformer.GainMatroid.Section2_FrameIndep

/-!
# No frame circuit lies in another

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, list the circuits of `F(G, B)` after
Zaslavsky (`BiasedGraph.IsFrameCircuit`). That none of them lies in another
(`BiasedGraph.IsFrameCircuit.eq_of_subset`) is proved here with the nullity. A tight handcuff, a
loose handcuff or a theta `X` has nullity at most two. A frame circuit inside `X` without the edge
`e` contains two distinct cycles, so the ends of `e` cannot be linked without `e`. They are linked
if `e` lies on a cycle of `X`. Otherwise `X` is a loose handcuff `C ∪ D ∪ P` and `e` lies on `P`.
The only cycles of `X` are then `C` and `D`: a cycle through an edge `f` of `P` would leave
`C ∪ D`, of nullity two, inside `X` without `f`, of nullity one; and a cycle inside `C ∪ D` is one
of them (`OrientedGraph.IsCycle.eq_or_eq_of_subset_union`). So the circuit inside links `C` to `D`
without `e`, and with the two parts of `P` this links the ends of `e`.
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- Along the union of two sets of edges without common vertices, a vertex of the first is linked
only to vertices of the first. -/
theorem mem_verts_of_reachableIn_union {C D : Set E} (hCD : Disjoint (G.verts C) (G.verts D))
    {u v : V} (h : G.ReachableIn (C ∪ D) u v) (hu : u ∈ G.verts C) : v ∈ G.verts C := by
  refine (ReachableIn.rel (r := fun a b => (a ∈ G.verts C ↔ b ∈ G.verts C)) (fun _ => Iff.rfl)
    (fun _ _ _ h₁ h₂ => h₁.trans h₂) (fun e he => ?_) h).1 hu
  have key : G.tail e ∈ G.verts C ↔ G.head e ∈ G.verts C := by
    rcases he with he | he
    · exact iff_of_true (mem_verts.2 ⟨e, he, Or.inl rfl⟩) (mem_verts.2 ⟨e, he, Or.inr rfl⟩)
    · exact iff_of_false (Set.disjoint_right.1 hCD (mem_verts.2 ⟨e, he, Or.inl rfl⟩))
        (Set.disjoint_right.1 hCD (mem_verts.2 ⟨e, he, Or.inr rfl⟩))
  exact ⟨key, key.symm⟩

/-- A cycle inside the union of two sets of edges without common vertices lies in the one that
contains one of its edges. -/
theorem IsCycle.subset_of_subset_union {C D Z : Set E} (hCD : Disjoint (G.verts C) (G.verts D))
    (hZ : G.IsCycle Z) (hZCD : Z ⊆ C ∪ D) {g : E} (hgZ : g ∈ Z) (hgC : g ∈ C) : Z ⊆ C := by
  intro h hh
  have hC : G.tail h ∈ G.verts C := mem_verts_of_reachableIn_union hCD
    ((hZ.reachableIn (mem_verts.2 ⟨g, hgZ, Or.inl rfl⟩) (mem_verts.2 ⟨h, hh, Or.inl rfl⟩)).mono
      hZCD) (mem_verts.2 ⟨g, hgC, Or.inl rfl⟩)
  exact (hZCD hh).resolve_right fun hhD =>
    Set.disjoint_left.1 hCD hC (mem_verts.2 ⟨h, hhD, Or.inl rfl⟩)

/-- A cycle inside the union of two vertex-disjoint cycles is one of them. -/
theorem IsCycle.eq_or_eq_of_subset_union {C D Z : Set E} (hC : G.IsCycle C) (hD : G.IsCycle D)
    (hCD : Disjoint (G.verts C) (G.verts D)) (hZ : G.IsCycle Z) (hZCD : Z ⊆ C ∪ D) :
    Z = C ∨ Z = D := by
  obtain ⟨g, hg⟩ := hZ.nonempty
  rcases hZCD hg with hgC | hgD
  · exact Or.inl (hC.eq_of_subset hZ (hZ.subset_of_subset_union hCD hZCD hg hgC))
  · exact Or.inr (hD.eq_of_subset hZ (hZ.subset_of_subset_union hCD.symm
      (hZCD.trans (Set.union_comm C D).subset) hg hgD))

end OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} [Finite V] [Finite E]

/-- **No circuit of `F(G, B)` lies in another.** -/
theorem IsFrameCircuit.eq_of_subset {X Y : Set E} (hX : Ω.IsFrameCircuit X)
    (hY : Ω.IsFrameCircuit Y) (hYX : Y ⊆ X) : Y = X := by
  rcases hX with hXB | ⟨hX, hXB⟩
  · obtain ⟨C, hCY, hC⟩ := hY.exists_isCycle
    have hCX : C = X := (Ω.isCycle_of_mem X hXB).eq_of_subset hC (hCY.trans hYX)
    exact hYX.antisymm (hCX ▸ hCY)
  rcases hY with hYB | ⟨hY, -⟩
  · exact absurd hYB (hXB Y hYX (Ω.isCycle_of_mem Y hYB))
  by_contra hne
  obtain ⟨e, heX, heY⟩ := Set.exists_of_ssubset (hYX.ssubset_of_ne hne)
  have hYe : Y ⊆ X \ {e} := fun f hf => ⟨hYX hf, fun h => heY (h ▸ hf)⟩
  obtain ⟨C₁, D₁, hC₁, hD₁, hCD₁, hC₁Y, hD₁Y, u, hu, v, hv, huv⟩ :=
    exists_linked_cycles_of_frame_bicycle hY
  have h₂ := two_le_nullity hC₁ hD₁ hCD₁ (hC₁Y.trans hYe) (hD₁Y.trans hYe)
  have hX₂ : Ω.graph.nullity X ≤ 2 :=
    hX.elim IsTightHandcuff.nullity_le fun h => h.elim IsLooseHandcuff.nullity_le
      IsThetaSet.nullity_le
  suffices hr : Ω.graph.ReachableIn (X \ {e}) (Ω.graph.tail e) (Ω.graph.head e) by
    have := nullity_diff_singleton_add_one heX hr
    omega
  by_cases hcyc : ∃ C ⊆ X, Ω.graph.IsCycle C ∧ e ∈ C
  · obtain ⟨C, hCX, hC, heC⟩ := hcyc
    exact (hC.reachableIn_diff heC).mono (Set.sdiff_subset_sdiff_left hCX)
  push Not at hcyc
  obtain ⟨C, D, x, y, P, hC, hD, hCD, hP, hx, hy, -, -, rfl⟩ : Ω.graph.IsLooseHandcuff X := by
    rcases hX with hX | hX | hX
    · obtain ⟨C, hCX, hC, heC⟩ := (exists_two_cycles_of_bicycle (Or.inl hX)).2 e heX
      exact absurd heC (hcyc C hCX hC)
    · exact hX
    · obtain ⟨C, hCX, hC, heC⟩ := (exists_two_cycles_of_bicycle (Or.inr (Or.inr hX))).2 e heX
      exact absurd heC (hcyc C hCX hC)
  have hCX : C ⊆ C ∪ D ∪ P.edgeSet := Set.subset_union_left.trans Set.subset_union_left
  have hDX : D ⊆ C ∪ D ∪ P.edgeSet := Set.subset_union_right.trans Set.subset_union_left
  have heC : e ∉ C := hcyc C hCX hC
  have heD : e ∉ D := hcyc D hDX hD
  have heP : e ∈ P.edges := heX.resolve_left fun h => h.elim heC heD
  have hCne : C ≠ D := fun h => by
    obtain ⟨g, hg⟩ := hC.nonempty
    have hg' : Ω.graph.tail g ∈ Ω.graph.verts C := mem_verts.2 ⟨g, hg, Or.inl rfl⟩
    exact Set.disjoint_left.1 hCD hg' (h ▸ hg')
  -- a cycle of `X` avoids `P`, so it is `C` or `D`
  have hsub : ∀ Z ⊆ Y, Ω.graph.IsCycle Z → Z = C ∨ Z = D := by
    intro Z hZY hZ
    refine hC.eq_or_eq_of_subset_union hD hCD hZ fun f hfZ => ?_
    by_contra hf
    have h₁ := nullity_diff_add_one_of_isCycle hZ (hZY.trans hYX) hfZ
    have h₃ := two_le_nullity (S := (C ∪ D ∪ P.edgeSet) \ {f}) hC hD hCne
      (fun g hg => ⟨hCX hg, fun h => hf (Or.inl (h ▸ hg))⟩)
      (fun g hg => ⟨hDX hg, fun h => hf (Or.inr (h ▸ hg))⟩)
    omega
  obtain ⟨c, hc, d, hd, hcd⟩ : ∃ c ∈ Ω.graph.verts C, ∃ d ∈ Ω.graph.verts D,
      Ω.graph.ReachableIn ((C ∪ D ∪ P.edgeSet) \ {e}) c d := by
    have huv' := huv.mono hYe
    rcases hsub C₁ hC₁Y hC₁ with h₁ | h₁ <;> rcases hsub D₁ hD₁Y hD₁ with h₂ | h₂
    · exact absurd (h₁.trans h₂.symm) hCD₁
    · exact ⟨u, h₁ ▸ hu, v, h₂ ▸ hv, huv'⟩
    · exact ⟨v, h₂ ▸ hv, u, h₁ ▸ hu, huv'.symm⟩
    · exact absurd (h₁.trans h₂.symm) hCD₁
  -- split `P` at `e`
  obtain ⟨s, t, g, hg, A, B, rfl, hA, hge⟩ :=
    P.exists_eq_append_cons_first (C := {f | f ≠ e}) ⟨e, heP, fun h => h rfl⟩
  obtain rfl : g = e := by simpa using hge
  have hnd := hP.edges_nodup
  rw [Walk.edges_append, Walk.edges] at hnd
  have hgB : g ∉ B.edges := (List.nodup_cons.1 hnd.of_append_right).1
  set S := (C ∪ D ∪ (A.append (Walk.cons g hg B)).edgeSet) \ {g}
  have hCS : C ⊆ S := fun f hf => ⟨hCX hf, fun h => heC (h ▸ hf)⟩
  have hDS : D ⊆ S := fun f hf => ⟨hDX hf, fun h => heD (h ▸ hf)⟩
  have hAS : Ω.graph.ReachableIn S x s :=
    ⟨A, fun f hf => ⟨Or.inr (Walk.mem_edges_append.2 (Or.inl hf)), hA f hf⟩⟩
  have hBS : Ω.graph.ReachableIn S t y := ⟨B, fun f hf =>
    ⟨Or.inr (Walk.mem_edges_append.2 (Or.inr (List.mem_cons_of_mem _ hf))),
      fun h => hgB (h ▸ hf)⟩⟩
  have hst : Ω.graph.ReachableIn S s t := hAS.symm.trans (((hC.reachableIn hx hc).mono hCS).trans
    (hcd.trans (((hD.reachableIn hd hy).mono hDS).trans hBS.symm)))
  obtain ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ := id hg
  · rw [h₁, h₂]
    exact hst
  · rw [h₁, h₂]
    exact hst.symm

end BiasedGraph

example := mem_verts_of_reachableIn_union loopLink_disjoint (reachableIn_refl _ (0 : Fin 2))
  (mem_verts.2 ⟨0, rfl, Or.inl rfl⟩)

example := (isCycle_singleton_of_loop (G := loopLink) (e := 0) rfl).subset_of_subset_union
  loopLink_disjoint Set.subset_union_left (Set.mem_singleton 0) (Set.mem_singleton 0)

example := (isCycle_singleton_of_loop (G := loopLink) (e := 0) rfl).eq_or_eq_of_subset_union
  (isCycle_singleton_of_loop (G := loopLink) (e := 1) rfl) loopLink_disjoint
  (isCycle_singleton_of_loop (G := loopLink) (e := 0) rfl) Set.subset_union_left

example := BiasedGraph.IsFrameCircuit.eq_of_subset (Ω := triangleBiased) (X := {1, 2, 0})
  (Y := {1, 2, 0}) (Or.inl rfl) (Or.inl rfl) le_rfl

end Transformer.GainMatroid
