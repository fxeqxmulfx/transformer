module

public import Transformer.GainMatroid.Section2_LiftIndep

/-!
# The independent sets of the frame matroid

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2: a circuit of `F(G, B)` is a balanced cycle,
or a tight handcuff, a loose handcuff or a theta containing no balanced cycle
(`BiasedGraph.IsFrameCircuit`). The sets containing none of them (`BiasedGraph.IsFrameIndep`) are
those with no balanced cycle in which no two distinct cycles are linked
(`BiasedGraph.isFrameIndep_iff`). Two distinct cycles that share a vertex contain a tight handcuff
or a theta (`OrientedGraph.IsCycle.exists_frame_bicycle`), and two vertex-disjoint cycles linked
along `S` form a loose handcuff with a path along `S`
(`OrientedGraph.IsCycle.exists_isLooseHandcuff`). Conversely, each of these contains two distinct
cycles linked along it (`OrientedGraph.exists_linked_cycles_of_frame_bicycle`).
-/

@[expose] public section

namespace Transformer.GainMatroid

open OrientedGraph

variable {V E : Type*}

namespace OrientedGraph

variable {G : OrientedGraph V E}

/-- Two vertices of a cycle are linked along it. -/
theorem IsCycle.reachableIn {C : Set E} (hC : G.IsCycle C) {u v : V} (hu : u ∈ G.verts C)
    (hv : v ∈ G.verts C) : G.ReachableIn C u v := by
  obtain ⟨W, -, rfl⟩ := hC.exists_walk_at hu
  obtain ⟨A, B, rfl⟩ := W.exists_eq_append_of_mem_support (Walk.mem_support_of_mem_verts hv)
  exact ⟨A, fun e he => Walk.mem_edges_append.2 (Or.inl he)⟩

/-- The vertices of a cycle through `e` inside `insert e I` are linked along `I` to the tail of
`e`. -/
theorem IsCycle.reachableIn_of_mem_insert {D I : Set E} {e : E} (hD : G.IsCycle D) (heD : e ∈ D)
    (hDI : D ⊆ insert e I) {w : V} (hw : w ∈ G.verts D) : G.ReachableIn I w (G.tail e) := by
  have h : G.ReachableIn (D \ {e}) w (G.tail e) :=
    (hD.reachableIn hw (mem_verts.2 ⟨e, heD, Or.inl rfl⟩)).of_forall fun f hf => by
      by_cases hfe : f = e
      · subst hfe
        exact hD.reachableIn_diff hf
      · exact reachableIn_of_mem ⟨hf, hfe⟩
  have hsub : D \ {e} ⊆ I := fun f hf => (hDI hf.1).resolve_left hf.2
  exact h.mono hsub

/-- Two distinct cycles that share a vertex contain a tight handcuff or a theta. -/
theorem IsCycle.exists_frame_bicycle {C D : Set E} (hC : G.IsCycle C) (hD : G.IsCycle D)
    (hCD : C ≠ D) {v : V} (hvC : v ∈ G.verts C) (hvD : v ∈ G.verts D) :
    ∃ X ⊆ C ∪ D, G.IsTightHandcuff X ∨ G.IsThetaSet X := by
  have hDC : ¬D ⊆ C := fun h => hCD (hC.eq_of_subset hD h).symm
  obtain ⟨u, a, y, g, hg, P, hu, hgD, hgC, hP, hgP, hPD, hy, hPC⟩ := hD.exists_ear hDC hvC hvD
  refine ⟨insert g P.edgeSet ∪ C, fun e he => ?_,
    hC.isTightHandcuff_or_isThetaSet_insert hg hu hgC hP hgP hy hPC⟩
  rcases he with (rfl | he) | he
  · exact Or.inr hgD
  · exact Or.inr (hPD e he)
  · exact Or.inl he

/-- Two vertex-disjoint cycles linked along `S` form a loose handcuff with a path along `S`. -/
theorem IsCycle.exists_isLooseHandcuff {S C D : Set E} (hC : G.IsCycle C) (hD : G.IsCycle D)
    (hCD : Disjoint (G.verts C) (G.verts D)) {u v : V} (hu : u ∈ G.verts C)
    (hv : v ∈ G.verts D) (h : G.ReachableIn S u v) :
    ∃ (x y : V) (P : G.Walk x y), (∀ e ∈ P.edges, e ∈ S) ∧
      G.IsLooseHandcuff (C ∪ D ∪ P.edgeSet) := by
  obtain ⟨p, hp⟩ := h
  obtain ⟨q, hq, hqp⟩ := p.exists_isPath
  obtain ⟨y, A, B, hy, rfl, hA⟩ := q.exists_eq_append_first hv
  have hApath : A.IsPath := hq.of_append_left
  obtain ⟨x, A', B', hx, hAB, hA'⟩ := A.reverse.exists_eq_append_first hu
  have hA'path : A'.IsPath := by
    have := hApath.reverse
    rw [hAB] at this
    exact this.of_append_left
  have hsupp : ∀ z ∈ A'.support, z ∈ A.support := fun z hz => by
    have : z ∈ A.reverse.support := hAB ▸ Walk.mem_support_append.2 (Or.inl hz)
    exact Walk.mem_support_reverse.1 this
  have hedges : ∀ e ∈ A'.edges, e ∈ S := fun e he => by
    have : e ∈ A.reverse.edges := hAB ▸ Walk.mem_edges_append.2 (Or.inl he)
    exact hp e (hqp e (Walk.mem_edges_append.2 (Or.inl (Walk.mem_edges_reverse.1 this))))
  exact ⟨x, y, A'.reverse, fun e he => hedges e (Walk.mem_edges_reverse.1 he),
    C, D, x, y, A'.reverse, hC, hD, hCD, hA'path.reverse, hx, hy,
    fun z hz hzC => hA' z (Walk.mem_support_reverse.1 hz) hzC,
    fun z hz hzD => hA z (hsupp z (Walk.mem_support_reverse.1 hz)) hzD, rfl⟩

/-- A tight handcuff, a loose handcuff or a theta contains two distinct cycles linked along it. -/
theorem exists_linked_cycles_of_frame_bicycle {X : Set E}
    (h : G.IsTightHandcuff X ∨ G.IsLooseHandcuff X ∨ G.IsThetaSet X) :
    ∃ C D, G.IsCycle C ∧ G.IsCycle D ∧ C ≠ D ∧ C ⊆ X ∧ D ⊆ X ∧
      ∃ u ∈ G.verts C, ∃ v ∈ G.verts D, G.ReachableIn X u v := by
  rcases h with ⟨C, D, v, hC, hD, hCD, hv, rfl⟩ |
    ⟨C, D, x, y, P, hC, hD, hCD, -, hx, hy, -, -, rfl⟩ | ⟨x, y, P, Q, R, hT, rfl⟩
  · have hv' : v ∈ G.verts C ∩ G.verts D := hv ▸ Set.mem_singleton v
    refine ⟨C, D, hC, hD, fun h => ?_, Set.subset_union_left, Set.subset_union_right,
      v, hv'.1, v, hv'.2, reachableIn_refl _ v⟩
    obtain ⟨e, he⟩ := hC.nonempty
    exact Set.disjoint_left.1 hCD he (h ▸ he)
  · exact ⟨C, D, hC, hD, fun h => Set.disjoint_left.1 hCD hx (h ▸ hx),
      Set.subset_union_left.trans Set.subset_union_left,
      Set.subset_union_right.trans Set.subset_union_left, x, hx, y, hy,
      P, fun e he => Or.inr he⟩
  · obtain ⟨hPQ, hPR, -⟩ := hT.isCycle
    have hx : x ∈ G.verts P.edgeSet :=
      P.mem_verts_of_mem_support (Walk.edges_ne_nil_of_ne hT.ne P) P.start_mem_support
    exact ⟨_, _, hPQ, hPR, hT.union_ne, Set.subset_union_left,
      Set.union_subset (Set.subset_union_left.trans Set.subset_union_left)
        Set.subset_union_right,
      x, verts_mono Set.subset_union_left hx, x, verts_mono Set.subset_union_left hx,
      reachableIn_refl _ x⟩

end OrientedGraph

namespace BiasedGraph

variable (Ω : BiasedGraph V E)

/-- The sets of edges containing no circuit of `F(G, B)`. -/
def IsFrameIndep (I : Set E) : Prop :=
  ∀ X ⊆ I, ¬Ω.IsFrameCircuit X

variable {Ω}

/-- A set contains no frame circuit exactly when it contains no balanced cycle and no two distinct
cycles linked along it. -/
theorem isFrameIndep_iff {I : Set E} : Ω.IsFrameIndep I ↔
    (∀ C ⊆ I, Ω.graph.IsCycle C → C ∉ Ω.balanced) ∧
      ∀ C D, Ω.graph.IsCycle C → Ω.graph.IsCycle D → C ⊆ I → D ⊆ I →
        ∀ u ∈ Ω.graph.verts C, ∀ v ∈ Ω.graph.verts D, Ω.graph.ReachableIn I u v → C = D := by
  constructor
  · intro hI
    have hIB : ∀ C ⊆ I, Ω.graph.IsCycle C → C ∉ Ω.balanced :=
      fun C hCI _ hCB => hI C hCI (Or.inl hCB)
    refine ⟨hIB, fun C D hC hD hCI hDI u hu v hv huv => ?_⟩
    by_contra hCD
    obtain ⟨X, hXI, hX⟩ : ∃ X ⊆ I, Ω.graph.IsTightHandcuff X ∨ Ω.graph.IsLooseHandcuff X ∨
        Ω.graph.IsThetaSet X := by
      by_cases hw : ∃ w, w ∈ Ω.graph.verts C ∧ w ∈ Ω.graph.verts D
      · obtain ⟨w, hwC, hwD⟩ := hw
        obtain ⟨X, hX, hX'⟩ := hC.exists_frame_bicycle hD hCD hwC hwD
        exact ⟨X, hX.trans (Set.union_subset hCI hDI), hX'.imp_right Or.inr⟩
      · push Not at hw
        obtain ⟨x, y, P, hP, hX⟩ := hC.exists_isLooseHandcuff hD
          (Set.disjoint_left.2 fun w hwC hwD => hw w hwC hwD) hu hv huv
        exact ⟨_, Set.union_subset (Set.union_subset hCI hDI) fun e he => hP e he,
          Or.inr (Or.inl hX)⟩
    by_cases hB : ∃ C' ⊆ X, Ω.graph.IsCycle C' ∧ C' ∈ Ω.balanced
    · obtain ⟨C', hC'X, hC', hC'B⟩ := hB
      exact hIB C' (hC'X.trans hXI) hC' hC'B
    · push Not at hB
      exact hI X hXI (Or.inr ⟨hX, hB⟩)
  · rintro ⟨hIB, hIl⟩ X hXI (hXB | ⟨hX, -⟩)
    · exact hIB X hXI (Ω.isCycle_of_mem X hXB) hXB
    · obtain ⟨C, D, hC, hD, hCD, hCX, hDX, u, hu, v, hv, huv⟩ :=
        exists_linked_cycles_of_frame_bicycle hX
      exact hCD (hIl C D hC hD (hCX.trans hXI) (hDX.trans hXI) u hu v hv (huv.mono hXI))

end BiasedGraph

example := pumpkin_pair_isCycle.reachableIn (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inl rfl⟩)
  (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inr rfl⟩)

example := pumpkin_pair_isCycle.reachableIn_of_mem_insert (I := {1}) (Set.mem_insert 0 {1})
  le_rfl (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inr rfl⟩)

example := pumpkin_pair_isCycle.exists_frame_bicycle pumpkin_pair_isCycle_zero_two pumpkin_pair_ne
  (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inl rfl⟩)
  (mem_verts.2 ⟨0, Set.mem_insert 0 {2}, Or.inl rfl⟩)

example := (isCycle_singleton_of_loop (G := loopLink) (e := 0) rfl).exists_isLooseHandcuff
  (isCycle_singleton_of_loop (G := loopLink) (e := 1) rfl) loopLink_disjoint
  (mem_verts.2 ⟨0, rfl, Or.inl rfl⟩) (mem_verts.2 ⟨1, rfl, Or.inl rfl⟩)
  (reachableIn_of_mem (S := {2}) (e := (2 : Fin 3)) (Set.mem_singleton 2))

example := exists_linked_cycles_of_frame_bicycle (Or.inr (Or.inl loopLink_isLooseHandcuff))

end Transformer.GainMatroid
