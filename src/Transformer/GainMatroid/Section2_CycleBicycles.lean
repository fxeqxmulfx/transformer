module

public import Transformer.GainMatroid.Section2_Nullity
public import Transformer.GainMatroid.Section2_TightHandcuffs

/-!
# Two cycles contain a theta, a handcuff or two disjoint cycles

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, list the circuits of `F(G, B)` and `L(G, B)`
other than balanced cycles: tight and loose handcuffs, pairs of vertex-disjoint cycles and thetas.
The proof here that they are the circuits of matroids finds one of them in any two distinct
cycles.

* A cycle can be traversed by a closed walk from any of its vertices (`IsCycle.exists_walk_at`).
* A cycle `D` that meets a cycle `C` but does not lie in it has an **ear** on `C`: an edge `g` of
  `D` off `C` from a vertex `u` of `C`, followed by a path along `D` that meets `C` only at its end
  (`IsCycle.exists_ear`). With `C` it forms a tight handcuff or a theta.
* Two distinct cycles therefore contain a tight handcuff, two vertex-disjoint cycles or a theta
  (`IsCycle.exists_lift_bicycle`).
* An edge from `a` to a vertex `w ≠ a` on no edge of `S` does not have its ends linked along `S`
  (`not_reachableIn_of_joins`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

namespace Walk

/-- A walk with an edge outside `C` splits at its first such edge. -/
theorem exists_eq_append_cons_first {C : Set E} {u v : V} (p : G.Walk u v)
    (h : ∃ f ∈ p.edges, f ∉ C) :
    ∃ (s t : V) (g : E) (hg : G.Joins g s t) (A : G.Walk u s) (B : G.Walk t v),
      p = A.append (cons g hg B) ∧ (∀ f ∈ A.edges, f ∈ C) ∧ g ∉ C := by
  induction p with
  | nil => simp [edges] at h
  | @cons a b c f hf q ih =>
    by_cases hfC : f ∈ C
    · obtain ⟨g, hg, hgC⟩ := h
      have hgq : g ∈ q.edges := (List.mem_cons.1 hg).resolve_left fun h => hgC (h ▸ hfC)
      obtain ⟨s, t, g', hg', A, B, rfl, hA, hg'C⟩ := ih ⟨g, hgq, hgC⟩
      refine ⟨s, t, g', hg', cons f hf A, B, rfl, fun e he => ?_, hg'C⟩
      rcases List.mem_cons.1 he with rfl | he
      · exact hfC
      · exact hA e he
    · exact ⟨a, b, f, hf, nil a, q, rfl, fun e he => by simp [edges] at he, hfC⟩

end Walk

/-- A cycle is traversed by a closed walk from any of its vertices. -/
theorem IsCycle.exists_walk_at {C : Set E} (hC : G.IsCycle C) {v : V} (hv : v ∈ G.verts C) :
    ∃ W : G.Walk v v, W.IsCycleWalk ∧ W.edgeSet = C := by
  obtain ⟨e, heC, hev⟩ := mem_verts.1 hv
  obtain ⟨R, hR, heR, rfl⟩ := hC.exists_isPath_insert_of_link heC hev
  exact ⟨.cons e (joins_otherEnd hev).symm R, ⟨List.cons_ne_nil _ _,
    List.nodup_cons.2 ⟨heR, hR.edges_nodup⟩, hR⟩, Walk.edgeSet_cons _ _ _⟩

/-- **An ear of `C` along `D`.** If a cycle `D` meets a cycle `C` but does not lie in it, then `D`
has an edge `g` off `C` from a vertex `u` of `C` to a vertex `a`, and a path `P` along `D` from `a`
to a vertex `y` of `C` that meets `C` only at `y` and avoids `g`. -/
theorem IsCycle.exists_ear {C D : Set E} (hD : G.IsCycle D) (hDC : ¬D ⊆ C) {v : V}
    (hvC : v ∈ G.verts C) (hvD : v ∈ G.verts D) :
    ∃ (u a y : V) (g : E) (_ : G.Joins g u a) (P : G.Walk a y), u ∈ G.verts C ∧ g ∈ D ∧
      g ∉ C ∧ P.IsPath ∧ g ∉ P.edges ∧ (∀ f ∈ P.edges, f ∈ D) ∧ y ∈ G.verts C ∧
      ∀ z ∈ P.support, z ∈ G.verts C → z = y := by
  obtain ⟨W, hW, rfl⟩ := hD.exists_walk_at hvD
  obtain ⟨f, hf, hfC⟩ := Set.not_subset.1 hDC
  obtain ⟨s, t, g, hg, A, B, rfl, hA, hgC⟩ := W.exists_eq_append_cons_first ⟨f, hf, hfC⟩
  have hedges := hW.edges_nodup
  rw [Walk.edges_append, Walk.edges] at hedges
  have hsupp := hW.support_tail_nodup
  rw [Walk.support_append, Walk.support, List.tail_cons, A.support_eq_cons, List.cons_append,
    List.tail_cons] at hsupp
  have hB : B.IsPath := (List.nodup_append.1 hsupp).2.1
  have hgB : g ∉ B.edges := (List.nodup_cons.1 (List.nodup_append.1 hedges).2.1).1
  have hs : s ∈ G.verts C := by
    by_cases hAe : A.edges = []
    · cases A with
      | nil => exact hvC
      | cons => exact absurd hAe (List.cons_ne_nil _ _)
    · exact verts_mono (show A.edgeSet ⊆ C from fun e he => hA e he)
        (A.mem_verts_of_mem_support hAe A.end_mem_support)
  obtain ⟨y, P, Q, hy, rfl, hP⟩ := B.exists_eq_append_first hvC
  refine ⟨s, t, y, g, hg, P, hs, ?_, hgC, hB.of_append_left, fun h => hgB ?_, fun e he => ?_, hy,
    hP⟩
  · simp [Walk.edgeSet, Walk.edges_append, Walk.edges]
  · exact Walk.mem_edges_append.2 (Or.inl h)
  · simp [Walk.edgeSet, Walk.edges_append, Walk.edges, he]

/-- Two distinct cycles contain a tight handcuff, two vertex-disjoint cycles or a theta. -/
theorem IsCycle.exists_lift_bicycle {C D : Set E} (hC : G.IsCycle C) (hD : G.IsCycle D)
    (hCD : C ≠ D) : ∃ X ⊆ C ∪ D,
      G.IsTightHandcuff X ∨ G.IsDisjointCycles X ∨ G.IsThetaSet X := by
  by_cases hdis : Disjoint (G.verts C) (G.verts D)
  · exact ⟨C ∪ D, le_rfl, Or.inr (Or.inl ⟨C, D, hC, hD, hdis, rfl⟩)⟩
  obtain ⟨v, hvC, hvD⟩ := Set.not_disjoint_iff.1 hdis
  have hDC : ¬D ⊆ C := fun h => hCD (hC.eq_of_subset hD h).symm
  obtain ⟨u, a, y, g, hg, P, hu, hgD, hgC, hP, hgP, hPD, hy, hPC⟩ := hD.exists_ear hDC hvC hvD
  refine ⟨insert g P.edgeSet ∪ C, fun e he => ?_, ?_⟩
  · rcases he with (rfl | he) | he
    · exact Or.inr hgD
    · exact Or.inr (hPD e he)
    · exact Or.inl he
  · rcases hC.isTightHandcuff_or_isThetaSet_insert hg hu hgC hP hgP hy hPC with h | h
    · exact Or.inl h
    · exact Or.inr (Or.inr h)

/-- The first vertex of a path does not occur again. -/
theorem Walk.IsPath.start_notMem_tail {u v : V} {p : G.Walk u v} (hp : p.IsPath) :
    u ∉ p.support.tail := by
  have h : p.support.Nodup := hp
  rw [p.support_eq_cons] at h
  exact (List.nodup_cons.1 h).1

theorem Walk.IsPath.tail_nodup {u v : V} {p : G.Walk u v} (hp : p.IsPath) :
    p.support.tail.Nodup :=
  (show p.support.Nodup from hp).sublist (List.tail_sublist _)

/-- A vertex that is neither end of `e` and not on `S` is not on `insert e S`. -/
theorem notMem_verts_insert {S : Set E} {e : E} {a w z : V} (hj : G.Joins e a w) (hza : z ≠ a)
    (hzw : z ≠ w) (hzS : z ∉ G.verts S) : z ∉ G.verts (insert e S) := by
  intro h
  obtain ⟨f, hf, hfz⟩ := mem_verts.1 h
  rcases hf with rfl | hf
  · rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩ <;> rcases hfz with h | h
    · exact hza (h.symm.trans h₁)
    · exact hzw (h.symm.trans h₂)
    · exact hzw (h.symm.trans h₁)
    · exact hza (h.symm.trans h₂)
  · exact hzS (mem_verts.2 ⟨f, hf, hfz⟩)

/-- An edge from `a` to a new vertex `w ≠ a` does not have linked ends. -/
theorem not_reachableIn_of_joins {S : Set E} {e : E} {a w : V} (hj : G.Joins e a w) (hwa : w ≠ a)
    (hwS : w ∉ G.verts S) : ¬G.ReachableIn S (G.tail e) (G.head e) := by
  intro h
  rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
  · rw [h₁, h₂] at h
    exact hwa (h.symm.eq_of_notMem_verts hwS).symm
  · rw [h₁, h₂] at h
    exact hwa (h.eq_of_notMem_verts hwS).symm

/-- The first and last links of `pumpkin` form a cycle. -/
theorem pumpkin_pair_isCycle_zero_two : pumpkin.IsCycle ({0, 2} : Set (Fin 3)) :=
  isCycle_pair (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)

theorem pumpkin_pair_ne : ({0, 1} : Set (Fin 3)) ≠ {0, 2} := fun h =>
  absurd (h ▸ Set.mem_insert_of_mem 0 (Set.mem_singleton 1) : (1 : Fin 3) ∈ ({0, 2} : Set (Fin 3)))
    (by simp)

example := pathThreeFirst.exists_eq_append_cons_first (C := ∅)
  ⟨0, by simp [pathThreeFirst, Walk.edges], Set.notMem_empty 0⟩

example := pumpkin_pair_isCycle.exists_walk_at (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inl rfl⟩)

example := pumpkin_pair_isCycle_zero_two.exists_ear (C := {0, 1})
  (fun h => absurd (h (Set.mem_insert_of_mem 0 (Set.mem_singleton 2))) (by simp))
  (mem_verts.2 ⟨0, Set.mem_insert 0 {1}, Or.inl rfl⟩)
  (mem_verts.2 ⟨0, Set.mem_insert 0 {2}, Or.inl rfl⟩)

example := pumpkin_pair_isCycle.exists_lift_bicycle pumpkin_pair_isCycle_zero_two pumpkin_pair_ne

example := pathThree_isPath.start_notMem_tail

example := pathThree_isPath.tail_nodup

example := notMem_verts_insert (G := pathThree) (S := ∅) (e := 0) (a := 0) (w := 1) (z := 2)
  (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide) (by simp [verts])

example := not_reachableIn_of_joins (G := pathThree) (S := ∅) (e := 0) (a := 0) (w := 1)
  (Or.inl ⟨rfl, rfl⟩) (by decide) (by simp [verts])

end OrientedGraph

end Transformer.GainMatroid
