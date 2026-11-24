module

public import Transformer.GainMatroid.Section2_CycleEars

/-!
# Tight handcuffs and thetas from ears

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, compare the circuits of a biased graph and
of its roll-ups from a vertex `u`. Unrolling two joints joined by a path through `u`, or a joint
joined by a path to a cycle through `u`, gives a tight handcuff at `u` or a theta. Two edges at `u`
and a path through `u` between their other ends form a tight handcuff at `u`
(`isTightHandcuff_of_mem_support`). An edge at `u`, a path from its other end to a cycle `D`
through `u` that meets `D` only at its last vertex, and `D` form a tight handcuff at `u` if the
path ends at `u` (`IsCycle.isTightHandcuff_insert`) and a theta otherwise
(`IsCycle.isTightHandcuff_or_isThetaSet_insert`). The vertices of the cycle that an edge at `u`
closes with a path back to `u` lie on the path (`mem_support_of_mem_verts_insert`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- The vertices of the cycle that an edge from `u` to `a` closes with a path from `a` to `u` lie
on the path. -/
theorem mem_support_of_mem_verts_insert {u a z : V} {e : E} (he : G.Joins e u a)
    {A : G.Walk a u} (hz : z ∈ G.verts (insert e A.edgeSet)) : z ∈ A.support := by
  obtain ⟨g, hg, hgz⟩ := mem_verts.1 hz
  rcases hg with rfl | hg
  · rcases he.eq_or_eq hgz with rfl | rfl
    · exact A.end_mem_support
    · exact A.start_mem_support
  · exact Walk.mem_support_of_mem_edges hg hgz

/-- Two edges at `u` and a path through `u` between their other ends, using neither edge, form a
tight handcuff at `u`. -/
theorem isTightHandcuff_of_mem_support {u a b : V} {e f : E} (he : G.Joins e u a)
    (hf : G.Joins f u b) (hef : e ≠ f) {P : G.Walk a b} (hP : P.IsPath) (huP : u ∈ P.support)
    (heP : e ∉ P.edges) (hfP : f ∉ P.edges) :
    G.IsTightHandcuff (insert e (insert f P.edgeSet)) := by
  obtain ⟨A, B, rfl⟩ := P.exists_eq_append_of_mem_support huP
  have heA : e ∉ A.edges := fun h => heP (Walk.mem_edges_append.2 (Or.inl h))
  have hfB : f ∉ B.reverse.edges := fun h =>
    hfP (Walk.mem_edges_append.2 (Or.inr (Walk.mem_edges_reverse.1 h)))
  refine ⟨insert e A.edgeSet, insert f B.reverse.edgeSet, u,
    isCycle_insert_of_isPath he hP.of_append_left heA,
    isCycle_insert_of_isPath hf hP.of_append_right.reverse hfB, ?_, ?_, ?_⟩
  · refine Set.disjoint_left.2 fun g hg hg' => ?_
    rcases hg with hge | hgA <;> rcases hg' with hgf | hgB
    · exact hef (hge.symm.trans hgf)
    · rw [hge] at hgB
      exact heP (Walk.mem_edges_append.2 (Or.inr (Walk.mem_edges_reverse.1 hgB)))
    · rw [hgf] at hgA
      exact hfP (Walk.mem_edges_append.2 (Or.inl hgA))
    · exact hP.not_mem_of_mem hgA (Walk.mem_edges_reverse.1 hgB)
  · refine Set.eq_singleton_iff_unique_mem.2 ⟨⟨mem_verts.2 ⟨e, Set.mem_insert _ _,
      he.tail_or_head⟩, mem_verts.2 ⟨f, Set.mem_insert _ _, hf.tail_or_head⟩⟩, fun z hz => ?_⟩
    exact hP.eq_of_mem_of_mem (mem_support_of_mem_verts_insert he hz.1)
      (Walk.mem_support_reverse.1 (mem_support_of_mem_verts_insert hf hz.2))
  · ext g
    simp only [Set.mem_insert_iff, Set.mem_union, Walk.edgeSet, Set.mem_ofPred_eq,
      Walk.mem_edges_append, Walk.mem_edges_reverse]
    tauto

/-- An edge from `u` to `a`, a path from `a` back to `u` and a cycle `D` through `u` that the edge
avoids and the path meets only at `u` form a tight handcuff at `u`. -/
theorem IsCycle.isTightHandcuff_insert {D : Set E} (hD : G.IsCycle D) {u a : V} {e : E}
    (he : G.Joins e u a) (huD : u ∈ G.verts D) (heD : e ∉ D) {A : G.Walk a u} (hA : A.IsPath)
    (heA : e ∉ A.edges) (hAD : ∀ z ∈ A.support, z ∈ G.verts D → z = u) :
    G.IsTightHandcuff (insert e A.edgeSet ∪ D) := by
  refine ⟨insert e A.edgeSet, D, u, isCycle_insert_of_isPath he hA heA, hD, ?_, ?_, rfl⟩
  · refine Set.disjoint_left.2 fun g hg hgD => ?_
    rcases hg with hge | hgA
    · rw [hge] at hgD
      exact heD hgD
    · exact hA.notMem_of_forall hAD hgA hgD
  · exact Set.eq_singleton_iff_unique_mem.2 ⟨⟨mem_verts.2 ⟨e, Set.mem_insert _ _,
      he.tail_or_head⟩, huD⟩, fun z hz => hAD z (mem_support_of_mem_verts_insert he hz.1) hz.2⟩

/-- An edge from `u` to `a`, a path from `a` to a cycle `D` through `u` that meets `D` only at its
last vertex `y`, and `D` form a tight handcuff at `u` if `y = u`, and a theta otherwise. -/
theorem IsCycle.isTightHandcuff_or_isThetaSet_insert {D : Set E} (hD : G.IsCycle D) {u a y : V}
    {e : E} (he : G.Joins e u a) (huD : u ∈ G.verts D) (heD : e ∉ D) {P : G.Walk a y}
    (hP : P.IsPath) (heP : e ∉ P.edges) (hyD : y ∈ G.verts D)
    (hPD : ∀ z ∈ P.support, z ∈ G.verts D → z = y) :
    G.IsTightHandcuff (insert e P.edgeSet ∪ D) ∨ G.IsThetaSet (insert e P.edgeSet ∪ D) := by
  rcases eq_or_ne y u with rfl | hyu
  · exact Or.inl (hD.isTightHandcuff_insert he huD heD hP heP hPD)
  have hR : (Walk.cons e he P).IsPath :=
    Walk.isPath_cons.2 ⟨fun h => hyu (hPD u h huD).symm, hP⟩
  have hRs : (Walk.cons e he P).edgeSet = insert e P.edgeSet := Set.ext fun _ => List.mem_cons
  right
  rw [Set.union_comm, ← hRs]
  refine hD.isThetaSet_union hyu.symm huD hyD hR (fun g hg hgD => ?_) fun z hz hzD => ?_
  · rcases List.mem_cons.1 hg with rfl | hg
    · exact heD hgD
    · exact hP.notMem_of_forall hPD hg hgD
  · rcases List.mem_cons.1 hz with rfl | hz
    · exact Or.inl rfl
    · exact Or.inr (hPD z hz hzD)

example := mem_support_of_mem_verts_insert (G := twoJointsHat) (u := 0) (a := 1) (e := 0)
  (Or.inl ⟨rfl, rfl⟩) (A := .cons 1 (Or.inr ⟨rfl, rfl⟩) (.nil 0))
  (mem_verts.2 ⟨0, Set.mem_insert _ _, Or.inl rfl⟩)
example := isTightHandcuff_of_mem_support (G := twoJointsHat) (u := 0) (a := 1) (b := 2) (e := 0)
  (f := 2) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩) (by decide)
  (P := .cons 1 (Or.inr ⟨rfl, rfl⟩) (.cons 3 (Or.inl ⟨rfl, rfl⟩) (.nil 2)))
  (by unfold Walk.IsPath; decide) (by decide) (by decide) (by decide)
example := (isCycle_pair (G := twoJoints) (e := 2) (f := 3) (u := 0) (v := 2) (Or.inl ⟨rfl, rfl⟩)
  (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)).isTightHandcuff_insert (u := 2) (a := 2) (e := 5)
  (Or.inl ⟨rfl, rfl⟩) (mem_verts.2 ⟨2, Set.mem_insert _ _, Or.inr rfl⟩) (by simp)
  (A := .nil 2) (List.nodup_singleton 2) List.not_mem_nil fun _ hz _ => List.mem_singleton.1 hz
example := (isCycle_pair (G := twoJointsHat) (e := 0) (f := 1) (u := 0) (v := 1) (Or.inl ⟨rfl, rfl⟩)
  (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide)).isTightHandcuff_or_isThetaSet_insert (u := 0)
  (a := 1) (e := 4) (Or.inr ⟨rfl, rfl⟩) (mem_verts.2 ⟨0, Set.mem_insert _ _, Or.inl rfl⟩)
  (by simp) (P := .nil 1) (List.nodup_singleton 1) List.not_mem_nil
  (mem_verts.2 ⟨0, Set.mem_insert _ _, Or.inr rfl⟩) fun _ hz _ => List.mem_singleton.1 hz

end OrientedGraph

end Transformer.GainMatroid
