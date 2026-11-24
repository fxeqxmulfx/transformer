module

public import Transformer.GainMatroid.Section2_AgreeingGraphs
public import Transformer.GainMatroid.Section2_WalkMaps
public import Transformer.GainMatroid.Section2_UnrollingTheta

/-!
# Cycles, ears and thetas

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, compare circuits of a biased graph and of
its roll-ups, which turn handcuffs into cycles and thetas and back. This file collects the graph
facts used there:

* a link `e` and a path from its other end back form a cycle (`isCycle_insert_of_isPath`), and a
  cycle through a link `e` at `u` is `e` and a path from the other end of `e` to `u`
  (`IsCycle.exists_isPath_insert_of_link`);
* a cycle through two vertices `a ≠ b` is two openly disjoint paths from `a` to `b`
  (`IsCycle.exists_openly`), so a path from `a` to `b` off the cycle closes a theta
  (`IsCycle.isThetaSet_union`);
* a path inside the edges of a path with the same ends passes through all of its vertices
  (`Walk.IsPath.mem_support_of_subset`), as the path between two vertices of a tree is unique;
* reversing the paths of a theta gives a theta (`IsTheta.reverse`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- A path inside the edges of a path with the same ends passes through all of its vertices. -/
theorem Walk.IsPath.mem_support_of_subset {x y z : V} {P Q : G.Walk x y} (hP : P.IsPath)
    (hQ : Q.IsPath) (hQP : ∀ g ∈ Q.edges, g ∈ P.edges) (hz : z ∈ P.support) : z ∈ Q.support := by
  induction P with
  | nil a =>
    rw [List.mem_singleton.1 hz]
    exact Q.start_mem_support
  | @cons a b c g hg P ih =>
    rcases List.mem_cons.1 hz with hza | hzP
    · rw [hza]
      exact Q.start_mem_support
    cases Q with
    | nil => exact absurd (hP.edges_eq_nil_of_eq rfl) (List.cons_ne_nil _ _)
    | @cons _ b' _ g' hg' Q =>
      have hgg : g' = g := hP.eq_of_mem_start (hQP g' List.mem_cons_self) hg'.tail_or_head
      subst hgg
      obtain rfl : b = b' := hg.eq_of_joins hg'
      refine List.mem_cons_of_mem _ (ih (isPath_cons.1 hP).2 (isPath_cons.1 hQ).2
        (fun e he => ?_) hzP)
      rcases List.mem_cons.1 (hQP e (List.mem_cons_of_mem _ he)) with heg | heP
      · rw [heg] at he
        exact absurd (mem_support_of_mem_edges he hg'.tail_or_head) (isPath_cons.1 hQ).1
      · exact heP

/-- An edge from `u` to `x` and a path from `x` back to `u` without it form a cycle. -/
theorem isCycle_insert_of_isPath {u x : V} {e : E} (he : G.Joins e u x) {A : G.Walk x u}
    (hA : A.IsPath) (heA : e ∉ A.edges) : G.IsCycle (insert e A.edgeSet) :=
  ⟨u, .cons e he A, ⟨List.cons_ne_nil _ _, List.nodup_cons.2 ⟨heA, hA.edges_nodup⟩, hA⟩,
    Set.ext fun _ => List.mem_cons⟩

/-- A cycle through an edge `e` at `u` is `e` and a path from the other end of `e` to `u`. -/
theorem IsCycle.exists_isPath_insert_of_link {C : Set E} (hC : G.IsCycle C) {u : V} {e : E}
    (heC : e ∈ C) (heu : G.tail e = u ∨ G.head e = u) :
    ∃ R : G.Walk (G.otherEnd u e) u, R.IsPath ∧ e ∉ R.edges ∧ C = insert e R.edgeSet := by
  obtain ⟨a, W, hW, rfl⟩ := hC
  obtain ⟨s, t, hj, R, hR, heR, hWR⟩ := hW.exists_isPath_insert heC
  rcases hj.eq_or_eq heu with rfl | rfl
  · obtain rfl := hj.symm.otherEnd_eq
    exact ⟨R, hR, heR, hWR⟩
  · obtain rfl := hj.otherEnd_eq
    refine ⟨R.reverse, hR.reverse, fun h => heR (Walk.mem_edges_reverse.1 h), ?_⟩
    rw [Walk.edgeSet_reverse]
    exact hWR

/-- A cycle through two vertices `a ≠ b` is two openly disjoint paths from `a` to `b`. -/
theorem IsCycle.exists_openly {C : Set E} (hC : G.IsCycle C) {a b : V} (hab : a ≠ b)
    (ha : a ∈ G.verts C) (hb : b ∈ G.verts C) :
    ∃ P Q : G.Walk a b, P.IsPath ∧ Q.IsPath ∧ Walk.Openly P Q ∧ C = P.edgeSet ∪ Q.edgeSet := by
  obtain ⟨e, heC, hea⟩ := mem_verts.1 ha
  have hl : G.IsLink e := fun hloop => by
    rw [hC.eq_singleton_of_loop heC hloop] at hb
    obtain ⟨g, hg, hgb⟩ := mem_verts.1 hb
    rw [Set.mem_singleton_iff.1 hg] at hgb
    exact hab ((hea.elim id fun h => hloop.trans h).symm.trans
      (hgb.elim id fun h => hloop.trans h))
  obtain ⟨f, hfC, hfe, hfa, -⟩ := hC.exists_partner heC hl hea
  obtain ⟨P, hP, haP, hCP⟩ := hC.exists_isPath_of_mem hfe.symm heC hfC hea hfa
  have hbP : b ∈ P.support := by
    obtain ⟨g, hg, hgb⟩ := mem_verts.1 hb
    rw [hCP] at hg
    rcases hg with rfl | rfl | hg
    · rcases (joins_otherEnd hea).eq_or_eq hgb with h | h
      · exact h ▸ P.start_mem_support
      · exact absurd h.symm hab
    · rcases (joins_otherEnd hfa).eq_or_eq hgb with h | h
      · exact h ▸ P.end_mem_support
      · exact absurd h.symm hab
    · exact Walk.mem_support_of_mem_edges hg hgb
  obtain ⟨A, B, rfl⟩ := P.exists_eq_append_of_mem_support hbP
  have heP : e ∉ (A.append B).edges := fun h => haP (Walk.mem_support_of_mem_edges h hea)
  have hfP : f ∉ (A.append B).edges := fun h => haP (Walk.mem_support_of_mem_edges h hfa)
  refine ⟨.cons e (joins_otherEnd hea).symm A, .cons f (joins_otherEnd hfa).symm B.reverse,
    Walk.isPath_cons.2 ⟨fun h => haP (Walk.mem_support_append.2 (Or.inl h)), hP.of_append_left⟩,
    Walk.isPath_cons.2 ⟨fun h => haP (Walk.mem_support_append.2
      (Or.inr (Walk.mem_support_reverse.1 h))), hP.of_append_right.reverse⟩,
    ⟨fun g hg hg' => ?_, fun z hz hz' => ?_⟩, ?_⟩
  · simp only [Walk.edges, List.mem_cons] at hg hg'
    rcases hg with hge | hgA <;> rcases hg' with hgf | hgB
    · exact hfe (hgf.symm.trans hge)
    · rw [hge] at hgB
      exact heP (Walk.mem_edges_append.2 (Or.inr (Walk.mem_edges_reverse.1 hgB)))
    · rw [hgf] at hgA
      exact hfP (Walk.mem_edges_append.2 (Or.inl hgA))
    · exact hP.not_mem_of_mem hgA (Walk.mem_edges_reverse.1 hgB)
  · simp only [Walk.support, List.mem_cons] at hz hz'
    rcases hz with hza | hzA
    · exact Or.inl hza
    rcases hz' with hza | hzB
    · exact Or.inl hza
    exact Or.inr (hP.eq_of_mem_of_mem hzA (Walk.mem_support_reverse.1 hzB))
  · ext g
    simp only [hCP, Walk.edgeSet, Set.mem_insert_iff, Set.mem_union, Set.mem_ofPred_eq,
      Walk.edges, List.mem_cons, Walk.mem_edges_append, Walk.mem_edges_reverse]
    tauto

/-- A cycle `D` through `a ≠ b` and a path from `a` to `b` with no edge of `D` and no other
vertex of `D` form a theta. -/
theorem IsCycle.isThetaSet_union {D : Set E} (hD : G.IsCycle D) {a b : V} (hab : a ≠ b)
    (ha : a ∈ G.verts D) (hb : b ∈ G.verts D) {R : G.Walk a b} (hR : R.IsPath)
    (hRe : ∀ g ∈ R.edges, g ∉ D) (hRv : ∀ z ∈ R.support, z ∈ G.verts D → z = a ∨ z = b) :
    G.IsThetaSet (D ∪ R.edgeSet) := by
  obtain ⟨P, Q, hP, hQ, hPQ, rfl⟩ := hD.exists_openly hab ha hb
  have hopen (S : G.Walk a b) (hS : S.edgeSet ⊆ P.edgeSet ∪ Q.edgeSet) : Walk.Openly S R := by
    refine ⟨fun g hg hgR => hRe g hgR (hS hg), fun z hz hzR => hRv z hzR ?_⟩
    by_cases hza : z = a
    · exact hza ▸ ha
    · obtain ⟨g, hg, hgz⟩ := S.exists_mem_edges_of_mem_support hz hza
      exact mem_verts.2 ⟨g, hS hg, hgz⟩
  exact ⟨a, b, P, Q, R, ⟨hab, hP, hQ, hR, hPQ, hopen P Set.subset_union_left,
    hopen Q Set.subset_union_right⟩, rfl⟩

theorem Walk.Openly.reverse {x y : V} {P Q : G.Walk x y} (h : Walk.Openly P Q) :
    Walk.Openly P.reverse Q.reverse :=
  ⟨fun e he he' => h.edges e (Walk.mem_edges_reverse.1 he) (Walk.mem_edges_reverse.1 he'),
    fun z hz hz' => (h.verts z (Walk.mem_support_reverse.1 hz)
      (Walk.mem_support_reverse.1 hz')).symm⟩

/-- Reversing the three paths of a theta gives a theta. -/
theorem IsTheta.reverse {x y : V} {P Q R : G.Walk x y} (h : IsTheta P Q R) :
    IsTheta P.reverse Q.reverse R.reverse :=
  ⟨h.ne.symm, h.pathP.reverse, h.pathQ.reverse, h.pathR.reverse, h.openPQ.reverse,
    h.openPR.reverse, h.openQR.reverse⟩

example := (pumpkinWalk_isPath 0).mem_support_of_subset (pumpkinWalk_isPath 0) (fun _ h => h)
  (z := 0) (by simp [pumpkinWalk, Walk.support])
example := isCycle_insert_of_isPath (G := pumpkin) (e := 1) (Or.inl ⟨rfl, rfl⟩)
  (pumpkinWalk_isPath 0).reverse (by decide)
example := pumpkin_isTheta.isCycle.1.exists_isPath_insert_of_link (u := 0) (e := 0)
  (Or.inl List.mem_cons_self) (Or.inl rfl)
example := pumpkin_isTheta.isCycle.1.exists_openly (a := 0) (b := 1) (by decide)
  (mem_verts.2 ⟨0, Or.inl List.mem_cons_self, Or.inl rfl⟩)
  (mem_verts.2 ⟨0, Or.inl List.mem_cons_self, Or.inr rfl⟩)
example := pumpkin_isTheta.isCycle.1.isThetaSet_union (a := 0) (b := 1) (by decide)
  (mem_verts.2 ⟨0, Or.inl List.mem_cons_self, Or.inl rfl⟩)
  (mem_verts.2 ⟨0, Or.inl List.mem_cons_self, Or.inr rfl⟩) (pumpkinWalk_isPath 2)
  (by simp [pumpkinWalk, Walk.edges, Walk.edgeSet])
  (by simp [pumpkinWalk, Walk.support, pumpkin])
example := (pumpkinWalk_openly (i := 0) (j := 1) (by decide)).reverse
example := pumpkin_isTheta.reverse

end OrientedGraph

end Transformer.GainMatroid
