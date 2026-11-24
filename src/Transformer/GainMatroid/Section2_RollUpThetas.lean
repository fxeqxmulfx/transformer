module

public import Transformer.GainMatroid.Section2_RollUpClass

/-!
# Thetas through an edge of a roll-up class

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.5, compare the frame circuits of `(Ĝ, B̂)` with
those of its roll-up `(G_U, B_U)`, which turns each edge of `U` into a joint at its end away from
`u`. A theta of `Ω` with no balanced cycle through an edge `e` of `U` becomes a handcuff. The vertex
`u` is balancing, so it lies on the cycle of the two paths that avoid `e`, and it is an end of `e`
on the third path, so it is an end of the theta. Then `e` is the first edge of its path, and no
other edge of the theta is in `U`: it would close a balanced cycle with `e`. So the theta is `e`,
the cycle `D` of the other two paths, which avoids `U`, and the rest of the path through `e`, a path
of the roll-up from the end of `e` away from `u` to `D` that meets `D` only at its last vertex
(`IsRollUpClass.exists_of_isTheta_at` for a theta from `u`, `IsRollUpClass.exists_of_isTheta` for
any theta).
-/

@[expose] public section

namespace Transformer.GainMatroid

variable {V E : Type*}

open OrientedGraph

namespace BiasedGraph

variable {Ω : BiasedGraph V E} {u : V} {U : Set E}

namespace IsRollUpClass

/-- A theta of `Ω` from `u` with no balanced cycle, through an edge `e` of `U` on its path `P`, is
`e`, the cycle `D` of the other two paths, which avoids `U`, and the rest of `P`: a path of the
roll-up from the end of `e` away from `u` to a vertex `y'` of `D`, meeting `D` only at `y'`. -/
theorem exists_of_isTheta_at (hU : Ω.IsRollUpClass u U) {y : V} {P Q R : Ω.graph.Walk u y}
    (hT : IsTheta P Q R)
    (hno : ∀ C ⊆ P.edgeSet ∪ Q.edgeSet ∪ R.edgeSet, Ω.graph.IsCycle C → C ∉ Ω.balanced)
    {e : E} (he : e ∈ U) (heP : e ∈ P.edges) :
    ∃ (D : Set E) (y' : V) (P₀ : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e) y'),
      Ω.graph.IsCycle D ∧ Disjoint D U ∧ P₀.IsPath ∧ y' ∈ Ω.graph.verts D ∧
      (∀ z ∈ P₀.support, z ∈ Ω.graph.verts D → z = y') ∧
      P.edgeSet ∪ Q.edgeSet ∪ R.edgeSet = {e} ∪ D ∪ P₀.edgeSet := by
  -- An edge of `U` on `Q` or `R` would close a balanced cycle with `e`.
  have hQR : Disjoint (Q.edgeSet ∪ R.edgeSet) U := by
    refine Set.disjoint_left.2 fun g hg hgU => ?_
    rcases hg with hg | hg
    · exact hno _ Set.subset_union_left hT.isCycle.1 (hU.pair _ hT.isCycle.1 e (Or.inl heP) g
        (Or.inr hg) (fun h => hT.openPQ.edges e heP (by rw [h]; exact hg)) he hgU)
    · exact hno _ (Set.union_subset (Set.subset_union_left.trans Set.subset_union_left)
        Set.subset_union_right) hT.isCycle.2.1 (hU.pair _ hT.isCycle.2.1 e (Or.inl heP) g
        (Or.inr hg) (fun h => hT.openPR.edges e heP (by rw [h]; exact hg)) he hgU)
  cases P with
  | nil => exact absurd rfl hT.ne
  | cons p hp P₀ =>
    obtain rfl := hT.pathP.eq_of_mem_start heP (hU.link he).2
    obtain ⟨huP₀, hP₀⟩ := Walk.isPath_cons.1 hT.pathP
    have hP₀U : Disjoint P₀.edgeSet U := Set.disjoint_left.2 fun g hg hgU =>
      huP₀ (Walk.mem_support_of_mem_edges hg (hU.link hgU).2)
    obtain ⟨P₁, hP₁e, hP₁s⟩ := P₀.exists_of_agreeOn hp.symm.otherEnd_eq.symm rfl
      (agreeOn_rollUp (Ω := Ω) (u := u) hP₀U).symm
    refine ⟨Q.edgeSet ∪ R.edgeSet, y, P₁, hT.isCycle.2.2, hQR, ?_, ?_, fun z hz hzD => ?_, ?_⟩
    · unfold Walk.IsPath
      rw [hP₁s]
      exact hP₀
    · obtain ⟨g, hg, hgy⟩ := Q.exists_mem_edges_of_mem_support Q.end_mem_support hT.ne.symm
      exact mem_verts.2 ⟨g, Or.inl hg, hgy⟩
    · rw [hP₁s] at hz
      have hzP : z ∈ (Walk.cons _ hp P₀).support := List.mem_cons_of_mem _ hz
      obtain ⟨g, hg, hgz⟩ := mem_verts.1 hzD
      have h : z = u ∨ z = y := by
        rcases hg with hg | hg
        · exact hT.openPQ.verts z hzP (Walk.mem_support_of_mem_edges hg hgz)
        · exact hT.openPR.verts z hzP (Walk.mem_support_of_mem_edges hg hgz)
      exact h.resolve_left fun h => huP₀ (h ▸ hz)
    · ext g
      simp only [Walk.edgeSet, Walk.edges, Set.mem_union, Set.mem_ofPred_eq,
        Set.mem_singleton_iff, List.mem_cons, hP₁e]
      tauto

/-- A theta of `Ω` with no balanced cycle, through an edge `e` of `U` on its path `P`, has `u` as
an end, and is `e`, a cycle `D` off `U` and a path of the roll-up from the end of `e` away from `u`
to a vertex `y'` of `D`, meeting `D` only at `y'`. -/
theorem exists_of_isTheta (hU : Ω.IsRollUpClass u U) {x y : V} {P Q R : Ω.graph.Walk x y}
    (hT : IsTheta P Q R)
    (hno : ∀ C ⊆ P.edgeSet ∪ Q.edgeSet ∪ R.edgeSet, Ω.graph.IsCycle C → C ∉ Ω.balanced)
    {e : E} (he : e ∈ U) (heP : e ∈ P.edges) :
    ∃ (D : Set E) (y' : V) (P₀ : (Ω.rollUp u U).graph.Walk (Ω.graph.otherEnd u e) y'),
      Ω.graph.IsCycle D ∧ Disjoint D U ∧ P₀.IsPath ∧ y' ∈ Ω.graph.verts D ∧
      (∀ z ∈ P₀.support, z ∈ Ω.graph.verts D → z = y') ∧
      P.edgeSet ∪ Q.edgeSet ∪ R.edgeSet = {e} ∪ D ∪ P₀.edgeSet := by
  have huP := Walk.mem_support_of_mem_edges heP (hU.link he).2
  obtain ⟨g, hg, hgu⟩ := mem_verts.1 (hU.vertex _ ⟨hT.isCycle.2.2, hno _ (Set.union_subset
    (Set.subset_union_right.trans Set.subset_union_left) Set.subset_union_right) hT.isCycle.2.2⟩)
  have hxy : u = x ∨ u = y := by
    rcases hg with hg | hg
    · exact hT.openPQ.verts u huP (Walk.mem_support_of_mem_edges hg hgu)
    · exact hT.openPR.verts u huP (Walk.mem_support_of_mem_edges hg hgu)
  rcases hxy with rfl | rfl
  · exact hU.exists_of_isTheta_at hT hno he heP
  · have h := hU.exists_of_isTheta_at hT.reverse (by simpa only [Walk.edgeSet_reverse] using hno)
      he (Walk.mem_edges_reverse.2 heP)
    simpa only [Walk.edgeSet_reverse] using h

end IsRollUpClass

end BiasedGraph

open BiasedGraph

/-- The edge `0` of the contrabalanced `pumpkin` rolls up from `0`. -/
theorem pumpkin_isRollUpClass : pumpkin.contrabalanced.IsRollUpClass 0 {0} :=
  ⟨fun C hC => by
    obtain ⟨a, W, hW, rfl⟩ := hC.1
    obtain ⟨g, hg⟩ := List.exists_mem_of_ne_nil _ hW.edges_ne_nil
    exact mem_verts.2 ⟨g, hg, Or.inl rfl⟩,
  Set.singleton_subset_iff.2 ⟨(by decide : (0 : Fin 2) ≠ 1), Or.inl rfl⟩,
  fun _ _ _ _ _ _ hne he hf => absurd (he.trans hf.symm) hne,
  fun C hC => absurd hC (Set.notMem_empty C)⟩

example := pumpkin_isRollUpClass.exists_of_isTheta_at pumpkin_isTheta
  (fun C _ _ hC => Set.notMem_empty C hC) (e := 0) rfl List.mem_cons_self
example := pumpkin_isRollUpClass.exists_of_isTheta pumpkin_isTheta
  (fun C _ _ hC => Set.notMem_empty C hC) (e := 0) rfl List.mem_cons_self

end Transformer.GainMatroid
