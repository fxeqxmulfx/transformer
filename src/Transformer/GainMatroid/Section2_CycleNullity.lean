module

public import Transformer.GainMatroid.Section2_Nullity

/-!
# Cycles and nullity

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, cite Zaslavsky for the matroids `F(G, B)` and
`L(G, B)`. Their circuits other than balanced cycles contain two distinct cycles; the nullity
(`OrientedGraph.nullity`) detects them.

* Removing an edge whose ends stay linked, for instance an edge of a cycle, lowers the nullity by
  one (`nullity_diff_singleton_add_one`, `nullity_diff_add_one_of_isCycle`), and a cycle has
  nullity one (`IsCycle.nullity_eq_one`).
* A set contains two distinct cycles exactly when its nullity is at least two
  (`two_le_nullity`, `exists_two_cycles`); with nullity at most one it has one cycle at most
  (`eq_of_nullity_le_one`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- The cycles `P ∪ Q` and `P ∪ R` of a theta are distinct: an edge of `Q` lies on neither `P` nor
`R`. -/
theorem IsTheta.union_ne {x y : V} {P Q R : G.Walk x y} (hT : IsTheta P Q R) :
    P.edgeSet ∪ Q.edgeSet ≠ P.edgeSet ∪ R.edgeSet := fun h => by
  obtain ⟨q, hq⟩ := List.exists_mem_of_ne_nil _ (Walk.edges_ne_nil_of_ne hT.ne Q)
  rcases (h ▸ Or.inr hq : q ∈ P.edgeSet ∪ R.edgeSet) with hq' | hq'
  · exact hT.openPQ.edges q hq' hq
  · exact hT.openQR.edges q hq hq'

variable [Finite V] [Finite E]

/-- Removing an edge whose ends stay linked lowers the nullity by one. -/
theorem nullity_diff_singleton_add_one {S : Set E} {e : E} (heS : e ∈ S)
    (he : G.ReachableIn (S \ {e}) (G.tail e) (G.head e)) :
    G.nullity (S \ {e}) + 1 = G.nullity S := by
  rw [← nullity_insert_of_reachableIn (fun h : e ∈ S \ {e} => h.2 rfl) he,
    Set.insert_sdiff_singleton, Set.insert_eq_of_mem heS]

/-- Removing an edge of a cycle of `S` lowers the nullity of `S` by one. -/
theorem nullity_diff_add_one_of_isCycle {S C : Set E} {e : E} (hC : G.IsCycle C) (hCS : C ⊆ S)
    (heC : e ∈ C) : G.nullity (S \ {e}) + 1 = G.nullity S :=
  nullity_diff_singleton_add_one (hCS heC)
    ((hC.reachableIn_diff heC).mono (Set.sdiff_subset_sdiff_left hCS))

/-- A cycle has nullity one. -/
theorem IsCycle.nullity_eq_one {C : Set E} (hC : G.IsCycle C) : G.nullity C = 1 := by
  obtain ⟨e, he⟩ := hC.nonempty
  rw [← nullity_diff_add_one_of_isCycle hC le_rfl he, nullity_eq_zero_iff.2 (hC.isForest_diff he)]

/-- Two distinct cycles in `S` make its nullity at least two. -/
theorem two_le_nullity {S C D : Set E} (hC : G.IsCycle C) (hD : G.IsCycle D) (hCD : C ≠ D)
    (hCS : C ⊆ S) (hDS : D ⊆ S) : 2 ≤ G.nullity S := by
  obtain ⟨e, heC, heD⟩ : ∃ e ∈ C, e ∉ D := by
    by_contra! h
    exact hCD (hD.eq_of_subset hC h)
  have h₁ := nullity_diff_add_one_of_isCycle hC hCS heC
  have h₂ : 1 ≤ G.nullity (S \ {e}) := by
    rw [← hD.nullity_eq_one]
    exact nullity_mono fun f hf => ⟨hDS hf, fun h => heD (h ▸ hf)⟩
  omega

/-- A set of nullity at least two contains two distinct cycles. -/
theorem exists_two_cycles {S : Set E} (h : 2 ≤ G.nullity S) :
    ∃ C D, G.IsCycle C ∧ G.IsCycle D ∧ C ≠ D ∧ C ⊆ S ∧ D ⊆ S := by
  have hS : ¬G.IsForest S := fun hF => by
    rw [nullity_eq_zero_iff.2 hF] at h
    omega
  obtain ⟨C, hC, hCS⟩ : ∃ C, G.IsCycle C ∧ C ⊆ S := by
    by_contra! hno
    exact hS fun C hC hCS => hno C hC hCS
  obtain ⟨e, he⟩ := hC.nonempty
  have h₁ := nullity_diff_add_one_of_isCycle hC hCS he
  have hS' : ¬G.IsForest (S \ {e}) := fun hF => by
    rw [nullity_eq_zero_iff.2 hF] at h₁
    omega
  obtain ⟨D, hD, hDS⟩ : ∃ D, G.IsCycle D ∧ D ⊆ S \ {e} := by
    by_contra! hno
    exact hS' fun D hD hDS => hno D hD hDS
  exact ⟨C, D, hC, hD, fun hCD => (hDS (hCD ▸ he)).2 rfl, hCS, hDS.trans Set.sdiff_subset⟩

/-- In a set of nullity at most one, any two cycles are equal. -/
theorem eq_of_nullity_le_one {S C D : Set E} (h : G.nullity S ≤ 1) (hC : G.IsCycle C)
    (hD : G.IsCycle D) (hCS : C ⊆ S) (hDS : D ⊆ S) : C = D := by
  by_contra hCD
  have := two_le_nullity hC hD hCD hCS hDS
  omega


example := pumpkin_isTheta.union_ne

example := nullity_diff_singleton_add_one (G := pumpkin) (S := {0, 1}) (e := 0)
  (Set.mem_insert 0 {1})
  (reachableIn_of_mem (e := (1 : Fin 3))
    (show (1 : Fin 3) ∈ ({0, 1} : Set (Fin 3)) \ {0} by simp))

example := nullity_diff_add_one_of_isCycle pumpkin_pair_isCycle le_rfl (Set.mem_insert 0 {1})

example := pumpkin_pair_isCycle.nullity_eq_one

example := two_le_nullity (S := Set.univ) pumpkin_isTheta.isCycle.1 pumpkin_isTheta.isCycle.2.1
  pumpkin_isTheta.union_ne (Set.subset_univ _) (Set.subset_univ _)

example := exists_two_cycles (two_le_nullity (S := Set.univ) pumpkin_isTheta.isCycle.1
  pumpkin_isTheta.isCycle.2.1 pumpkin_isTheta.union_ne (Set.subset_univ _) (Set.subset_univ _))

example := eq_of_nullity_le_one pumpkin_pair_isCycle.nullity_eq_one.le pumpkin_pair_isCycle
  pumpkin_pair_isCycle le_rfl le_rfl

end OrientedGraph

end Transformer.GainMatroid
