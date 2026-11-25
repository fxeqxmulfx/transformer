module

public import Transformer.GainMatroid.Section2_Components

/-!
# The nullity of a set of edges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, cite Zaslavsky for the matroids `F(G, B)` and
`L(G, B)`. Their independent sets are read off the nullity `|S| + c(S) - |V|` of a set `S` of
edges (`nullity`), where `c(S)` counts its components (`compCount`).

* `|V| ≤ |S| + c(S)`, with equality exactly for a forest (`card_le_ncard_add_compCount`,
  `nullity_eq_zero_iff`).
* An edge with linked ends raises the nullity by one, any other edge keeps it
  (`nullity_insert_of_reachableIn`, `nullity_insert_of_not_reachableIn`), so the nullity grows
  with the set (`nullity_mono`).

`Section2_CycleNullity` relates the nullity to the cycles of a set.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- A cycle through `e` inside `insert e S` links the ends of `e` along `S`. -/
theorem IsCycle.reachableIn_of_subset_insert {C S : Set E} {e : E} (hC : G.IsCycle C)
    (heC : e ∈ C) (hCS : C ⊆ insert e S) : G.ReachableIn S (G.tail e) (G.head e) := by
  obtain ⟨R, -, heR, rfl⟩ := hC.exists_isPath_insert_of_joins (Or.inl ⟨rfl, rfl⟩) heC
  exact ⟨R, fun f hf => (hCS (Set.mem_insert_of_mem e hf)).resolve_left fun h => heR (h ▸ hf)⟩

/-- A cycle through `e` links the ends of `e` without `e`. -/
theorem IsCycle.reachableIn_diff {C : Set E} {e : E} (hC : G.IsCycle C) (heC : e ∈ C) :
    G.ReachableIn (C \ {e}) (G.tail e) (G.head e) :=
  hC.reachableIn_of_subset_insert heC fun f hf => by
    by_cases hfe : f = e
    · exact Or.inl hfe
    · exact Or.inr ⟨hf, hfe⟩

/-- A cycle without one of its edges is a forest. -/
theorem IsCycle.isForest_diff {C : Set E} {e : E} (hC : G.IsCycle C) (heC : e ∈ C) :
    G.IsForest (C \ {e}) := by
  intro D hD hDC
  have hDC' : D ⊆ C := hDC.trans Set.sdiff_subset
  obtain rfl : D = C := hC.eq_of_subset hD hDC'
  exact (hDC heC).2 rfl

/-- An edge whose ends are not linked along a forest extends it to a forest. -/
theorem IsForest.insert {S : Set E} {e : E} (hS : G.IsForest S)
    (he : ¬G.ReachableIn S (G.tail e) (G.head e)) : G.IsForest (insert e S) := by
  intro C hC hCS
  by_cases heC : e ∈ C
  · exact he (hC.reachableIn_of_subset_insert heC hCS)
  · exact hS C hC fun f hf => (hCS hf).resolve_left fun h => heC (h ▸ hf)

/-- A vertex that no edge of `S` meets reaches only itself along `S`. -/
theorem ReachableIn.eq_of_notMem_verts {S : Set E} {u v : V} (h : G.ReachableIn S u v)
    (hu : u ∉ G.verts S) : v = u := by
  have := ReachableIn.rel (r := fun a b => a = b ∨ (a ∈ G.verts S ∧ b ∈ G.verts S))
    (fun _ => Or.inl rfl) (fun a b c => by
      rintro (rfl | ⟨ha, hb⟩) (rfl | ⟨hb', hc⟩)
      · exact Or.inl rfl
      · exact Or.inr ⟨hb', hc⟩
      · exact Or.inr ⟨ha, hb⟩
      · exact Or.inr ⟨ha, hc⟩)
    (fun e he => ⟨Or.inr ⟨mem_verts.2 ⟨e, he, Or.inl rfl⟩, mem_verts.2 ⟨e, he, Or.inr rfl⟩⟩,
      Or.inr ⟨mem_verts.2 ⟨e, he, Or.inr rfl⟩, mem_verts.2 ⟨e, he, Or.inl rfl⟩⟩⟩) h
  rcases this with rfl | ⟨hu', -⟩
  · rfl
  · exact absurd hu' hu

variable [Finite V] [Finite E]

/-- `|V| ≤ |S| + c(S)`, with equality exactly when `S` is a forest. -/
theorem card_le_ncard_add_compCount (S : Set E) :
    Nat.card V ≤ S.ncard + G.compCount S ∧
      (S.ncard + G.compCount S = Nat.card V ↔ G.IsForest S) := by
  induction S, S.toFinite using Set.Finite.induction_on with
  | empty => simp [compCount_empty, isForest_empty]
  | @insert a s ha hs ih =>
    rw [Set.ncard_insert_of_notMem ha hs]
    by_cases hr : G.ReachableIn s (G.tail a) (G.head a)
    · rw [compCount_insert_of_reachableIn hr]
      refine ⟨by omega, fun h => by omega, fun hF => ?_⟩
      exact absurd hr.symm (hF.not_reachableIn ha)
    · have hc := compCount_insert_of_not_reachableIn (G := G) hr
      refine ⟨by omega, ?_⟩
      rw [show s.ncard + 1 + G.compCount (insert a s) = s.ncard + G.compCount s by omega, ih.2]
      exact ⟨fun hF => hF.insert hr, fun hF => hF.mono (Set.subset_insert a s)⟩

/-- The **nullity** `|S| + c(S) - |V|` of a set of edges. -/
noncomputable def nullity (G : OrientedGraph V E) (S : Set E) : ℕ :=
  S.ncard + G.compCount S - Nat.card V

theorem nullity_add_card (S : Set E) : G.nullity S + Nat.card V = S.ncard + G.compCount S := by
  have := (card_le_ncard_add_compCount (G := G) S).1
  unfold nullity
  omega

/-- A set of edges has nullity zero exactly when it is a forest. -/
theorem nullity_eq_zero_iff {S : Set E} : G.nullity S = 0 ↔ G.IsForest S := by
  rw [← (card_le_ncard_add_compCount S).2]
  have := nullity_add_card (G := G) S
  omega

/-- An edge with linked ends raises the nullity by one. -/
theorem nullity_insert_of_reachableIn {S : Set E} {e : E} (heS : e ∉ S)
    (he : G.ReachableIn S (G.tail e) (G.head e)) : G.nullity (insert e S) = G.nullity S + 1 := by
  have h₁ := nullity_add_card (G := G) S
  have h₂ := nullity_add_card (G := G) (insert e S)
  rw [Set.ncard_insert_of_notMem heS, compCount_insert_of_reachableIn he] at h₂
  omega

/-- An edge whose ends are not linked keeps the nullity. -/
theorem nullity_insert_of_not_reachableIn {S : Set E} {e : E}
    (he : ¬G.ReachableIn S (G.tail e) (G.head e)) : G.nullity (insert e S) = G.nullity S := by
  have heS : e ∉ S := fun h => he (reachableIn_of_mem h)
  have h₁ := nullity_add_card (G := G) S
  have h₂ := nullity_add_card (G := G) (insert e S)
  have hc := compCount_insert_of_not_reachableIn (G := G) he
  rw [Set.ncard_insert_of_notMem heS] at h₂
  omega

theorem nullity_le_insert (S : Set E) (e : E) : G.nullity S ≤ G.nullity (insert e S) := by
  by_cases heS : e ∈ S
  · rw [Set.insert_eq_of_mem heS]
  by_cases he : G.ReachableIn S (G.tail e) (G.head e)
  · rw [nullity_insert_of_reachableIn heS he]
    omega
  · rw [nullity_insert_of_not_reachableIn he]

theorem nullity_insert_le (S : Set E) (e : E) : G.nullity (insert e S) ≤ G.nullity S + 1 := by
  by_cases heS : e ∈ S
  · rw [Set.insert_eq_of_mem heS]
    omega
  by_cases he : G.ReachableIn S (G.tail e) (G.head e)
  · rw [nullity_insert_of_reachableIn heS he]
  · rw [nullity_insert_of_not_reachableIn he]
    omega

/-- The nullity grows with the set. -/
theorem nullity_mono {S T : Set E} (hST : S ⊆ T) : G.nullity S ≤ G.nullity T := by
  rw [← Set.union_sdiff_cancel hST]
  induction T \ S, (T \ S).toFinite using Set.Finite.induction_on with
  | empty => rw [Set.union_empty]
  | @insert a s ha hs ih =>
    rw [Set.union_insert]
    exact ih.trans (nullity_le_insert _ a)


example := pumpkin_pair_isCycle.reachableIn_of_subset_insert (e := 0) (S := {1})
  (Set.mem_insert 0 {1}) le_rfl

example := pumpkin_pair_isCycle.reachableIn_diff (Set.mem_insert 0 {1})

example := pumpkin_pair_isCycle.isForest_diff (Set.mem_insert 0 {1})

example := (isForest_empty (G := pathThree)).insert pathThree_not_reachableIn_empty

example := (reachableIn_refl (G := pathThree) ∅ 0).eq_of_notMem_verts (by simp [verts])

example := nullity_insert_of_reachableIn (G := pumpkin) (S := {1}) (e := 0) (by simp)
  (reachableIn_of_mem (e := (1 : Fin 3)) (Set.mem_singleton 1))

example := nullity_insert_of_not_reachableIn pathThree_not_reachableIn_empty

example := nullity_mono (G := pathThree) (Set.empty_subset ({0} : Set (Fin 2)))

end OrientedGraph

end Transformer.GainMatroid
