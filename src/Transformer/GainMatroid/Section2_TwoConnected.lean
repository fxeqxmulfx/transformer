module

public import Transformer.GainMatroid.Section4_Graphs

/-!
# Two-connected graphs from connectivity off each vertex

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.1. A graph is 2-connected when it has at least
four vertices and no vertical separation of order zero or one. A graph on at least four vertices,
each an end of an edge, that stays connected off each vertex is 2-connected
(`OrientedGraph.isKConnected_two_of_connectedOff`): a walk between vertices on the two sides of a
vertical separation of order at most one, avoiding the common vertex, would have to cross from one
side to the other at a vertex of both. This is the converse of `IsKConnected.connectedOff`, and
gives `tube_isKConnected` and `K4_isKConnected`.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- A walk avoiding `w` from a vertex of `A` stays on vertices of `A`, when `A` and its complement
share no vertex other than `w`. -/
theorem Walk.mem_verts_of_avoid {A B : Set E} (hAB : A ∪ B = Set.univ) {w : V}
    (hsub : G.verts A ∩ G.verts B ⊆ {w}) {x y : V} (p : G.Walk x y) (hx : x ∈ G.verts A)
    (hp : w ∉ p.support) : y ∈ G.verts A := by
  induction p with
  | nil => exact hx
  | @cons a b c e hj p ih =>
    have haw : a ≠ w := fun h => hp (by simp [Walk.support, h])
    have hpw : w ∉ p.support := fun h => hp (by simp [Walk.support, h])
    have heA : e ∈ A := by
      rcases (Set.mem_union e A B).1 (hAB ▸ Set.mem_univ e) with he | he
      · exact he
      · have haB : a ∈ G.verts B := by
          rcases hj with ⟨h₁, -⟩ | ⟨-, h₂⟩
          · exact mem_verts.2 ⟨e, he, Or.inl h₁⟩
          · exact mem_verts.2 ⟨e, he, Or.inr h₂⟩
        exact absurd (hsub ⟨hx, haB⟩) haw
    have hbA : b ∈ G.verts A := by
      rcases hj with ⟨-, h₂⟩ | ⟨h₁, -⟩
      · exact mem_verts.2 ⟨e, heA, Or.inr h₂⟩
      · exact mem_verts.2 ⟨e, heA, Or.inl h₁⟩
    exact ih hbA hpw

/-- A graph on at least four vertices, each an end of an edge, that stays connected off each vertex
is 2-connected (§2.1). -/
theorem isKConnected_two_of_connectedOff (hV : 4 ≤ (Set.univ : Set V).encard)
    (hverts : ∀ v, v ∈ G.verts Set.univ) (hc : G.ConnectedOff) : G.IsKConnected 2 := by
  refine Or.inl ⟨le_trans (by norm_num) hV, fun l hl A B ⟨hsep, ⟨a, haA, haB⟩, ⟨b, hbB, hbA⟩⟩ => ?_⟩
  have hle : (G.verts A ∩ G.verts B).encard ≤ 1 := by
    rw [hsep.inter]
    exact_mod_cast Nat.lt_succ_iff.1 hl
  obtain ⟨w, hwa, hwb, hsub⟩ : ∃ w, w ≠ a ∧ w ≠ b ∧ G.verts A ∩ G.verts B ⊆ {w} := by
    rcases Set.encard_le_one_iff_eq.1 hle with h | ⟨w, h⟩
    · have h3 : (2 : ℕ∞) < (Set.univ : Set V).encard := lt_of_lt_of_le (by norm_num) hV
      have hpair : ({a, b} : Set V).encard ≤ 2 :=
        (Set.encard_insert_le _ _).trans (by rw [Set.encard_singleton]; norm_num)
      have hns : ¬(Set.univ : Set V) ⊆ {a, b} := fun hs =>
        absurd ((Set.encard_le_encard hs).trans hpair) (not_le.2 h3)
      obtain ⟨w, -, hw⟩ := Set.not_subset_iff_exists_mem_notMem.1 hns
      simp only [Set.mem_insert_iff, Set.mem_singleton_iff, not_or] at hw
      exact ⟨w, hw.1, hw.2, h ▸ Set.empty_subset _⟩
    · have hw : w ∈ G.verts A ∩ G.verts B := h ▸ rfl
      exact ⟨w, fun hwa => haB (hwa ▸ hw.2), fun hwb => hbA (hwb ▸ hw.1), h.le⟩
  obtain ⟨p, hp⟩ := hc w a b (hverts a) (hverts b) hwa.symm hwb.symm
  exact hbA (Walk.mem_verts_of_avoid hsep.union hsub p haA hp)

/-- Every vertex of `K₄` is an end of an edge. -/
theorem K4_mem_verts (v : Fin 4) : v ∈ K4.verts Set.univ := by
  refine mem_verts.2 ?_
  fin_cases v
  exacts [⟨0, trivial, Or.inl rfl⟩, ⟨0, trivial, Or.inr rfl⟩, ⟨1, trivial, Or.inr rfl⟩,
    ⟨2, trivial, Or.inr rfl⟩]

/-- Every vertex of the tube is an end of an edge. -/
theorem tube_mem_verts (v : Fin 4) : v ∈ tube.verts Set.univ := by
  refine mem_verts.2 ?_
  fin_cases v
  exacts [⟨0, trivial, Or.inl rfl⟩, ⟨0, trivial, Or.inr rfl⟩, ⟨2, trivial, Or.inr rfl⟩,
    ⟨3, trivial, Or.inr rfl⟩]

/-- `K₄` is 2-connected. -/
theorem K4_isKConnected : K4.IsKConnected 2 :=
  isKConnected_two_of_connectedOff (by simp [Set.encard_univ]) K4_mem_verts
    (connectedOff_of_star K4_star)

/-- The tube is 2-connected. -/
theorem tube_isKConnected : tube.IsKConnected 2 :=
  isKConnected_two_of_connectedOff (by simp [Set.encard_univ]) tube_mem_verts
    (connectedOff_of_star tube_star)

end OrientedGraph

end Transformer.GainMatroid
