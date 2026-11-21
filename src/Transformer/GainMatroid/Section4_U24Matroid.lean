module

public import Transformer.GainMatroid.Section4_TwoVertexCircuits

/-!
# `𝖴₂` and `𝖴₃` represent `U_{2,4}`

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §4.1: "Two biased graphic representations of
`U_{2,4}` are `𝖴₂` and `𝖴₃`, […] where all cycles in each are unbalanced." The circuits of
`U_{2,4}` are the sets of three edges, that is, the complements of single edges.

In `𝖴₂` such a complement is a tight handcuff when the missing edge is a loop and a loose handcuff
when it is a link, so `𝖴₂` represents `U_{2,4}` as a frame matroid. Its two loops are
vertex-disjoint cycles, so it is not a lift representation. In `𝖴₃` the complement is a theta or a
tight handcuff, and every cycle passes through `v₁`, so `𝖴₃` represents `U_{2,4}` both as a frame
matroid and as a lift matroid.
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- The frame circuits of a contrabalanced biased graph: tight and loose handcuffs and thetas. -/
theorem contrabalanced_isFrameCircuit_iff {X : Set E} :
    G.contrabalanced.IsFrameCircuit X ↔
      G.IsTightHandcuff X ∨ G.IsLooseHandcuff X ∨ G.IsThetaSet X :=
  ⟨fun h => h.elim (fun h => absurd h (Set.notMem_empty _)) And.left,
    fun h => Or.inr ⟨h, fun _ _ _ hmem => Set.notMem_empty _ hmem⟩⟩

/-- The lift circuits of a contrabalanced biased graph: tight handcuffs, two vertex-disjoint cycles
and thetas. -/
theorem contrabalanced_isLiftCircuit_iff {X : Set E} :
    G.contrabalanced.IsLiftCircuit X ↔
      G.IsTightHandcuff X ∨ G.IsDisjointCycles X ∨ G.IsThetaSet X :=
  ⟨fun h => h.elim (fun h => absurd h (Set.notMem_empty _)) And.left,
    fun h => Or.inr ⟨h, fun _ _ _ hmem => Set.notMem_empty _ hmem⟩⟩

/-- `U₂` has one loop at each vertex. -/
theorem U2_loops : ∀ e f : Fin 4, e ≠ f → U2.tail e = U2.head e → U2.tail f = U2.head f →
    U2.tail e ≠ U2.tail f := by
  simp only [U2]
  decide

/-- `U₃` has a single loop. -/
theorem U3_loops : ∀ e f : Fin 4, e ≠ f → U3.tail e = U3.head e → U3.tail f = U3.head f →
    U3.tail e ≠ U3.tail f := by
  simp only [U3]
  decide

/-- Every cycle of `U₃` passes through `v₁`. -/
theorem U3_zero_mem_verts {C : Set (Fin 4)} (hC : U3.IsCycle C) : (0 : Fin 2) ∈ U3.verts C := by
  rcases hC.encard_eq_one_or_two with h | h
  · obtain ⟨e, he, rfl⟩ := hC.exists_loop h
    have hloop : ∀ e : Fin 4, U3.tail e = U3.head e → U3.tail e = 0 := by
      simp only [U3]
      decide
    exact mem_verts.2 ⟨e, Set.mem_singleton e, Or.inl (hloop e he)⟩
  · rw [hC.verts_eq_univ_of_encard_two h]
    exact Set.mem_univ _

/-- `U₃` has no two vertex-disjoint cycles. -/
theorem U3_not_isDisjointCycles (X : Set (Fin 4)) : ¬U3.IsDisjointCycles X :=
  fun ⟨_, _, hC, hD, hdisj, _⟩ =>
    Set.disjoint_left.1 hdisj (U3_zero_mem_verts hC) (U3_zero_mem_verts hD)

/-- The complement of an edge of `U₂` is a tight handcuff if the edge is a loop and a loose handcuff
if it is a link. -/
theorem U2_compl_shape (e : Fin 4) : U2.IsTightHandcuff ({e}ᶜ) ∨ U2.IsLooseHandcuff ({e}ᶜ) := by
  have h4 : ∀ e : Fin 4, e = 0 ∨ e = 1 ∨ e = 2 ∨ e = 3 := by decide
  rcases h4 e with rfl | rfl | rfl | rfl
  · rw [show ({0}ᶜ : Set (Fin 4)) = {1, 2, 3} by ext a; fin_cases a <;> simp]
    exact Or.inl (isTightHandcuff_of_loop_pair (u := 1) (v := 0) rfl rfl (Or.inr ⟨rfl, rfl⟩)
      (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide) (by decide) (by decide))
  · rw [show ({1}ᶜ : Set (Fin 4)) = {0, 2, 3} by ext a; fin_cases a <;> simp]
    exact Or.inl (isTightHandcuff_of_loop_pair (u := 0) (v := 1) rfl rfl (Or.inl ⟨rfl, rfl⟩)
      (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide) (by decide) (by decide))
  · rw [show ({2}ᶜ : Set (Fin 4)) = {0, 1, 3} by ext a; fin_cases a <;> simp]
    exact Or.inr (isLooseHandcuff_of_loops (x := 0) (y := 1) rfl rfl rfl rfl (Or.inr ⟨rfl, rfl⟩)
      (by decide))
  · rw [show ({3}ᶜ : Set (Fin 4)) = {0, 1, 2} by ext a; fin_cases a <;> simp]
    exact Or.inr (isLooseHandcuff_of_loops (x := 0) (y := 1) rfl rfl rfl rfl (Or.inl ⟨rfl, rfl⟩)
      (by decide))

/-- The complement of an edge of `U₃` is a theta if the edge is the loop and a tight handcuff
otherwise. -/
theorem U3_compl_shape (e : Fin 4) : U3.IsTightHandcuff ({e}ᶜ) ∨ U3.IsThetaSet ({e}ᶜ) := by
  have h4 : ∀ e : Fin 4, e = 0 ∨ e = 1 ∨ e = 2 ∨ e = 3 := by decide
  rcases h4 e with rfl | rfl | rfl | rfl
  · rw [show ({0}ᶜ : Set (Fin 4)) = {1, 2, 3} by ext a; fin_cases a <;> simp]
    exact Or.inr (isThetaSet_of_three (x := 0) (y := 1) (by decide) (Or.inl ⟨rfl, rfl⟩)
      (Or.inl ⟨rfl, rfl⟩) (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide) (by decide))
  · rw [show ({1}ᶜ : Set (Fin 4)) = {0, 2, 3} by ext a; fin_cases a <;> simp]
    exact Or.inl (isTightHandcuff_of_loop_pair (u := 0) (v := 1) rfl rfl (Or.inl ⟨rfl, rfl⟩)
      (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide) (by decide) (by decide))
  · rw [show ({2}ᶜ : Set (Fin 4)) = {0, 1, 3} by ext a; fin_cases a <;> simp]
    exact Or.inl (isTightHandcuff_of_loop_pair (u := 0) (v := 1) rfl rfl (Or.inl ⟨rfl, rfl⟩)
      (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide) (by decide) (by decide))
  · rw [show ({3}ᶜ : Set (Fin 4)) = {0, 1, 2} by ext a; fin_cases a <;> simp]
    exact Or.inl (isTightHandcuff_of_loop_pair (u := 0) (v := 1) rfl rfl (Or.inl ⟨rfl, rfl⟩)
      (Or.inl ⟨rfl, rfl⟩) (by decide) (by decide) (by decide) (by decide))

end OrientedGraph

open OrientedGraph

private theorem ncard_iff_of_shape {P : Set (Fin 4) → Prop} (h₁ : ∀ e, P ({e}ᶜ))
    (h₂ : ∀ Y, P Y → Y.encard = 3) : ∀ X : Set (Fin 4), X.ncard = 2 + 1 ↔ P X := by
  intro X
  constructor
  · intro h
    have hc : Xᶜ.ncard = 1 := by
      have h' := Set.ncard_add_ncard_compl X (Set.toFinite X) (Set.toFinite Xᶜ)
      simp only [Nat.card_eq_fintype_card, Fintype.card_fin] at h'
      omega
    obtain ⟨e, he⟩ := Set.ncard_eq_one.1 hc
    rw [show X = {e}ᶜ by rw [← he, compl_compl]]
    exact h₁ e
  · intro h
    have hn := (Set.toFinite X).cast_ncard_eq
    rw [h₂ X h] at hn
    exact_mod_cast hn

/-- `𝖴₂` is a frame representation of `U_{2,4}`. -/
theorem U2_isFrameMatroid : U2.contrabalanced.IsFrameMatroid (uniformMatroid (Fin 4) 2) := by
  refine ⟨rfl, ?_⟩
  simp only [uniformMatroid_isCircuit_iff, contrabalanced_isFrameCircuit_iff]
  refine ncard_iff_of_shape (fun e => ?_) fun _ h => ?_
  · exact (U2_compl_shape e).imp_right Or.inl
  · exact h.elim (IsTightHandcuff.encard_eq_three_of_two U2_loops) fun h =>
      h.elim IsLooseHandcuff.encard_eq_three_of_two IsThetaSet.encard_eq_three_of_two

/-- `𝖴₃` is a frame representation of `U_{2,4}`. -/
theorem U3_isFrameMatroid : U3.contrabalanced.IsFrameMatroid (uniformMatroid (Fin 4) 2) := by
  refine ⟨rfl, ?_⟩
  simp only [uniformMatroid_isCircuit_iff, contrabalanced_isFrameCircuit_iff]
  refine ncard_iff_of_shape (fun e => ?_) fun _ h => ?_
  · exact (U3_compl_shape e).imp_right Or.inr
  · exact h.elim (IsTightHandcuff.encard_eq_three_of_two U3_loops) fun h =>
      h.elim IsLooseHandcuff.encard_eq_three_of_two IsThetaSet.encard_eq_three_of_two

/-- `𝖴₃` is a lift representation of `U_{2,4}`. -/
theorem U3_isLiftMatroid : U3.contrabalanced.IsLiftMatroid (uniformMatroid (Fin 4) 2) := by
  refine ⟨rfl, ?_⟩
  simp only [uniformMatroid_isCircuit_iff, contrabalanced_isLiftCircuit_iff]
  refine ncard_iff_of_shape (fun e => ?_) fun _ h => ?_
  · exact (U3_compl_shape e).imp_right Or.inr
  · exact h.elim (IsTightHandcuff.encard_eq_three_of_two U3_loops) fun h =>
      h.elim (fun hd => absurd hd (U3_not_isDisjointCycles _)) IsThetaSet.encard_eq_three_of_two

/-- **§4.1, before Lemma 4.11.** `𝖴₂` and `𝖴₃` are biased graphic representations of `U_{2,4}`. -/
theorem U24_represents :
    U2.contrabalanced.Represents (uniformMatroid (Fin 4) 2) ∧
      U3.contrabalanced.Represents (uniformMatroid (Fin 4) 2) :=
  ⟨Or.inl U2_isFrameMatroid, Or.inl U3_isFrameMatroid⟩

/-- `𝖴₂` is not a lift representation of `U_{2,4}`: its two loops are vertex-disjoint cycles. -/
theorem U2_not_isLiftMatroid : ¬U2.contrabalanced.IsLiftMatroid (uniformMatroid (Fin 4) 2) := by
  rintro ⟨-, h⟩
  have hcirc : U2.contrabalanced.IsLiftCircuit {0, 1} := by
    refine Or.inr ⟨Or.inr (Or.inl ⟨{0}, {1}, isCycle_singleton_of_loop rfl,
      isCycle_singleton_of_loop rfl, ?_, Set.singleton_union.symm⟩),
      fun _ _ _ hmem => Set.notMem_empty _ hmem⟩
    rw [verts_singleton_of_loop (u := 0) rfl rfl, verts_singleton_of_loop (u := 1) rfl rfl]
    exact Set.disjoint_singleton.2 (by decide)
  have hn := uniformMatroid_isCircuit_iff.1 ((h _).2 hcirc)
  rw [Set.ncard_pair (by decide)] at hn
  omega

example : U3.verts {2, 3} = Set.univ :=
  (isCycle_pair (G := U3) (u := 0) (v := 1) (e := 2) (f := 3) (Or.inl ⟨rfl, rfl⟩)
    (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide)).verts_eq_univ_of_encard_two
    (Set.encard_pair (by decide))

example : (Walk.cons (G := U2) (2 : Fin 4) (Or.inl ⟨rfl, rfl⟩ : U2.Joins 2 0 1)
    (.nil 1)).edges.length = 1 :=
  Walk.IsPath.length_eq_one_of_two (by simp [Walk.IsPath, Walk.support]) (by decide)

example : U3.tail 0 = 0 :=
  tail_eq_of_mem_verts_loop (G := U3) (e := 0) (v := 0) rfl (mem_verts.2 ⟨0, rfl, Or.inl rfl⟩)

example : ({0, 2, 3} : Set (Fin 4)).encard = 3 :=
  IsTightHandcuff.encard_eq_three_of_two U2_loops
    (isTightHandcuff_of_loop_pair (u := 0) (v := 1) rfl rfl (Or.inl ⟨rfl, rfl⟩)
      (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide) (by decide) (by decide))

example : ({0, 1, 2} : Set (Fin 4)).encard = 3 :=
  IsLooseHandcuff.encard_eq_three_of_two (G := U2)
    (isLooseHandcuff_of_loops (x := 0) (y := 1) rfl rfl rfl rfl (Or.inl ⟨rfl, rfl⟩) (by decide))

example : ({1, 2, 3} : Set (Fin 4)).encard = 3 :=
  IsThetaSet.encard_eq_three_of_two (G := U3)
    (isThetaSet_of_three (x := 0) (y := 1) (by decide) (Or.inl ⟨rfl, rfl⟩) (Or.inl ⟨rfl, rfl⟩)
      (Or.inr ⟨rfl, rfl⟩) (by decide) (by decide) (by decide))

end Transformer.GainMatroid
