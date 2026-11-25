module

public import Transformer.GainMatroid.Section2_ExchangeCycles

/-!
# Counting the components of a set of edges

Funk, Pivotto and Slilaty (arXiv:1609.05574v5), §2.2, cite Zaslavsky for the fact that the frame and
lift circuits of a biased graph are the circuits of matroids. The proof given here counts the
components of a set `S` of edges: the classes of `reachableSetoid G S` on all the vertices, isolated
ones included (`compCount`).

* A relation that holds across every edge of `S` and is reflexive and transitive holds between the
  ends of every walk along `S` (`ReachableIn.rel`).
* Adding an edge whose ends are already linked keeps the components
  (`compCount_insert_of_reachableIn`); adding any other edge joins two of them
  (`compCount_insert_of_not_reachableIn`).
* Without edges every vertex is a component (`compCount_empty`), and reachability along `T` inside
  reachability along `S` leaves `S` with at most as many components (`compCount_le_of_forall`).
-/

@[expose] public section

namespace Transformer.GainMatroid

namespace OrientedGraph

variable {V E : Type*} {G : OrientedGraph V E}

/-- A reflexive and transitive relation that holds across every edge of `S`, both ways, holds
between the ends of every walk along `S`. -/
theorem ReachableIn.rel {S : Set E} {r : V → V → Prop} (hrefl : ∀ v, r v v)
    (htrans : ∀ u v w, r u v → r v w → r u w)
    (hstep : ∀ e ∈ S, r (G.tail e) (G.head e) ∧ r (G.head e) (G.tail e)) {u v : V}
    (h : G.ReachableIn S u v) : r u v := by
  obtain ⟨p, hp⟩ := h
  induction p with
  | nil u => exact hrefl u
  | @cons a b c e hj p ih =>
    have hab : r a b := by
      rcases hj with ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩
      · rw [← h₁, ← h₂]
        exact (hstep e (hp e List.mem_cons_self)).1
      · rw [← h₁, ← h₂]
        exact (hstep e (hp e List.mem_cons_self)).2
    exact htrans a b c hab (ih fun f hf => hp f (List.mem_cons_of_mem _ hf))

/-- Reachability along `T` gives reachability along `S` when the ends of every edge of `T` are
linked along `S`. -/
theorem ReachableIn.of_forall {S T : Set E}
    (hT : ∀ e ∈ T, G.ReachableIn S (G.tail e) (G.head e)) {u v : V} (h : G.ReachableIn T u v) :
    G.ReachableIn S u v :=
  ReachableIn.rel (r := G.ReachableIn S) (reachableIn_refl S) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
    (fun e he => ⟨hT e he, (hT e he).symm⟩) h

/-- The ends of an edge of `S` are linked along `S`. -/
theorem reachableIn_of_mem {S : Set E} {e : E} (he : e ∈ S) :
    G.ReachableIn S (G.tail e) (G.head e) :=
  (reachableIn_refl S _).step he (Or.inl ⟨rfl, rfl⟩)

theorem ReachableIn.mono {S T : Set E} (hST : S ⊆ T) {u v : V} (h : G.ReachableIn S u v) :
    G.ReachableIn T u v :=
  h.of_forall fun _ he => reachableIn_of_mem (hST he)

/-- Reachability along `insert e S`: along `S`, or along `S` to one end of `e` and on from the
other. -/
theorem reachableIn_insert_iff {S : Set E} {e : E} {u v : V} :
    G.ReachableIn (insert e S) u v ↔ G.ReachableIn S u v ∨
      (G.ReachableIn S u (G.tail e) ∧ G.ReachableIn S (G.head e) v) ∨
      (G.ReachableIn S u (G.head e) ∧ G.ReachableIn S (G.tail e) v) := by
  constructor
  · refine ReachableIn.rel (r := fun a b => G.ReachableIn S a b ∨
      (G.ReachableIn S a (G.tail e) ∧ G.ReachableIn S (G.head e) b) ∨
      (G.ReachableIn S a (G.head e) ∧ G.ReachableIn S (G.tail e) b))
      (fun v => Or.inl (reachableIn_refl S v)) ?_ ?_
    · rintro a b c (hab | ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩) (hbc | ⟨h₃, h₄⟩ | ⟨h₃, h₄⟩)
      · exact Or.inl (hab.trans hbc)
      · exact Or.inr (Or.inl ⟨hab.trans h₃, h₄⟩)
      · exact Or.inr (Or.inr ⟨hab.trans h₃, h₄⟩)
      · exact Or.inr (Or.inl ⟨h₁, h₂.trans hbc⟩)
      · exact Or.inr (Or.inl ⟨h₁, h₄⟩)
      · exact Or.inl (h₁.trans h₄)
      · exact Or.inr (Or.inr ⟨h₁, h₂.trans hbc⟩)
      · exact Or.inl (h₁.trans h₄)
      · exact Or.inr (Or.inr ⟨h₁, h₄⟩)
    · rintro f (rfl | hf)
      · exact ⟨Or.inr (Or.inl ⟨reachableIn_refl _ _, reachableIn_refl _ _⟩),
          Or.inr (Or.inr ⟨reachableIn_refl _ _, reachableIn_refl _ _⟩)⟩
      · exact ⟨Or.inl (reachableIn_of_mem hf), Or.inl (reachableIn_of_mem hf).symm⟩
  · have hS : S ⊆ insert e S := Set.subset_insert e S
    rintro (h | ⟨h₁, h₂⟩ | ⟨h₁, h₂⟩)
    · exact h.mono hS
    · exact ((h₁.mono hS).step (Set.mem_insert e S) (Or.inl ⟨rfl, rfl⟩)).trans (h₂.mono hS)
    · exact ((h₁.mono hS).step (Set.mem_insert e S) (Or.inr ⟨rfl, rfl⟩)).trans (h₂.mono hS)

/-- The number of components of `S`: the classes of `reachableSetoid G S`, isolated vertices
included. -/
noncomputable def compCount (G : OrientedGraph V E) (S : Set E) : ℕ :=
  Nat.card (Quotient (G.reachableSetoid S))

/-- An edge whose ends are already linked keeps the components. -/
theorem compCount_insert_of_reachableIn {S : Set E} {e : E}
    (he : G.ReachableIn S (G.tail e) (G.head e)) : G.compCount (insert e S) = G.compCount S := by
  have h : G.reachableSetoid (insert e S) = G.reachableSetoid S := by
    ext u v
    refine ⟨fun h => ReachableIn.of_forall (S := S) (T := insert e S) ?_ h,
      fun h => ReachableIn.mono (Set.subset_insert e S) h⟩
    rintro f (rfl | hf)
    · exact he
    · exact reachableIn_of_mem hf
  simp only [compCount, h]

/-- An edge whose ends are not linked joins two components. -/
theorem compCount_insert_of_not_reachableIn [Finite V] {S : Set E} {e : E}
    (he : ¬G.ReachableIn S (G.tail e) (G.head e)) :
    G.compCount (insert e S) + 1 = G.compCount S := by
  let s := G.reachableSetoid S
  let s' := G.reachableSetoid (insert e S)
  let π : Quotient s → Quotient s' :=
    Quotient.map' id fun _ _ h => ReachableIn.mono (Set.subset_insert e S) h
  have hπ (v : V) : π (Quotient.mk s v) = Quotient.mk s' v := rfl
  have hth : Quotient.mk s (G.tail e) ≠ Quotient.mk s (G.head e) := fun h => he (Quotient.exact h)
  let A : Set (Quotient s) := Set.univ \ {Quotient.mk s (G.head e)}
  have himage : π '' A = Set.univ := by
    refine Set.eq_univ_of_forall fun c => ?_
    obtain ⟨v, rfl⟩ := Quotient.mk_surjective c
    by_cases hv : Quotient.mk s v = Quotient.mk s (G.head e)
    · refine ⟨Quotient.mk s (G.tail e), ⟨trivial, hth⟩, ?_⟩
      rw [hπ]
      exact Quotient.sound ((reachableIn_of_mem (Set.mem_insert e S)).trans
        ((Quotient.exact hv).symm.mono (Set.subset_insert e S)))
    · exact ⟨Quotient.mk s v, ⟨trivial, hv⟩, rfl⟩
  have hinj : Set.InjOn π A := by
    intro c hc d hd hcd
    obtain ⟨u, rfl⟩ := Quotient.mk_surjective c
    obtain ⟨v, rfl⟩ := Quotient.mk_surjective d
    rw [hπ, hπ] at hcd
    rcases reachableIn_insert_iff.1 (Quotient.exact hcd) with h | ⟨-, h⟩ | ⟨h, -⟩
    · exact Quotient.sound h
    · exact absurd (Quotient.sound h.symm) hd.2
    · exact absurd (Quotient.sound h) hc.2
  have hcard : A.ncard + 1 = (Set.univ : Set (Quotient s)).ncard :=
    Set.ncard_sdiff_singleton_add_one (Set.mem_univ _)
  rw [← hinj.ncard_image, himage, Set.ncard_univ, Set.ncard_univ] at hcard
  exact hcard

/-- Without edges every vertex is a component. -/
theorem compCount_empty : G.compCount ∅ = Nat.card V := by
  have hinj : Function.Injective (Quotient.mk (G.reachableSetoid ∅)) := fun u v h =>
    ReachableIn.rel (S := ∅) (r := Eq) (fun _ => rfl) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
      (fun _ he => absurd he (Set.notMem_empty _)) (Quotient.exact h)
  exact (Nat.card_congr (Equiv.ofBijective _ ⟨hinj, Quotient.mk_surjective⟩)).symm

/-- If the ends of every edge of `T` are linked along `S`, then `S` has at most as many components
as `T`. -/
theorem compCount_le_of_forall [Finite V] {S T : Set E}
    (hT : ∀ e ∈ T, G.ReachableIn S (G.tail e) (G.head e)) : G.compCount S ≤ G.compCount T := by
  let f : Quotient (G.reachableSetoid T) → Quotient (G.reachableSetoid S) :=
    Quotient.map' id fun _ _ h => ReachableIn.of_forall hT h
  have hf : Function.Surjective f := fun c => by
    obtain ⟨v, rfl⟩ := Quotient.mk_surjective c
    exact ⟨Quotient.mk _ v, rfl⟩
  exact Nat.card_le_card_of_surjective f hf

/-- Without edges, the ends of the first edge of `pathThree` are not linked. -/
theorem pathThree_not_reachableIn_empty :
    ¬pathThree.ReachableIn ∅ (pathThree.tail 0) (pathThree.head 0) := fun h =>
  absurd (ReachableIn.rel (S := ∅) (r := Eq) (fun _ => rfl) (fun _ _ _ h₁ h₂ => h₁.trans h₂)
    (fun _ he => absurd he (Set.notMem_empty _)) h) (by decide)

example := ReachableIn.rel (G := pathThree) (S := ∅) (r := Eq) (fun _ => rfl)
  (fun _ _ _ h₁ h₂ => h₁.trans h₂) (fun _ he => absurd he (Set.notMem_empty _))
  (reachableIn_refl ∅ (0 : Fin 3))

example := ReachableIn.of_forall (G := pathThree) (S := {0}) (T := {0})
  (fun _ he => reachableIn_of_mem he) (reachableIn_of_mem (Set.mem_singleton 0))

example := reachableIn_of_mem (G := pathThree) (Set.mem_singleton (0 : Fin 2))

example := (reachableIn_of_mem (G := pathThree) (Set.mem_singleton (0 : Fin 2))).mono
  (Set.subset_insert 1 {0})

example := compCount_insert_of_reachableIn (G := pathThree) (e := 0)
  (reachableIn_of_mem (Set.mem_singleton (0 : Fin 2)))

example := compCount_insert_of_not_reachableIn pathThree_not_reachableIn_empty

example := compCount_le_of_forall (G := pathThree) (S := {0}) (T := {0})
  fun _ he => reachableIn_of_mem he

end OrientedGraph

end Transformer.GainMatroid
