module

public import Transformer.Phasor.ClosedRelations

/-!
# Minimum support of a moving closed-relation address

Source: the closed-read timing trial in `experiments/phasor_value_transport/README.md`.
The query, its image and its two-step image must be three distinct written
sources when a closed relation moves a two-step read. A streaming reader
can use the size of its entire edge table before inspecting any relation.
-/

@[expose] public section

namespace Transformer.Phasor

variable {α β : Type*} {p : ℕ}

/-- A two-step movement within a finite permutation witnesses three distinct
members of its domain, independently of the token labels. -/
theorem three_le_card_of_moving_twoStep [DecidableEq α] (s : Finset α) (f : α → α) (k : α)
    (hk : k ∈ s) (perm : Set.BijOn f (s : Set α) s) (moves : f (f k) ≠ k) :
    3 ≤ s.card := by
  have hm : f k ∈ s := perm.1 hk
  have he : f (f k) ∈ s := perm.1 hm
  have km : k ≠ f k := fun h => moves (h.trans (congrArg f h)).symm
  have me : f k ≠ f (f k) := fun h => km (perm.2.1 hk hm h)
  have sub : {k, f k, f (f k)} ⊆ s :=
    Finset.insert_subset hk (Finset.insert_subset hm (Finset.singleton_subset_iff.mpr he))
  simpa [km, me, Ne.symm moves] using Finset.card_le_card sub

/-- With at most two written sources a closed relation either fixes the
query or exchanges two elements whose second image is the query again. -/
theorem closedPathEnd_of_card_le_two (s : Finset (Fin p)) (f : Fin p → Fin p) (k : Fin p)
    (small : s.card ≤ 2) : closedPathEnd s f k = k := by
  by_contra moves
  unfold closedPathEnd at moves
  split at moves
  · rename_i h
    have bound := three_le_card_of_moving_twoStep s f k h.1 ((closed_counts_iff s f).mp h.2) moves
    omega
  · exact moves rfl

/-- An upper bound on the total number of distinct stored pairs bounds the
number of sources under every individual relation, even after overwrites. -/
theorem relationDomain_card_le_edges [DecidableEq α] [DecidableEq β]
    (edges : Finset (α × β)) (h : β) :
    ((edges.filter (fun pair => pair.2 = h)).image Prod.fst).card ≤ edges.card :=
  Finset.card_image_le.trans (Finset.card_le_card (Finset.filter_subset _ _))

/-- Before the third distinct edge is written the closed consensus always
returns the supplied fallback, for every relation update and every query. -/
theorem closedRelationAddress_of_small_table (edges : Finset (Fin p × Fin p))
    (f : Fin p → Fin p → Fin p) (k fallback : Fin p) (small : edges.card ≤ 2) :
    closedRelationAddress k fallback
      (fun h => (edges.filter (fun pair => pair.2 = h)).image Prod.fst) f = fallback := by
  have fixed (h : Fin p) : closedPathEnd
      ((edges.filter (fun pair => pair.2 = h)).image Prod.fst) (f h) k = k :=
    closedPathEnd_of_card_le_two _ _ _ ((relationDomain_card_le_edges edges h).trans small)
  simp [closedRelationAddress, fixed, supportedRelations, movingRelations, consensusAddress]

/-- A three-cycle attains the bound. A transposition cannot move a two-step read. -/
example : 3 ≤ (Finset.univ : Finset (Fin 3)).card ∧
    closedPathEnd Finset.univ (fun a : Fin 3 => a+1) 0 = 2 := by decide

example : closedPathEnd Finset.univ (fun a : Fin 2 => a+1) 0 = 0 :=
  closedPathEnd_of_card_le_two _ _ _ (by decide)

example : closedRelationAddress 0 4 (fun h =>
    (({(0, 2), (1, 2)} : Finset (Fin 5 × Fin 5)).filter
      (fun pair => pair.2 = h)).image Prod.fst) (fun _ a => a+1) = 4 :=
  closedRelationAddress_of_small_table _ _ _ _ (by decide)

end Transformer.Phasor
