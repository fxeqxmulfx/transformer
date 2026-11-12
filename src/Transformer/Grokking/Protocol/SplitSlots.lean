module

public import Transformer.Grokking.Protocol.Target

/-!
# The slots of a shuffle that keeps a set of equations out of training

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; `experiments/grok_division/PROTOCOL.md`, *The split*.

A shuffle `π` puts `N` items in `N` slots: the item `π t` is in the slot `t`, the item `x` in
the slot `π.symm x`, and the first `k` slots are the training set. `keptOut k Q` is the set
of the shuffles that keep every item of `Q` out of training. A shuffle has `N - |Q|` slots
whose item is not in `Q` (`card_filter_apply_notMem`); if it keeps `Q` out of training,
`N - k - |Q|` of them are held out (`card_filter_heldOut_notMem`). `SplitCount` counts the
shuffles of `keptOut` with these two numbers.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {N : ℕ}

/-- **The shuffles that keep every item of `Q` out of the first `k` slots.** -/
def keptOut (k : ℕ) (Q : Finset (Fin N)) : Finset (Equiv.Perm (Fin N)) :=
  Finset.univ.filter fun π => ∀ x ∈ Q, k ≤ (π.symm x : ℕ)

lemma mem_keptOut {k : ℕ} {Q : Finset (Fin N)} {π : Equiv.Perm (Fin N)} :
    π ∈ keptOut k Q ↔ ∀ x ∈ Q, k ≤ (π.symm x : ℕ) := by
  simp [keptOut]

/-- The slot of an item after the items of two slots are exchanged. -/
lemma symm_mul_swap_apply (π : Equiv.Perm (Fin N)) (a t x : Fin N) :
    (π * Equiv.swap a t).symm x = Equiv.swap a t (π.symm x) := by
  rw [Equiv.symm_apply_eq]
  simp

/-- A shuffle has `N - |Q|` slots whose item is not in `Q`. -/
lemma card_filter_apply_notMem (π : Equiv.Perm (Fin N)) (Q : Finset (Fin N)) :
    (Finset.univ.filter fun t => π t ∉ Q).card = N - Q.card := by
  have h1 : (Finset.univ.filter fun t => π t ∈ Q).card = Q.card :=
    Finset.card_nbij' (fun t => π t) (fun x => π.symm x)
      (fun t ht => (Finset.mem_filter.1 ht).2)
      (fun x hx => Finset.mem_filter.2 ⟨Finset.mem_univ _, by simpa using hx⟩)
      (fun t _ => by simp) (fun x _ => by simp)
  have h2 := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin N)))
    (fun t => π t ∈ Q)
  rw [h1, Finset.card_univ, Fintype.card_fin] at h2
  omega

/-- A shuffle that keeps `Q` out of training has `N - k - |Q|` held-out slots whose item is
not in `Q`. -/
lemma card_filter_heldOut_notMem {k : ℕ} {Q : Finset (Fin N)} {π : Equiv.Perm (Fin N)}
    (hπ : π ∈ keptOut k Q) :
    (Finset.univ.filter fun t : Fin N => k ≤ (t : ℕ) ∧ π t ∉ Q).card = N - k - Q.card := by
  have hS : (Finset.univ.filter fun t : Fin N => k ≤ (t : ℕ)).card = N - min N k := by
    have h := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin N)))
      (fun t => (t : ℕ) < k)
    rw [Fin.card_filter_val_lt, Finset.card_univ, Fintype.card_fin] at h
    have hc : (Finset.univ.filter fun t : Fin N => ¬ (t : ℕ) < k) =
        Finset.univ.filter fun t : Fin N => k ≤ (t : ℕ) :=
      Finset.filter_congr fun t _ => not_lt
    rw [hc] at h
    omega
  have hQ : (Finset.univ.filter fun t : Fin N => k ≤ (t : ℕ) ∧ π t ∈ Q).card = Q.card := by
    have hc : (Finset.univ.filter fun t : Fin N => k ≤ (t : ℕ) ∧ π t ∈ Q) =
        Finset.univ.filter fun t => π t ∈ Q := by
      refine Finset.filter_congr fun t _ => and_iff_right_of_imp fun ht => ?_
      have := mem_keptOut.1 hπ _ ht
      rwa [Equiv.symm_apply_apply] at this
    have h := card_filter_apply_notMem π Q
    have h2 := Finset.card_filter_add_card_filter_not (s := (Finset.univ : Finset (Fin N)))
      (fun t => π t ∈ Q)
    have hle := Finset.card_le_univ Q
    rw [Finset.card_univ, Fintype.card_fin] at h2
    rw [Fintype.card_fin] at hle
    rw [hc]
    omega
  have h := Finset.card_filter_add_card_filter_not
    (s := Finset.univ.filter fun t : Fin N => k ≤ (t : ℕ)) (fun t => π t ∈ Q)
  rw [Finset.filter_filter, Finset.filter_filter, hQ, hS] at h
  omega

/-! The witnesses: four items and two training slots; the shuffle that exchanges the items `0`
and `2` keeps the item `0` out of training, and one of its two held-out slots has another
item. -/

example : (Equiv.swap (0 : Fin 4) 2 * Equiv.swap 0 1).symm 3 =
    Equiv.swap 0 1 ((Equiv.swap (0 : Fin 4) 2).symm 3) :=
  symm_mul_swap_apply _ 0 1 3

example : (Finset.univ.filter fun t : Fin 4 =>
    2 ≤ (t : ℕ) ∧ Equiv.swap (0 : Fin 4) 2 t ∉ ({0} : Finset (Fin 4))).card =
      4 - 2 - ({0} : Finset (Fin 4)).card :=
  card_filter_heldOut_notMem (mem_keptOut.2 (by decide))

end Transformer.Grokking.Protocol
