module

public import Transformer.Grokking.Protocol.SplitSlots

/-!
# How many shuffles keep a set of equations out of training

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; `experiments/grok_division/PROTOCOL.md`, *The split*.

The shuffles that keep every item of a set `Q` out of the first `k` of `N` slots (`keptOut`
of `SplitSlots`) are counted here:
`|keptOut| · N (N - 1) ⋯ (N - |Q| + 1) = N! · (N - k) (N - k - 1) ⋯ (N - k - |Q| + 1)`
(`card_keptOut_mul_descFactorial`). `SplitProbability` reads the count as a probability.

The count is by one item at a time (`card_keptOut_insert_mul`). Take a shuffle that keeps `q`
and every item of `Q` out of training, and a slot whose item is not in `Q`; exchange `q` with
the item of that slot and remember the slot that `q` left (`exchangeItem`). The result is a
shuffle that keeps `Q` out of training, with a held-out slot whose item is not in `Q`
(`exchangeItem_mem_pairsOut`), and the same step leads back (`exchangeItem_exchangeItem`,
`exchangeItem_mem_pairsIn`). The first kind has `N - |Q|` slots for each shuffle, the second
`N - k - |Q|`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {N : ℕ}

/-- Exchange the item `q` of a shuffle with the item of a slot, and return the slot that `q`
left. -/
def exchangeItem (q : Fin N) (p : Σ _ : Equiv.Perm (Fin N), Fin N) :
    Σ _ : Equiv.Perm (Fin N), Fin N :=
  ⟨p.1 * Equiv.swap (p.1.symm q) p.2, p.1.symm q⟩

/-- The same step leads back. -/
lemma exchangeItem_exchangeItem (q : Fin N) (p : Σ _ : Equiv.Perm (Fin N), Fin N) :
    exchangeItem q (exchangeItem q p) = p := by
  obtain ⟨π, t⟩ := p
  have h : (π * Equiv.swap (π.symm q) t).symm q = t := by
    rw [symm_mul_swap_apply, Equiv.swap_apply_left]
  simp only [exchangeItem, h]
  rw [Equiv.swap_comm t, mul_assoc, Equiv.swap_mul_self, mul_one]

/-- The shuffles that keep `q` and `Q` out of training, each with a slot whose item is not in
`Q`. -/
def pairsIn (k : ℕ) (q : Fin N) (Q : Finset (Fin N)) :
    Finset (Σ _ : Equiv.Perm (Fin N), Fin N) :=
  (keptOut k (insert q Q)).sigma fun π => Finset.univ.filter fun t => π t ∉ Q

/-- The shuffles that keep `Q` out of training, each with a held-out slot whose item is not in
`Q`. -/
def pairsOut (k : ℕ) (Q : Finset (Fin N)) : Finset (Σ _ : Equiv.Perm (Fin N), Fin N) :=
  (keptOut k Q).sigma fun π => Finset.univ.filter fun t : Fin N => k ≤ (t : ℕ) ∧ π t ∉ Q

/-- The slot of an item of `Q` is not moved by the exchange. -/
lemma symm_exchange_of_mem {q : Fin N} {Q : Finset (Fin N)} (hq : q ∉ Q)
    {π : Equiv.Perm (Fin N)} {t x : Fin N} (ht : π t ∉ Q) (hx : x ∈ Q) :
    (π * Equiv.swap (π.symm q) t).symm x = π.symm x := by
  rw [symm_mul_swap_apply]
  refine Equiv.swap_apply_of_ne_of_ne (fun h => hq ?_) (fun h => ht ?_)
  · rwa [← π.symm.injective h]
  · rwa [← h, Equiv.apply_symm_apply]

variable {k : ℕ} {q : Fin N} {Q : Finset (Fin N)}

lemma exchangeItem_mem_pairsOut (hq : q ∉ Q) {p : Σ _ : Equiv.Perm (Fin N), Fin N}
    (hp : p ∈ pairsIn k q Q) : exchangeItem q p ∈ pairsOut k Q := by
  obtain ⟨π, t⟩ := p
  obtain ⟨hπ, ht⟩ := Finset.mem_sigma.1 hp
  have ht' : π t ∉ Q := (Finset.mem_filter.1 ht).2
  refine Finset.mem_sigma.2 ⟨mem_keptOut.2 fun x hx => ?_, Finset.mem_filter.2
    ⟨Finset.mem_univ _, mem_keptOut.1 hπ q (Finset.mem_insert_self _ _), ?_⟩⟩
  · change k ≤ ((π * Equiv.swap (π.symm q) t).symm x : ℕ)
    rw [symm_exchange_of_mem hq ht' hx]
    exact mem_keptOut.1 hπ x (Finset.mem_insert_of_mem hx)
  · simpa [exchangeItem] using ht'

lemma exchangeItem_mem_pairsIn (hq : q ∉ Q) {p : Σ _ : Equiv.Perm (Fin N), Fin N}
    (hp : p ∈ pairsOut k Q) : exchangeItem q p ∈ pairsIn k q Q := by
  obtain ⟨π, t⟩ := p
  obtain ⟨hπ, ht⟩ := Finset.mem_sigma.1 hp
  obtain ⟨-, htk, ht'⟩ := Finset.mem_filter.1 ht
  refine Finset.mem_sigma.2 ⟨mem_keptOut.2 fun x hx => ?_, Finset.mem_filter.2
    ⟨Finset.mem_univ _, ?_⟩⟩
  · rcases Finset.mem_insert.1 hx with rfl | hxQ
    · change k ≤ ((π * Equiv.swap (π.symm x) t).symm x : ℕ)
      rw [symm_mul_swap_apply, Equiv.swap_apply_left]
      exact htk
    · change k ≤ ((π * Equiv.swap (π.symm q) t).symm x : ℕ)
      rw [symm_exchange_of_mem hq ht' hxQ]
      exact mem_keptOut.1 hπ x hxQ
  · simpa [exchangeItem] using ht'

/-- **One more item kept out of training**: of the slots that the items of `Q` leave, the
held-out ones are fewer by `k`. -/
theorem card_keptOut_insert_mul (k : ℕ) (hq : q ∉ Q) :
    (keptOut k (insert q Q)).card * (N - Q.card) = (keptOut k Q).card * (N - k - Q.card) := by
  have h := Finset.card_nbij' (s := pairsIn k q Q) (t := pairsOut k Q) (exchangeItem q)
    (exchangeItem q) (fun p hp => exchangeItem_mem_pairsOut hq hp)
    (fun p hp => exchangeItem_mem_pairsIn hq hp) (fun p _ => exchangeItem_exchangeItem q p)
    (fun p _ => exchangeItem_exchangeItem q p)
  rw [pairsIn, pairsOut, Finset.card_sigma, Finset.card_sigma,
    Finset.sum_congr rfl fun π _ => card_filter_apply_notMem π Q,
    Finset.sum_congr rfl fun π hπ => card_filter_heldOut_notMem hπ] at h
  simpa using h

/-- **The count of the shuffles that keep `Q` out of training.** -/
theorem card_keptOut_mul_descFactorial (k : ℕ) (Q : Finset (Fin N)) :
    (keptOut k Q).card * N.descFactorial Q.card =
      N.factorial * (N - k).descFactorial Q.card := by
  induction Q using Finset.induction_on with
  | empty =>
    have h : keptOut k (∅ : Finset (Fin N)) = Finset.univ := by
      ext π
      simp [mem_keptOut]
    rw [h, Finset.card_univ, Fintype.card_perm, Fintype.card_fin]
    simp
  | insert q Q hq ih =>
    rw [Finset.card_insert_of_notMem hq, Nat.descFactorial_succ, Nat.descFactorial_succ]
    calc (keptOut k (insert q Q)).card * ((N - Q.card) * N.descFactorial Q.card)
        = (keptOut k (insert q Q)).card * (N - Q.card) * N.descFactorial Q.card := by ring
      _ = (keptOut k Q).card * (N - k - Q.card) * N.descFactorial Q.card := by
          rw [card_keptOut_insert_mul k hq]
      _ = (N - k - Q.card) * ((keptOut k Q).card * N.descFactorial Q.card) := by ring
      _ = (N - k - Q.card) * (N.factorial * (N - k).descFactorial Q.card) := by rw [ih]
      _ = N.factorial * ((N - k - Q.card) * (N - k).descFactorial Q.card) := by ring

/-! The witnesses: four items and two training slots, the item `1` beside the set of the item
`0`. The shuffle that puts the items `0` and `1` in the slots `2` and `3`, with the slot `0`,
whose item is `2`; and the shuffle that puts the item `0` in the slot `2`, with the held-out
slot `3`, whose item is `3`. -/

example : (1 * Equiv.swap ((1 : Equiv.Perm (Fin 4)).symm 1) 3).symm 0 =
    (1 : Equiv.Perm (Fin 4)).symm 0 :=
  symm_exchange_of_mem (q := 1) (Q := {0}) (by decide) (by decide) (by decide)

example : exchangeItem 1 ⟨Equiv.swap (0 : Fin 4) 2 * Equiv.swap 1 3, 0⟩ ∈ pairsOut 2 {0} :=
  exchangeItem_mem_pairsOut (by decide) (Finset.mem_sigma.2
    ⟨mem_keptOut.2 (by decide), Finset.mem_filter.2 ⟨Finset.mem_univ _, by decide⟩⟩)

example : exchangeItem 1 ⟨Equiv.swap (0 : Fin 4) 2, 3⟩ ∈ pairsIn 2 1 {0} :=
  exchangeItem_mem_pairsIn (by decide) (Finset.mem_sigma.2
    ⟨mem_keptOut.2 (by decide), Finset.mem_filter.2 ⟨Finset.mem_univ _, by decide⟩⟩)

example : (keptOut 2 (insert (1 : Fin 4) {0})).card * (4 - ({0} : Finset (Fin 4)).card) =
    (keptOut 2 ({0} : Finset (Fin 4))).card * (4 - 2 - ({0} : Finset (Fin 4)).card) :=
  card_keptOut_insert_mul 2 (by decide)

end Transformer.Grokking.Protocol
