module

public import Transformer.Grokking.Protocol.Target

/-!
# The probability that a split keeps a set of equations out of training

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; `experiments/grok_division/PROTOCOL.md`, *The split*. The uniform law on the shuffles
is `uniformPerm` of `Transformer.Grokking.Protocol.Target`.

A shuffle puts `N` items in `N` slots, and the first `k` slots are the training set. Let the
training set be at least half: `N ≤ 2 k`. Then the shuffles that keep every item of a set `Q`
out of training (`keptOut`) are at most `N! / 2 ^ |Q|` (`card_keptOut_mul_le`), so a uniformly
random split keeps `Q` out of training with a probability of at most `2 ^ (-|Q|)`
(`uniformPerm_keptOut_le`).

The count is by one item at a time. Take a shuffle that keeps `q` and every item of `Q` out
of training, and exchange `q` with the item `k` slots before it (`flipItem`). The slot of `q`
is now a training slot, and no item of `Q` has moved, for the item that left that slot was in
training. So the map sends these shuffles, one to one, to shuffles that keep `Q` out of
training and not `q` (`two_mul_card_keptOut_insert_le`). The exact probability is a ratio of
binomial coefficients, which is not needed here.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

variable {N : ℕ}

/-- The slot `k` slots before a slot: the mirror of a held-out slot among the training slots. -/
def mirrorSlot (k : ℕ) (i : Fin N) : Fin N := ⟨i - k, lt_of_le_of_lt (Nat.sub_le _ _) i.isLt⟩

/-- Exchange the item `q` of a shuffle with the item `k` slots before it. -/
def flipItem (k : ℕ) (q : Fin N) (π : Equiv.Perm (Fin N)) : Equiv.Perm (Fin N) :=
  π * Equiv.swap (π.symm q) (mirrorSlot k (π.symm q))

/-- After the exchange the item `q` is in the mirror slot. -/
lemma flipItem_symm_self (k : ℕ) (q : Fin N) (π : Equiv.Perm (Fin N)) :
    (flipItem k q π).symm q = mirrorSlot k (π.symm q) := by
  rw [Equiv.symm_apply_eq]
  simp [flipItem]

/-- The exchange moves no item that is in neither of the two slots. -/
lemma flipItem_symm_of_ne (k : ℕ) (q : Fin N) (π : Equiv.Perm (Fin N)) {x : Fin N}
    (h1 : x ≠ q) (h2 : π.symm x ≠ mirrorSlot k (π.symm q)) :
    (flipItem k q π).symm x = π.symm x := by
  rw [Equiv.symm_apply_eq]
  have h3 : π.symm x ≠ π.symm q := fun h => h1 (π.symm.injective h)
  simp [flipItem, Equiv.swap_apply_of_ne_of_ne h3 h2]

/-- **The shuffles that keep every item of `Q` out of the first `k` slots.** -/
def keptOut (k : ℕ) (Q : Finset (Fin N)) : Finset (Equiv.Perm (Fin N)) :=
  Finset.univ.filter fun π => ∀ x ∈ Q, k ≤ (π.symm x : ℕ)

lemma mem_keptOut {k : ℕ} {Q : Finset (Fin N)} {π : Equiv.Perm (Fin N)} :
    π ∈ keptOut k Q ↔ ∀ x ∈ Q, k ≤ (π.symm x : ℕ) := by
  simp [keptOut]

/-- **One more item kept out of training halves the shuffles**, when the training set is at
least half. -/
theorem two_mul_card_keptOut_insert_le (k : ℕ) (hN : N ≤ 2 * k) {q : Fin N}
    {Q : Finset (Fin N)} (hq : q ∉ Q) :
    2 * (keptOut k (insert q Q)).card ≤ (keptOut k Q).card := by
  have hsub : keptOut k (insert q Q) ⊆ keptOut k Q := fun π hπ =>
    mem_keptOut.2 fun x hx => mem_keptOut.1 hπ x (Finset.mem_insert_of_mem hx)
  have hlt : ∀ π ∈ keptOut k (insert q Q), (mirrorSlot k (π.symm q) : ℕ) < k := fun π hπ => by
    have h1 := mem_keptOut.1 hπ q (Finset.mem_insert_self _ _)
    have h2 := (π.symm q).isLt
    simp only [mirrorSlot]
    omega
  have hmaps : Set.MapsTo (flipItem k q) (keptOut k (insert q Q) : Set (Equiv.Perm (Fin N)))
      (keptOut k Q \ keptOut k (insert q Q) : Finset (Equiv.Perm (Fin N))) := fun π hπ => by
    have hm := hlt π hπ
    refine Finset.mem_coe.2 (Finset.mem_sdiff.2 ⟨mem_keptOut.2 fun x hx => ?_, fun hmem => ?_⟩)
    · have hxk := mem_keptOut.1 hπ x (Finset.mem_insert_of_mem hx)
      rw [flipItem_symm_of_ne k q π (fun h => hq (h ▸ hx)) fun h => by rw [h] at hxk; omega]
      exact hxk
    · have hqk := mem_keptOut.1 hmem q (Finset.mem_insert_self _ _)
      rw [flipItem_symm_self] at hqk
      omega
  have hinj : Set.InjOn (flipItem k q) (keptOut k (insert q Q) : Set (Equiv.Perm (Fin N))) :=
    fun π₁ h₁ π₂ h₂ h => by
      have ha₁ := mem_keptOut.1 h₁ q (Finset.mem_insert_self _ _)
      have ha₂ := mem_keptOut.1 h₂ q (Finset.mem_insert_self _ _)
      have hm : mirrorSlot k (π₁.symm q) = mirrorSlot k (π₂.symm q) := by
        rw [← flipItem_symm_self, ← flipItem_symm_self, h]
      have hval := congrArg Fin.val hm
      simp only [mirrorSlot] at hval
      have ha : π₁.symm q = π₂.symm q := Fin.ext (by omega)
      simp only [flipItem, ha] at h
      exact mul_right_cancel h
  have h := Finset.card_le_card_of_injOn (flipItem k q) hmaps hinj
  rw [Finset.card_sdiff_of_subset hsub] at h
  have hle := Finset.card_le_card hsub
  omega

/-- **The shuffles that keep `Q` out of training are at most `N! / 2 ^ |Q|`**, when the
training set is at least half. -/
theorem card_keptOut_mul_le (k : ℕ) (hN : N ≤ 2 * k) (Q : Finset (Fin N)) :
    (keptOut k Q).card * 2 ^ Q.card ≤ N.factorial := by
  induction Q using Finset.induction_on with
  | empty =>
    have h := Finset.card_le_univ (keptOut k (∅ : Finset (Fin N)))
    rw [Fintype.card_perm, Fintype.card_fin] at h
    simpa using h
  | insert q Q hq ih =>
    rw [Finset.card_insert_of_notMem hq, pow_succ]
    calc (keptOut k (insert q Q)).card * (2 ^ Q.card * 2)
        = 2 * (keptOut k (insert q Q)).card * 2 ^ Q.card := by ring
      _ ≤ (keptOut k Q).card * 2 ^ Q.card :=
          Nat.mul_le_mul_right _ (two_mul_card_keptOut_insert_le k hN hq)
      _ ≤ N.factorial := ih

/-- The probability of a finite set of shuffles is its share of the shuffles. -/
lemma uniformPerm_coe (F : Finset (Equiv.Perm (Fin N))) :
    uniformPerm N (F : Set (Equiv.Perm (Fin N))) =
      (F.card : ENNReal) / (N.factorial : ENNReal) := by
  classical
  rw [uniformPerm, Measure.uniformOfFinset_apply, Finset.card_univ, Fintype.card_perm,
    Fintype.card_fin]
  refine congrArg (fun n : ℕ => (n : ENNReal) / (N.factorial : ENNReal)) (congrArg Finset.card ?_)
  ext π
  simp

/-- A set of shuffles and its complement have the total probability `1`. -/
lemma uniformPerm_add_compl (A : Set (Equiv.Perm (Fin N))) :
    uniformPerm N A + uniformPerm N Aᶜ = 1 := by
  rw [measure_add_measure_compl MeasurableSpace.measurableSet_top, measure_univ]

/-- **A uniformly random split keeps a set `Q` of items out of training with a probability of
at most `2 ^ (-|Q|)`**, when the training set is at least half. -/
theorem uniformPerm_keptOut_le (k : ℕ) (hN : N ≤ 2 * k) (Q : Finset (Fin N)) :
    uniformPerm N {π | ∀ x ∈ Q, k ≤ (π.symm x : ℕ)} ≤ (2 ^ Q.card)⁻¹ := by
  have hset : {π : Equiv.Perm (Fin N) | ∀ x ∈ Q, k ≤ (π.symm x : ℕ)} =
      (keptOut k Q : Set (Equiv.Perm (Fin N))) := by
    ext π
    simp [mem_keptOut]
  rw [hset, uniformPerm_coe]
  have h : ((keptOut k Q).card : ENNReal) * 2 ^ Q.card ≤ (N.factorial : ENNReal) := by
    exact_mod_cast card_keptOut_mul_le k hN Q
  have h2 : (2 : ENNReal) ^ Q.card ≠ 0 := pow_ne_zero _ (by norm_num)
  have h3 : (2 : ENNReal) ^ Q.card ≠ ⊤ := ENNReal.pow_ne_top (by norm_num)
  rw [ENNReal.div_le_iff (Nat.cast_ne_zero.2 (Nat.factorial_ne_zero N)) (ENNReal.natCast_ne_top _)]
  calc ((keptOut k Q).card : ENNReal)
      = (keptOut k Q).card * 2 ^ Q.card * (2 ^ Q.card)⁻¹ := by
        rw [mul_assoc, ENNReal.mul_inv_cancel h2 h3, mul_one]
    _ ≤ N.factorial * (2 ^ Q.card)⁻¹ := by gcongr
    _ = (2 ^ Q.card)⁻¹ * N.factorial := mul_comm _ _

/-! The witnesses: four items and two training slots. In the shuffle that exchanges the items
`0` and `2`, the item `0` is in the slot `2`, and its exchange with the item two slots before
does not move the item `3`. The item `0` is kept out of training by at most half of the 24
shuffles, and the items `0` and `1` together by at most a quarter. -/

example : (flipItem 2 0 (Equiv.swap (0 : Fin 4) 2)).symm 3 = (Equiv.swap (0 : Fin 4) 2).symm 3 :=
  flipItem_symm_of_ne 2 0 _ (by decide) (by decide)

example : 2 * (keptOut 2 (insert (0 : Fin 4) {1})).card ≤ (keptOut 2 ({1} : Finset (Fin 4))).card :=
  two_mul_card_keptOut_insert_le 2 (by norm_num) (by decide)

example : (keptOut 2 ({0, 1} : Finset (Fin 4))).card * 2 ^ 2 ≤ 24 :=
  card_keptOut_mul_le 2 (by norm_num) {0, 1}

example : uniformPerm 4 {π | ∀ x ∈ ({0} : Finset (Fin 4)), 2 ≤ (π.symm x : ℕ)} ≤ (2 ^ 1)⁻¹ :=
  uniformPerm_keptOut_le 2 (by norm_num) {0}

example : uniformPerm 2 {1} + uniformPerm 2 ({1} : Set (Equiv.Perm (Fin 2)))ᶜ = 1 :=
  uniformPerm_add_compl {1}

end Transformer.Grokking.Protocol
