module

public import Transformer.Grokking.Protocol.SplitCount

/-!
# The probability that a split keeps a set of equations out of training

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*: "we
chose a fraction of all available equations at random and declared them to be the training
set"; `experiments/grok_division/PROTOCOL.md`, *The split*. The uniform law on the shuffles
is `uniformPerm` of `Transformer.Grokking.Protocol.Target`.

`SplitCount` counts the shuffles that keep a set `Q` of items out of the first `k` slots. For
a uniformly random shuffle of `N` items the count is a probability,
`(N - k) (N - k - 1) ⋯ (N - k - |Q| + 1) / (N (N - 1) ⋯ (N - |Q| + 1))`
(`uniformPerm_keptOut_eq`): each item in turn takes one of the slots that the earlier ones
left, and of those the held-out ones are fewer by `k`. When the training set is at least half,
`N ≤ 2 k`, each factor is at most one half (`descFactorial_mul_two_pow_le`), so the probability
is at most `2 ^ (-|Q|)` (`uniformPerm_keptOut_le`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

variable {N : ℕ}

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

/-- A comparison of two ratios of natural numbers. -/
lemma natCast_div_le_div {a b c d : ℕ} (hb : b ≠ 0) (hd : d ≠ 0) (h : a * d ≤ c * b) :
    (a : ENNReal) / b ≤ (c : ENNReal) / d := by
  rw [ENNReal.div_le_iff (Nat.cast_ne_zero.2 hb) (ENNReal.natCast_ne_top _),
    ← ENNReal.mul_div_right_comm,
    ENNReal.le_div_iff_mul_le (Or.inl (Nat.cast_ne_zero.2 hd)) (Or.inl (ENNReal.natCast_ne_top _))]
  exact_mod_cast h

/-- **The probability that a uniformly random split keeps a set `Q` of items out of
training.** -/
theorem uniformPerm_keptOut_eq (k : ℕ) (Q : Finset (Fin N)) :
    uniformPerm N {π | ∀ x ∈ Q, k ≤ (π.symm x : ℕ)} =
      ((N - k).descFactorial Q.card : ENNReal) / (N.descFactorial Q.card : ENNReal) := by
  have hset : {π : Equiv.Perm (Fin N) | ∀ x ∈ Q, k ≤ (π.symm x : ℕ)} =
      (keptOut k Q : Set (Equiv.Perm (Fin N))) := by
    ext π
    simp [mem_keptOut]
  have hQ : N.descFactorial Q.card ≠ 0 := by
    rw [Ne, Nat.descFactorial_eq_zero_iff_lt, not_lt]
    simpa using Finset.card_le_univ Q
  rw [hset, uniformPerm_coe, ENNReal.div_eq_div_iff (Nat.cast_ne_zero.2 hQ)
    (ENNReal.natCast_ne_top _) (Nat.cast_ne_zero.2 (Nat.factorial_ne_zero N))
    (ENNReal.natCast_ne_top _)]
  have h := card_keptOut_mul_descFactorial k Q
  rw [mul_comm] at h
  exact_mod_cast h

/-- When the training set is at least half, every factor of the count is at most one half. -/
lemma descFactorial_mul_two_pow_le {k : ℕ} (hN : N ≤ 2 * k) (q : ℕ) :
    (N - k).descFactorial q * 2 ^ q ≤ N.descFactorial q := by
  induction q with
  | zero => simp
  | succ q ih =>
    rw [Nat.descFactorial_succ, Nat.descFactorial_succ, pow_succ]
    calc (N - k - q) * (N - k).descFactorial q * (2 ^ q * 2)
        = 2 * (N - k - q) * ((N - k).descFactorial q * 2 ^ q) := by ring
      _ ≤ (N - q) * N.descFactorial q := Nat.mul_le_mul (by omega) ih

/-- **A uniformly random split keeps a set `Q` of items out of training with a probability of
at most `2 ^ (-|Q|)`**, when the training set is at least half. -/
theorem uniformPerm_keptOut_le (k : ℕ) (hN : N ≤ 2 * k) (Q : Finset (Fin N)) :
    uniformPerm N {π | ∀ x ∈ Q, k ≤ (π.symm x : ℕ)} ≤ (2 ^ Q.card)⁻¹ := by
  have h := natCast_div_le_div (a := (N - k).descFactorial Q.card)
    (b := N.descFactorial Q.card) (c := 1) (d := 2 ^ Q.card)
    (by
      rw [Ne, Nat.descFactorial_eq_zero_iff_lt, not_lt]
      simpa using Finset.card_le_univ Q)
    (pow_ne_zero _ (by norm_num))
    (by simpa using descFactorial_mul_two_pow_le hN Q.card)
  rw [uniformPerm_keptOut_eq]
  simpa using h

/-! The witnesses: four items and two training slots. The item `0` is kept out of training
with the probability `2 / 4`, and the items `0` and `1` together with `2 · 1 / (4 · 3)`, which
is at most a quarter; `2 / 4` is at most `1 / 2`; a set of the two shuffles of two items and
its complement. -/

example : uniformPerm 4 {π | ∀ x ∈ ({0, 1} : Finset (Fin 4)), 2 ≤ (π.symm x : ℕ)} =
    ((4 - 2).descFactorial 2 : ENNReal) / ((4 : ℕ).descFactorial 2 : ENNReal) :=
  uniformPerm_keptOut_eq 2 {0, 1}

example : uniformPerm 4 {π | ∀ x ∈ ({0, 1} : Finset (Fin 4)), 2 ≤ (π.symm x : ℕ)} ≤
    (2 ^ 2)⁻¹ :=
  uniformPerm_keptOut_le 2 (by norm_num) {0, 1}

example : ((2 : ℕ) : ENNReal) / (4 : ℕ) ≤ ((1 : ℕ) : ENNReal) / (2 : ℕ) :=
  natCast_div_le_div (by norm_num) (by norm_num) (by norm_num)

example : (4 - 2).descFactorial 1 * 2 ^ 1 ≤ (4 : ℕ).descFactorial 1 :=
  descFactorial_mul_two_pow_le (by norm_num) 1

example : uniformPerm 2 {1} + uniformPerm 2 ({1} : Set (Equiv.Perm (Fin 2)))ᶜ = 1 :=
  uniformPerm_add_compl {1}

end Transformer.Grokking.Protocol
