module

public import Mathlib

/-!
# A message about a dictionary answers at most what it tells apart

In no paper: the counting behind Zoology's index argument (arXiv:2312.04927v1,
Appendix `app:retnet-proof`, `cor: space-ar`), for values from any finite set
and for answers that are only partly right.

A dictionary is a map `D : K → V`. Alice sends a message `msg D` from a finite
set `M`, and Bob answers the key `q` with `ans (msg D) q`. If every key of
every dictionary is answered right, the message tells the dictionaries apart,
so `|V|^|K| ≤ |M|` (`card_pow_le_card_of_answers`); with `V = Bool` and
`M = Fin (2 ^ b)` this gives `Transformer.Zoology.exact_index_requires_bits`.

For answers that are partly right, weigh a dictionary `D` against a guess `g`:
`x` for each key where they agree, `y` for each other key. All dictionaries
weigh `(x + (|V| - 1) y)^|K|` together (`sum_prod_ite_eq_pow`), and one that
agrees with `g` on `a` keys weighs at least `x^a y^(|K| - a)` when `y ≤ x`.
Each message names one guess, so the dictionaries with `a` keys answered right
number at most `|M| (x + (|V| - 1) y)^|K| / (x^a y^(|K| - a))`
(`card_answered_mul_le`). With `x = |V|` and `y = 1`, `a` keys of a uniformly
random dictionary are answered right with chance at most `|M| 2^|K| / |V|^a`
(`card_answered_mul_pow_le`). With `V = Bool`, `x = a` and `y = |K| - a`, the
chance is at most `|M| 2^(-|K| (1 - H(a / |K|)))`, `H` the binary entropy: the
Chernoff bound. Zoology cites a randomized protocol bound instead; these count
dictionaries and need no probability.
-/

@[expose] public section

namespace Transformer.CausalRecall

open Finset

variable {K V M : Type*} [Fintype K] [Fintype V] [Fintype M]

/-- **A message that answers every key tells the dictionaries apart**: if Bob
answers every key `q` of every dictionary `D` right from Alice's message
`msg D`, then `|V|^|K| ≤ |M|`. -/
theorem card_pow_le_card_of_answers (msg : (K → V) → M) (ans : M → K → V)
    (h : ∀ D q, ans (msg D) q = D q) : Fintype.card V ^ Fintype.card K ≤ Fintype.card M := by
  classical
  have hinj : Function.Injective msg := fun D D' hD =>
    funext fun q => by rw [← h D q, ← h D' q, hD]
  simpa [Fintype.card_fun] using Fintype.card_le_of_injective msg hinj

/-- The hypothesis of `card_pow_le_card_of_answers` is satisfiable: a
dictionary of one key sends its value. -/
example : Fintype.card Bool ^ Fintype.card (Fin 1) ≤ Fintype.card Bool :=
  card_pow_le_card_of_answers (fun D : Fin 1 → Bool => D 0) (fun m _ => m)
    fun D q => by fin_cases q; rfl

variable [DecidableEq K] [DecidableEq V]

/-- **All dictionaries weigh `(x + (|V| - 1) y)^|K|`** when `D` weighs `x` for
each key where it agrees with `g` and `y` for each other key. -/
theorem sum_prod_ite_eq_pow (g : K → V) (x y : ℕ) :
    ∑ D : K → V, ∏ q, (if g q = D q then x else y) =
      (x + (Fintype.card V - 1) * y) ^ Fintype.card K := by
  have hq (q : K) : ∑ v : V, (if g q = v then x else y) = x + (Fintype.card V - 1) * y := by
    rw [← add_sum_erase _ _ (mem_univ (g q)), ite_eq_left rfl,
      sum_congr rfl fun v hv => ite_eq_right (ne_of_mem_erase hv).symm, sum_const,
      card_erase_of_mem (mem_univ _), card_univ, smul_eq_mul]
  rw [← Fintype.prod_sum (fun q v => if g q = v then x else y)]
  simp only [hq, prod_const, card_univ]

/-- **A message answers few dictionaries on many keys**: for weights `y ≤ x`,
the dictionaries `D` with `a` keys answered right from `msg D` number at most
`|M| (x + (|V| - 1) y)^|K| / (x^a y^(|K| - a))`. -/
theorem card_answered_mul_le (msg : (K → V) → M) (ans : M → K → V) {x y : ℕ} (hyx : y ≤ x)
    (a : ℕ) :
    #{D | a ≤ #{q | ans (msg D) q = D q}} * (x ^ a * y ^ (Fintype.card K - a)) ≤
      Fintype.card M * (x + (Fintype.card V - 1) * y) ^ Fintype.card K := by
  have hw (g D : K → V) (hD : a ≤ #{q | g q = D q}) :
      x ^ a * y ^ (Fintype.card K - a) ≤ ∏ q, (if g q = D q then x else y) := by
    rw [prod_ite, prod_const, prod_const]
    have hc := card_filter_add_card_filter_not (s := (univ : Finset K)) fun q => g q = D q
    rw [card_univ] at hc
    obtain ⟨j, hj⟩ := Nat.exists_eq_add_of_le hD
    rw [hj] at hc ⊢
    rw [show Fintype.card K - a = j + #{q | ¬g q = D q} by omega, pow_add, pow_add, mul_assoc]
    exact Nat.mul_le_mul_left _ (Nat.mul_le_mul_right _ (Nat.pow_le_pow_left hyx j))
  calc #{D | a ≤ #{q | ans (msg D) q = D q}} * (x ^ a * y ^ (Fintype.card K - a))
      ≤ (∑ m : M, #{D : K → V | a ≤ #{q | ans m q = D q}}) *
          (x ^ a * y ^ (Fintype.card K - a)) := by
        refine Nat.mul_le_mul_right _ (le_trans (card_le_card fun D hD => ?_) card_biUnion_le)
        exact mem_biUnion.mpr ⟨msg D, mem_univ _, mem_filter.mpr ⟨mem_univ _, (mem_filter.mp hD).2⟩⟩
    _ ≤ ∑ _m : M, (x + (Fintype.card V - 1) * y) ^ Fintype.card K := by
        rw [sum_mul]
        refine sum_le_sum fun m _ => ?_
        rw [← sum_prod_ite_eq_pow (ans m) x y, ← smul_eq_mul]
        exact (card_nsmul_le_sum _ _ _ fun D hD => hw (ans m) D (mem_filter.mp hD).2).trans
          (sum_le_sum_of_subset (subset_univ _))
    _ = _ := by rw [sum_const, card_univ, smul_eq_mul]

/-- The hypothesis of `card_answered_mul_le` is satisfiable: two keys, answered
from the first value, weighed `3` and `1`. -/
example : #{D : Fin 2 → Bool | 2 ≤ #{q | D 0 = D q}} * (3 ^ 2 * 1 ^ (Fintype.card (Fin 2) - 2)) ≤
    Fintype.card Bool * (3 + (Fintype.card Bool - 1) * 1) ^ Fintype.card (Fin 2) :=
  card_answered_mul_le (fun D : Fin 2 → Bool => D 0) (fun m _ => m) (by norm_num) 2

/-- **The chance form**: `a` keys of a uniformly random dictionary are answered
right with chance at most `|M| 2^|K| / |V|^a`. -/
theorem card_answered_mul_pow_le [Nonempty V] (msg : (K → V) → M) (ans : M → K → V) (a : ℕ) :
    #{D | a ≤ #{q | ans (msg D) q = D q}} * Fintype.card V ^ a ≤
      Fintype.card M * 2 ^ Fintype.card K * Fintype.card V ^ Fintype.card K := by
  have h := card_answered_mul_le msg ans (x := Fintype.card V) (y := 1) Fintype.card_pos a
  rw [one_pow, mul_one, mul_one] at h
  calc _ ≤ _ := h
    _ ≤ Fintype.card M * (2 * Fintype.card V) ^ Fintype.card K :=
        Nat.mul_le_mul_left _ (Nat.pow_le_pow_left (by omega) _)
    _ = _ := by rw [mul_pow, mul_assoc]

/-- The hypothesis of `card_answered_mul_pow_le` is satisfiable: values in the
nonempty `Bool`. -/
example : #{D : Fin 1 → Bool | 1 ≤ #{q | D 0 = D q}} * Fintype.card Bool ^ 1 ≤
    Fintype.card Bool * 2 ^ Fintype.card (Fin 1) * Fintype.card Bool ^ Fintype.card (Fin 1) :=
  card_answered_mul_pow_le (fun D : Fin 1 → Bool => D 0) (fun m _ => m) 1

end Transformer.CausalRecall
