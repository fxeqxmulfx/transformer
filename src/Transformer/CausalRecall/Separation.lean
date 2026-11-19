module

public import Transformer.CausalRecall.MQAR
public import Transformer.CRASP.RecallSeparation

/-!
# Comparing a query with a key separates recall from summaries

In no paper. `Transformer.CRASP.exists_matcher_not_queryFree` separates the
matcher of `Transformer.CRASP.MatchingRecallAnswers`, one layer of width `5`
comparing a query with a key, from transformers whose attention ignores the
query. Here the matcher is separated from every causal model of
`Transformer.CausalRecall.Summary`, whatever its layers compute, and from
recall that is right on only most queries.

Take depth `L`, summaries of `w` numbers at the matcher's `2t + 3` bits, so of
at most `2^{w (2t + 3)}` values each, and `t = 2 (w L + 7) + 3`. Over `2^t + 1`
tokens and `2^t + 1` rows the matcher decides every instance
(`matcher_answers`). A model of the class decides the index instances of a set
`S` of `2^t` keys right on three quarters of the queries for fewer than half
of the sets, and so does not decide every consistent instance
(`exists_matcher_not_summary`). By `Model.card_recall_mul_le` with weights `3`
and `1`, those sets number at most `|R|^L 4^{2^t} / 27^{2^t / 4}`; since
`27² > 2 · 16²`, this is below `2^{2^t} / 2` once
`2^{1 + w L (2t + 3)} < 2^{2^t / 8}`, which `mul_lt_two_pow_sub_one` gives.

At the same logarithmic precision and any fixed depth and summary width,
comparing a query with a key recalls exactly, while a summary formed before
the query recalls three quarters of the keys for fewer than half of the
dictionaries.
-/

@[expose] public section

namespace Transformer.CausalRecall

open Finset Zoology CRASP

/-- **Comparing a query with a key separates recall from summaries, even in
part**: at every depth `L` and summary width `w` there is `t` such that, over
`2^t + 1` tokens and `2^t + 1` rows at `2t + 3` bits, none of them fractional,
the matcher of one layer and width `5` decides whether `v` answers on every
instance, while every model of `Transformer.CausalRecall.Summary` with at most
`L` layers whose summaries take at most `2^{w (2t + 3)}` values decides the
index instances of fewer than half of the sets `S` of `2^t` keys right on
three quarters of the queries, and so does not decide every consistent
instance. -/
theorem exists_matcher_not_summary (L w : ℕ) :
    ∃ t : ℕ, ∀ v : Fin (2 ^ t + 1),
      (∀ x : MQARInstance (2 ^ t + 1) (2 ^ t + 1),
        (matcher v : RTfr _ (2 * t + 3) 0 5 1).Accepts (bos (word x)) ↔
          PriorAnswer x (Fin.last _) v) ∧
      ∀ {Z R : Type*} [Fintype R]
        (T : Model (Fin (2 ^ t + 1) × Fin (2 ^ t + 1) × Fin (2 ^ t + 1)) Z R Bool),
        T.layers.length ≤ L → Fintype.card R ≤ 2 ^ (w * (2 * t + 3)) →
        2 * #{S : Fin (2 ^ t) → Bool | 3 * 2 ^ t ≤ 4 * #{q |
          T.Accepts (word (indexInstance (Nat.two_pow_pos t) v S (2 ^ t) q)) ↔
            PriorAnswer (indexInstance (Nat.two_pow_pos t) v S (2 ^ t) q) (Fin.last _) v}} <
          2 ^ 2 ^ t ∧
        ¬ ∀ x : MQARInstance (2 ^ t + 1) (2 ^ t + 1), Consistent x →
          (T.Accepts (word x) ↔ PriorAnswer x (Fin.last _) v) := by
  obtain ⟨K, hK⟩ : ∃ K, K = w * L := ⟨_, rfl⟩
  obtain ⟨i, hi⟩ : ∃ i, i = 2 ^ (2 * (K + 7)) := ⟨_, rfl⟩
  obtain ⟨t, ht⟩ : ∃ t, t = 2 * (K + 7) + 3 := ⟨_, rfl⟩
  have hN : 2 ^ t = 4 * (2 * i) := by rw [ht, pow_add, ← hi]; ring
  have hi1 : 1 ≤ i := hi ▸ Nat.one_le_two_pow
  refine ⟨t, fun v => ⟨fun x => matcher_answers ?_ v x, ?_⟩⟩
  · have h2 : 2 ^ (2 * t + 3 - 1) = 2 ^ t * 2 ^ t * 4 := by
      rw [show 2 * t + 3 - 1 = t + t + 2 by omega, pow_add, pow_add]; norm_num
    rw [h2, hN]
    nlinarith
  intro Z R _ T hL hR
  have hm := Nat.two_pow_pos t
  -- the weighted count, at three quarters of the `2^t = 8 i` keys
  have h := T.card_recall_mul_le hm v (n := 2 ^ t) le_rfl (x := 3) (y := 1) (by norm_num)
    (3 * (2 * i))
  have e₁ : (3 + 1) ^ 2 ^ t = 256 ^ (2 * i) := by rw [hN, pow_mul]; norm_num
  have e₂ : 3 ^ (3 * (2 * i)) = 27 ^ (2 * i) := by rw [pow_mul]; norm_num
  rw [one_pow, mul_one, e₁, e₂] at h
  -- the summaries take at most `2^{w L (2t + 3)}` values together
  have hRL : Fintype.card R ^ T.layers.length ≤ 2 ^ (K * (2 * t + 3)) :=
    calc Fintype.card R ^ T.layers.length
        ≤ (2 ^ (w * (2 * t + 3))) ^ T.layers.length := Nat.pow_le_pow_left hR _
      _ ≤ (2 ^ (w * (2 * t + 3))) ^ L := Nat.pow_le_pow_right (Nat.two_pow_pos _) hL
      _ = 2 ^ (K * (2 * t + 3)) := by rw [← pow_mul, hK, mul_right_comm]
  have hB : 1 + K * (2 * t + 3) < i := by
    have h₁ := mul_lt_two_pow_sub_one K
    rw [← hi] at h₁
    have h₂ : K * (2 * t + 3) ≤ (3 * (2 * (K + 7)) + 4) * K := by
      rw [Nat.mul_comm K]; exact Nat.mul_le_mul_right K (by omega)
    omega
  have hpow : 2 ^ i * 256 ^ (2 * i) ≤ 16 ^ (2 * i) * 27 ^ (2 * i) := by
    rw [pow_mul, pow_mul, pow_mul, ← mul_pow, ← mul_pow]
    exact Nat.pow_le_pow_left (by norm_num) i
  have hiff (r : ℕ) : 3 * 2 ^ t ≤ 4 * r ↔ 3 * (2 * i) ≤ r := by omega
  have happrox : 2 * #{S : Fin (2 ^ t) → Bool | 3 * 2 ^ t ≤ 4 * #{q |
      T.Accepts (word (indexInstance hm v S (2 ^ t) q)) ↔
        PriorAnswer (indexInstance hm v S (2 ^ t) q) (Fin.last _) v}} < 2 ^ 2 ^ t := by
    simp only [hiff]
    rw [show 2 ^ 2 ^ t = 16 ^ (2 * i) by rw [hN, pow_mul]; norm_num]
    refine Nat.lt_of_mul_lt_mul_right (a := 27 ^ (2 * i)) ?_
    calc _ ≤ 2 * (Fintype.card R ^ T.layers.length * 256 ^ (2 * i)) := by
          rw [mul_assoc]; exact Nat.mul_le_mul_left 2 h
      _ ≤ 2 * (2 ^ (K * (2 * t + 3)) * 256 ^ (2 * i)) := by gcongr
      _ = 2 ^ (1 + K * (2 * t + 3)) * 256 ^ (2 * i) := by ring
      _ < 2 ^ i * 256 ^ (2 * i) :=
          Nat.mul_lt_mul_of_pos_right (Nat.pow_lt_pow_right (by norm_num) hB) (by positivity)
      _ ≤ 16 ^ (2 * i) * 27 ^ (2 * i) := hpow
  refine ⟨happrox, fun hT => ?_⟩
  -- deciding every consistent instance decides every query of every set
  have hq (S : Fin (2 ^ t) → Bool) : 3 * 2 ^ t ≤ 4 * #{q |
      T.Accepts (word (indexInstance hm v S (2 ^ t) q)) ↔
        PriorAnswer (indexInstance hm v S (2 ^ t) q) (Fin.last _) v} := by
    rw [filter_true_of_mem fun q _ => hT _ (consistent_indexInstance hm v S _ q), card_univ,
      Fintype.card_fin]
    omega
  rw [filter_true_of_mem fun S _ => hq S, card_univ, Fintype.card_fun, Fintype.card_bool,
    Fintype.card_fin] at happrox
  omega

end Transformer.CausalRecall
