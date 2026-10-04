/-
# Comparing a query with a key separates recall

In no paper.  At every depth `k` and width `d`, aligned recall over `c = 2^t`
tokens and `c` rows, at `p = 4t + 5` bits of which `s = t + 1` are fractional,
is recognized by the matcher of `CRASP.MatchingRecallAnswers`, one layer of
width `5` comparing a query with a key (`matcher_answers`), and by no
transformer of depth `k` and width `d` whose attention ignores the query
(`two_pow_le_of_queryFree`), once `t = 2 (k (2d + 1) + 7)`
(`exists_matcher_not_queryFree`): there `2^{c-1} ≤ (2^p c + 1)^{k (2d + 1)}`
fails, as `2^t` outgrows `(5t + 6) k (2d + 1) + 1` (`mul_lt_two_pow_sub_one`).
At the same precision, logarithmic in the vocabulary, attention comparing a
query with a key recalls in constant width, and attention ignoring the query
does not at any fixed width and depth.
-/

import Transformer.CRASP.MatchingRecallAnswers

namespace Transformer.CRASP

/-- `11 b ≤ 2^b` from `b = 7` on. -/
theorem eleven_mul_le_two_pow {b : ℕ} (hb : 7 ≤ b) : 11 * b ≤ 2 ^ b := by
  induction b, hb using Nat.le_induction with
  | base => norm_num
  | succ b hb ih => rw [pow_succ]; omega

/-- The hypothesis of `eleven_mul_le_two_pow` is satisfiable: `b = 7`. -/
example : 11 * 7 ≤ 2 ^ 7 := eleven_mul_le_two_pow le_rfl

/-- `2^t` outgrows `(5t + 6) K + 1` at `t = 2 (K + 7)`. -/
theorem mul_lt_two_pow_sub_one (K : ℕ) :
    (5 * (2 * (K + 7)) + 6) * K < 2 ^ (2 * (K + 7)) - 1 := by
  have h₁ := eleven_mul_le_two_pow (b := K + 7) (by omega)
  have h₂ : K + 2 ≤ 2 ^ (K + 7) := by
    have := Nat.lt_two_pow_self (n := K + 7)
    omega
  have h₃ := Nat.mul_le_mul h₁ h₂
  rw [two_mul (K + 7), pow_add]
  have h₄ : (5 * (K + 7 + (K + 7)) + 6) * K + 1 < 11 * (K + 7) * (K + 2) := by nlinarith
  omega

/-- **Comparing a query with a key separates recall**: at every depth `k` and
width `d` there is `t` such that, over `2^t` tokens and `2^t` rows at `4t + 5`
bits of which `t + 1` are fractional, the matcher of one layer and width `5`
recalls every value `v`, and no transformer of depth `k` and width `d` whose
attention ignores the query does, even on consistent instances alone. -/
theorem exists_matcher_not_queryFree (k d : ℕ) :
    ∃ t : ℕ, ∀ v : Fin (2 ^ t),
      (∀ x : Zoology.MQARInstance (2 ^ t - 1 + 1) (2 ^ t),
        (matcher v : RTfr _ (4 * t + 5) (t + 1) 5 1).Accepts (bos (word x)) ↔
          Zoology.PriorAnswer x (Fin.last _) v) ∧
      ∀ T : RTfr (Option (Fin (2 ^ t) × Fin (2 ^ t) × Fin (2 ^ t))) (4 * t + 5) (t + 1) d k,
        T.QueryFree → ¬ ∀ x : Zoology.MQARInstance (2 ^ t - 1 + 1) (2 ^ t), Zoology.Consistent x →
          (T.Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last _) v) := by
  obtain ⟨t, ht⟩ : ∃ t, t = 2 * (k * (2 * d + 1) + 7) := ⟨_, rfl⟩
  have h₁ : 1 ≤ 2 ^ t := Nat.one_le_two_pow
  have hp : 2 * (t + 1 + 1) * (2 ^ t) ^ 2 * 2 ^ (t + 1) < 2 ^ (4 * t + 5 - 1) := by
    have e₁ : 2 * (t + 1 + 1) * (2 ^ t) ^ 2 * 2 ^ (t + 1) = (t + 2) * 2 ^ (3 * t + 2) := by ring
    have e₂ : 2 ^ (4 * t + 5 - 1) = 2 ^ (t + 2) * 2 ^ (3 * t + 2) := by
      rw [← pow_add]
      congr 1
      omega
    rw [e₁, e₂]
    exact Nat.mul_lt_mul_of_pos_right Nat.lt_two_pow_self (by positivity)
  have hs : 2 ^ t - 1 + 2 ≤ 2 ^ (t + 1) := by
    rw [pow_succ]
    omega
  refine ⟨t, fun v => ⟨fun x => matcher_answers hp hs v x, fun T hQ hT => ?_⟩⟩
  have h := two_pow_le_of_queryFree T hQ v (by omega) hT
  have hb : 2 ^ (4 * t + 5) * (2 ^ t - 1 + 1) + 1 ≤ 2 ^ (5 * t + 6) := by
    rw [Nat.sub_add_cancel h₁, ← pow_add, show 5 * t + 6 = 4 * t + 5 + t + 1 by omega, pow_succ]
    have := Nat.one_le_two_pow (n := 4 * t + 5 + t)
    omega
  have hK := Nat.pow_le_pow_left hb (k * (2 * d + 1))
  rw [← pow_mul] at hK
  have hlt : 2 ^ ((5 * t + 6) * (k * (2 * d + 1))) < 2 ^ (2 ^ t - 1) :=
    Nat.pow_lt_pow_right (by norm_num) (ht ▸ mul_lt_two_pow_sub_one (k * (2 * d + 1)))
  omega

end Transformer.CRASP
