/-
# One layer of width five recalls over every vocabulary, at every length

In no paper: the attention comparing a query with a key of
`CRASP.MatchingRecall` recalls, where attention ignoring the query needs a
width growing with the vocabulary (`CRASP.QueryFreeRecall`).  Over `c` tokens
and `n + 1` rows, with `2 (s + 1) c² 2^s < 2^{p-1}`, the matcher's last
activation is the last token followed by `(m_v - 1) / (1 + m)` and
`(m_v - 2) / (1 + m)` rounded (`actAt_matcher`), for the `m` rows whose key is
the last query and the `m_v` of them with value `v`.  Rounding down keeps the
sign (`m_round_nonneg_iff`), so the matcher accepts exactly when
`m_v ≥ 1 + e`, for the last row's own share `e ∈ {0, 1}`, that is, when an
earlier row binds the last query to `v` (`matcher_answers`), consistent
dictionary or not, whatever `n`.  Aligned recall over `c` tokens thus takes one
layer of width `5` at `p = O(log c)` bits at every length, with no fractional
bit (`s = 0`) needed.
-/

import Transformer.CRASP.MatchingRecallSums

namespace Transformer.CRASP

open Finset

variable {c p s : ℕ}

/-- **The last activation**: the last row's token, then `(m_v - 1) / (1 + m)`
and `(m_v - 2) / (1 + m)` rounded, where `m` rows hold the last query as key and
`m_v` of them the value `v`. -/
theorem actAt_matcher (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) (v : Fin c) {n : ℕ}
    (x : Zoology.MQARInstance (n + 1) c) :
    (matcher v : RTfr _ p s 5 1).actAt (word x) 1 (n + 1) =
      ![embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n))) 0,
        embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n))) 1,
        embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n))) 2,
        Fx.round p s ((#{j | x.key j = x.query (Fin.last n) ∧ x.value j = v} - 1) /
          (1 + #{j | x.key j = x.query (Fin.last n)})),
        Fx.round p s ((#{j | x.key j = x.query (Fin.last n) ∧ x.value j = v} - 2) /
          (1 + #{j | x.key j = x.query (Fin.last n)}))] := by
  classical
  have h2 := two_pow_lt_of_le hp (Fin.pos v)
  have hi : n + 1 < (bos (word x)).length := by simp [word]
  have hmask := masked_eq_univ (⟨n + 1, hi⟩ : Fin (bos (word x)).length) (by simp [word])
  have hact : ∀ j, (matcher v : RTfr _ p s 5 1).act (bos (word x)) 0 j =
      embed (bos (word x))[j.1] := fun j => rfl
  have hlast : (bos (word x))[n + 1]'hi =
      some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n)) := by
    simp only [bos, List.getElem_cons_succ, List.getElem_map, word, List.getElem_ofFn]
    rfl
  have hD : ∑ j ∈ RTfr.masked (⟨n + 1, hi⟩ : Fin (bos (word x)).length),
      (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0
        ((matcher v : RTfr _ p s 5 1).act (bos (word x)) 0 ⟨n + 1, hi⟩)
        ((matcher v : RTfr _ p s 5 1).act (bos (word x)) 0 j)))).val =
      1 + #{j | x.key j = x.query (Fin.last n)} := by
    have hs := sum_bos_word x fun t => (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0
      (embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n)))) (embed t)))).val
    rw [hmask]
    simp only [hact, hlast]
    rw [hs, val_weight_none h2]
    simp only [val_weight_some hp, Finset.sum_boole]
  have hM : ∀ k : Fin 5, ∑ j ∈ RTfr.masked (⟨n + 1, hi⟩ : Fin (bos (word x)).length),
      (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0
        ((matcher v : RTfr _ p s 5 1).act (bos (word x)) 0 ⟨n + 1, hi⟩)
        ((matcher v : RTfr _ p s 5 1).act (bos (word x)) 0 j)) *
        ((matcher v : RTfr _ p s 5 1).WV 0 ((matcher v : RTfr _ p s 5 1).act (bos (word x)) 0 j)
          k).val)).val =
      (if k = 3 then -1 else if k = 4 then -2 else 0) +
        #{j | (k = 3 ∨ k = 4) ∧ x.key j = x.query (Fin.last n) ∧ x.value j = v} := by
    intro k
    have hs := sum_bos_word x fun t => (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0
      (embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n)))) (embed t)) *
        ((matcher v : RTfr _ p s 5 1).WV 0 (embed t) k).val)).val
    rw [hmask]
    simp only [hact, hlast]
    rw [hs, val_weighted_none hp]
    simp only [val_weighted_some hp, Finset.sum_boole]
  have hD₀ : (1 : ℝ) + #{j | x.key j = x.query (Fin.last n)} ≠ 0 := by positivity
  rw [RTfr.actAt, dite_eq_left hi]
  funext k
  rw [show (matcher v : RTfr _ p s 5 1).act (bos (word x)) 1 =
    (matcher v : RTfr _ p s 5 1).layer 0 ((matcher v : RTfr _ p s 5 1).act (bos (word x)) 0)
      from rfl, layer_matcher v _ _ k (hD ▸ hD₀), hD, hM k]
  simp only [hact, hlast]
  fin_cases k <;> simp [embed, neg_add_eq_sub]

/-- The hypothesis of `actAt_matcher` is satisfiable: one token, `s = 1`,
`p = 5`. -/
example : 2 * (1 + 1) * 1 ^ 2 * 2 ^ 1 < 2 ^ (5 - 1) := by norm_num

/-- **Rounding down keeps the sign**: the mantissa of `round(x)` is not
negative exactly when `x` is not (Appendix B.1, `def:fixed_precision`). -/
theorem m_round_nonneg_iff (x : ℝ) : 0 ≤ (Fx.round p s x).m ↔ 0 ≤ x := by
  rw [Fx.m_round, ← not_lt, Fx.clamp_neg_iff, not_lt, Int.floor_nonneg,
    mul_nonneg_iff_of_pos_right (by positivity)]

/-- The rows binding the last query to `v`: the earlier ones, then the last. -/
theorem card_filter_last {n : ℕ} (x : Zoology.MQARInstance (n + 1) c) (v : Fin c) :
    #{j | x.key j = x.query (Fin.last n) ∧ x.value j = v} =
      #{j : Fin n | x.key j.castSucc = x.query (Fin.last n) ∧ x.value j.castSucc = v} +
      if x.key (Fin.last n) = x.query (Fin.last n) ∧ x.value (Fin.last n) = v then 1 else 0 := by
  rw [card_filter, card_filter, Fin.sum_univ_castSucc]

/-- `v` answers the last query exactly when an earlier row binds it to `v`. -/
theorem priorAnswer_iff_pos {n : ℕ} (x : Zoology.MQARInstance (n + 1) c) (v : Fin c) :
    Zoology.PriorAnswer x (Fin.last n) v ↔
      0 < #{j : Fin n | x.key j.castSucc = x.query (Fin.last n) ∧ x.value j.castSucc = v} := by
  rw [card_pos, filter_nonempty_iff]
  constructor
  · rintro ⟨j, hj, hk, hv⟩
    have hjn : j.val < n := hj
    refine ⟨⟨j.val, hjn⟩, mem_univ _, ?_⟩
    rw [show (⟨j.val, hjn⟩ : Fin n).castSucc = j from Fin.ext rfl]
    exact ⟨hk, hv⟩
  · rintro ⟨j, -, hk, hv⟩
    exact ⟨j.castSucc, Fin.castSucc_lt_last j, hk, hv⟩

/-- **One layer of width five recalls**: over `c` tokens and `n + 1` rows,
whatever `n`, with `2 (s + 1) c² 2^s < 2^{p-1}`, the matcher accepts exactly
the instances whose last query an earlier row binds to `v`, consistent or not. -/
theorem matcher_answers (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) {n : ℕ} (v : Fin c)
    (x : Zoology.MQARInstance (n + 1) c) :
    (matcher v : RTfr _ p s 5 1).Accepts (bos (word x)) ↔
      Zoology.PriorAnswer x (Fin.last n) v := by
  classical
  have h2 := two_pow_lt_of_le hp (Fin.pos v)
  set m := #{j | x.key j = x.query (Fin.last n)} with hm_def
  set mv := #{j | x.key j = x.query (Fin.last n) ∧ x.value j = v} with hmv_def
  have hone : (Fx.ofInt p s 1).val = 1 := by
    simpa using Fx.val_ofInt (p := p) (s := s) (z := 1) (by simpa using h2)
  have hD : (0 : ℝ) < 1 + m := by positivity
  have hsign : ∀ b : ℕ, 0 < (if 0 ≤ (Fx.round p s (((mv : ℝ) - b) / (1 + m))).m
      then Fx.ofInt p s 1 else 0).val ↔ b ≤ mv := by
    intro b
    have hiff : 0 ≤ (Fx.round p s (((mv : ℝ) - b) / (1 + m))).m ↔ b ≤ mv := by
      rw [m_round_nonneg_iff, le_div_iff₀ hD, zero_mul, sub_nonneg, Nat.cast_le]
    split_ifs with hb
    · rw [hone]
      exact iff_of_true one_pos (hiff.mp hb)
    · rw [Fx.val_zero]
      exact iff_of_false (lt_irrefl 0) fun h => hb (hiff.mpr h)
  have hsplit := card_filter_last x v
  rw [← hmv_def] at hsplit
  obtain ⟨hk, hu, hr⟩ := entry_embed (p := p) (s := s) hp (x.key (Fin.last n))
    (x.value (Fin.last n)) (x.query (Fin.last n))
  rw [RTfr.Accepts, RTfr.out_bos, show (word x).length = n + 1 by simp [word],
    actAt_matcher hp v x, ← hm_def, ← hmv_def, priorAnswer_iff_pos]
  simp only [entry] at hk hu hr
  simp only [matcher, entry, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.cons_val_three, Matrix.cons_val_four, Matrix.head_cons, Matrix.tail_cons, hk, hu, hr,
    add_left_inj, Nat.cast_inj, Fin.val_inj]
  by_cases hC : x.key (Fin.last n) = x.query (Fin.last n) ∧ x.value (Fin.last n) = v
  · have h₂ := hsign 2
    rw [Nat.cast_ofNat] at h₂
    rw [ite_eq_left hC, h₂]
    rw [ite_eq_left hC] at hsplit
    omega
  · have h₁ := hsign 1
    rw [Nat.cast_one] at h₁
    rw [ite_eq_right hC, h₁]
    rw [ite_eq_right hC] at hsplit
    omega

/-- The hypothesis of `matcher_answers` is satisfiable: one token, two rows,
no fractional bit, `p = 4`. -/
example (x : Zoology.MQARInstance (1 + 1) 1) :
    (matcher 0 : RTfr _ 4 0 5 1).Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last 1) 0 :=
  matcher_answers (by norm_num) 0 x

end Transformer.CRASP
