/-
# One layer of width five recalls over every vocabulary

In no paper: the attention comparing a query with a key of
`CRASP.MatchingRecall` recalls, where attention ignoring the query needs a
width growing with the vocabulary (`CRASP.QueryFreeRecall`).  Over `c` tokens
and `n + 1` rows, with `2 (s + 1) c² 2^s < 2^{p-1}`, the matcher's last
activation is the last token followed by `m_v / (1 + m)` and `1 / (1 + m)`
rounded (`actAt_matcher`), for the `m` rows whose key is the last query and the
`m_v` of them with value `v`.  Once `2^s ≥ n + 2` the rounding keeps `m_v`
apart from the last row's own share `e ∈ {0, 1}` (`mul_div_le_iff`), so the
matcher accepts exactly when `m_v ≥ 1 + e`, that is, when an earlier row binds
the last query to `v` (`matcher_answers`), consistent dictionary or not.
Aligned recall over `c` tokens and `n + 1` rows thus takes one layer of width
`5` at `p = O(log c + log n)` bits.
-/

import Transformer.CRASP.MatchingRecallSums

namespace Transformer.CRASP

open Finset

variable {c p s : ℕ}

/-- **The last activation**: the last row's token, then `m_v / (1 + m)` and
`1 / (1 + m)` rounded, where `m` rows hold the last query as key and `m_v` of
them the value `v`. -/
theorem actAt_matcher (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) (v : Fin c) {n : ℕ}
    (x : Zoology.MQARInstance (n + 1) c) :
    (matcher v : RTfr _ p s 5 1).actAt (word x) 1 (n + 1) =
      ![embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n))) 0,
        embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n))) 1,
        embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n))) 2,
        Fx.round p s (#{j | x.key j = x.query (Fin.last n) ∧ x.value j = v} /
          (1 + #{j | x.key j = x.query (Fin.last n)})),
        Fx.round p s (1 / (1 + #{j | x.key j = x.query (Fin.last n)}))] := by
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
      (if k = 4 then 1 else 0) + #{j | k = 3 ∧ x.key j = x.query (Fin.last n) ∧ x.value j = v} := by
    intro k
    have hs := sum_bos_word x fun t => (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0
      (embed (some (x.key (Fin.last n), x.value (Fin.last n), x.query (Fin.last n)))) (embed t)) *
        ((matcher v : RTfr _ p s 5 1).WV 0 (embed t) k).val)).val
    rw [hmask]
    simp only [hact, hlast]
    rw [hs, val_weighted_none h2]
    simp only [val_weighted_some hp, Finset.sum_boole]
  have hD₀ : (1 : ℝ) + #{j | x.key j = x.query (Fin.last n)} ≠ 0 := by positivity
  rw [RTfr.actAt, dite_eq_left hi]
  funext k
  rw [show (matcher v : RTfr _ p s 5 1).act (bos (word x)) 1 =
    (matcher v : RTfr _ p s 5 1).layer 0 ((matcher v : RTfr _ p s 5 1).act (bos (word x)) 0)
      from rfl, layer_matcher v _ _ k (hD ▸ hD₀), hD, hM k]
  simp only [hact, hlast]
  fin_cases k <;> simp [embed]

/-- The hypothesis of `actAt_matcher` is satisfiable: one token, `s = 1`,
`p = 5`. -/
example : 2 * (1 + 1) * 1 ^ 2 * 2 ^ 1 < 2 ^ (5 - 1) := by norm_num

/-- **Rounding keeps `m_v` apart from the last row's share `e`**: with
`0 < D ≤ N`, `e ≤ 1` and `e ≤ m_v`, `(1 + e) ⌊N / D⌋ ≤ ⌊m_v N / D⌋` exactly when
`1 + e ≤ m_v`. -/
theorem mul_div_le_iff {N D mv e : ℕ} (hD : 0 < D) (hDN : D ≤ N) (he : e ≤ 1) (hem : e ≤ mv) :
    (1 + e) * (N / D) ≤ mv * N / D ↔ 1 + e ≤ mv := by
  constructor
  · intro h
    by_contra hlt
    have hpos : 0 < N / D := Nat.div_pos hDN hD
    rcases Nat.le_one_iff_eq_zero_or_eq_one.mp he with rfl | rfl
    · obtain rfl : mv = 0 := by omega
      simp only [zero_mul, Nat.zero_div, add_zero, one_mul] at h
      omega
    · obtain rfl : mv = 1 := by omega
      simp only [one_mul] at h
      omega
  · intro h
    calc (1 + e) * (N / D) ≤ (1 + e) * N / D := Nat.mul_div_le_mul_div_assoc _ _ _
      _ ≤ mv * N / D := Nat.div_le_div_right (Nat.mul_le_mul_right _ h)

/-- The hypotheses of `mul_div_le_iff` are satisfiable: `N = 4`, `D = 3`,
`e = 1`, `m_v = 2`. -/
example : (1 + 1) * (4 / 3) ≤ 2 * 4 / 3 ↔ 1 + 1 ≤ 2 :=
  mul_div_le_iff (by norm_num) (by norm_num) le_rfl (by norm_num)

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

/-- **One layer of width five recalls**: over `c` tokens and `n + 1` rows, with
`2^s ≥ n + 2` and `2 (s + 1) c² 2^s < 2^{p-1}`, the matcher accepts exactly the
instances whose last query an earlier row binds to `v`, consistent or not. -/
theorem matcher_answers (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) {n : ℕ}
    (hs : n + 2 ≤ 2 ^ s) (v : Fin c) (x : Zoology.MQARInstance (n + 1) c) :
    (matcher v : RTfr _ p s 5 1).Accepts (bos (word x)) ↔
      Zoology.PriorAnswer x (Fin.last n) v := by
  classical
  have h2 := two_pow_lt_of_le hp (Fin.pos v)
  set m := #{j | x.key j = x.query (Fin.last n)} with hm_def
  set mv := #{j | x.key j = x.query (Fin.last n) ∧ x.value j = v} with hmv_def
  have hm : m ≤ n + 1 := by
    simpa using card_le_univ (filter (fun j => x.key j = x.query (Fin.last n)) univ)
  have hmv : mv ≤ m := card_le_card fun j hj => by
    simp only [mem_filter, mem_univ, true_and] at hj ⊢
    exact hj.1
  have hround : ∀ a : ℕ, a ≤ 1 + m → (Fx.round p s ((a : ℝ) / (1 + (m : ℝ)))).m =
      ((a * 2 ^ s / (1 + m) : ℕ) : ℤ) := by
    intro a ha
    have hfl : ⌊(a : ℝ) / (1 + m) * 2 ^ s⌋ = ((a * 2 ^ s / (1 + m) : ℕ) : ℤ) := by
      rw [show (a : ℝ) / (1 + m) * 2 ^ s = ((a * 2 ^ s : ℕ) : ℝ) / ((1 + m : ℕ) : ℝ) by
        push_cast; ring, ← Int.natCast_floor_eq_floor (by positivity), Nat.floor_div_eq_div]
    have hle : a * 2 ^ s / (1 + m) ≤ 2 ^ s := Nat.div_le_of_le_mul (Nat.mul_le_mul_right _ ha)
    have hle' : ((a * 2 ^ s / (1 + m) : ℕ) : ℤ) ≤ 2 ^ s := by exact_mod_cast hle
    have hp₀ : (0 : ℤ) ≤ 2 ^ (p - 1) := by positivity
    have hnn : (0 : ℤ) ≤ ((a * 2 ^ s / (1 + m) : ℕ) : ℤ) := by positivity
    rw [Fx.m_round, hfl, Fx.clamp_eq_self (by omega) (by omega)]
  have h₃ := hround mv (by omega)
  have h₄ := hround 1 (by omega)
  rw [Nat.cast_one, one_mul] at h₄
  obtain ⟨hk, hu, hr⟩ := entry_embed (p := p) (s := s) hp (x.key (Fin.last n))
    (x.value (Fin.last n)) (x.query (Fin.last n))
  have hone : (Fx.ofInt p s 1).val = 1 := by
    simpa using Fx.val_ofInt (p := p) (s := s) (z := 1) (by simpa using h2)
  set e : ℕ := if x.key (Fin.last n) = x.query (Fin.last n) ∧ x.value (Fin.last n) = v then 1
    else 0 with he_def
  have he : e ≤ 1 := by rw [he_def]; split <;> omega
  have hsplit := card_filter_last x v
  rw [← hmv_def, ← he_def] at hsplit
  have hdec : Zoology.PriorAnswer x (Fin.last n) v ↔
      (((1 + e) * (2 ^ s / (1 + m)) : ℕ) : ℤ) ≤ ((mv * 2 ^ s / (1 + m) : ℕ) : ℤ) := by
    rw [Nat.cast_le, mul_div_le_iff (by omega) (by omega) he (by omega), priorAnswer_iff_pos]
    omega
  have hfac : (1 + if x.key (Fin.last n) = x.query (Fin.last n) ∧ x.value (Fin.last n) = v
      then 1 else 0 : ℤ) = ((1 + e : ℕ) : ℤ) := by
    rw [he_def]
    split_ifs <;> simp
  rw [RTfr.Accepts, RTfr.out_bos, show (word x).length = n + 1 by simp [word],
    actAt_matcher hp v x, ← hm_def, ← hmv_def]
  simp only [entry] at hk hu hr
  simp only [matcher, entry, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
    Matrix.cons_val_three, Matrix.cons_val_four, Matrix.head_cons, Matrix.tail_cons, hk, hu, hr,
    h₃, h₄, add_left_inj, Nat.cast_inj, Fin.val_inj, hfac]
  split_ifs with hC
  · rw [hone]
    exact iff_of_true one_pos (hdec.mpr (by rw [Nat.cast_mul]; exact hC))
  · rw [Fx.val_zero]
    exact iff_of_false (lt_irrefl 0) fun hP => hC (by rw [← Nat.cast_mul]; exact hdec.mp hP)

/-- The hypotheses of `matcher_answers` are satisfiable: one token, one row,
`s = 1`, `p = 5`. -/
example (x : Zoology.MQARInstance (0 + 1) 1) :
    (matcher 0 : RTfr _ 5 1 5 1).Accepts (bos (word x)) ↔ Zoology.PriorAnswer x (Fin.last 0) 0 :=
  matcher_answers (by norm_num) (by norm_num) 0 x

end Transformer.CRASP
