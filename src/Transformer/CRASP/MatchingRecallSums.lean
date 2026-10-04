/-
# The matcher's weights, and its layer at a position

In no paper: the attention of `CRASP.MatchingRecall`'s matcher, token by
token.  Once `2 (s + 1) c² 2^s < 2^{p-1}` over `c` tokens, a query weighs `⊲`
`1` (`val_weight_none`), and a row `1` when its key is the query and `0`
otherwise (`val_weight_some`); the values mark `⊲` in the last coordinate
(`val_weighted_none`) and the rows with value `v` in the fourth
(`val_weighted_some`).  A sum over `⊲ · word x` is the sum at `⊲` plus one per
row (`sum_bos_word`), at the last position every position is visible
(`masked_eq_univ`), and a coordinate of the matcher's layer is its weighted
values over its weights once these do not sum to `0` (`layer_matcher`).
-/

import Transformer.CRASP.MatchingRecall

namespace Transformer.CRASP

open Finset

variable {c p s : ℕ}

/-- `2^s < 2^{p-1}` once `2 (s + 1) c² 2^s < 2^{p-1}` over `c ≥ 1` tokens. -/
theorem two_pow_lt_of_le (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) (hc : 0 < c) :
    (2 : ℤ) ^ s < 2 ^ (p - 1) := by
  have h : 2 ^ s ≤ 2 * (s + 1) * c ^ 2 * 2 ^ s :=
    Nat.le_mul_of_pos_left _ (by positivity)
  exact_mod_cast h.trans_lt hp

/-- The hypotheses of `two_pow_lt_of_le` are satisfiable: one token, `s = 1`,
`p = 5`. -/
example : (2 : ℤ) ^ 1 < 2 ^ (5 - 1) := two_pow_lt_of_le (c := 1) (by norm_num) one_pos

/-- `1` is exact once `2^s < 2^{p-1}`. -/
theorem val_round_one (hp : (2 : ℤ) ^ s < 2 ^ (p - 1)) : (Fx.round p s 1).val = 1 := by
  have h := Fx.val_ofInt (p := p) (s := s) (z := 1) (by simpa using hp)
  simpa [Fx.ofInt] using h

/-- The hypothesis of `val_round_one` is satisfiable: `s = 1`, `p = 3`. -/
example : (Fx.round 3 1 1).val = 1 := val_round_one (by norm_num)

/-- **`⊲` weighs `1`.** -/
theorem val_weight_none (hp : (2 : ℤ) ^ s < 2 ^ (p - 1)) (v : Fin c) (q : Fin 5 → Fx p s) :
    (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0 q (embed (c := c) none)))).val = 1 := by
  rw [score_matcher_none, Real.exp_zero, val_round_one hp]

/-- The hypothesis of `val_weight_none` is satisfiable: one token, `s = 1`,
`p = 3`. -/
example : (Fx.round 3 1 (Real.exp ((matcher (0 : Fin 1) : RTfr _ 3 1 5 1).score 0 0
    (embed (c := 1) none)))).val = 1 :=
  val_weight_none (by norm_num) 0 0

/-- **A row weighs `[its key is the query]`.** -/
theorem val_weight_some (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) (v : Fin c)
    (a u r a' u' r' : Fin c) :
    (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0 (embed (some (a, u, r)))
      (embed (some (a', u', r')))))).val = if a' = r then 1 else 0 := by
  rw [score_matcher_some hp, val_round_exp (two_pow_lt_of_le hp (Fin.pos r))]
  simp only [sub_eq_zero, Nat.cast_inj, Fin.val_inj, eq_comm]

/-- The hypothesis of `val_weight_some` is satisfiable: one token, `s = 1`,
`p = 5`. -/
example : (Fx.round 5 1 (Real.exp ((matcher (0 : Fin 1) : RTfr _ 5 1 5 1).score 0
    (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1)))) (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1))))))).val =
    if (0 : Fin 1) = 0 then 1 else 0 :=
  val_weight_some (by norm_num) 0 0 0 0 0 0 0

/-- **`⊲` contributes its mark**, in the last coordinate. -/
theorem val_weighted_none (hp : (2 : ℤ) ^ s < 2 ^ (p - 1)) (v : Fin c) (q : Fin 5 → Fx p s)
    (k : Fin 5) :
    (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0 q (embed (c := c) none)) *
      ((matcher v : RTfr _ p s 5 1).WV 0 (embed (c := c) none) k).val)).val = if k = 4 then 1 else 0 := by
  have hv : (0 : ℤ) ≠ v.val + 1 := by omega
  have h₁ : (Fx.ofInt p s 1).val = 1 := by
    simpa using Fx.val_ofInt (p := p) (s := s) (z := 1) (by simpa using hp)
  rw [score_matcher_none, Real.exp_zero, one_mul]
  fin_cases k <;> simp [matcher, embed, entry, h₁, val_round_one hp, hv]

/-- The hypothesis of `val_weighted_none` is satisfiable: one token, `s = 1`,
`p = 3`. -/
example : (Fx.round 3 1 (Real.exp ((matcher (0 : Fin 1) : RTfr _ 3 1 5 1).score 0 0
    (embed (c := 1) none)) * ((matcher (0 : Fin 1) : RTfr _ 3 1 5 1).WV 0 (embed (c := 1) none) 4).val)).val =
    if (4 : Fin 5) = 4 then 1 else 0 :=
  val_weighted_none (by norm_num) 0 0 4

/-- **A row contributes `[its key is the query ∧ its value is v]`**, in the
fourth coordinate. -/
theorem val_weighted_some (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) (v : Fin c)
    (a u r a' u' r' : Fin c) (k : Fin 5) :
    (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0 (embed (some (a, u, r)))
      (embed (some (a', u', r')))) *
      ((matcher v : RTfr _ p s 5 1).WV 0 (embed (some (a', u', r'))) k).val)).val =
      if k = 3 ∧ a' = r ∧ u' = v then 1 else 0 := by
  have h2 := two_pow_lt_of_le hp (Fin.pos r)
  obtain ⟨ha, hu, -⟩ := entry_embed (p := p) (s := s) hp a' u' r'
  have hA : (a'.val : ℤ) + 1 ≠ 0 := by omega
  have h₁ : (Fx.ofInt p s 1).val = 1 := by
    simpa using Fx.val_ofInt (p := p) (s := s) (z := 1) (by simpa using h2)
  have hw := val_weight_some hp v a u r a' u' r'
  have hu' : ((u'.val : ℤ) + 1 = v.val + 1) = (u' = v) := by
    simp only [add_left_inj, Nat.cast_inj, Fin.val_inj]
  have hWV : (matcher v : RTfr _ p s 5 1).WV 0 (embed (some (a', u', r'))) =
      ![0, 0, 0, if u' = v then Fx.ofInt p s 1 else 0, 0] := by
    simp only [matcher, ha, hu, hA, hu', ite_false]
  rw [hWV]
  by_cases huv : u' = v
  · subst huv
    fin_cases k <;> simp [h₁, hw]
  · fin_cases k <;> simp [huv]

/-- The hypothesis of `val_weighted_some` is satisfiable: one token, `s = 1`,
`p = 5`. -/
example : (Fx.round 5 1 (Real.exp ((matcher (0 : Fin 1) : RTfr _ 5 1 5 1).score 0
    (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1)))) (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1))))) *
    ((matcher (0 : Fin 1) : RTfr _ 5 1 5 1).WV 0 (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1)))) 3).val)).val =
    if (3 : Fin 5) = 3 ∧ (0 : Fin 1) = 0 ∧ (0 : Fin 1) = 0 then 1 else 0 :=
  val_weighted_some (by norm_num) 0 0 0 0 0 0 0 3

/-- At the last position every position is visible. -/
theorem masked_eq_univ {N : ℕ} (i : Fin N) (h : N = i.val + 1) :
    RTfr.masked i = Finset.univ := by
  ext j
  simp only [RTfr.masked, Finset.mem_filter, Finset.mem_univ, true_and, iff_true, Fin.le_def]
  omega

/-- The hypothesis of `masked_eq_univ` is satisfiable: the last of three. -/
example : RTfr.masked (Fin.last 2) = Finset.univ := masked_eq_univ _ rfl

/-- A sum over `⊲ · word x`: `⊲`, then the rows. -/
theorem sum_bos_word {n : ℕ} (x : Zoology.MQARInstance n c)
    (G : Option (Fin c × Fin c × Fin c) → ℝ) :
    ∑ j : Fin (bos (word x)).length, G (bos (word x))[j.1] =
      G none + ∑ j, G (some (x.key j, x.value j, x.query j)) := by
  rw [Fin.sum_univ_fun_getElem]
  simp [bos, word, List.map_ofFn, List.sum_ofFn]

/-- A coordinate of the matcher's layer once its denominator does not vanish
(Equation `eq:att`; the feed-forward network is the identity). -/
theorem layer_matcher (v : Fin c) {N : ℕ} (h : Fin N → Fin 5 → Fx p s) (i : Fin N) (k : Fin 5)
    (hD : ∑ j ∈ RTfr.masked i,
      (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0 (h i) (h j)))).val ≠ 0) :
    (matcher v : RTfr _ p s 5 1).layer 0 h i k = Fx.add (Fx.round p s
      ((∑ j ∈ RTfr.masked i, (Fx.round p s
        (Real.exp ((matcher v : RTfr _ p s 5 1).score 0 (h i) (h j)) *
          ((matcher v : RTfr _ p s 5 1).WV 0 (h j) k).val)).val) /
      ∑ j ∈ RTfr.masked i,
        (Fx.round p s (Real.exp ((matcher v : RTfr _ p s 5 1).score 0 (h i) (h j)))).val))
      (h i k) := by
  simp only [RTfr.score] at hD ⊢
  simp only [RTfr.layer, hD, ite_false]
  rfl

/-- The hypothesis of `layer_matcher` is satisfiable: `⊲` alone weighs `1`. -/
example : ∑ j ∈ RTfr.masked (0 : Fin 1), (Fx.round 3 1 (Real.exp ((matcher (0 : Fin 1) :
    RTfr _ 3 1 5 1).score 0 (embed (c := 1) none)
      ((fun _ : Fin 1 => embed (c := 1) none) j)))).val ≠ 0 := by
  rw [masked_eq_univ (0 : Fin 1) rfl, Fin.sum_univ_one, val_weight_none (by norm_num)]
  norm_num

end Transformer.CRASP
