/-
# A transformer comparing a query with a key, in width five

In no paper: the attention comparing a query with a key that
`CRASP.QueryFreeRecall` leaves out.  Zoology's attention solution of MQAR
(arXiv:2312.04927v1, §4, `prop: attention-ar`, proved as
`prop: app-attention`) compares them through one-hot embeddings, in width `3c`
over `c` tokens.  A future-masked rounded transformer of arXiv:2506.16055v3
(Appendix B.1, `def:transformer`) compares them in width `5`, spending
precision instead: the query `(-β R², 2βR, -β)` of a token `R` meets the key
`(1, A, A²)` of a token `A` in the score `-β (R - A)²` (`score_matcher_some`),
`0` on a match and at most `-β` otherwise.  At `β = s + 1` the rounded weight
`round(exp(score))` is `1` on a match and `0` otherwise (`val_round_exp`), as
`e^{s+1} > 2^s`; the key of `⊲` is `0`, so `⊲` weighs `1`
(`score_matcher_none`) and no denominator vanishes.  This file builds the
transformer (`matcher`) and its scores; `CRASP.MatchingRecallSums` sums them,
and `CRASP.MatchingRecallAnswers` reads the last position.
-/

import Transformer.CRASP.FixedSign
import Transformer.CRASP.FixedTail
import Transformer.CRASP.QueryFreeRecall

namespace Transformer.CRASP

namespace Fx

variable {p s : ℕ}

/-- The integer `z` at precision `(p, s)`, `round(z)` (Definition
`def:fixed_precision`), exact while `|z| 2^s < 2^{p-1}` (`val_ofInt`). -/
noncomputable def ofInt (p s : ℕ) (z : ℤ) : Fx p s := round p s z

theorem m_ofInt {z : ℤ} (h : |z| * 2 ^ s < 2 ^ (p - 1)) : (ofInt p s z).m = z * 2 ^ s := by
  have hf : ⌊(z : ℝ) * 2 ^ s⌋ = z * 2 ^ s := by
    rw [show (z : ℝ) * 2 ^ s = ((z * 2 ^ s : ℤ) : ℝ) by push_cast; ring, Int.floor_intCast]
  have h₁ := neg_abs_le z
  have h₂ := le_abs_self z
  have hs : (0 : ℤ) < 2 ^ s := by positivity
  rw [ofInt, m_round, hf]
  exact clamp_eq_self (by nlinarith) (by nlinarith)

theorem val_ofInt {z : ℤ} (h : |z| * 2 ^ s < 2 ^ (p - 1)) : (ofInt p s z).val = z := by
  have hs : (0 : ℝ) < 2 ^ s := by positivity
  rw [val, m_ofInt h]
  push_cast
  field_simp

/-- The hypothesis of `m_ofInt` and `val_ofInt` is satisfiable: `-3` at
`p = 4`, `s = 1`. -/
example : (ofInt 4 1 (-3)).m = -3 * 2 ^ 1 ∧ (ofInt 4 1 (-3)).val = ((-3 : ℤ) : ℝ) :=
  ⟨m_ofInt (p := 4) (s := 1) (z := -3) (by norm_num),
    val_ofInt (p := 4) (s := 1) (z := -3) (by norm_num)⟩

end Fx

variable {c p s : ℕ}

/-- The integer part of coordinate `i`, by which the matcher reads a token. -/
noncomputable def entry (h : Fin 5 → Fx p s) (i : Fin 5) : ℤ := ⌊(h i).val⌋

/-- The embedding: a token `(a, u, r)` as `(a + 1, u + 1, r + 1, 0, 0)`, and `⊲`
as `0`. -/
noncomputable def embed : Option (Fin c × Fin c × Fin c) → Fin 5 → Fx p s
  | none => 0
  | some (a, u, r) =>
      ![Fx.ofInt p s (a.val + 1), Fx.ofInt p s (u.val + 1), Fx.ofInt p s (r.val + 1), 0, 0]

/-- **The matcher** for the value `v`, of one layer and width `5`
(Definition `def:transformer`): the query of a token `(a, u, r)` is
`(-β R², 2βR, -β, 0, 0)` for `R = r + 1` and `β = s + 1`, its key is
`(1, A, A², 0, 0)` for `A = a + 1`, the key of `⊲` is `0`, and the values are
`[u = v]` and `[⊲]` in the last two coordinates.  The feed-forward network is
the identity, and `W_out` accepts when `(1 + [a = r ∧ u = v]) · m₄ ≤ m₃` for
the last two mantissas `m₃`, `m₄`. -/
noncomputable def matcher (v : Fin c) : RTfr (Option (Fin c × Fin c × Fin c)) p s 5 1 where
  E := embed
  WQ _ h := ![Fx.ofInt p s (-(s + 1) * entry h 2 ^ 2), Fx.ofInt p s (2 * (s + 1) * entry h 2),
    Fx.ofInt p s (-(s + 1)), 0, 0]
  WK _ h := if entry h 0 = 0 then 0 else
    ![Fx.ofInt p s 1, Fx.ofInt p s (entry h 0), Fx.ofInt p s (entry h 0 ^ 2), 0, 0]
  WV _ h := ![0, 0, 0, if entry h 1 = v.val + 1 then Fx.ofInt p s 1 else 0,
    if entry h 0 = 0 then Fx.ofInt p s 1 else 0]
  ff _ h := h
  Wout h := if (1 + if entry h 0 = entry h 2 ∧ entry h 1 = v.val + 1 then 1 else 0) * (h 4).m ≤
    (h 3).m then Fx.ofInt p s 1 else 0

/-- The integers the matcher writes, up to `2 (s + 1) c²`, are exact once
`2 (s + 1) c² 2^s < 2^{p-1}`. -/
theorem val_ofInt_of_le (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) {z : ℤ}
    (hz : |z| ≤ 2 * (s + 1) * c ^ 2) : (Fx.ofInt p s z).val = z := by
  have h : ((2 * (s + 1) * c ^ 2 * 2 ^ s : ℕ) : ℤ) < ((2 ^ (p - 1) : ℕ) : ℤ) := by
    exact_mod_cast hp
  have hs : (0 : ℤ) < 2 ^ s := by positivity
  push_cast at h
  exact Fx.val_ofInt (by nlinarith)

/-- The hypotheses of `val_ofInt_of_le` are satisfiable: one token, `s = 1`,
`p = 5`. -/
example : (Fx.ofInt 5 1 (-4)).val = ((-4 : ℤ) : ℝ) :=
  val_ofInt_of_le (c := 1) (p := 5) (s := 1) (z := -4) (by norm_num) (by norm_num)

/-- The matcher reads a token back from its embedding. -/
theorem entry_embed (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) (a u r : Fin c) :
    entry (embed (some (a, u, r)) : Fin 5 → Fx p s) 0 = a.val + 1 ∧
      entry (embed (some (a, u, r)) : Fin 5 → Fx p s) 1 = u.val + 1 ∧
      entry (embed (some (a, u, r)) : Fin 5 → Fx p s) 2 = r.val + 1 := by
  have hb : ∀ t : Fin c, |(t.val + 1 : ℤ)| ≤ 2 * (s + 1) * c ^ 2 := fun t => by
    have ht : (t.val : ℤ) + 1 ≤ c := by exact_mod_cast t.isLt
    rw [abs_of_nonneg (by positivity)]
    nlinarith
  refine ⟨?_, ?_, ?_⟩ <;>
    simp only [entry, embed, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two,
      Matrix.head_cons, Matrix.tail_cons, val_ofInt_of_le hp (hb _), Int.floor_intCast]

/-- The hypothesis of `entry_embed` is satisfiable: one token, `s = 1`, `p = 5`. -/
example : entry (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1))) : Fin 5 → Fx 5 1) 0 =
    (0 : Fin 1).val + 1 :=
  (entry_embed (c := 1) (p := 5) (s := 1) (by norm_num) 0 0 0).1

/-- `⊲`, whose key is `0`, scores `0` against every query. -/
theorem score_matcher_none (v : Fin c) (x : Fin 5 → Fx p s) :
    (matcher v : RTfr _ p s 5 1).score 0 x (embed (c := c) none) = 0 := by
  simp [RTfr.score, matcher, embed, entry]

/-- **A query meets a key in `-(s + 1) (r - a')²`**: the token `(a, u, r)`
scores `0` against `(a', u', r')` when `a' = r`, at most `-(s + 1)` otherwise. -/
theorem score_matcher_some (hp : 2 * (s + 1) * c ^ 2 * 2 ^ s < 2 ^ (p - 1)) (v : Fin c)
    (a u r a' u' r' : Fin c) :
    (matcher v : RTfr _ p s 5 1).score 0 (embed (some (a, u, r))) (embed (some (a', u', r'))) =
      -(s + 1) * (((r.val : ℤ) - a'.val : ℤ) : ℝ) ^ 2 := by
  obtain ⟨-, -, hr⟩ := entry_embed (p := p) (s := s) hp a u r
  obtain ⟨ha, -, -⟩ := entry_embed (p := p) (s := s) hp a' u' r'
  have hR : (r.val : ℤ) + 1 ≤ c := by exact_mod_cast r.isLt
  have hA : (a'.val : ℤ) + 1 ≤ c := by exact_mod_cast a'.isLt
  have hA₀ : (a'.val : ℤ) + 1 ≠ 0 := by omega
  have hS : (0 : ℤ) ≤ s + 1 := by positivity
  have hc : (c : ℤ) ≤ c ^ 2 := by nlinarith
  have hc₁ : 1 ≤ (c : ℤ) ^ 2 := by nlinarith
  have hR2 : ((r.val : ℤ) + 1) ^ 2 ≤ (c : ℤ) ^ 2 := by nlinarith
  have hA2 : ((a'.val : ℤ) + 1) ^ 2 ≤ (c : ℤ) ^ 2 := by nlinarith
  have h₁ := mul_le_mul_of_nonneg_left hR2 hS
  have h₂ := mul_le_mul_of_nonneg_left (hR.trans hc) hS
  have h₃ := mul_le_mul_of_nonneg_left hc₁ hS
  have hb : ∀ z : ℤ, |z| ≤ 2 * (s + 1) * c ^ 2 → (Fx.ofInt p s z).val = z :=
    fun z hz => val_ofInt_of_le hp hz
  simp only [RTfr.score, matcher, Fin.sum_univ_five, hr, ha, hA₀, ite_false,
    Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
    Matrix.cons_val_four, Matrix.head_cons, Matrix.tail_cons, Fx.val_zero, mul_zero, add_zero]
  rw [hb _ (by rw [abs_le]; constructor <;> nlinarith), hb _ (by rw [abs_le]; constructor <;> nlinarith),
    hb _ (by rw [abs_le]; constructor <;> nlinarith), hb _ (by rw [abs_le]; constructor <;> nlinarith),
    hb _ (by rw [abs_le]; constructor <;> nlinarith), hb _ (by rw [abs_le]; constructor <;> nlinarith)]
  push_cast
  ring

/-- The hypothesis of `score_matcher_some` is satisfiable: one token, `s = 1`,
`p = 5`. -/
example : (matcher (0 : Fin 1) : RTfr _ 5 1 5 1).score 0
    (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1))))
    (embed (some ((0 : Fin 1), (0 : Fin 1), (0 : Fin 1)))) =
      -((1 : ℕ) + 1) * ((((0 : Fin 1).val : ℤ) - (0 : Fin 1).val : ℤ) : ℝ) ^ 2 :=
  score_matcher_some (by norm_num) 0 0 0 0 0 0 0

/-- `e^{s+1} > 2^s`. -/
theorem two_pow_lt_exp (s : ℕ) : (2 : ℝ) ^ s < Real.exp (s + 1) := by
  calc (2 : ℝ) ^ s < 2 ^ (s + 1) := pow_lt_pow_right₀ (by norm_num) (by omega)
    _ ≤ Real.exp 1 ^ (s + 1) := pow_le_pow_left₀ (by norm_num) Real.exp_one_gt_two.le _
    _ = Real.exp (s + 1) := by rw [← Real.exp_nat_mul]; push_cast; ring_nf

/-- **The rounded weight is `[z = 0]`**: `round(exp(-(s + 1) z²))` is `1` at
`z = 0`, and `0` elsewhere, where `e^{-(s+1)} 2^s < 1`. -/
theorem val_round_exp (hp : (2 : ℤ) ^ s < 2 ^ (p - 1)) (z : ℤ) :
    (Fx.round p s (Real.exp (-(s + 1) * (z : ℝ) ^ 2))).val = if z = 0 then 1 else 0 := by
  by_cases hz : z = 0
  · subst hz
    have h₁ : Fx.round p s 1 = Fx.ofInt p s 1 := by simp [Fx.ofInt]
    simp [h₁, Fx.val_ofInt (show |(1 : ℤ)| * 2 ^ s < 2 ^ (p - 1) by simpa using hp)]
  · rw [ite_eq_right hz]
    have hz2 : (1 : ℝ) ≤ (z : ℝ) ^ 2 := by
      have : (1 : ℤ) ≤ z ^ 2 := by rcases lt_or_gt_of_ne hz with h | h <;> nlinarith
      exact_mod_cast this
    have hs : (0 : ℝ) ≤ s + 1 := by positivity
    have hle : Real.exp (-(s + 1) * (z : ℝ) ^ 2) ≤ Real.exp (-(s + 1)) :=
      Real.exp_le_exp.mpr (by nlinarith [mul_le_mul_of_nonneg_left hz2 hs])
    have hlt : Real.exp (-(s + 1)) * 2 ^ s < 1 := by
      rw [Real.exp_neg, inv_mul_lt_iff₀ (Real.exp_pos _), mul_one]
      exact two_pow_lt_exp s
    have hx : |Real.exp (-(s + 1) * (z : ℝ) ^ 2) * 2 ^ s| < 1 := by
      rw [abs_of_pos (by positivity)]
      calc _ ≤ Real.exp (-(s + 1)) * 2 ^ s := by gcongr
        _ < 1 := hlt
    rw [Fx.round_small _ hx, ite_eq_right (not_lt.mpr (Real.exp_pos _).le), Fx.val_zero]

/-- The hypothesis of `val_round_exp` is satisfiable: `s = 1`, `p = 3`. -/
example : (Fx.round 3 1 (Real.exp (-(((1 : ℕ) : ℝ) + 1) * ((0 : ℤ) : ℝ) ^ 2))).val =
    if (0 : ℤ) = 0 then 1 else 0 :=
  val_round_exp (p := 3) (s := 1) (by norm_num) 0

end Transformer.CRASP
