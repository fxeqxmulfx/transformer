module

public import Transformer.Grokking.FlatSpectrum.ScoreShift

/-!
# The wrap from the numbers of the two steps, at any frequency

Source: `review/flat_spectrum.md`, *The numbers of the tokens* (added), items NT1 and NT2.
In CW7 the phase of the sum has the order `2p`: its frequency `π / p` is that of a character,
and the term `2p [d₂ < 0]` of the sum disappears. At another frequency `ω` it does not, and the
phase is still a product of a function of each step (`exp_sum_eq_mul_halfLine`):

`exp (iω (b + x)) = exp (2iω d₁) · exp (3iω d₂) · (exp (2iω p) if d₂ < 0, and 1 otherwise)`.

The second factor is the phase `exp (3iω d₂)` of the number `d₂` times one of two constants,
by the sign of `d₂`: linear in the two phases of `d₂` cut to a half-line, with two
coefficients for all tokens. For `0 < ω` and `ω p ≤ π` the wrap is a half-plane of this phase
(`sin_pos_iff_le`, `modStep_neg_iff_sin_pos`): the next step wraps exactly when
`0 < sin (ω (b + x + ½ − p))`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Complex Real

/-- **NT1**, `review/flat_spectrum.md`: the phase of the sum at any frequency is the phase of
`2 d₁`, the phase of `3 d₂`, and one of two constants by the sign of `d₂`. -/
theorem exp_sum_eq_mul_halfLine {p a b : ℕ} (ha : a < p) (hb : b < p) (ω : ℝ) :
    cexp (ω * I * ((b + (a + b) % p : ℕ) : ℂ)) =
      cexp (2 * ω * I * ((b : ℂ) - a)) * (cexp (3 * ω * I * (modStep p a b : ℂ)) *
        if modStep p a b < 0 then cexp (2 * ω * I * p) else 1) := by
  have h : ((b + (a + b) % p : ℕ) : ℂ) = 2 * ((b : ℂ) - a) + 3 * (modStep p a b : ℂ) +
      2 * if modStep p a b < 0 then (p : ℂ) else 0 := by
    rw [← Int.cast_natCast (R := ℂ) (b + (a + b) % p), add_eq_steps ha hb]
    push_cast
    rfl
  rw [h]
  split_ifs
  · rw [← Complex.exp_add, ← Complex.exp_add]
    congr 1
    ring
  · rw [mul_one, ← Complex.exp_add]
    congr 1
    ring

example := exp_sum_eq_mul_halfLine (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)
  (1 / 100)

/-- **NT2**, `review/flat_spectrum.md`: for `n < 2p` and a frequency `0 < ω` with `ω p ≤ π`,
`sin (ω (n + ½ − p))` is positive exactly when `p ≤ n`. -/
theorem sin_pos_iff_le {p n : ℕ} (hn : n < 2 * p) {ω : ℝ} (hω : 0 < ω) (hp : ω * p ≤ π) :
    0 < Real.sin (ω * (n + 1 / 2 - p)) ↔ p ≤ n := by
  have hn' : (n : ℝ) + 1 ≤ 2 * p := by exact_mod_cast hn
  constructor
  · intro h
    by_contra hlt
    have h1 : (n : ℝ) + 1 ≤ p := by exact_mod_cast (by omega : n + 1 ≤ p)
    have h2 : Real.sin (ω * (n + 1 / 2 - p)) < 0 := by
      refine sin_neg_of_neg_of_neg_pi_lt (by nlinarith) ?_
      nlinarith [Nat.cast_nonneg (α := ℝ) n]
    linarith
  · intro h
    have h1 : (p : ℝ) ≤ n := by exact_mod_cast h
    refine sin_pos_of_pos_of_lt_pi (by nlinarith) ?_
    nlinarith

/-- Modulo `97` at the frequency `1 / 100`: the sum `73` does not wrap. -/
example : 0 < Real.sin (1 / 100 * ((73 : ℕ) + 1 / 2 - (97 : ℕ))) ↔ 97 ≤ 73 :=
  sin_pos_iff_le (by norm_num) (by norm_num) (by nlinarith [two_le_pi])

/-- **NT2**, `review/flat_spectrum.md`: the next step wraps exactly when the sum, at a
frequency `0 < ω` with `ω p ≤ π`, is in the upper half of its turn around `p − ½`. -/
theorem modStep_neg_iff_sin_pos {p a b : ℕ} (hb : b < p) {ω : ℝ} (hω : 0 < ω)
    (hp : ω * p ≤ π) :
    modStep p b ((a + b) % p) < 0 ↔
      0 < Real.sin (ω * (((b + (a + b) % p : ℕ) : ℝ) + 1 / 2 - p)) := by
  have hx : (a + b) % p < p := Nat.mod_lt _ (by omega)
  rw [sin_pos_iff_le (by omega) hω hp, modStep_neg_iff hb hx]

example := modStep_neg_iff_sin_pos (p := 97) (a := 50) (b := 60) (by norm_num)
  (ω := 1 / 100) (by norm_num) (by nlinarith [two_le_pi])

end Transformer.Grokking.FlatSpectrum
