module

public import Transformer.Grokking.FlatSpectrum.StepWrap

/-!
# The wrap of a step is a half-plane of a phase of rank one

Source: `review/flat_spectrum.md`, *The wrap of a step is a comparison* (added), item CW7.
Along the values `a`, `b`, `x = (a + b) % p` let `d₁ = b − a` and `d₂ = modStep p a b` be the
two steps that a difference encoding has written. The next step wraps when `p ≤ b + x`, and
the sum, not reduced, is an integer form in the two steps (`add_eq_steps`):
`b + x = 2 d₁ + 3 d₂ + 2p [d₂ < 0]`. So its phase of order `2p` is the product of a function of
`d₁` and a function of `d₂` (`exp_sum_eq_mul`),

`exp (iπ (b + x) / p) = exp (2πi d₁ / p) · exp (3πi d₂ / p)`,

a unimodular kernel of rank one in the two steps, and the wrap is a half-plane of it
(`modStep_neg_iff_im_neg`): the next step is negative exactly when
`Im (exp (iπ / 2p) · exp (2πi d₁ / p) · exp (3πi d₂ / p)) < 0`.

The sign of the step is a comparison (`wrapKernel_eq_submatrix`), which a minimum of the
nuclear norm does not recover at the boundary. The phase of the sum has rank one, which it
does recover, and it holds the same bit. The difference is the readout.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Complex Real

/-- **The sum in the coordinates of the steps**, `review/flat_spectrum.md`, item CW7: along
`a`, `b`, `x = (a + b) % p`, with `d₁ = b − a` and `d₂ = modStep p a b`,
`b + x = 2 d₁ + 3 d₂ + 2p [d₂ < 0]`. -/
theorem add_eq_steps {p a b : ℕ} (ha : a < p) (hb : b < p) :
    ((b + (a + b) % p : ℕ) : ℤ) = 2 * ((b : ℤ) - a) + 3 * modStep p a b +
      2 * if modStep p a b < 0 then (p : ℤ) else 0 := by
  have h1 : (((a + b) % p : ℕ) : ℤ) = modStep p a b + b := by
    unfold modStep
    ring
  rw [Nat.cast_add, h1, modStep_eq ha hb]
  split_ifs <;> omega

/-- Modulo `97`, from `50` and `60`: `60 + 13 = 2 · 10 + 3 · (50 − 97) + 2 · 97`. -/
example := add_eq_steps (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- The next step wraps exactly when the integer form of the two steps before it reaches
`p`. -/
theorem modStep_neg_iff_steps {p a b : ℕ} (ha : a < p) (hb : b < p) :
    modStep p b ((a + b) % p) < 0 ↔ (p : ℤ) ≤ 2 * ((b : ℤ) - a) + 3 * modStep p a b +
      2 * if modStep p a b < 0 then (p : ℤ) else 0 := by
  rw [modStep_neg_iff hb (Nat.mod_lt _ (by omega)), ← add_eq_steps ha hb, Nat.cast_le]

example := modStep_neg_iff_steps (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- **The phase of the sum has rank one**, `review/flat_spectrum.md`, item CW7:
`exp (iπ (b + x) / p) = exp (2πi d₁ / p) · exp (3πi d₂ / p)`. -/
theorem exp_sum_eq_mul {p a b : ℕ} (ha : a < p) (hb : b < p) :
    cexp (π * I * ((b + (a + b) % p : ℕ) : ℂ) / p) =
      cexp (2 * π * I * ((b : ℂ) - a) / p) * cexp (3 * π * I * (modStep p a b : ℂ) / p) := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have h : ((b + (a + b) % p : ℕ) : ℂ) = 2 * ((b : ℂ) - a) + 3 * (modStep p a b : ℂ) +
      2 * if modStep p a b < 0 then (p : ℂ) else 0 := by
    rw [← Int.cast_natCast (R := ℂ) (b + (a + b) % p), add_eq_steps ha hb]
    push_cast
    rfl
  rw [h, ← Complex.exp_add]
  split_ifs
  · have h2 : π * I * (2 * ((b : ℂ) - a) + 3 * (modStep p a b : ℂ) + 2 * (p : ℂ)) / p =
        2 * π * I * ((b : ℂ) - a) / p + 3 * π * I * (modStep p a b : ℂ) / p + 2 * π * I := by
      field_simp
    rw [h2, Complex.exp_add, Complex.exp_two_pi_mul_I, mul_one]
  · congr 1
    field_simp
    ring

example := exp_sum_eq_mul (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- A half-plane: for `n < 2p`, `sin (π (n + ½) / p)` is negative exactly when `p ≤ n`. -/
theorem sin_neg_iff_le {p n : ℕ} (hn : n < 2 * p) :
    Real.sin (π * (n + 1 / 2) / p) < 0 ↔ p ≤ n := by
  have hp : (0 : ℝ) < p := by exact_mod_cast (by omega : 0 < p)
  have hn' : (n : ℝ) + 1 ≤ 2 * p := by exact_mod_cast hn
  constructor
  · intro h
    by_contra hlt
    have h1 : (n : ℝ) + 1 ≤ p := by exact_mod_cast (by omega : n + 1 ≤ p)
    have h2 : 0 < Real.sin (π * (n + 1 / 2) / p) := by
      refine sin_pos_of_pos_of_lt_pi (by positivity) ?_
      rw [div_lt_iff₀ hp]
      nlinarith [pi_pos]
    linarith
  · intro h
    have h1 : (p : ℝ) ≤ n := by exact_mod_cast h
    rw [← Real.sin_sub_two_pi]
    refine sin_neg_of_neg_of_neg_pi_lt ?_ ?_
    · rw [sub_neg, div_lt_iff₀ hp]
      nlinarith [pi_pos]
    · rw [lt_sub_iff_add_lt, lt_div_iff₀ hp]
      nlinarith [pi_pos]

example : Real.sin (π * ((73 : ℕ) + 1 / 2) / (97 : ℕ)) < 0 ↔ 97 ≤ 73 :=
  sin_neg_iff_le (by norm_num)

/-- **The wrap is a half-plane of a phase of rank one**, `review/flat_spectrum.md`, item CW7:
along `a`, `b`, `(a + b) % p` the next step is negative exactly when the product of a
unimodular function of `d₁ = b − a` and one of `d₂ = modStep p a b`, turned by `π / 2p`, lies
below the real axis. -/
theorem modStep_neg_iff_im_neg {p a b : ℕ} (ha : a < p) (hb : b < p) :
    modStep p b ((a + b) % p) < 0 ↔ (cexp (π * I / (2 * p)) *
      (cexp (2 * π * I * ((b : ℂ) - a) / p) * cexp (3 * π * I * (modStep p a b : ℂ) / p))).im
        < 0 := by
  have hp : (p : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (by omega)
  have hx : (a + b) % p < p := Nat.mod_lt _ (by omega)
  have h : π * I / (2 * p) + π * I * ((b + (a + b) % p : ℕ) : ℂ) / p =
      ((π * (((b + (a + b) % p : ℕ) : ℝ) + 1 / 2) / p : ℝ) : ℂ) * I := by
    push_cast
    field_simp
    ring
  rw [← exp_sum_eq_mul ha hb, ← Complex.exp_add, h, exp_ofReal_mul_I_im,
    sin_neg_iff_le (by omega), modStep_neg_iff hb hx]

example := modStep_neg_iff_im_neg (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

end Transformer.Grokking.FlatSpectrum
