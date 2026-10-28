module

public import Transformer.Grokking.FlatSpectrum.ReadoutFree
public import Transformer.Grokking.FlatSpectrum.ZeroIndicator

/-!
# The readout-free dichotomy for `P(a) + a b`

Source: `review/flat_spectrum.md`, *Readout-free bound for `P(a) + a b`* (added). For every `P`,
every readout `ρ` and every centered `h`, `∑_a |∑_b ρ(P(a) + a b) h(b)|² ≤ (p − 1) ‖ρ‖² ‖h‖²`:
row `0` vanishes, and each other row is at most `‖ρ‖ ‖h‖` by Cauchy–Schwarz. Affine `P` reach this
bound. For `P(a) = p₀ + p₁ a`, `f(a, b) − p₀ = a (b + p₁)`, and `r = 1 − p δ₀` satisfies
`r(a u) = r(u)` for `a ≠ 0`. So `ρ(c) = r(c − p₀)` and the centered `h(b) = r(b + p₁)` make every
row `a ≠ 0` equal to `‖r‖²`. With `sum_norm_sq_sum_poly_add_mul_le` for `deg P ≥ 2` this gives
the dichotomy: once `(m − 1)(2p − 1) < (p − 2)²`, the bound `p − 1` is reached iff `deg P ≤ 1`.
For a phase readout, `|ρ| = 1`, and unit `g`, `h` the bound reads
`|∑_{a, b} ρ(f(a, b)) g(a) h(b)|² / p² ≤ (1 + √((m − 1)(2p − 1))) / p`
(`norm_sq_sum_poly_add_mul_div_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset Polynomial

variable {p : ℕ} [Fact p.Prime]

/-- **The largest share**: `∑_a |∑_b ρ(P(a) + a b) h(b)|² ≤ (p − 1) ‖ρ‖² ‖h‖²` for every `P`,
every readout `ρ` and every centered `h`. -/
theorem sum_norm_sq_sum_poly_add_mul_le_sub_one (P : (ZMod p)[X]) (ρ h : ZMod p → ℂ)
    (hh : ∑ b, h b = 0) :
    ∑ a, ‖∑ b, ρ (P.eval a + a * b) * h b‖ ^ 2 ≤
      (p - 1) * (∑ c, ‖ρ c‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) := by
  have hrow : ∀ a ∈ univ.erase (0 : ZMod p), ‖∑ b, ρ (P.eval a + a * b) * h b‖ ^ 2 ≤
      (∑ c, ‖ρ c‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) := by
    intro a ha
    have hcs := norm_sum_star_mul_le univ (fun b => star (ρ (P.eval a + a * b))) h
    simp only [star_star, norm_star] at hcs
    have hρ : ∑ b, ‖ρ (P.eval a + a * b)‖ ^ 2 = ∑ c, ‖ρ c‖ ^ 2 := by
      rw [← sum_mul_add_eq (ne_of_mem_erase ha) (P.eval a) fun c => ‖ρ c‖ ^ 2]
      exact sum_congr rfl fun b _ => by rw [add_comm]
    calc _ ≤ (√(∑ b, ‖ρ (P.eval a + a * b)‖ ^ 2) * √(∑ b, ‖h b‖ ^ 2)) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) hcs 2
      _ = _ := by
          rw [mul_pow, Real.sq_sqrt (sum_nonneg fun _ _ => sq_nonneg _),
            Real.sq_sqrt (sum_nonneg fun _ _ => sq_nonneg _), hρ]
  have h0 : ∑ b, ρ (P.eval 0 + 0 * b) * h b = 0 := by
    simp_rw [zero_mul, add_zero]
    rw [← mul_sum, hh, mul_zero]
  rw [← add_sum_erase _ _ (mem_univ (0 : ZMod p)), h0, norm_zero, zero_pow two_ne_zero,
    zero_add]
  refine (sum_le_sum hrow).trans_eq ?_
  rw [sum_const, card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul,
    Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one, mul_assoc]

/-- **Phase readouts of `P(a) + a b` are flat**: for `deg P = m ≥ 2`, `|ρ| = 1` and a centered
`h`, `|∑_{a, b} ρ(P(a) + a b) g(a) h(b)|² / (p² ‖g‖² ‖h‖²) ≤ (1 + √((m − 1)(2p − 1))) / p`. -/
theorem norm_sq_sum_poly_add_mul_div_le {P : (ZMod p)[X]} (hm : 2 ≤ P.natDegree)
    {ρ : ZMod p → ℂ} (hρ : ∀ c, ‖ρ c‖ = 1) (g h : ZMod p → ℂ) (hh : ∑ b, h b = 0) :
    ‖∑ a, ∑ b, ρ (P.eval a + a * b) * g a * h b‖ ^ 2 /
        ((p : ℝ) ^ 2 * (∑ a, ‖g a‖ ^ 2) * (∑ b, ‖h b‖ ^ 2)) ≤
      (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) / p := by
  have key := norm_sq_sum_poly_add_mul_le hm ρ g h hh
  have hρn : ∑ c, ‖ρ c‖ ^ 2 = p := by simp [hρ]
  rw [hρn] at key
  have hp : (0 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).pos
  have hR : 0 ≤ (1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1))) / p := by positivity
  have hD : 0 ≤ (p : ℝ) ^ 2 * (∑ a, ‖g a‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) := by positivity
  rcases hD.eq_or_lt with h0 | hpos
  · rw [← h0, div_zero]
    exact hR
  · rw [div_le_div_iff₀ hpos hp]
    nlinarith [mul_le_mul_of_nonneg_left key hp.le]

/-- `r = 1 − p δ₀ : ℤ/p → ℝ`, `−p` times the centered indicator of `0`. -/
noncomputable def centeredDelta (u : ZMod p) : ℝ := if u = 0 then 1 - p else 1

/-- `r(a u) = r(u)` for `a ≠ 0`. -/
lemma centeredDelta_mul {a : ZMod p} (ha : a ≠ 0) (u : ZMod p) :
    centeredDelta (a * u) = centeredDelta u := by
  simp only [centeredDelta, mul_eq_zero, ha, false_or]

/-- `r` is centered. -/
lemma sum_centeredDelta : ∑ u : ZMod p, (centeredDelta u : ℂ) = 0 := by
  rw [sum_eq_add_sub_one_mul (c := 1) fun u hu => by simp [centeredDelta, hu]]
  simp only [centeredDelta, ↓reduceIte]
  push_cast
  ring

/-- `‖r‖² > 0`. -/
lemma sum_sq_centeredDelta_pos : 0 < ∑ u : ZMod p, centeredDelta u ^ 2 := by
  have hp : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  refine lt_of_lt_of_le ?_ (single_le_sum (fun u _ => sq_nonneg (centeredDelta u)) (mem_univ 0))
  simp only [centeredDelta, ↓reduceIte]
  nlinarith

/-- Shifts preserve `‖r‖²`. -/
lemma sum_norm_sq_centeredDelta_add (t : ZMod p) :
    ∑ x, ‖(centeredDelta (x + t) : ℂ)‖ ^ 2 = ∑ u : ZMod p, centeredDelta u ^ 2 := by
  simp only [Complex.norm_real, Real.norm_eq_abs, sq_abs]
  exact Equiv.sum_comp (Equiv.addRight t) fun u => centeredDelta u ^ 2

/-- **Affine `P` reach the largest share**: for `deg P ≤ 1`, `ρ(c) = r(c − P(0))` and the
centered `h(b) = r(b + P'(0))` give `∑_a |∑_b ρ(P(a) + a b) h(b)|² = (p − 1) ‖ρ‖² ‖h‖²`. -/
theorem sum_norm_sq_sum_affine_add_mul_eq {P : (ZMod p)[X]} (hP : P.natDegree ≤ 1) :
    ∑ a, ‖∑ b, (centeredDelta (P.eval a + a * b + -P.coeff 0) : ℂ) *
        centeredDelta (b + P.coeff 1)‖ ^ 2 =
      (p - 1) * (∑ c, ‖(centeredDelta (c + -P.coeff 0) : ℂ)‖ ^ 2) *
        (∑ b, ‖(centeredDelta (b + P.coeff 1) : ℂ)‖ ^ 2) := by
  set Q := ∑ u : ZMod p, centeredDelta u ^ 2 with hQdef
  have hev : ∀ a, P.eval a = P.coeff 1 * a + P.coeff 0 := fun a => by
    conv_lhs => rw [eq_X_add_C_of_natDegree_le_one hP]
    simp
  have hshift : ∑ b, (centeredDelta (b + P.coeff 1) : ℂ) = 0 :=
    (Equiv.sum_comp (Equiv.addRight (P.coeff 1)) fun u => (centeredDelta u : ℂ)).trans
      sum_centeredDelta
  have hrow : ∀ a ∈ univ.erase (0 : ZMod p),
      ∑ b, (centeredDelta (P.eval a + a * b + -P.coeff 0) : ℂ) * centeredDelta (b + P.coeff 1) =
        (Q : ℂ) := by
    intro a ha
    have hQ : ∑ b, (centeredDelta (b + P.coeff 1) : ℂ) ^ 2 = (Q : ℂ) := by
      rw [hQdef]
      push_cast
      exact Equiv.sum_comp (Equiv.addRight (P.coeff 1)) fun u => (centeredDelta u : ℂ) ^ 2
    rw [← hQ]
    refine sum_congr rfl fun b _ => ?_
    rw [hev, show P.coeff 1 * a + P.coeff 0 + a * b + -P.coeff 0 = a * (b + P.coeff 1) by ring,
      centeredDelta_mul (ne_of_mem_erase ha), sq]
  have h0 : ∑ b, (centeredDelta (P.eval 0 + 0 * b + -P.coeff 0) : ℂ) *
      centeredDelta (b + P.coeff 1) = 0 := by
    simp_rw [zero_mul, add_zero, hev 0, mul_zero, zero_add, add_neg_cancel]
    rw [← mul_sum, hshift, mul_zero]
  rw [sum_norm_sq_centeredDelta_add, sum_norm_sq_centeredDelta_add,
    ← add_sum_erase _ _ (mem_univ (0 : ZMod p)), h0, norm_zero, zero_pow two_ne_zero, zero_add,
    sum_congr rfl fun a ha => by rw [hrow a ha], sum_const, card_erase_of_mem (mem_univ _),
    card_univ, ZMod.card, nsmul_eq_mul, Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one,
    Complex.norm_of_nonneg (sum_nonneg fun _ _ => sq_nonneg _)]
  ring

/-- **The readout-free dichotomy for `P(a) + a b`**: when `(m − 1)(2p − 1) < (p − 2)²` for
`m = deg P`, a readout `ρ` and a centered `h` with `‖ρ‖ ‖h‖ > 0` reach the largest share
`p − 1` iff `deg P ≤ 1`. -/
theorem exists_sum_norm_sq_eq_iff_natDegree_le_one {P : (ZMod p)[X]}
    (hp : ((P.natDegree : ℝ) - 1) * (2 * p - 1) < ((p : ℝ) - 2) ^ 2) :
    (∃ ρ h : ZMod p → ℂ, ∑ b, h b = 0 ∧ 0 < (∑ c, ‖ρ c‖ ^ 2) * (∑ b, ‖h b‖ ^ 2) ∧
      ∑ a, ‖∑ b, ρ (P.eval a + a * b) * h b‖ ^ 2 =
        (p - 1) * (∑ c, ‖ρ c‖ ^ 2) * (∑ b, ‖h b‖ ^ 2)) ↔ P.natDegree ≤ 1 := by
  constructor
  · rintro ⟨ρ, h, hh, hpos, heq⟩
    by_contra hd
    have hm : 2 ≤ P.natDegree := by omega
    have hle := sum_norm_sq_sum_poly_add_mul_le hm ρ h hh
    rw [heq] at hle
    have hp2 : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
    have hd2 : (2 : ℝ) ≤ P.natDegree := by exact_mod_cast hm
    have hκ : 0 ≤ ((P.natDegree : ℝ) - 1) * (2 * p - 1) :=
      mul_nonneg (by linarith) (by linarith)
    have hs : √(((P.natDegree : ℝ) - 1) * (2 * p - 1)) < (p : ℝ) - 2 :=
      (Real.sqrt_lt_sqrt hκ hp).trans_eq (Real.sqrt_sq (by linarith))
    have hlt := mul_lt_mul_of_pos_right (show 1 + √(((P.natDegree : ℝ) - 1) * (2 * p - 1)) <
      (p : ℝ) - 1 by linarith) hpos
    nlinarith
  · intro hP
    refine ⟨fun c => centeredDelta (c + -P.coeff 0), fun b => centeredDelta (b + P.coeff 1),
      (Equiv.sum_comp (Equiv.addRight (P.coeff 1)) fun u => (centeredDelta u : ℂ)).trans
        sum_centeredDelta, ?_, sum_norm_sq_sum_affine_add_mul_eq hP⟩
    rw [sum_norm_sq_centeredDelta_add, sum_norm_sq_centeredDelta_add]
    exact mul_pos sum_sq_centeredDelta_pos sum_sq_centeredDelta_pos

section Examples

attribute [local instance] fact_prime_5

example := norm_sq_sum_poly_add_mul_div_le (p := 5) (P := X ^ 2) (by rw [natDegree_X_pow])
  (ρ := fun _ => 1) (fun _ => norm_one) (fun _ => 1)
  (fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

example := centeredDelta_mul (p := 5) (a := 2) (by decide) 3

example := sum_norm_sq_centeredDelta_add (p := 5) 3

example := sum_norm_sq_sum_poly_add_mul_le_sub_one (p := 5) (X ^ 3) (fun c => (c.val : ℂ))
  (fun x => (if x = 0 then 1 else 0) - if x = 1 then 1 else 0) (by simp [sum_sub_distrib])

example := sum_norm_sq_sum_affine_add_mul_eq (p := 5) (P := 3 * X + 2)
  (natDegree_linear_le (a := 3) (b := 2))

end Examples

end Transformer.Grokking.FlatSpectrum
