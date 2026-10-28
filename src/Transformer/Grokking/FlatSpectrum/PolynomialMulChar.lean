module

public import Transformer.Grokking.FlatSpectrum.PolynomialFibers

/-!
# Theorem B for any `P` and a multiplicative character of `a b`

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p", here with `a³` replaced by any function `P` of `a`.
On the fiber `P(a) + a b = c` the line `a = 0` contributes `χ(0) = 0`, and each `a ≠ 0` one
point with `a b = c − P(a)`, so the fiber sum of `χ(a b)` is `∑_{a ≠ 0} χ(c − P(a))`. If
`|∑_a χ(c − P(a))| ≤ B` for every `c ≠ P(0)`, then
`‖E[χ(a b) | f]‖² ≤ (B + 1)²/p² + (p − 1)²/(p² (2p − 1))`: the fibers `c ≠ P(0)` give
`(B + 1)²/(p − 1)` each, and the fiber `c = P(0)` at most `(p − 1)²/(2p − 1)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- The fiber sum `∑_{P(a) + a b = c} χ(a b) = ∑_{a ≠ 0} χ(c − P(a))`. -/
lemma sum_fiber_poly_add_mul_mulChar (P : ZMod p → ZMod p) (χ : MulChar (ZMod p) ℂ)
    (c : ZMod p) :
    ∑ ab : ZMod p × ZMod p with P ab.1 + ab.1 * ab.2 = c, χ (ab.1 * ab.2) =
      ∑ a ∈ univ.erase 0, χ (c - P a) := by
  have hrow (a : ZMod p) (ha : a ≠ 0) :
      ∑ b : ZMod p, (if P a + a * b = c then χ (a * b) else 0) = χ (c - P a) := by
    simp_rw [poly_add_mul_eq_iff P ha]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    rw [← mul_assoc, mul_inv_cancel₀ ha, one_mul]
  rw [Finset.sum_filter, Fintype.sum_prod_type, ← Finset.add_sum_erase univ _ (mem_univ 0),
    Finset.sum_congr rfl fun a ha => hrow a (ne_of_mem_erase ha)]
  simp [MulChar.map_zero]

example := @sum_fiber_poly_add_mul_mulChar 5 ⟨Nat.prime_five⟩ (· ^ 3) 1 0

/-- `|∑_{a ≠ 0} χ(F a)| ≤ p − 1`. -/
lemma norm_sum_erase_mulChar_le (χ : MulChar (ZMod p) ℂ) (F : ZMod p → ZMod p) :
    ‖∑ a ∈ univ.erase 0, χ (F a)‖ ≤ p - 1 := by
  refine (norm_sum_le _ _).trans ((Finset.sum_le_card_nsmul _ _ 1 fun a _ =>
    DirichletCharacter.norm_le_one χ _).trans_eq ?_)
  rw [card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul, mul_one,
    Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]

example := @norm_sum_erase_mulChar_le 5 ⟨Nat.prime_five⟩ 1 (· ^ 3)

/-- **Theorem B for any `P` and a multiplicative character**, `review/flat_spectrum.md` with
`a³` replaced by `P(a)`: if `|∑_a χ(c − P(a))| ≤ B` for every `c ≠ P(0)`, then
`‖E[χ(a b) | f]‖² ≤ (B + 1)²/p² + (p − 1)²/(p² (2p − 1))`. -/
theorem meanSq_condMean_poly_add_mul_mulChar_le (P : ZMod p → ZMod p) (χ : MulChar (ZMod p) ℂ)
    {B : ℝ} (hB : ∀ c, c ≠ P 0 → ‖∑ a, χ (c - P a)‖ ≤ B) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P ab.1 + ab.1 * ab.2)
      fun ab => χ (ab.1 * ab.2)) ≤ (B + 1) ^ 2 / p ^ 2 + (p - 1) ^ 2 / (p ^ 2 * (2 * p - 1)) := by
  have hp := (Fact.out : p.Prime).one_lt
  have hp1 : (1 : ℝ) < p := by exact_mod_cast hp
  have hterm (c : ZMod p) (hc : c ≠ P 0) :
      ‖∑ a ∈ univ.erase 0, χ (c - P a)‖ ^ 2 / ((p : ℝ) - 1 + if c = P 0 then (p : ℝ) else 0) ≤
        (B + 1) ^ 2 / (p - 1) := by
    rw [ite_eq_right hc, add_zero]
    have hle : ‖∑ a ∈ univ.erase 0, χ (c - P a)‖ ≤ B + 1 := by
      rw [Finset.sum_erase_eq_sub (mem_univ 0)]
      exact (norm_sub_le _ _).trans
        (add_le_add (hB c hc) (DirichletCharacter.norm_le_one χ _))
    exact div_le_div_of_nonneg_right (pow_le_pow_left₀ (norm_nonneg _) hle 2) (by linarith)
  have hzero : ‖∑ a ∈ univ.erase 0, χ (P 0 - P a)‖ ^ 2 /
      ((p : ℝ) - 1 + if P 0 = P 0 then (p : ℝ) else 0) ≤ (p - 1) ^ 2 / (2 * p - 1) := by
    rw [ite_eq_left rfl, show (p : ℝ) - 1 + p = 2 * p - 1 by ring]
    exact div_le_div_of_nonneg_right
      (pow_le_pow_left₀ (norm_nonneg _) (norm_sum_erase_mulChar_le χ _) 2) (by linarith)
  have hsum : ∑ c ∈ univ.erase (P 0), ‖∑ a ∈ univ.erase 0, χ (c - P a)‖ ^ 2 /
      ((p : ℝ) - 1 + if c = P 0 then (p : ℝ) else 0) ≤ (B + 1) ^ 2 := by
    refine (Finset.sum_le_sum fun c hc => hterm c (ne_of_mem_erase hc)).trans_eq ?_
    rw [Finset.sum_const, card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul,
      Nat.cast_sub hp.le, Nat.cast_one]
    have h1 : (p : ℝ) - 1 ≠ 0 := by linarith
    field_simp
  rw [meanSq, sum_norm_condMean_sq]
  simp_rw [sum_fiber_poly_add_mul_mulChar, card_fiber_poly_add_mul]
  rw [← Finset.add_sum_erase _ _ (mem_univ (P 0)), Fintype.card_prod, ZMod.card]
  refine (div_le_div_of_nonneg_right (add_le_add hzero hsum) (by positivity)).trans_eq ?_
  have h0 : (p : ℝ) ≠ 0 := by positivity
  have h3 : 2 * (p : ℝ) - 1 ≠ 0 := by linarith
  push_cast
  field_simp
  ring

example := @meanSq_condMean_poly_add_mul_mulChar_le 7 ⟨Nat.prime_seven⟩ (· ^ 3) _ _
  fun c hc => @norm_sum_mulChar_sub_cube_le 7 ⟨Nat.prime_seven⟩ _
    (@quadraticChar_ringHomComp_ne_one 7 ⟨Nat.prime_seven⟩ (by decide)) c (by simpa using hc)

end Transformer.Grokking.FlatSpectrum
