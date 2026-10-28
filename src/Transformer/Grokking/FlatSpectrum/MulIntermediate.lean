module

public import Transformer.Grokking.FlatSpectrum.Intermediate
public import Transformer.Grokking.FlatSpectrum.JacobiSums

/-!
# Theorem B for multiplicative characters of `a b`

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p." On the fiber `a³ + a b = c` the line `a = 0`
contributes `χ(0) = 0`, and each `a ≠ 0` one point with `a b = c − a³`, so the fiber sum of
`χ(a b)` is `∑_{a ≠ 0} χ(c − a³) = T_c − χ(c)` with `T_c = ∑_a χ(c − a³)`. For nontrivial `χ`,
`|T_c| ≤ 2√p` for `c ≠ 0` and the fiber sum at `c = 0` has modulus at most `p − 1`, so
`‖E[χ(a b) | f]‖² ≤ (2√p + 1)²/p² + (p − 1)²/(p² (2p − 1))`. That is below the bound
`9/(2p) + 8/p^{3/2}` proved for the additive characters `ψ_r(a b)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- The fiber sum `∑_{a³ + a b = c} χ(a b) = ∑_{a ≠ 0} χ(c − a³)`. -/
lemma sum_fiber_cube_add_mul_mulChar (χ : MulChar (ZMod p) ℂ) (c : ZMod p) :
    ∑ ab : ZMod p × ZMod p with ab.1 ^ 3 + ab.1 * ab.2 = c, χ (ab.1 * ab.2) =
      ∑ a ∈ univ.erase 0, χ (c - a ^ 3) := by
  have hrow (a : ZMod p) (ha : a ≠ 0) :
      ∑ b : ZMod p, (if a ^ 3 + a * b = c then χ (a * b) else 0) = χ (c - a ^ 3) := by
    simp_rw [cube_add_mul_eq_iff ha]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true]
    rw [← mul_assoc, mul_inv_cancel₀ ha, one_mul]
  rw [Finset.sum_filter, Fintype.sum_prod_type, ← Finset.add_sum_erase univ _ (mem_univ 0),
    Finset.sum_congr rfl fun a ha => hrow a (ne_of_mem_erase ha)]
  simp [MulChar.map_zero]

example := @sum_fiber_cube_add_mul_mulChar 5 ⟨Nat.prime_five⟩ 1 0

/-- `∑_{a ≠ 0} χ(c − a³) = ∑_a χ(c − a³) − χ(c)`. -/
lemma sum_erase_mulChar_sub_cube (χ : MulChar (ZMod p) ℂ) (c : ZMod p) :
    ∑ a ∈ univ.erase 0, χ (c - a ^ 3) = ∑ a, χ (c - a ^ 3) - χ c := by
  rw [← Finset.add_sum_erase univ _ (mem_univ 0)]
  simp

example := @sum_erase_mulChar_sub_cube 5 ⟨Nat.prime_five⟩ 1 0

/-- `|∑_{a ≠ 0} χ(c − a³)| ≤ p − 1`. -/
lemma norm_sum_erase_mulChar_sub_cube_le (χ : MulChar (ZMod p) ℂ) (c : ZMod p) :
    ‖∑ a ∈ univ.erase 0, χ (c - a ^ 3)‖ ≤ p - 1 := by
  refine (norm_sum_le _ _).trans ((Finset.sum_le_card_nsmul _ _ 1 fun a _ =>
    DirichletCharacter.norm_le_one χ _).trans_eq ?_)
  rw [card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul, mul_one,
    Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]

example := @norm_sum_erase_mulChar_sub_cube_le 5 ⟨Nat.prime_five⟩ 1 0

variable {χ : MulChar (ZMod p) ℂ}

/-- **Theorem B for multiplicative characters**, `review/flat_spectrum.md`: "Multiplicative
characters of ab are the same order". For nontrivial `χ`,
`‖E[χ(a b) | f]‖² ≤ 9/(2p) + 8/p^{3/2}`, the bound of `meanSq_condMean_cube_add_mul_le`. -/
theorem meanSq_condMean_cube_add_mul_mulChar_le (hχ : χ ≠ 1) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => ab.1 ^ 3 + ab.1 * ab.2)
      fun ab => χ (ab.1 * ab.2)) ≤ 9 / (2 * p) + 8 / (p * √p) := by
  have hp := (Fact.out : p.Prime).one_lt
  have hp1 : (1 : ℝ) < p := by exact_mod_cast hp
  have hp2 : (2 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).two_le
  have hq : 0 ≤ √(p : ℝ) := Real.sqrt_nonneg _
  have hterm (c : ZMod p) (hc : c ≠ 0) :
      ‖∑ a ∈ univ.erase 0, χ (c - a ^ 3)‖ ^ 2 / ((p : ℝ) - 1 + if c = 0 then (p : ℝ) else 0) ≤
        (2 * √p + 1) ^ 2 / (p - 1) := by
    rw [ite_eq_right hc, add_zero]
    gcongr
    rw [sum_erase_mulChar_sub_cube]
    refine (norm_sub_le _ _).trans ?_
    rw [norm_mulChar_eq_one χ hc]
    linarith [norm_sum_mulChar_sub_cube_le hχ hc]
  have hzero : ‖∑ a ∈ univ.erase (0 : ZMod p), χ (0 - a ^ 3)‖ ^ 2 /
      ((p : ℝ) - 1 + if (0 : ZMod p) = 0 then (p : ℝ) else 0) ≤ (p - 1) ^ 2 / (2 * p - 1) := by
    rw [ite_eq_left rfl, show (p : ℝ) - 1 + p = 2 * p - 1 by ring]
    exact div_le_div_of_nonneg_right
      (pow_le_pow_left₀ (norm_nonneg _) (norm_sum_erase_mulChar_sub_cube_le χ 0) 2) (by linarith)
  have hsum : ∑ c ∈ univ.erase (0 : ZMod p), ‖∑ a ∈ univ.erase 0, χ (c - a ^ 3)‖ ^ 2 /
      ((p : ℝ) - 1 + if c = 0 then (p : ℝ) else 0) ≤ (2 * √p + 1) ^ 2 := by
    refine (Finset.sum_le_sum fun c hc => hterm c (ne_of_mem_erase hc)).trans_eq ?_
    rw [Finset.sum_const, card_erase_of_mem (mem_univ _), card_univ, ZMod.card, nsmul_eq_mul,
      Nat.cast_sub hp.le, Nat.cast_one]
    have h1 : (p : ℝ) - 1 ≠ 0 := by linarith
    field_simp
  rw [meanSq, sum_norm_condMean_sq]
  simp_rw [sum_fiber_cube_add_mul_mulChar, card_fiber_cube_add_mul]
  rw [← Finset.add_sum_erase _ _ (mem_univ (0 : ZMod p)), Fintype.card_prod, ZMod.card]
  refine (div_le_div_of_nonneg_right (add_le_add hzero hsum) (by positivity)).trans
    (le_trans ?_ (cube_bound_le hp2))
  have h0 : (p : ℝ) ≠ 0 := by positivity
  have h3 : 2 * (p : ℝ) - 1 ≠ 0 := by linarith
  rw [show ((p - 1) ^ 2 / (2 * p - 1) + (2 * √p + 1) ^ 2) / ((p * p : ℕ) : ℝ) =
      (2 * √p + 1) ^ 2 / p ^ 2 + (p - 1) ^ 2 / (p ^ 2 * (2 * p - 1)) by
    push_cast
    field_simp
    ring]
  have hle : ((p : ℝ) - 1) ^ 2 ≤ ((p : ℝ) - 1 + 2 * √p) ^ 2 :=
    pow_le_pow_left₀ (by linarith) (by linarith) 2
  have hden : (0 : ℝ) ≤ (p : ℝ) ^ 2 * (2 * p - 1) := mul_nonneg (by positivity) (by linarith)
  exact add_le_add le_rfl (div_le_div_of_nonneg_right hle hden)

example := @meanSq_condMean_cube_add_mul_mulChar_le 7 ⟨Nat.prime_seven⟩ _
  (@quadraticChar_ringHomComp_ne_one 7 ⟨Nat.prime_seven⟩ (by decide))

/-- The corollary of Theorem B for multiplicative characters: for nontrivial `χ` and every
readout `φ`, `|E[conj φ(f) χ(a b)]| ≤ ‖φ(f)‖ √(9/(2p) + 8/p^{3/2})`, by Cauchy–Schwarz. -/
theorem norm_mean_star_mul_cube_add_mul_mulChar_le (hχ : χ ≠ 1) (φ : ZMod p → ℂ) :
    ‖mean fun ab : ZMod p × ZMod p =>
      star (φ (ab.1 ^ 3 + ab.1 * ab.2)) * χ (ab.1 * ab.2)‖ ≤
      √(meanSq fun ab : ZMod p × ZMod p => φ (ab.1 ^ 3 + ab.1 * ab.2)) *
        √(9 / (2 * p) + 8 / (p * √p)) :=
  (norm_mean_star_mul_le (fun ab : ZMod p × ZMod p => ab.1 ^ 3 + ab.1 * ab.2) _ φ).trans
    (mul_le_mul_of_nonneg_left
      (Real.sqrt_le_sqrt (meanSq_condMean_cube_add_mul_mulChar_le hχ)) (Real.sqrt_nonneg _))

example := @norm_mean_star_mul_cube_add_mul_mulChar_le 7 ⟨Nat.prime_seven⟩ _
  (@quadraticChar_ringHomComp_ne_one 7 ⟨Nat.prime_seven⟩ (by decide)) fun _ => 1

end Transformer.Grokking.FlatSpectrum
