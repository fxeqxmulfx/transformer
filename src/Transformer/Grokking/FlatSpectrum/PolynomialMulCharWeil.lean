module

public import Transformer.Grokking.FlatSpectrum.PolynomialMulChar
public import Transformer.Weil

/-!
# Theorem B for a polynomial `P` and a character of order prime to `deg P`

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p", here with `a³` replaced by a polynomial `P` of
degree `m ≥ 1` and `χ ≠ 1` a multiplicative character whose order `d` is prime to `m`. Every
`c − P` has degree `m`, so Weil's bound
(`Transformer.Weil.norm_sum_mulChar_eval_le_of_coprime_orderOf`) gives
`|∑_a χ(c − P(a))| ≤ (m − 1) √p` for every `c`, and
`‖E[χ(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`, of order `m²/p`.
For `d = 2` this is the odd-degree case of `PolynomialLegendre`.

For `d` and `m` with a common factor, `PolynomialMulCharAnyOrder` proves Weil's bound for every
`c` but one, and the readout bound with an added `(p − 1)/p²`. The excluded fiber can fail
Weil's bound: for `d ∣ m`, `P = (a − 1)^d` and `c = 0` give `|∑_a χ(−(a − 1)^d)| = p − 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Polynomial Finset

variable {p : ℕ} [Fact p.Prime]

/-- Weil's bound for `c − P`, `χ ≠ 1` of order prime to `m = deg P ≥ 1`:
`|∑_a χ(c − P(a))| ≤ (m − 1) √p`. -/
theorem norm_sum_mulChar_sub_poly_le_of_coprime {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1)
    {P : (ZMod p)[X]} (hP : 0 < P.natDegree) (hdm : Nat.Coprime (orderOf χ) P.natDegree)
    (c : ZMod p) :
    ‖∑ a, χ (c - P.eval a)‖ ≤ ((P.natDegree : ℝ) - 1) * √p := by
  have hdeg : (C c - P).natDegree = P.natDegree :=
    natDegree_sub_eq_right_of_natDegree_lt (by rw [natDegree_C]; exact hP)
  have h := Transformer.Weil.norm_sum_mulChar_eval_le_of_coprime_orderOf hχ (f := C c - P)
    (by rw [hdeg]; exact hP) (by rwa [hdeg])
  simpa only [hdeg, eval_sub, eval_C] using h

/-- **Theorem B for `P` and a character of order prime to `deg P`**, `review/flat_spectrum.md`
with `a³` replaced by `P(a)`, `P` of degree `m ≥ 1`, and `χ ≠ 1` of order prime to `m`:
`‖E[χ(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`. -/
theorem meanSq_condMean_poly_add_mul_mulChar_le_of_coprime {χ : MulChar (ZMod p) ℂ}
    (hχ : χ ≠ 1) {P : (ZMod p)[X]} (hP : 0 < P.natDegree)
    (hdm : Nat.Coprime (orderOf χ) P.natDegree) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P.eval ab.1 + ab.1 * ab.2)
      fun ab => χ (ab.1 * ab.2)) ≤
      (((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / p ^ 2 + (p - 1) ^ 2 / (p ^ 2 * (2 * p - 1)) :=
  meanSq_condMean_poly_add_mul_mulChar_le (fun a => P.eval a) χ
    fun c _ => norm_sum_mulChar_sub_poly_le_of_coprime hχ hP hdm c

section Examples

attribute [local instance] Transformer.Weil.fact_prime_71

/-- For a character `χ` of order `5` of `𝔽₇₁`: `|∑_a χ(1 − a²)| ≤ √71`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧ ‖∑ a : ZMod 71, χ (1 - a ^ 2)‖ ≤ √71 := by
  obtain ⟨χ, hχ⟩ := Transformer.Weil.exists_mulChar_zmod71_orderOf_eq_five
  refine ⟨χ, hχ, ?_⟩
  have h := norm_sum_mulChar_sub_poly_le_of_coprime (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (P := X ^ 2)
    (by rw [natDegree_X_pow]; norm_num) (by rw [hχ, natDegree_X_pow]; norm_num) 1
  simp only [natDegree_X_pow, eval_pow, eval_X] at h
  norm_num at h
  exact h

/-- **Theorem B for a character `χ` of order `5` of `𝔽₇₁`** and `P(a) = a²`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧
    meanSq (condMean (fun ab : ZMod 71 × ZMod 71 => ab.1 ^ 2 + ab.1 * ab.2)
      fun ab => χ (ab.1 * ab.2)) ≤ (√71 + 1) ^ 2 / 71 ^ 2 + 70 ^ 2 / (71 ^ 2 * 141) := by
  obtain ⟨χ, hχ⟩ := Transformer.Weil.exists_mulChar_zmod71_orderOf_eq_five
  refine ⟨χ, hχ, ?_⟩
  have h := meanSq_condMean_poly_add_mul_mulChar_le_of_coprime (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (P := X ^ 2)
    (by rw [natDegree_X_pow]; norm_num) (by rw [hχ, natDegree_X_pow]; norm_num)
  simp only [natDegree_X_pow, eval_pow, eval_X] at h
  refine h.trans_eq ?_
  push_cast
  ring

end Examples

end Transformer.Grokking.FlatSpectrum
