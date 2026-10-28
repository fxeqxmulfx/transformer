module

public import Transformer.Grokking.FlatSpectrum.PolynomialMulChar
public import Transformer.Weil

/-!
# Theorem B for an odd-degree `P` and the Legendre symbol of `a b`

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p", here with `a³` replaced by a polynomial `P` of
odd degree `m` and `χ` the Legendre symbol `χ₂`. For every `c`, `c − P` has odd degree `m`, so
Weil's bound (`Transformer.Weil.norm_sum_quadraticChar_eval_le`) gives
`|∑_a χ₂(c − P(a))| ≤ (m − 1) √p` for odd `p`, and then
`‖E[χ₂(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`, of order `m²/p`.
For `P(a) = a³` the constant `m − 1 = 2` is the source's `2 √p`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Polynomial Finset

variable {p : ℕ} [Fact p.Prime]

/-- Weil's bound for `c − P`, `P` of odd degree `m`: `|∑_a χ₂(c − P(a))| ≤ (m − 1) √p` for
`p ≠ 2`. -/
theorem norm_sum_quadraticChar_sub_poly_le {P : (ZMod p)[X]} (hP : Odd P.natDegree)
    (hp : p ≠ 2) (c : ZMod p) :
    ‖∑ a, (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (c - P.eval a)‖ ≤
      ((P.natDegree : ℝ) - 1) * √p := by
  have hdeg : (C c - P).natDegree = P.natDegree :=
    natDegree_sub_eq_right_of_natDegree_lt (by rw [natDegree_C]; exact hP.pos)
  have h := Transformer.Weil.norm_sum_quadraticChar_eval_le hp (f := C c - P) (by rwa [hdeg])
  simpa only [hdeg, eval_sub, eval_C] using h

/-- **Theorem B for an odd-degree `P` and the Legendre symbol**, `review/flat_spectrum.md` with
`a³` replaced by `P(a)`, `P` of odd degree `m`, and `χ = χ₂`: for `p ≠ 2`,
`‖E[χ₂(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`. -/
theorem meanSq_condMean_poly_add_mul_quadraticChar_le {P : (ZMod p)[X]} (hP : Odd P.natDegree)
    (hp : p ≠ 2) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P.eval ab.1 + ab.1 * ab.2)
      fun ab => (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (ab.1 * ab.2)) ≤
      (((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / p ^ 2 + (p - 1) ^ 2 / (p ^ 2 * (2 * p - 1)) :=
  meanSq_condMean_poly_add_mul_mulChar_le (fun a => P.eval a) _
    fun c _ => norm_sum_quadraticChar_sub_poly_le hP hp c

section Examples

local instance fact_prime_97 : Fact (Nat.Prime 97) := ⟨by norm_num⟩

example := norm_sum_quadraticChar_sub_poly_le (p := 97) (P := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩) (by norm_num) 1

example := meanSq_condMean_poly_add_mul_quadraticChar_le (p := 97) (P := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩) (by norm_num)

end Examples

end Transformer.Grokking.FlatSpectrum
