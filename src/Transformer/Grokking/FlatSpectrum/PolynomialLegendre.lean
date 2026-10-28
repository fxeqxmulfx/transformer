module

public import Transformer.Grokking.FlatSpectrum.PolynomialMulChar
public import Transformer.Weil

/-!
# Theorem B for an odd-degree `P` and the Legendre symbol of `a b`

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p", here with `a³` replaced by a polynomial `P` of
odd degree `m` and `χ` the Legendre symbol `χ₂`. For every `c`, `c − P` has odd degree `m`, so
Stepanov's bound (`Transformer.Weil.abs_sum_quadraticChar_le`) gives
`|∑_a χ₂(c − P(a))| ≤ 7 m √p` for `p ≥ 64` and `8 m ≤ p`, and then
`‖E[χ₂(a b) | P(a) + a b]‖² ≤ (7 m √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`, of order `m²/p`.
The constant `7 m` is not sharp: for `P(a) = a³` the Jacobi-sum bound `2 √p` is better, and
the bound here is below `1` only once `p > 49 m²`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Polynomial Finset

variable {p : ℕ} [Fact p.Prime]

/-- Weil's bound for `c − P`, `P` of odd degree `m`: `|∑_a χ₂(c − P(a))| ≤ 7 m √p` for
`p ≥ 64` and `8 m ≤ p`. -/
theorem norm_sum_quadraticChar_sub_poly_le {P : (ZMod p)[X]} (hP : Odd P.natDegree)
    (h64 : 64 ≤ p) (h8 : 8 * P.natDegree ≤ p) (c : ZMod p) :
    ‖∑ a, (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (c - P.eval a)‖ ≤
      7 * P.natDegree * √p := by
  have hdeg : (C c - P).natDegree = P.natDegree :=
    natDegree_sub_eq_right_of_natDegree_lt (by rw [natDegree_C]; exact hP.pos)
  have hF : ringChar (ZMod p) ≠ 2 := by rw [ZMod.ringChar_zmod_n]; omega
  have h := Transformer.Weil.abs_sum_quadraticChar_le hF (f := C c - P) (by rwa [hdeg])
    (by rwa [ZMod.card]) (by rwa [ZMod.card, hdeg])
  rw [hdeg, ZMod.card] at h
  have hsum : ∑ a, (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (c - P.eval a) =
      ((∑ x, quadraticChar (ZMod p) ((C c - P).eval x) : ℤ) : ℂ) := by
    simp [MulChar.ringHomComp_apply]
  rw [hsum, Complex.norm_intCast]
  exact h

/-- **Theorem B for an odd-degree `P` and the Legendre symbol**, `review/flat_spectrum.md` with
`a³` replaced by `P(a)`, `P` of odd degree `m`, and `χ = χ₂`: for `p ≥ 64` and `8 m ≤ p`,
`‖E[χ₂(a b) | P(a) + a b]‖² ≤ (7 m √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`. -/
theorem meanSq_condMean_poly_add_mul_quadraticChar_le {P : (ZMod p)[X]} (hP : Odd P.natDegree)
    (h64 : 64 ≤ p) (h8 : 8 * P.natDegree ≤ p) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P.eval ab.1 + ab.1 * ab.2)
      fun ab => (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (ab.1 * ab.2)) ≤
      (7 * P.natDegree * √p + 1) ^ 2 / p ^ 2 + (p - 1) ^ 2 / (p ^ 2 * (2 * p - 1)) :=
  meanSq_condMean_poly_add_mul_mulChar_le (fun a => P.eval a) _
    fun c _ => norm_sum_quadraticChar_sub_poly_le hP h64 h8 c

section Examples

local instance fact_prime_97 : Fact (Nat.Prime 97) := ⟨by norm_num⟩

example := norm_sum_quadraticChar_sub_poly_le (p := 97) (P := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩) (by norm_num) (by rw [natDegree_X_pow]; norm_num) 1

example := meanSq_condMean_poly_add_mul_quadraticChar_le (p := 97) (P := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩) (by norm_num) (by rw [natDegree_X_pow]; norm_num)

end Examples

end Transformer.Grokking.FlatSpectrum
