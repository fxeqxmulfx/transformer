module

public import Transformer.Grokking.FlatSpectrum.PolynomialFibers
public import Transformer.Weil

/-!
# Theorem B for a polynomial `P`: the additive readout, by Weil's bound

Source: `review/flat_spectrum.md`, Theorem B, which takes `P(a) = a³` and bounds
`S_r = ∑_x ψ(r x³)` by `2 √p`. Here `P` is a polynomial of degree `m ≥ 1` prime to `p`, as every
`P` of degree `1 ≤ m < p` is. For `r ≠ 0`, `r P` has degree `m`, so Weil's bound
(`Transformer.Weil.norm_sum_addChar_eval_le`) gives `|∑_x ψ(r P(x))| ≤ (m − 1) √p`, and then
`‖E[ψ_r(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p + (m − 1) √p + 1)²/(p² (2p − 1))`,
of order `m²/p`. For `P(a) = a³` and `p ≠ 3`, `(m − 1) √p = 2 √p` is the source's bound on
`S_r`. By Cauchy–Schwarz, a readout `φ` of the label correlates with `ψ_r(a b)` by at most its
norm times the square root of this bound.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Polynomial Finset

variable {p : ℕ} [Fact p.Prime]

/-- Weil's bound for `r P`: `|∑_x ψ(r P(x))| ≤ (m − 1) √p` for `r ≠ 0` and `P` of degree
`m ≥ 1` prime to `p`. -/
theorem norm_sum_stdAddChar_mul_poly_le {P : (ZMod p)[X]} (hm : 1 ≤ P.natDegree)
    (hpm : ¬p ∣ P.natDegree) {r : ZMod p} (hr : r ≠ 0) :
    ‖∑ x, ZMod.stdAddChar (r * P.eval x)‖ ≤ ((P.natDegree : ℝ) - 1) * √p := by
  have hdeg : (C r * P).natDegree = P.natDegree := natDegree_C_mul hr
  have h := Transformer.Weil.norm_sum_addChar_eval_le Transformer.Weil.stdAddChar_ne_one
    (f := C r * P) (by rwa [hdeg]) (by rwa [hdeg])
  simp only [eval_mul, eval_C] at h
  rwa [hdeg] at h

/-- **Theorem B for a polynomial `P`**, `review/flat_spectrum.md` with `a³` replaced by `P(a)`,
`P` of degree `m ≥ 1` prime to `p`: for `r ≠ 0`,
`‖E[ψ_r(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p + (m − 1) √p + 1)²/(p² (2p − 1))`. -/
theorem meanSq_condMean_poly_add_mul_stdAddChar_le {P : (ZMod p)[X]} (hm : 1 ≤ P.natDegree)
    (hpm : ¬p ∣ P.natDegree) {r : ZMod p} (hr : r ≠ 0) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P.eval ab.1 + ab.1 * ab.2)
      fun ab => ZMod.stdAddChar (r * (ab.1 * ab.2))) ≤
      (((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / p ^ 2 +
        (p + ((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / (p ^ 2 * (2 * p - 1)) :=
  meanSq_condMean_poly_add_mul_le (fun a => P.eval a) (norm_sum_stdAddChar_mul_poly_le hm hpm hr)

/-- Theorem B for a polynomial `P`, Cauchy–Schwarz: for every readout `φ` of the label
`f = P(a) + a b`, `|E[conj φ(f) ψ_r(a b)]| ≤ ‖φ(f)‖ √β` with `β` the bound above. -/
theorem norm_mean_star_mul_poly_add_mul_le {P : (ZMod p)[X]} (hm : 1 ≤ P.natDegree)
    (hpm : ¬p ∣ P.natDegree) {r : ZMod p} (hr : r ≠ 0) (φ : ZMod p → ℂ) :
    ‖mean fun ab : ZMod p × ZMod p =>
      star (φ (P.eval ab.1 + ab.1 * ab.2)) * ZMod.stdAddChar (r * (ab.1 * ab.2))‖ ≤
      √(meanSq fun ab : ZMod p × ZMod p => φ (P.eval ab.1 + ab.1 * ab.2)) *
        √((((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / p ^ 2 +
          (p + ((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / (p ^ 2 * (2 * p - 1))) :=
  (norm_mean_star_mul_le (fun ab : ZMod p × ZMod p => P.eval ab.1 + ab.1 * ab.2) _ φ).trans
    (mul_le_mul_of_nonneg_left
      (Real.sqrt_le_sqrt (meanSq_condMean_poly_add_mul_stdAddChar_le hm hpm hr))
      (Real.sqrt_nonneg _))

section Examples

local instance fact_prime_97 : Fact (Nat.Prime 97) := ⟨by norm_num⟩

example := norm_sum_stdAddChar_mul_poly_le (p := 97) (P := X ^ 4)
  (by rw [natDegree_X_pow]; norm_num) (by rw [natDegree_X_pow]; norm_num) one_ne_zero

example := meanSq_condMean_poly_add_mul_stdAddChar_le (p := 97) (P := X ^ 4)
  (by rw [natDegree_X_pow]; norm_num) (by rw [natDegree_X_pow]; norm_num) one_ne_zero

example := norm_mean_star_mul_poly_add_mul_le (p := 97) (P := X ^ 4)
  (by rw [natDegree_X_pow]; norm_num) (by rw [natDegree_X_pow]; norm_num) one_ne_zero fun _ => 1

end Examples

end Transformer.Grokking.FlatSpectrum
