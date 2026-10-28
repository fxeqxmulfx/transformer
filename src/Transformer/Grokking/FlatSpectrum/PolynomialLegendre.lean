module

public import Transformer.Grokking.FlatSpectrum.PolynomialMulChar
public import Transformer.Weil

/-!
# Theorem B for a polynomial `P` and the Legendre symbol of `a b`

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p", here with `a³` replaced by a polynomial `P` of
degree `m ≥ 1` and `χ` the Legendre symbol `χ₂`. For odd `p`, Weil's bound
(`Transformer.Weil.norm_sum_quadraticChar_eval_le_of_ne`) gives
`|∑_a χ₂(c − P(a))| ≤ (m − 1) √p` whenever `c − P` is not a constant times a square. For odd `m`
that is every `c`, and `‖E[χ₂(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)²/(p² (2p − 1))`,
of order `m²/p`. For even `m` at most one `c` fails (for `P = a²` and `c = 0`,
`|∑_a χ₂(−a²)| = p − 1`), and its fiber adds `(p − 1)/p²`. For `P(a) = a³` the constant
`m − 1 = 2` is the source's `2 √p`.
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

/-- At most one `c` makes `c − P` a constant times a square, for `P` of positive degree over a
field with `2 ≠ 0`: if `c₁ − P = λ₁ g₁²` and `c₂ − P = λ₂ g₂²`, the leading coefficients give
`λ₂ = λ₁ μ²`, so `c₁ − c₂ = λ₁ (g₁ − μ g₂)(g₁ + μ g₂)`; for `c₁ ≠ c₂` both factors are constant,
hence so is `g₁`, and so is `P`. -/
theorem eq_of_C_sub_eq_C_mul_sq {K : Type*} [Field K] (h2 : (2 : K) ≠ 0) {P : K[X]}
    (hP : 0 < P.natDegree) {c₁ c₂ l₁ l₂ : K} {g₁ g₂ : K[X]} (h₁ : C c₁ - P = C l₁ * g₁ ^ 2)
    (h₂ : C c₂ - P = C l₂ * g₂ ^ 2) : c₁ = c₂ := by
  have hdeg (c : K) : (C c - P).natDegree = P.natDegree :=
    natDegree_sub_eq_right_of_natDegree_lt (by rw [natDegree_C]; exact hP)
  have hlc (c : K) : (C c - P).leadingCoeff = -P.leadingCoeff :=
    leadingCoeff_sub_of_degree_lt' (degree_C_le.trans_lt (natDegree_pos_iff_degree_pos.mp hP))
  have hne {c l : K} {g : K[X]} (h : C c - P = C l * g ^ 2) : l ≠ 0 ∧ g ≠ 0 := by
    have h0 : C l * g ^ 2 ≠ 0 := fun h0 => by
      have := congrArg natDegree h
      rw [hdeg, h0, natDegree_zero] at this
      omega
    exact ⟨fun hl => h0 (by rw [hl, C_0, zero_mul]), fun hg => h0 (by rw [hg]; ring)⟩
  obtain ⟨hl₁, hg₁⟩ := hne h₁
  obtain ⟨hl₂, hg₂⟩ := hne h₂
  have hlc₁ := congrArg leadingCoeff h₁
  have hlc₂ := congrArg leadingCoeff h₂
  rw [hlc, leadingCoeff_mul, leadingCoeff_C, leadingCoeff_pow] at hlc₁ hlc₂
  have hb : g₂.leadingCoeff ≠ 0 := leadingCoeff_ne_zero.mpr hg₂
  set μ := g₁.leadingCoeff / g₂.leadingCoeff
  have hl : C l₂ = C l₁ * C μ ^ 2 := by
    rw [← C_pow, ← C_mul]
    congr 1
    simp only [μ]
    field_simp
    linear_combination hlc₁ - hlc₂
  have key : C (c₁ - c₂) = C l₁ * ((g₁ - C μ * g₂) * (g₁ + C μ * g₂)) := by
    rw [C_sub]
    linear_combination h₁ - h₂ - g₂ ^ 2 * hl
  by_contra hc
  have hAB : (g₁ - C μ * g₂) * (g₁ + C μ * g₂) ≠ 0 := fun h => by
    rw [h, mul_zero, C_eq_zero, sub_eq_zero] at key
    exact hc key
  have hdAB := congrArg natDegree key
  rw [natDegree_C, natDegree_C_mul hl₁,
    natDegree_mul (left_ne_zero_of_mul hAB) (right_ne_zero_of_mul hAB)] at hdAB
  have hg : (C 2 * g₁).natDegree = 0 := by
    rw [show C 2 * g₁ = (g₁ - C μ * g₂) + (g₁ + C μ * g₂) by rw [C_ofNat]; ring]
    exact Nat.le_zero.mp ((natDegree_add_le _ _).trans (by omega))
  rw [natDegree_C_mul h2] at hg
  have := congrArg natDegree h₁
  rw [hdeg, natDegree_C_mul hl₁, natDegree_pow, hg, mul_zero] at this
  omega

/-- **Theorem B for any `P` of positive degree and the Legendre symbol**,
`review/flat_spectrum.md` with `a³` replaced by `P(a)`, `P` of degree `m ≥ 1`, and `χ = χ₂`: for
`p ≠ 2`, `‖E[χ₂(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)/p² + (p − 1)²/(p² (2p − 1))`.
Weil's bound covers every `c` with `c − P` not a constant times a square; for even `m` one `c`
can fail, with a sum of order `p`, and its fiber adds `(p − 1)/p²`. -/
theorem meanSq_condMean_poly_add_mul_quadraticChar_le_of_natDegree_pos {P : (ZMod p)[X]}
    (hP : 0 < P.natDegree) (hp : p ≠ 2) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P.eval ab.1 + ab.1 * ab.2)
      fun ab => (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) (ab.1 * ab.2)) ≤
      (((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / p ^ 2 + (p - 1) / p ^ 2 +
        (p - 1) ^ 2 / (p ^ 2 * (2 * p - 1)) := by
  classical
  have h2 : (2 : ZMod p) ≠ 0 := Ring.two_ne_zero (by rwa [ZMod.ringChar_zmod_n])
  have hE : #(univ.filter fun c : ZMod p => ∃ l g, C c - P = C l * g ^ 2) ≤ 1 :=
    Finset.card_le_one.mpr fun c₁ h₁ c₂ h₂ => by
      obtain ⟨l₁, g₁, e₁⟩ := (mem_filter.mp h₁).2
      obtain ⟨l₂, g₂, e₂⟩ := (mem_filter.mp h₂).2
      exact eq_of_C_sub_eq_C_mul_sq h2 hP e₁ e₂
  have hb := meanSq_condMean_poly_add_mul_mulChar_le_of_finset (fun a => P.eval a)
    ((quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ))
    (univ.filter fun c : ZMod p => ∃ l g, C c - P = C l * g ^ 2)
    (B := ((P.natDegree : ℝ) - 1) * √p) fun c _ hc => by
      have hdeg : (C c - P).natDegree = P.natDegree :=
        natDegree_sub_eq_right_of_natDegree_lt (by rw [natDegree_C]; exact hP)
      have h := Transformer.Weil.norm_sum_quadraticChar_eval_le_of_ne hp (f := C c - P)
        fun l g e => hc (mem_filter.mpr ⟨mem_univ _, l, g, e⟩)
      simpa only [hdeg, eval_sub, eval_C] using h
  have hp1 : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_lt.le
  have hE' : (#(univ.filter fun c : ZMod p => ∃ l g, C c - P = C l * g ^ 2) : ℝ) ≤ 1 := by
    exact_mod_cast hE
  have key := div_le_div_of_nonneg_right (c := (p : ℝ) ^ 2)
    (mul_le_of_le_one_left (by linarith : (0 : ℝ) ≤ p - 1) hE') (by positivity)
  linarith

section Examples

local instance fact_prime_97 : Fact (Nat.Prime 97) := ⟨by norm_num⟩

example := norm_sum_quadraticChar_sub_poly_le (p := 97) (P := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩) (by norm_num) 1

example := meanSq_condMean_poly_add_mul_quadraticChar_le (p := 97) (P := X ^ 3)
  (by rw [natDegree_X_pow]; exact ⟨1, rfl⟩) (by norm_num)

example := eq_of_C_sub_eq_C_mul_sq (K := ZMod 97) (by decide) (P := X ^ 2)
  (by rw [natDegree_X_pow]; norm_num) (c₁ := 0) (c₂ := 0) (l₁ := -1) (l₂ := -1) (g₁ := X)
  (g₂ := X) (by simp) (by simp)

example := meanSq_condMean_poly_add_mul_quadraticChar_le_of_natDegree_pos (p := 97) (P := X ^ 2)
  (by rw [natDegree_X_pow]; norm_num) (by norm_num)

end Examples

end Transformer.Grokking.FlatSpectrum
