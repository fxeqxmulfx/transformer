module

public import Transformer.Grokking.FlatSpectrum.PolynomialMulChar
public import Transformer.Grokking.FlatSpectrum.PowerfulFibers
public import Transformer.Weil

/-!
# Theorem B for a polynomial `P` and every character `χ ≠ 1`

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p", here with `a³` replaced by a polynomial `P` of
degree `m ≥ 1` and `χ ≠ 1` of any order `d`, which may share a factor with `m`.

Weil's bound (`Transformer.Weil.norm_sum_mulChar_eval_le_of_pow_mul`) gives
`|∑_a χ(c − P(a))| ≤ (m − 1) √p` as soon as `c − P` has an irreducible factor `R` with
`R² ∤ c − P`. For `P' ≠ 0` this holds for every `c` but one (`PowerfulFibers`). For `P' = 0`,
`p ∣ m`, and the trivial bound `p ≤ (m − 1) √p` holds for every `c`. The one fiber left adds
`(p − 1)/p²`, as for the Legendre symbol and even `m` (`PolynomialLegendre`):
`‖E[χ(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)/p² + (p − 1)²/(p² (2p − 1))`.
For `d ∣ m` that fiber can fail Weil's bound: `P = (a − 1)^d` and `c = 0` give
`|∑_a χ(−(a − 1)^d)| = p − 1`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Polynomial Finset

variable {p : ℕ} [Fact p.Prime]

/-- Weil's bound for `c − P` with an irreducible factor `R`, `R² ∤ c − P`, and any `χ ≠ 1`:
`|∑_a χ(c − P(a))| ≤ (m − 1) √p`. -/
theorem norm_sum_mulChar_sub_poly_le_of_not_forall {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1)
    {P : (ZMod p)[X]} (hP : 0 < P.natDegree) {c : ZMod p}
    (hc : ¬∀ R : (ZMod p)[X], Irreducible R → R ∣ C c - P → R ^ 2 ∣ C c - P) :
    ‖∑ a, χ (c - P.eval a)‖ ≤ ((P.natDegree : ℝ) - 1) * √p := by
  push Not at hc
  obtain ⟨R, hR, hRf, hR2⟩ := hc
  obtain ⟨Q, hQ⟩ := normalize_dvd_iff.mpr hRf
  have hRQ : ¬normalize R ∣ Q := fun ⟨Q', hQ'⟩ => hR2
    (((normalize_associated R).pow_pow (n := 2)).dvd_iff_dvd_left.mp ⟨Q', by rw [hQ, hQ']; ring⟩)
  have hdeg : (C c - P).natDegree = P.natDegree :=
    natDegree_sub_eq_right_of_natDegree_lt (by rw [natDegree_C]; exact hP)
  have h := Transformer.Weil.norm_sum_mulChar_eval_le_of_pow_mul hχ (monic_normalize hR.ne_zero)
    ((normalize_associated R).symm.irreducible hR) hRQ (e := 1) (Nat.coprime_one_right _)
  rw [pow_one, ← hQ, hdeg] at h
  simpa only [eval_sub, eval_C] using h

/-- **The trivial bound for `P' = 0`**: then `p ∣ m = deg P ≥ 1`, and
`|∑_a χ(c − P(a))| ≤ p ≤ (m − 1) √p` for `χ ≠ 1`, which needs `p ≥ 3`. -/
theorem norm_sum_mulChar_sub_poly_le_of_derivative_eq_zero {χ : MulChar (ZMod p) ℂ}
    (hχ : χ ≠ 1) {P : (ZMod p)[X]} (hP : 0 < P.natDegree) (hP' : derivative P = 0)
    (c : ZMod p) : ‖∑ a, χ (c - P.eval a)‖ ≤ ((P.natDegree : ℝ) - 1) * √p := by
  have hpm : p ≤ P.natDegree := by
    have h : (derivative P).coeff (P.natDegree - 1) = 0 := by rw [hP', coeff_zero]
    rw [coeff_derivative, ← Nat.cast_add_one, Nat.sub_add_cancel hP] at h
    have hlc : P.coeff P.natDegree ≠ 0 :=
      leadingCoeff_ne_zero.mpr fun h0 => by rw [h0, natDegree_zero] at hP; omega
    exact Nat.le_of_dvd hP
      ((ZMod.natCast_eq_zero_iff _ _).mp ((mul_eq_zero.mp h).resolve_left hlc))
  have hp3 : 3 ≤ p := by
    have h2 := (Fact.out : p.Prime).two_le
    have hd := Nat.le_of_dvd (n := p - 1) (by omega)
      (by simpa only [ZMod.card] using MulChar.orderOf_dvd_card_sub_one (χ := χ))
    have h1 : orderOf χ ≠ 1 := fun h => hχ (orderOf_eq_one_iff.mp h)
    have h0 := MulChar.orderOf_pos χ
    omega
  have hp : (3 : ℝ) ≤ p := by exact_mod_cast hp3
  have hm : (p : ℝ) ≤ P.natDegree := by exact_mod_cast hpm
  have hs : (3 : ℝ) / 2 ≤ √p := by
    rw [Real.le_sqrt (by norm_num) (Nat.cast_nonneg _)]
    linarith
  calc ‖∑ a, χ (c - P.eval a)‖ ≤ ∑ a : ZMod p, ‖χ (c - P.eval a)‖ := norm_sum_le _ _
    _ ≤ ∑ _a : ZMod p, (1 : ℝ) := sum_le_sum fun a _ => DirichletCharacter.norm_le_one χ _
    _ = p := by simp
    _ ≤ ((P.natDegree : ℝ) - 1) * √p := by nlinarith

/-- **Weil's bound for `c − P` at every `c` but one**, for `χ ≠ 1` of any order and `P` of
degree `m ≥ 1`: `|∑_a χ(c − P(a))| ≤ (m − 1) √p` for every `c ≠ c₀`. -/
theorem exists_forall_ne_norm_sum_mulChar_sub_poly_le {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1)
    {P : (ZMod p)[X]} (hP : 0 < P.natDegree) :
    ∃ c₀ : ZMod p, ∀ c, c ≠ c₀ → ‖∑ a, χ (c - P.eval a)‖ ≤ ((P.natDegree : ℝ) - 1) * √p := by
  by_cases hP' : derivative P = 0
  · exact ⟨0, fun c _ => norm_sum_mulChar_sub_poly_le_of_derivative_eq_zero hχ hP hP' c⟩
  by_cases hE : ∃ c₀ : ZMod p,
      ∀ R : (ZMod p)[X], Irreducible R → R ∣ C c₀ - P → R ^ 2 ∣ C c₀ - P
  · obtain ⟨c₀, h₀⟩ := hE
    exact ⟨c₀, fun c hc => norm_sum_mulChar_sub_poly_le_of_not_forall hχ hP fun h =>
      hc (eq_of_forall_sq_dvd_C_sub hP hP' h h₀)⟩
  · exact ⟨0, fun c _ => norm_sum_mulChar_sub_poly_le_of_not_forall hχ hP fun h => hE ⟨c, h⟩⟩

/-- **Theorem B for `P` and every character `χ ≠ 1`**, `review/flat_spectrum.md` with `a³`
replaced by `P(a)`, `P` of degree `m ≥ 1`:
`‖E[χ(a b) | P(a) + a b]‖² ≤ ((m − 1) √p + 1)²/p² + (p − 1)/p² + (p − 1)²/(p² (2p − 1))`.
Weil's bound covers every `c` but one, whose fiber adds `(p − 1)/p²`. -/
theorem meanSq_condMean_poly_add_mul_mulChar_le_of_natDegree_pos {χ : MulChar (ZMod p) ℂ}
    (hχ : χ ≠ 1) {P : (ZMod p)[X]} (hP : 0 < P.natDegree) :
    meanSq (condMean (fun ab : ZMod p × ZMod p => P.eval ab.1 + ab.1 * ab.2)
      fun ab => χ (ab.1 * ab.2)) ≤
      (((P.natDegree : ℝ) - 1) * √p + 1) ^ 2 / p ^ 2 + (p - 1) / p ^ 2 +
        (p - 1) ^ 2 / (p ^ 2 * (2 * p - 1)) := by
  classical
  obtain ⟨c₀, h₀⟩ := exists_forall_ne_norm_sum_mulChar_sub_poly_le hχ hP
  have hb := meanSq_condMean_poly_add_mul_mulChar_le_of_finset (fun a => P.eval a) χ {c₀}
    (B := ((P.natDegree : ℝ) - 1) * √p) fun c _ hc => h₀ c (notMem_singleton.mp hc)
  rwa [card_singleton, Nat.cast_one, one_mul] at hb

section Examples

attribute [local instance] Transformer.Weil.fact_prime_71

/-- For a character `χ` of order `5` of `𝔽₇₁`, `P = X` and `c = 0`: `|∑_a χ(−a)| ≤ 0`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧ ‖∑ a : ZMod 71, χ (0 - a)‖ ≤ 0 := by
  obtain ⟨χ, hχ⟩ := Transformer.Weil.exists_mulChar_zmod71_orderOf_eq_five
  refine ⟨χ, hχ, ?_⟩
  have h := norm_sum_mulChar_sub_poly_le_of_not_forall (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (P := X) (by rw [natDegree_X]; norm_num)
    (c := 0) fun h => by
      have h2 := natDegree_le_of_dvd (h X irreducible_X (by rw [C_0, zero_sub, dvd_neg]))
        (by rw [C_0, zero_sub, neg_ne_zero]; exact X_ne_zero)
      rw [natDegree_X_pow, C_0, zero_sub, natDegree_neg, natDegree_X] at h2
      omega
  simp only [natDegree_X, eval_X] at h
  exact h.trans_eq (by norm_num)

/-- For a character `χ` of order `5` of `𝔽₇₁` and `P = X^71`, with `P' = 0`:
`|∑_a χ(−a^71)| ≤ 70 √71`. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧
    ‖∑ a : ZMod 71, χ (0 - a ^ 71)‖ ≤ 70 * √71 := by
  obtain ⟨χ, hχ⟩ := Transformer.Weil.exists_mulChar_zmod71_orderOf_eq_five
  refine ⟨χ, hχ, ?_⟩
  have h := norm_sum_mulChar_sub_poly_le_of_derivative_eq_zero (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (P := X ^ 71)
    (by rw [natDegree_X_pow]; norm_num)
    (by rw [derivative_X_pow, ZMod.natCast_self, C_0, zero_mul]) 0
  simp only [natDegree_X_pow, eval_pow, eval_X] at h
  exact h.trans_eq (by norm_num)

/-- For a character `χ` of order `5` of `𝔽₇₁` and `P = X⁵`: Weil's bound at every `c` but
one. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧ ∃ c₀ : ZMod 71,
    ∀ c, c ≠ c₀ → ‖∑ a : ZMod 71, χ (c - a ^ 5)‖ ≤ 4 * √71 := by
  obtain ⟨χ, hχ⟩ := Transformer.Weil.exists_mulChar_zmod71_orderOf_eq_five
  refine ⟨χ, hχ, ?_⟩
  have h := exists_forall_ne_norm_sum_mulChar_sub_poly_le (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (P := X ^ 5)
    (by rw [natDegree_X_pow]; norm_num)
  simp only [natDegree_X_pow, eval_pow, eval_X] at h
  obtain ⟨c₀, h₀⟩ := h
  exact ⟨c₀, fun c hc => (h₀ c hc).trans_eq (by norm_num)⟩

/-- **Theorem B for a character `χ` of order `5` of `𝔽₇₁`** and `P(a) = a⁵`, whose degree is
the order. -/
example : ∃ χ : MulChar (ZMod 71) ℂ, orderOf χ = 5 ∧
    meanSq (condMean (fun ab : ZMod 71 × ZMod 71 => ab.1 ^ 5 + ab.1 * ab.2)
      fun ab => χ (ab.1 * ab.2)) ≤
      (4 * √71 + 1) ^ 2 / 71 ^ 2 + 70 / 71 ^ 2 + 70 ^ 2 / (71 ^ 2 * 141) := by
  obtain ⟨χ, hχ⟩ := Transformer.Weil.exists_mulChar_zmod71_orderOf_eq_five
  refine ⟨χ, hχ, ?_⟩
  have h := meanSq_condMean_poly_add_mul_mulChar_le_of_natDegree_pos (p := 71) (χ := χ)
    (by rintro rfl; rw [orderOf_one] at hχ; omega) (P := X ^ 5)
    (by rw [natDegree_X_pow]; norm_num)
  simp only [natDegree_X_pow, eval_pow, eval_X] at h
  refine h.trans_eq ?_
  push_cast
  ring

end Examples

end Transformer.Grokking.FlatSpectrum
