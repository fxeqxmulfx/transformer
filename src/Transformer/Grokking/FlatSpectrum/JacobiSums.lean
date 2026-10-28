module

public import Transformer.Grokking.FlatSpectrum.GaussSums
public import Mathlib.NumberTheory.JacobiSum.Basic
public import Mathlib.NumberTheory.DirichletCharacter.Bounds

/-!
# Weil's bound for `∑_a χ(c − a³)`, by Jacobi sums

Source: `review/flat_spectrum.md`, Theorem B: "Multiplicative characters of ab are the same
order, by Weil's bound |Σ χ(c − a³)| ≤ 2√p." For this sum the Jacobi sums
`J(χ₁, χ₂) = ∑_t χ₁(t) χ₂(1 − t)` give the bound without Weil's theorem. For `p ≡ 1 mod 3` and
`χ₃` of order 3, `#{a³ = y} = 1 + χ₃(y) + χ₃²(y)` and `y = c t` give
`∑_a χ(c − a³) = χ₃(c) χ(c) J(χ₃, χ) + χ₃²(c) χ(c) J(χ₃², χ)` for `c ≠ 0` and nontrivial `χ`.
For nontrivial `χ₁`, `χ₂`, `|J(χ₁, χ₂)| = √p` if `χ₁ χ₂ ≠ 1`, as
`J(χ₁, χ₂) conj J(χ₁, χ₂) = J(χ₁, χ₂) J(χ₁⁻¹, χ₂⁻¹) = p`, and `|J(χ₁, χ₁⁻¹)| = |χ₁(−1)| = 1`.
For `p ≢ 1 mod 3` cubing is a bijection, so the sum is `∑_y χ(c − y) = 0`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- `|χ(c)| = 1` for `c ≠ 0`. -/
lemma norm_mulChar_eq_one (χ : MulChar (ZMod p) ℂ) {c : ZMod p} (hc : c ≠ 0) : ‖χ c‖ = 1 := by
  simpa using DirichletCharacter.unit_norm_eq_one χ (Units.mk0 c hc)

example := @norm_mulChar_eq_one 5 ⟨Nat.prime_five⟩ 1 1 (by decide)

/-- `conj J(χ₁, χ₂) = J(χ₁⁻¹, χ₂⁻¹)` for complex characters. -/
lemma star_jacobiSum (χ₁ χ₂ : MulChar (ZMod p) ℂ) :
    star (jacobiSum χ₁ χ₂) = jacobiSum χ₁⁻¹ χ₂⁻¹ := by
  simp only [jacobiSum, star_sum, star_mul', MulChar.star_apply']

example := @star_jacobiSum 5 ⟨Nat.prime_five⟩ 1 1

/-- `|J(χ₁, χ₂)| = √p` for nontrivial `χ₁`, `χ₂` with `χ₁ χ₂ ≠ 1`. -/
theorem norm_jacobiSum {χ₁ χ₂ : MulChar (ZMod p) ℂ} (h₁ : χ₁ ≠ 1) (h₂ : χ₂ ≠ 1)
    (h₁₂ : χ₁ * χ₂ ≠ 1) : ‖jacobiSum χ₁ χ₂‖ = √p := by
  have hchar : ringChar ℂ ≠ ringChar (ZMod p) := by
    rw [ringChar.eq_zero, ZMod.ringChar_zmod_n]
    exact (Fact.out : p.Prime).ne_zero.symm
  have h := jacobiSum_mul_jacobiSum_inv hchar h₁ h₂ h₁₂
  rw [← star_jacobiSum, RCLike.star_def, Complex.mul_conj', ZMod.card] at h
  rw [← Real.sqrt_sq (norm_nonneg _)]
  congr 1
  exact_mod_cast h

example : ∃ χ : MulChar (ZMod 7) ℂ, ‖jacobiSum χ χ‖ = √((7 : ℕ) : ℝ) := by
  obtain ⟨χ, hχ⟩ := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)
  obtain ⟨h1, h2⟩ := ne_one_of_orderOf_eq_three hχ
  exact ⟨χ, @norm_jacobiSum 7 ⟨Nat.prime_seven⟩ χ χ h1 h1 (by rwa [← sq])⟩

/-- `|J(χ₁, χ₂)| ≤ √p` for nontrivial `χ₁`, `χ₂`; if `χ₁ χ₂ = 1`, `J(χ₁, χ₁⁻¹) = −χ₁(−1)`. -/
theorem norm_jacobiSum_le {χ₁ χ₂ : MulChar (ZMod p) ℂ} (h₁ : χ₁ ≠ 1) (h₂ : χ₂ ≠ 1) :
    ‖jacobiSum χ₁ χ₂‖ ≤ √p := by
  by_cases h₁₂ : χ₁ * χ₂ = 1
  · rw [eq_inv_of_mul_eq_one_right h₁₂, jacobiSum_nontrivial_inv h₁, norm_neg]
    exact (DirichletCharacter.norm_le_one χ₁ _).trans
      (Real.one_le_sqrt.mpr (by exact_mod_cast (Fact.out : p.Prime).one_le))
  · exact (norm_jacobiSum h₁ h₂ h₁₂).le

example : ∃ χ : MulChar (ZMod 7) ℂ, ‖jacobiSum χ χ⁻¹‖ ≤ √((7 : ℕ) : ℝ) := by
  obtain ⟨χ, hχ⟩ := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)
  obtain ⟨h1, -⟩ := ne_one_of_orderOf_eq_three hχ
  exact ⟨χ, @norm_jacobiSum_le 7 ⟨Nat.prime_seven⟩ χ χ⁻¹ h1 (inv_ne_one.mpr h1)⟩

/-- `∑_y χ₁(y) χ₂(c − y) = χ₁(c) χ₂(c) J(χ₁, χ₂)` for `c ≠ 0`, by `y = c t`. -/
lemma sum_mulChar_mul_sub (χ₁ χ₂ : MulChar (ZMod p) ℂ) {c : ZMod p} (hc : c ≠ 0) :
    ∑ y, χ₁ y * χ₂ (c - y) = χ₁ c * χ₂ c * jacobiSum χ₁ χ₂ := by
  rw [← Equiv.sum_comp (Equiv.mulLeft₀ c hc), jacobiSum, Finset.mul_sum]
  refine Finset.sum_congr rfl fun t _ => ?_
  simp only [Equiv.mulLeft₀_apply]
  rw [show c - c * t = c * (1 - t) by ring, map_mul, map_mul]
  ring

example := @sum_mulChar_mul_sub 5 ⟨Nat.prime_five⟩ 1 1 1 (by decide)

/-- `∑_y χ(c − y) = 0` for nontrivial `χ`. -/
lemma sum_mulChar_sub {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) (c : ZMod p) :
    ∑ y, χ (c - y) = 0 := by
  rw [show ∑ y, χ (c - y) = ∑ y, χ y from Equiv.sum_comp (Equiv.subLeft c) fun y => χ y,
    MulChar.sum_eq_zero_of_ne_one hχ]

example := @sum_mulChar_sub 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) 1

variable {χ : MulChar (ZMod p) ℂ}

/-- **Weil's sum for `p ≡ 1 mod 3`**: for `χ₃` of order 3, nontrivial `χ` and `c ≠ 0`,
`∑_a χ(c − a³) = χ₃(c) χ(c) J(χ₃, χ) + χ₃²(c) χ(c) J(χ₃², χ)`. -/
theorem sum_mulChar_sub_cube_eq {χ₃ : MulChar (ZMod p) ℂ} (h₃ : orderOf χ₃ = 3) (hχ : χ ≠ 1)
    {c : ZMod p} (hc : c ≠ 0) :
    ∑ a, χ (c - a ^ 3) =
      χ₃ c * χ c * jacobiSum χ₃ χ + (χ₃ ^ 2) c * χ c * jacobiSum (χ₃ ^ 2) χ := by
  have h1 : ∑ a, χ (c - a ^ 3) = ∑ y, (#{x : ZMod p | x ^ 3 = y} : ℂ) * χ (c - y) := by
    rw [← Finset.sum_fiberwise univ fun x : ZMod p => x ^ 3]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [Finset.sum_congr rfl fun x hx => by rw [(Finset.mem_filter.mp hx).2], Finset.sum_const,
      nsmul_eq_mul]
  rw [h1, Finset.sum_congr rfl fun y _ => by rw [card_pow_three_eq_eq h₃ y]]
  simp only [add_mul, Finset.sum_add_distrib, one_mul, sum_mulChar_sub hχ, zero_add,
    sum_mulChar_mul_sub _ _ hc]

example : ∃ χ₃ : MulChar (ZMod 7) ℂ, ∑ a : ZMod 7, χ₃ (1 - a ^ 3) =
    χ₃ 1 * χ₃ 1 * jacobiSum χ₃ χ₃ + (χ₃ ^ 2) 1 * χ₃ 1 * jacobiSum (χ₃ ^ 2) χ₃ := by
  obtain ⟨χ₃, h₃⟩ := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)
  exact ⟨χ₃, @sum_mulChar_sub_cube_eq 7 ⟨Nat.prime_seven⟩ χ₃ χ₃ h₃
    (ne_one_of_orderOf_eq_three h₃).1 1 (by decide)⟩

/-- Cubing is injective on `ℤ/p` for `p ≢ 1 mod 3`: `x³ = x` for `p = 3`, and
`x = (x³)^{(2p−1)/3}` for `p ≡ 2 mod 3`. -/
lemma injective_pow_three (hp3 : p % 3 ≠ 1) : Function.Injective fun x : ZMod p => x ^ 3 := by
  have hp := (Fact.out : p.Prime)
  have hroot (x : ZMod p) : (x ^ 3) ^ ((2 * p - 1) / 3) = x := by
    by_cases h3 : p = 3
    · subst h3
      rw [show (2 * 3 - 1) / 3 = 1 by norm_num, pow_one, ZMod.pow_card]
    · have h2 : p % 3 = 2 := by
        rcases (by omega : p % 3 = 0 ∨ p % 3 = 2) with h0 | h2
        · exact absurd ((hp.eq_one_or_self_of_dvd 3 (Nat.dvd_of_mod_eq_zero h0)).resolve_left
            (by norm_num)).symm h3
        · exact h2
      rw [← pow_mul, show 3 * ((2 * p - 1) / 3) = p + (p - 1) by omega, pow_add, ZMod.pow_card,
        ← pow_succ', Nat.sub_add_cancel hp.one_le, ZMod.pow_card]
  intro x y h
  have h' : x ^ 3 = y ^ 3 := h
  rw [← hroot x, ← hroot y, h']

example := @injective_pow_three 5 ⟨Nat.prime_five⟩ (by decide)

/-- **Weil's sum for `p ≢ 1 mod 3`**: cubing is a bijection, so `∑_a χ(c − a³) = 0`. -/
theorem sum_mulChar_sub_cube_eq_zero (hp3 : p % 3 ≠ 1) (hχ : χ ≠ 1) (c : ZMod p) :
    ∑ a, χ (c - a ^ 3) = 0 := by
  rw [Fintype.sum_bijective _ (injective_pow_three hp3).bijective_of_finite _
    (fun y => χ (c - y)) fun _ => rfl, sum_mulChar_sub hχ]

example := @sum_mulChar_sub_cube_eq_zero 5 ⟨Nat.prime_five⟩ _ (by decide)
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) 1

/-- **Weil's bound** of `review/flat_spectrum.md`, Theorem B: `|∑_a χ(c − a³)| ≤ 2 √p` for
nontrivial `χ` and `c ≠ 0`. The source cites Weil's theorem; here Jacobi sums give it. -/
theorem norm_sum_mulChar_sub_cube_le (hχ : χ ≠ 1) {c : ZMod p} (hc : c ≠ 0) :
    ‖∑ a, χ (c - a ^ 3)‖ ≤ 2 * √p := by
  by_cases hp3 : p % 3 = 1
  · obtain ⟨χ₃, h₃⟩ := exists_orderOf_eq_three (p := p) hp3
    obtain ⟨hne1, hne2⟩ := ne_one_of_orderOf_eq_three h₃
    rw [sum_mulChar_sub_cube_eq h₃ hχ hc, two_mul]
    refine (norm_add_le _ _).trans (add_le_add ?_ ?_)
    · rw [norm_mul, norm_mul, norm_mulChar_eq_one _ hc, norm_mulChar_eq_one _ hc, one_mul,
        one_mul]
      exact norm_jacobiSum_le hne1 hχ
    · rw [norm_mul, norm_mul, norm_mulChar_eq_one _ hc, norm_mulChar_eq_one _ hc, one_mul,
        one_mul]
      exact norm_jacobiSum_le hne2 hχ
  · rw [sum_mulChar_sub_cube_eq_zero hp3 hχ, norm_zero]
    positivity

example := @norm_sum_mulChar_sub_cube_le 7 ⟨Nat.prime_seven⟩ _
  (@quadraticChar_ringHomComp_ne_one 7 ⟨Nat.prime_seven⟩ (by decide)) 1 (by decide)

end Transformer.Grokking.FlatSpectrum
