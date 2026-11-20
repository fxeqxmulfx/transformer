module

public import Transformer.Grokking.FlatSpectrum.GaussSums
public import Mathlib.NumberTheory.MulChar.Duality
public import Mathlib.RingTheory.RootsOfUnity.AlgebraicallyClosed

/-!
# The product `x y` through multiplicative characters

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV7. A product layer reads
`χ(x)` and `χ(y)` linearly from the features of `x` and `y` and multiplies them, so it computes
`χ(x y) = χ(x) χ(y)` with one hidden unit per character. The `p − 1` complex characters of
`ℤ/p` are orthogonal both ways (`sum_mulChar_apply`, `sum_star_mulChar_mul`,
`sum_mulChar_mul_star`). So `ψ_r(t) − 1`, which vanishes at `0` as every character does, is
`∑_χ c_r(χ) χ(t)` (`stdAddChar_eq_one_add_sum`), and

    ψ_r(x y) = 1 + ∑_χ c_r(χ) χ(x) χ(y)

(`stdAddChar_mul_eq`). The coefficients are Gauss sums over `p − 1` (`mulCoeff_eq`):
`c_r(1) = −p/(p − 1)` (`mulCoeff_one`) and `|c_r(χ)| = √p/(p − 1)` for `χ ≠ 1`
(`norm_mulCoeff`), so none is zero (`mulCoeff_ne_zero`). The expansion is unique
(`eq_mulCoeff_of_forall`), so this route to `ψ_r(x y)` needs all `p − 1` characters.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

noncomputable instance : Fintype (MulChar (ZMod p) ℂ) := Fintype.ofFinite _

noncomputable instance : DecidableEq (MulChar (ZMod p) ℂ) := Classical.decEq _

/-- `ℤ/p` has `p − 1` complex characters. -/
lemma card_mulChar : Fintype.card (MulChar (ZMod p) ℂ) = p - 1 := by
  rw [← Nat.card_eq_fintype_card, MulChar.card_eq_card_units_of_hasEnoughRootsOfUnity,
    Nat.card_eq_fintype_card, ZMod.card_units]

lemma natCast_sub_one_ne_zero : (p : ℂ) - 1 ≠ 0 := by
  rw [sub_ne_zero]
  exact_mod_cast (Fact.out : p.Prime).one_lt.ne'

/-- **Orthogonality over the characters**: `∑_χ χ(a) = (p − 1) [a = 1]`. -/
lemma sum_mulChar_apply (a : ZMod p) :
    ∑ χ : MulChar (ZMod p) ℂ, χ a = if a = 1 then (p : ℂ) - 1 else 0 := by
  split_ifs with ha
  · subst ha
    simp only [MulChar.map_one, sum_const, card_univ, card_mulChar, nsmul_eq_mul, mul_one]
    rw [Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]
  · obtain ⟨χ₀, hχ₀⟩ := MulChar.exists_apply_ne_one_of_hasEnoughRootsOfUnity (ZMod p) ℂ ha
    have h : χ₀ a * ∑ χ : MulChar (ZMod p) ℂ, χ a = ∑ χ : MulChar (ZMod p) ℂ, χ a := by
      rw [mul_sum]
      exact Fintype.sum_equiv (Equiv.mulLeft χ₀) _ _ fun χ => (MulChar.mul_apply χ₀ χ a).symm
    have : (1 - χ₀ a) * ∑ χ : MulChar (ZMod p) ℂ, χ a = 0 := by rw [sub_mul, one_mul, h, sub_self]
    exact (mul_eq_zero.1 this).resolve_left (sub_ne_zero.2 (Ne.symm hχ₀))

example := @sum_mulChar_apply 5 ⟨Nat.prime_five⟩ 2

/-- `∑_χ conj χ(s) χ(t) = (p − 1) [t = s]` for `s ≠ 0`. -/
lemma sum_star_mulChar_mul {s : ZMod p} (hs : s ≠ 0) (t : ZMod p) :
    ∑ χ : MulChar (ZMod p) ℂ, star (χ s) * χ t = if t = s then (p : ℂ) - 1 else 0 := by
  have (χ : MulChar (ZMod p) ℂ) : star (χ s) * χ t = χ (s⁻¹ * t) := by
    rw [MulChar.star_apply', MulChar.inv_apply', map_mul]
  simp only [this, sum_mulChar_apply, inv_mul_eq_one₀ hs]
  exact if_congr eq_comm rfl rfl

example := @sum_star_mulChar_mul 5 ⟨Nat.prime_five⟩ 1 (by decide) 2

/-- **Orthogonality over `ℤ/p`**: `∑_y χ(y) conj χ'(y) = (p − 1) [χ = χ']`. -/
lemma sum_mulChar_mul_star (χ χ' : MulChar (ZMod p) ℂ) :
    ∑ y, χ y * star (χ' y) = if χ = χ' then (p : ℂ) - 1 else 0 := by
  have (y : ZMod p) : χ y * star (χ' y) = (χ * χ'⁻¹) y := by
    rw [MulChar.star_apply', MulChar.mul_apply]
  simp only [this]
  split_ifs with h
  · subst h
    rw [mul_inv_cancel, MulChar.sum_one_eq_card_units, ZMod.card_units,
      Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one]
  · exact MulChar.sum_eq_zero_of_ne_one fun h' => h (mul_inv_eq_one.1 h')

/-- The coefficient of `χ` in `ψ_r(t) − 1 = ∑_χ c_r(χ) χ(t)`,
`c_r(χ) = (p − 1)⁻¹ ∑_s (ψ_r(s) − 1) conj χ(s)`. -/
noncomputable def mulCoeff (r : ZMod p) (χ : MulChar (ZMod p) ℂ) : ℂ :=
  (∑ s, (ZMod.stdAddChar (r * s) - 1) * star (χ s)) / ((p : ℂ) - 1)

/-- **The expansion**: `ψ_r(t) = 1 + ∑_χ c_r(χ) χ(t)` for every `t`, the zero included. -/
theorem stdAddChar_eq_one_add_sum (r t : ZMod p) :
    ZMod.stdAddChar (r * t) = 1 + ∑ χ, mulCoeff r χ * χ t := by
  have key : ∑ χ : MulChar (ZMod p) ℂ, mulCoeff r χ * χ t =
      (∑ s, (ZMod.stdAddChar (r * s) - 1) * ∑ χ : MulChar (ZMod p) ℂ, star (χ s) * χ t) /
        ((p : ℂ) - 1) := by
    simp only [mulCoeff, div_mul_eq_mul_div, ← sum_div, sum_mul, mul_sum, mul_assoc]
    rw [sum_comm]
  have hs (s : ZMod p) : (ZMod.stdAddChar (r * s) - 1) *
      ∑ χ : MulChar (ZMod p) ℂ, star (χ s) * χ t =
      if t = s then (ZMod.stdAddChar (r * t) - 1) * ((p : ℂ) - 1) else 0 := by
    by_cases h0 : s = 0
    · subst h0
      split_ifs with ht <;> simp [ht]
    · rw [sum_star_mulChar_mul h0 t]
      split_ifs with ht
      · subst ht
        rfl
      · rw [mul_zero]
  rw [key]
  simp only [hs, sum_ite_eq, mem_univ, ite_true]
  rw [mul_div_cancel_right₀ _ natCast_sub_one_ne_zero]
  ring

example := @stdAddChar_eq_one_add_sum 5 ⟨Nat.prime_five⟩ 1 0

/-- **`ψ_r(x y)` through characters**: `ψ_r(x y) = 1 + ∑_χ c_r(χ) χ(x) χ(y)`. -/
theorem stdAddChar_mul_eq (r x y : ZMod p) :
    ZMod.stdAddChar (r * (x * y)) = 1 + ∑ χ, mulCoeff r χ * (χ x * χ y) := by
  rw [stdAddChar_eq_one_add_sum r (x * y)]
  simp only [map_mul]

example := @stdAddChar_mul_eq 5 ⟨Nat.prime_five⟩ 1 2 3

/-- The coefficients are Gauss sums: `c_r(χ) = (conj g(χ, ψ_{−r}) − conj ∑_s χ(s)) / (p − 1)`. -/
lemma mulCoeff_eq (r : ZMod p) (χ : MulChar (ZMod p) ℂ) :
    mulCoeff r χ = (star (gaussSum χ (ZMod.stdAddChar.mulShift (-r))) - star (∑ s, χ s)) /
      ((p : ℂ) - 1) := by
  rw [mulCoeff, ← sum_star_stdAddChar_mul, star_sum, star_sum, ← sum_sub_distrib]
  congr 1
  refine sum_congr rfl fun s _ => ?_
  rw [star_mul', star_star]
  ring

/-- The trivial character: `c_r(1) = −p/(p − 1)` for `r ≠ 0`. -/
theorem mulCoeff_one {r : ZMod p} (hr : r ≠ 0) : mulCoeff r 1 = -(p : ℂ) / ((p : ℂ) - 1) := by
  rw [mulCoeff_eq, gaussSum_one_mulShift (neg_ne_zero.2 hr), MulChar.sum_one_eq_card_units,
    ZMod.card_units, Nat.cast_sub (Fact.out : p.Prime).one_le, Nat.cast_one, star_neg, star_one,
    star_sub, star_natCast, star_one]
  ring

example := @mulCoeff_one 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- A nontrivial character: `|c_r(χ)| = √p/(p − 1)` for `r ≠ 0`. -/
theorem norm_mulCoeff {r : ZMod p} (hr : r ≠ 0) {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) :
    ‖mulCoeff r χ‖ = √p / (p - 1) := by
  have h1 : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_le
  have hn : ‖(p : ℂ) - 1‖ = (p : ℝ) - 1 := by
    rw [show (p : ℂ) - 1 = (((p : ℝ) - 1 : ℝ) : ℂ) by push_cast; ring, Complex.norm_real,
      Real.norm_of_nonneg (by linarith)]
  rw [mulCoeff_eq, MulChar.sum_eq_zero_of_ne_one hχ, star_zero, sub_zero, norm_div, norm_star,
    norm_gaussSum_mulShift hχ (neg_ne_zero.2 hr), hn]

example := @norm_mulCoeff 5 ⟨Nat.prime_five⟩ 1 (by decide) _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- No coefficient is zero. -/
theorem mulCoeff_ne_zero {r : ZMod p} (hr : r ≠ 0) (χ : MulChar (ZMod p) ℂ) :
    mulCoeff r χ ≠ 0 := by
  by_cases hχ : χ = 1
  · subst hχ
    rw [mulCoeff_one hr]
    exact div_ne_zero (neg_ne_zero.2 (Nat.cast_ne_zero.2 (Fact.out : p.Prime).ne_zero))
      natCast_sub_one_ne_zero
  · have hp : (1 : ℝ) < p := by exact_mod_cast (Fact.out : p.Prime).one_lt
    rw [← norm_pos_iff, norm_mulCoeff hr hχ]
    exact div_pos (Real.sqrt_pos.2 (by linarith)) (by linarith)

example := @mulCoeff_ne_zero 5 ⟨Nat.prime_five⟩ 1 (by decide) 1

/-- **The route needs every character**: if `ψ_r(x y) = a + ∑_χ d(χ) χ(x) χ(y)` for all `x` and
`y`, then `a = 1` and `d = c_r`, none of whose values is zero. -/
theorem eq_mulCoeff_of_forall {r : ZMod p} {a : ℂ} {d : MulChar (ZMod p) ℂ → ℂ}
    (h : ∀ x y : ZMod p, ZMod.stdAddChar (r * (x * y)) = a + ∑ χ, d χ * (χ x * χ y)) :
    a = 1 ∧ d = mulCoeff r := by
  have ha : a = 1 := by
    have := h 0 0
    simp only [mul_zero, AddChar.map_zero_eq_one, MulChar.map_zero, sum_const_zero,
      add_zero] at this
    exact this.symm
  refine ⟨ha, funext fun χ' => ?_⟩
  have hy (y : ZMod p) : ∑ χ, (d χ - mulCoeff r χ) * χ y = 0 := by
    have h1 := h 1 y
    rw [one_mul, stdAddChar_eq_one_add_sum, ha] at h1
    simp only [MulChar.map_one, one_mul] at h1
    simp only [sub_mul, sum_sub_distrib]
    linear_combination -h1
  have hsum : ∑ y, (∑ χ, (d χ - mulCoeff r χ) * χ y) * star (χ' y) =
      (d χ' - mulCoeff r χ') * ((p : ℂ) - 1) := by
    simp only [sum_mul, mul_assoc]
    rw [sum_comm]
    simp only [← mul_sum, sum_mulChar_mul_star, mul_ite, mul_zero, sum_ite_eq', mem_univ,
      ite_true]
  simp only [hy, zero_mul, sum_const_zero] at hsum
  exact sub_eq_zero.1 ((mul_eq_zero.1 hsum.symm).resolve_right natCast_sub_one_ne_zero)

/-- The expansion itself satisfies the hypothesis. -/
example := @eq_mulCoeff_of_forall 5 ⟨Nat.prime_five⟩ 1 1 (@mulCoeff 5 ⟨Nat.prime_five⟩ 1)
  fun x y => @stdAddChar_mul_eq 5 ⟨Nat.prime_five⟩ 1 x y

end Transformer.Grokking.FlatSpectrum
