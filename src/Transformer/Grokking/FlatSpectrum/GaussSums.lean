module

public import Transformer.Grokking.FlatSpectrum.CubicSums
public import Mathlib.NumberTheory.GaussSum
public import Mathlib.NumberTheory.MulChar.Lemmas
public import Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.Basic
public import Mathlib.RingTheory.RootsOfUnity.Complex

/-!
# Theorem B: Gauss sums between the two character bases

Source: `review/flat_spectrum.md`, Theorem B: "These two bases are mutually unbiased: their
inner products are Gauss sums of modulus √p", and "For p ≡ 1 mod 3, S_r is a sum of two cubic
Gauss sums, so |S_r| ≤ 2√p." The inner product of `ψ_r` and a multiplicative character `χ` is
`g(χ, ψ_{−r})`, and `g(χ, ψ_r) conj g(χ, ψ_r) = p` for nontrivial `χ` and `r ≠ 0`. For `χ` of
order 3, `#{x³ = y} = 1 + χ(y) + χ²(y)`: both sides are `1` at `y = 0`; at `y ≠ 0` the left
side is at most `3` where `χ(y) = 1` and `0` elsewhere, and both sides sum to `p`. So
`S_r = ∑_y #{x³ = y} ψ_r(y) = g(χ, ψ_r) + g(χ², ψ_r)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {p : ℕ} [Fact p.Prime]

/-- `conj χ(x) χ(x) = 1` for `x ≠ 0`. -/
lemma star_mulChar_mul_self (χ : MulChar (ZMod p) ℂ) {x : ZMod p} (hx : x ≠ 0) :
    star (χ x) * χ x = 1 := by
  rw [MulChar.star_apply', ← MulChar.mul_apply, MulChar.inv_mul, MulChar.one_apply hx.isUnit]

example := @star_mulChar_mul_self 5 ⟨Nat.prime_five⟩ 1 1 (by decide)

/-- The quadratic character of `ℤ/p`, `p` odd, is a nontrivial complex character. -/
lemma quadraticChar_ringHomComp_ne_one (hp : p ≠ 2) :
    (quadraticChar (ZMod p)).ringHomComp (Int.castRingHom ℂ) ≠ 1 := fun h =>
  quadraticChar_ne_one (by rwa [ZMod.ringChar_zmod_n])
    (MulChar.injective_ringHomComp Int.cast_injective (h.trans (MulChar.ringHomComp_one _).symm))

example := @quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)

/-- `ψ_r = ψ(r ·)` is primitive for `r ≠ 0`. -/
lemma isPrimitive_stdAddChar_mulShift {r : ZMod p} (hr : r ≠ 0) :
    (ZMod.stdAddChar.mulShift r : AddChar (ZMod p) ℂ).IsPrimitive := fun a ha => by
  rw [AddChar.mulShift_mulShift]
  exact ZMod.isPrimitive_stdAddChar p (mul_ne_zero hr ha)

example := @isPrimitive_stdAddChar_mulShift 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- The inner product `∑_x conj ψ_r(x) χ(x)` of the two bases is the Gauss sum `g(χ, ψ_{−r})`,
`review/flat_spectrum.md`, Theorem B. -/
lemma sum_star_stdAddChar_mul (χ : MulChar (ZMod p) ℂ) (r : ZMod p) :
    ∑ x, star (ZMod.stdAddChar (r * x)) * χ x = gaussSum χ (ZMod.stdAddChar.mulShift (-r)) := by
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [AddChar.mulShift_apply, mul_comm, Complex.star_def, ← AddChar.map_neg_eq_conj, neg_mul]

example := @sum_star_stdAddChar_mul 5 ⟨Nat.prime_five⟩ 1 1

/-- **The two bases are mutually unbiased**, `review/flat_spectrum.md`, Theorem B: for
nontrivial `χ` and `r ≠ 0`, `|g(χ, ψ_r)| = √p`. The source says it of all inner products; the
trivial character gives `g(1, ψ_r) = −1` and `ψ_0` gives `g(χ, 1) = 0`
(`gaussSum_one_mulShift`, `gaussSum_mulShift_zero`), and the `p − 1` multiplicative characters
vanish at `0`. -/
theorem norm_gaussSum_mulShift {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) {r : ZMod p} (hr : r ≠ 0) :
    ‖gaussSum χ (ZMod.stdAddChar.mulShift r)‖ = √p := by
  have h := gaussSum_mul_gaussSum_eq_card hχ (isPrimitive_stdAddChar_mulShift hr)
  rw [← star_gaussSum_eq, ZMod.card, Complex.star_def, Complex.mul_conj,
    Complex.normSq_eq_norm_sq] at h
  have h' : ‖gaussSum χ (ZMod.stdAddChar.mulShift r)‖ ^ 2 = p := by exact_mod_cast h
  rw [← h', Real.sqrt_sq (norm_nonneg _)]

example := @norm_gaussSum_mulShift 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide)) 1 (by decide)

/-- The trivial character: `g(1, ψ_r) = −1` for `r ≠ 0`. -/
lemma gaussSum_one_mulShift {r : ZMod p} (hr : r ≠ 0) :
    gaussSum (1 : MulChar (ZMod p) ℂ) (ZMod.stdAddChar.mulShift r) = -1 :=
  gaussSum_one_left (ZMod.isPrimitive_stdAddChar p hr)

example := @gaussSum_one_mulShift 5 ⟨Nat.prime_five⟩ 1 (by decide)

/-- The trivial additive character: `g(χ, ψ_0) = 0` for nontrivial `χ`. -/
lemma gaussSum_mulShift_zero {χ : MulChar (ZMod p) ℂ} (hχ : χ ≠ 1) :
    gaussSum χ (ZMod.stdAddChar.mulShift 0) = 0 := by
  rw [AddChar.mulShift_zero, gaussSum_one_right hχ]

example := @gaussSum_mulShift_zero 5 ⟨Nat.prime_five⟩ _
  (@quadraticChar_ringHomComp_ne_one 5 ⟨Nat.prime_five⟩ (by decide))

/-- For `p ≡ 1 mod 3` there is a complex character of `ℤ/p` of order 3. -/
lemma exists_orderOf_eq_three (hp3 : p % 3 = 1) : ∃ χ : MulChar (ZMod p) ℂ, orderOf χ = 3 :=
  MulChar.exists_mulChar_orderOf (ZMod p) (by rw [ZMod.card]; omega)
    (Complex.isPrimitiveRoot_exp 3 (by norm_num))

example := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)

variable {χ : MulChar (ZMod p) ℂ}

omit [Fact p.Prime] in
/-- A character of order 3 and its square are nontrivial. -/
lemma ne_one_of_orderOf_eq_three (hχ : orderOf χ = 3) : χ ≠ 1 ∧ χ ^ 2 ≠ 1 := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · rw [h, orderOf_one] at hχ
    omega
  · have := orderOf_dvd_of_pow_eq_one h
    rw [hχ] at this
    omega

example : ∃ χ : MulChar (ZMod 7) ℂ, χ ≠ 1 ∧ χ ^ 2 ≠ 1 := by
  obtain ⟨χ, hχ⟩ := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)
  exact ⟨χ, ne_one_of_orderOf_eq_three hχ⟩

/-- The number of cube roots: `#{x³ = y} = 1 + χ(y) + χ²(y)` for `χ` of order 3. -/
lemma card_pow_three_eq_eq (hχ : orderOf χ = 3) (y : ZMod p) :
    (#{x : ZMod p | x ^ 3 = y} : ℂ) = 1 + χ y + (χ ^ 2) y := by
  obtain ⟨hne1, hne2⟩ := ne_one_of_orderOf_eq_three hχ
  have h3 (x : ZMod p) (hx : x ≠ 0) : χ x ^ 3 = 1 := by
    rw [← MulChar.pow_apply' χ (by norm_num), ← hχ, pow_orderOf_eq_one,
      MulChar.one_apply hx.isUnit]
  set M : ZMod p → ℕ := fun y => if y = 0 then 1 else if χ y = 1 then 3 else 0 with hMdef
  have hM (y : ZMod p) : (M y : ℂ) = 1 + χ y + (χ ^ 2) y := by
    rw [MulChar.pow_apply' χ (by norm_num)]
    by_cases hy : y = 0
    · simp [hMdef, hy, MulChar.map_zero]
    by_cases hχy : χ y = 1
    · simp only [hMdef, hy, hχy, ite_false, ite_true]
      norm_num
    · simp only [hMdef, hy, hχy, ite_false, Nat.cast_zero]
      have h : (χ y - 1) * (1 + χ y + χ y ^ 2) = 0 := by linear_combination h3 y hy
      exact ((mul_eq_zero.mp h).resolve_left (sub_ne_zero.mpr hχy)).symm
  have hle (y : ZMod p) : #{x : ZMod p | x ^ 3 = y} ≤ M y := by
    by_cases hy : y = 0
    · simp only [hMdef, hy, ite_true]
      refine Finset.card_le_one.mpr fun a ha b hb => ?_
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, pow_eq_zero_iff,
        ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true] at ha hb
      rw [ha, hb]
    by_cases hχy : χ y = 1
    · simpa only [hMdef, hy, hχy, ite_false, ite_true] using card_pow_three_eq_le y
    · simp only [hMdef, hy, hχy, ite_false, Nat.le_zero, Finset.card_eq_zero,
        Finset.filter_eq_empty_iff, Finset.mem_univ, true_implies]
      intro x hx
      have hx0 : x ≠ 0 := fun h0 => hy (by rw [← hx, h0, zero_pow (by norm_num)])
      exact hχy (by rw [← hx, map_pow, h3 x hx0])
  have hsumN : ∑ y, #{x : ZMod p | x ^ 3 = y} = p := by
    rw [← Finset.card_eq_sum_card_fiberwise fun x _ => Finset.mem_coe.mpr (mem_univ (x ^ 3)),
      card_univ, ZMod.card]
  have hsumM : ∑ y, M y = p := by
    have h : ((∑ y, M y : ℕ) : ℂ) = p := by
      push_cast
      simp only [hM, Finset.sum_add_distrib, MulChar.sum_eq_zero_of_ne_one hne1,
        MulChar.sum_eq_zero_of_ne_one hne2, Finset.sum_const, card_univ, ZMod.card, nsmul_eq_mul,
        mul_one, add_zero]
    exact_mod_cast h
  rw [(Finset.sum_eq_sum_iff_of_le fun y _ => hle y).mp (hsumN.trans hsumM.symm) y (mem_univ y),
    hM]

example : ∃ χ : MulChar (ZMod 7) ℂ, (#{x : ZMod 7 | x ^ 3 = 1} : ℂ) = 1 + χ 1 + (χ ^ 2) 1 := by
  obtain ⟨χ, hχ⟩ := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)
  exact ⟨χ, @card_pow_three_eq_eq 7 ⟨Nat.prime_seven⟩ χ hχ 1⟩

/-- **For `p ≡ 1 mod 3`, `S_r` is a sum of two cubic Gauss sums**, `review/flat_spectrum.md`,
Theorem B: `S_r = g(χ, ψ_r) + g(χ², ψ_r)` for `χ` of order 3 and `r ≠ 0`. -/
theorem cubicSum_eq_gaussSum_add (hχ : orderOf χ = 3) {r : ZMod p} (hr : r ≠ 0) :
    cubicSum r = gaussSum χ (ZMod.stdAddChar.mulShift r) +
      gaussSum (χ ^ 2) (ZMod.stdAddChar.mulShift r) := by
  have h1 : cubicSum r = ∑ y, (#{x : ZMod p | x ^ 3 = y} : ℂ) * ZMod.stdAddChar (r * y) := by
    rw [cubicSum, ← Finset.sum_fiberwise univ fun x : ZMod p => x ^ 3]
    refine Finset.sum_congr rfl fun y _ => ?_
    rw [Finset.sum_congr rfl fun x hx => by rw [(Finset.mem_filter.mp hx).2], Finset.sum_const,
      nsmul_eq_mul]
  have h0 : ∑ y : ZMod p, ZMod.stdAddChar (r * y) = 0 := by
    simpa [mul_comm, hr] using sum_stdAddChar_mul p r
  rw [h1, Finset.sum_congr rfl fun y _ => by rw [card_pow_three_eq_eq hχ y]]
  simp only [add_mul, Finset.sum_add_distrib, one_mul, h0, zero_add, gaussSum,
    AddChar.mulShift_apply]

example : ∃ χ : MulChar (ZMod 7) ℂ, @cubicSum 7 ⟨Nat.prime_seven⟩ 1 =
    gaussSum χ (ZMod.stdAddChar.mulShift 1) + gaussSum (χ ^ 2) (ZMod.stdAddChar.mulShift 1) := by
  obtain ⟨χ, hχ⟩ := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)
  exact ⟨χ, @cubicSum_eq_gaussSum_add 7 ⟨Nat.prime_seven⟩ χ hχ 1 (by decide)⟩

/-- The cubic-Gauss-sum route to `|S_r| ≤ 2√p`, `review/flat_spectrum.md`, Theorem B; for
every prime `p` the bound is `norm_cubicSum_le`. -/
theorem norm_cubicSum_le_of_orderOf (hχ : orderOf χ = 3) {r : ZMod p} (hr : r ≠ 0) :
    ‖cubicSum r‖ ≤ 2 * √p := by
  obtain ⟨hne1, hne2⟩ := ne_one_of_orderOf_eq_three hχ
  rw [cubicSum_eq_gaussSum_add hχ hr]
  refine (norm_add_le _ _).trans_eq ?_
  rw [norm_gaussSum_mulShift hne1 hr, norm_gaussSum_mulShift hne2 hr, two_mul]

example : ‖@cubicSum 7 ⟨Nat.prime_seven⟩ 1‖ ≤ 2 * √7 := by
  obtain ⟨χ, hχ⟩ := @exists_orderOf_eq_three 7 ⟨Nat.prime_seven⟩ (by norm_num)
  exact_mod_cast @norm_cubicSum_le_of_orderOf 7 ⟨Nat.prime_seven⟩ χ hχ 1 (by decide)

end Transformer.Grokking.FlatSpectrum
