/-
# The bits of a fixed-precision number determine it

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix B.1, `def:fixed_precision`.

`Fx.bit x b` is the paper's `[x]_b`, the parity of `⌊x · 2^{b-s-1}⌋`.  The
scale `2^{-s}` cancels: `bit x (b + 1)` is the parity of `m / 2^b`, the
mantissa shifted right by `b` places (`bit_succ`).  So `bit x 0` is always
`0` — it reads the parity of `2m` — and the bits that carry information are
`1, …, p`, the last of them the two's-complement sign.

What the `TL[◁#]` construction of `Transformer.CRASP.FiniteFunction` needs is
that these determine the number (`ext_of_bit`): two mantissas whose shifts
`m / 2^j` all have the same parity for `j < K` agree modulo `2^K`
(`emod_pow_eq_of_parity`), and two mantissas of `Fx p s` differ by less than
`2^{p+1}`.  The range `b < p + 2` of `ext_of_bit` is not tight for `p ≥ 1`,
where `b < p + 1` suffices; it is what also covers `p = 0`, where `Fx 0 s`
still holds the two mantissas `-1` and `0` and only bit `1` tells them apart.
-/

import Transformer.CRASP.Fixed

namespace Transformer
namespace CRASP
namespace Fx

variable {p s : ℕ}

/-- **The bits read the mantissa alone.**  Bit `b + 1` of `x` is the parity of
`m / 2^b`: the `2^{-s}` of `Fx.val` and the `2^{s+1-b}` of `Fx.bit` cancel. -/
theorem bit_succ (x : Fx p s) (b : ℕ) : x.bit (b + 1) = decide (Odd (x.m / 2 ^ b)) := by
  have h2 : (2 : ℝ) ≠ 0 := two_ne_zero
  have hval : x.val * (2 : ℝ) ^ ((s : ℤ) + 1 - ((b + 1 : ℕ) : ℤ))
      = (x.m : ℝ) / ((2 ^ b : ℕ) : ℝ) := by
    rw [show ((s : ℤ) + 1 - ((b + 1 : ℕ) : ℤ)) = (s : ℤ) - (b : ℤ) by push_cast; ring,
      zpow_sub₀ h2, zpow_natCast, zpow_natCast, val]
    push_cast
    field_simp
  rw [bit, hval, Int.floor_div_natCast, Int.floor_intCast]
  push_cast
  rfl

/-- **Parity of the shifts determines the residue.**  If `a / 2^j` and
`b / 2^j` have the same parity for every `j < K`, then `a` and `b` agree
modulo `2^K`: the bits below `K` are the binary digits of the residue. -/
theorem emod_pow_eq_of_parity {a b : ℤ} {K : ℕ}
    (h : ∀ j < K, a / 2 ^ j % 2 = b / 2 ^ j % 2) :
    a % 2 ^ K = b % 2 ^ K := by
  induction K with
  | zero => simp
  | succ K ih =>
    have key : ∀ c : ℤ, c % 2 ^ (K + 1) = 2 ^ K * (c / 2 ^ K % 2) + c % 2 ^ K := by
      intro c
      have hpow : (0 : ℤ) < 2 ^ K := by positivity
      have hr0 : 0 ≤ c % 2 ^ K := Int.emod_nonneg c (by positivity)
      have hr1 : c % 2 ^ K < 2 ^ K := Int.emod_lt_of_pos c hpow
      have ht0 : 0 ≤ c / 2 ^ K % 2 := Int.emod_nonneg _ (by norm_num)
      have ht1 : c / 2 ^ K % 2 < 2 := Int.emod_lt_of_pos _ (by norm_num)
      have h1 : c / 2 ^ K * 2 ^ K + c % 2 ^ K = c := Int.ediv_mul_add_emod c (2 ^ K)
      have h2 : c / 2 ^ K / 2 * 2 + c / 2 ^ K % 2 = c / 2 ^ K := Int.ediv_mul_add_emod _ 2
      have hc : c = 2 ^ K * (c / 2 ^ K % 2) + c % 2 ^ K + 2 ^ (K + 1) * (c / 2 ^ K / 2) := by
        rw [pow_succ]
        linear_combination -h1 - 2 ^ K * h2
      have hlt : 2 ^ K * (c / 2 ^ K % 2) + c % 2 ^ K < 2 ^ (K + 1) := by
        have : 2 ^ K * (c / 2 ^ K % 2) ≤ 2 ^ K * 1 := by
          exact mul_le_mul_of_nonneg_left (by omega) hpow.le
        rw [pow_succ]
        linarith
      conv_lhs => rw [hc]
      rw [Int.add_mul_emod_self_left, Int.emod_eq_of_lt (by positivity) hlt]
    rw [key a, key b, ih fun j hj => h j (Nat.lt_succ_of_lt hj),
      h K (Nat.lt_succ_self K)]

/-- **A fixed-precision number is its bits.**  Two numbers of `Fx p s` whose
bits agree below `p + 2` are equal: the mantissas lie in an interval of length
`2^{max(p,1)}` and agree modulo `2^{p+1}`. -/
theorem ext_of_bit {x y : Fx p s} (h : ∀ c < p + 2, x.bit c = y.bit c) : x = y := by
  have hpar : ∀ j < p + 1, x.m / 2 ^ j % 2 = y.m / 2 ^ j % 2 := by
    intro j hj
    have hb := h (j + 1) (by omega)
    rw [bit_succ, bit_succ, decide_eq_decide] at hb
    rcases Int.even_or_odd (x.m / 2 ^ j) with hev | hod
    · have hy : ¬ Odd (y.m / 2 ^ j) := fun hc => (Int.not_odd_iff_even.mpr hev) (hb.mpr hc)
      rw [Int.even_iff.mp hev, Int.even_iff.mp (Int.not_odd_iff_even.mp hy)]
    · rw [Int.odd_iff.mp hod, Int.odd_iff.mp (hb.mp hod)]
  have hmod := emod_pow_eq_of_parity hpar
  have hdvd : (2 : ℤ) ^ (p + 1) ∣ y.m - x.m := Int.ModEq.dvd hmod
  have hmono : (2 : ℤ) ^ (p - 1) ≤ 2 ^ p :=
    pow_le_pow_right₀ (by norm_num) (Nat.sub_le p 1)
  have hlt : |y.m - x.m| < 2 ^ (p + 1) := by
    have hx := x.lo; have hx' := x.hi; have hy := y.lo; have hy' := y.hi
    rw [abs_lt, pow_succ]
    constructor <;> linarith
  exact ext (by have := Int.eq_zero_of_abs_lt_dvd hdvd hlt; omega)

/-- Bit `0` carries nothing: it is the parity of `2m`. -/
theorem bit_zero (x : Fx p s) : x.bit 0 = false := by
  have h2 : (2 : ℝ) ≠ 0 := two_ne_zero
  have hval : x.val * (2 : ℝ) ^ ((s : ℤ) + 1 - ((0 : ℕ) : ℤ)) = ((2 * x.m : ℤ) : ℝ) := by
    rw [show ((s : ℤ) + 1 - ((0 : ℕ) : ℤ)) = (s : ℤ) + 1 by push_cast; ring,
      zpow_add₀ h2, zpow_natCast, val]
    push_cast
    field_simp
  rw [bit, hval, Int.floor_intCast, decide_eq_false_iff_not, Int.not_odd_iff_even]
  exact even_two_mul _

/-- The two hypotheses meet: at `p = 0` the two mantissas `-1` and `0` are
told apart by bit `1`, and by no bit below it. -/
example : (⟨-1, by norm_num, by norm_num⟩ : Fx 0 0).bit 1
    ≠ (⟨0, by norm_num, by norm_num⟩ : Fx 0 0).bit 1 := by
  rw [bit_succ, bit_succ]
  norm_num

end Fx
end CRASP
end Transformer
