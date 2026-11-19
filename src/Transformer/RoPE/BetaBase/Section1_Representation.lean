module

public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Data.Nat.Digits.Lemmas

/-!
# RoPE is a β-base encoding: digits, extrapolation, interpolation and base conversion

Su Jianlin, *Transformer升级之路：10、RoPE是一种β进制编码* ("The road to upgrading the Transformer,
10: RoPE is a β-base encoding"), kexue.fm/archives/9675, 2023-07-06. The post's sections 进制表示
(Base representation), 直接外推 (Direct extrapolation), 线性内插 (Linear interpolation), 进制转换
(Base conversion), 位置编码 (Position encoding), 追根溯源 (Tracing the origin), 个人测试 (Personal
tests) and 写在最后 (Closing) are unnumbered; the files count them 1–8. This file is §§1–4. The
`i`-th digit of `n` in base `β`, counted from `0` on the right, is `n / β^i % β`
(`Nat.getD_digits`); it is below `β` (`Nat.mod_lt`).

* `eq_of_digits_eq`: §1, input `n < 1000` as its digits `[a, b, c]`: "we narrowed the span of the
  numbers without shrinking the gap between adjacent ones": the digits are below `β`, and below
  `β^L` the `L` digits determine the number, so distinct numbers differ by a whole unit in some
  digit;
* `digits_length_le_of_le`: §1, "we can shrink the base further, to 8, 6 or even 2, at the cost of
  more input dimensions";
* `digit_eq_zero_iff`: §2, a digit reserved above the trained ones is "set to 0 in training": it is
  `0` exactly below `β^L`; numbers below `2000` take a fourth decimal digit;
* `interpolated_floor_step`: §3, dividing by `k ≥ 1` crowds the units digit, adjacent numbers now
  `1/k` apart, while "the hundreds and tens still keep the adjacent difference 1": `⌊x/β^i⌋` steps
  by `0` or `1`. That the divided numbers stay below the trained length is
  `ReRoPE.exists_interpolated_lt`;
* `lt_of_digit_lt`: §4, "the comparison rules are exactly the same (the model does not know in which
  base you input)": in every base the highest differing digit decides the order;
* `ceil_rpow_le_iff`, `ceil_cbrt_two_thousand`: §4, three hexadecimal digits reach
  `16³ − 1 = 4095 > 1999`, and the smaller base `⌈∛2000⌉ = 13` is the least that suffices.
-/

@[expose] public section

namespace Transformer.RoPE.BetaBase

/-- Base representation: below `β^L` the `L` digits in base `β` determine the number. -/
theorem eq_of_digits_eq {β : ℕ} (hβ : 0 < β) {L : ℕ} : ∀ {n m : ℕ}, n < β ^ L → m < β ^ L →
    (∀ i < L, n / β ^ i % β = m / β ^ i % β) → n = m := by
  induction L with
  | zero =>
    intro n m hn hm _
    simp only [pow_zero, Nat.lt_one_iff] at hn hm
    rw [hn, hm]
  | succ L ih =>
    intro n m hn hm h
    have h0 := h 0 (by omega)
    simp only [pow_zero, Nat.div_one] at h0
    have hdiv : n / β = m / β := by
      refine ih ?_ ?_ fun i hi => ?_
      · rwa [Nat.div_lt_iff_lt_mul hβ, ← pow_succ]
      · rwa [Nat.div_lt_iff_lt_mul hβ, ← pow_succ]
      · rw [Nat.div_div_eq_div_mul, Nat.div_div_eq_div_mul, ← pow_succ']
        exact h (i + 1) (by omega)
    rw [← Nat.div_add_mod n β, ← Nat.div_add_mod m β, hdiv, h0]

example : 749 = 749 :=
  eq_of_digits_eq (β := 10) (L := 3) (by norm_num) (by norm_num) (by norm_num) fun _ _ => rfl

/-- Base representation: a smaller base needs at least as many digits. -/
theorem digits_length_le_of_le {b c : ℕ} (hb : 2 ≤ b) (hbc : b ≤ c) (n : ℕ) :
    (c.digits n).length ≤ (b.digits n).length := by
  rw [Nat.digits_length_le_iff (by omega)]
  calc n < b ^ (b.digits n).length := (Nat.digits_length_le_iff (by omega) n).mp le_rfl
    _ ≤ c ^ (b.digits n).length := Nat.pow_le_pow_left hbc _

example : (Nat.digits 10 999).length ≤ (Nat.digits 2 999).length :=
  digits_length_le_of_le (by norm_num) (by norm_num) 999

/-- Direct extrapolation: below `β^{L+1}`, the digit `L` is `0` exactly below `β^L`. A digit
reserved for the numbers `β^L, …, β^{L+1} − 1` is never anything but `0` in training. -/
theorem digit_eq_zero_iff {β L n : ℕ} (hβ : 0 < β) (hn : n < β ^ (L + 1)) :
    n / β ^ L % β = 0 ↔ n < β ^ L := by
  have hlt : n / β ^ L < β := Nat.div_lt_of_lt_mul (by rwa [← pow_succ])
  rw [Nat.mod_eq_of_lt hlt, Nat.div_eq_zero_iff]
  have hpos : 0 < β ^ L := Nat.pow_pos hβ
  omega

example : 1749 / 10 ^ 3 % 10 ≠ 0 :=
  fun h => absurd ((digit_eq_zero_iff (by norm_num) (by norm_num)).mp h) (by norm_num)

/-- Linear interpolation: for `k ≥ 1` and a place value `y ≥ 1`, the higher digit `⌊x/y⌋` steps by
`0` or `1` from `x` to `x + 1/k`, while the units move by `1/k`. -/
theorem interpolated_floor_step {k y : ℝ} (hk : 1 ≤ k) (hy : 1 ≤ y) (x : ℝ) :
    ⌊x / y⌋ ≤ ⌊(x + 1 / k) / y⌋ ∧ ⌊(x + 1 / k) / y⌋ ≤ ⌊x / y⌋ + 1 := by
  have hk0 : 0 < k := by linarith
  have hy0 : 0 < y := by linarith
  have h1 : 0 ≤ 1 / k := by positivity
  have h2 : 1 / k / y ≤ 1 := by
    rw [div_div, div_le_one (by positivity)]
    nlinarith
  constructor
  · exact Int.floor_le_floor (div_le_div_of_nonneg_right (by linarith) hy0.le)
  · rw [← Int.floor_add_one]
    exact Int.floor_le_floor (by rw [add_div]; linarith)

example (n : ℕ) : ⌊(n / 2 : ℝ) / 10⌋ ≤ ⌊((n + 1 : ℕ) / 2 : ℝ) / 10⌋ ∧
    ⌊((n + 1 : ℕ) / 2 : ℝ) / 10⌋ ≤ ⌊(n / 2 : ℝ) / 10⌋ + 1 := by
  have h := interpolated_floor_step (k := 2) (y := 10) (by norm_num) (by norm_num) (n / 2)
  rwa [show (n : ℝ) / 2 + 1 / 2 = ((n + 1 : ℕ) : ℝ) / 2 by push_cast; ring] at h

/-- Base conversion: if `n` and `m` agree in the digits above `i` and `n` has the smaller `i`-th
digit, then `n < m`, in every base `β > 0`. -/
theorem lt_of_digit_lt {β n m i : ℕ} (hhigh : n / β ^ (i + 1) = m / β ^ (i + 1))
    (hi : n / β ^ i % β < m / β ^ i % β) : n < m := by
  have hn := Nat.div_add_mod (n / β ^ i) β
  have hm := Nat.div_add_mod (m / β ^ i) β
  rw [Nat.div_div_eq_div_mul, ← pow_succ, hhigh] at hn
  rw [Nat.div_div_eq_div_mul, ← pow_succ] at hm
  have hlt : n / β ^ i < m / β ^ i := by omega
  by_contra h
  exact absurd (Nat.div_le_div_right (not_lt.mp h)) (not_le.mpr hlt)

example : 874 < 875 := lt_of_digit_lt (β := 16) (i := 0) (by norm_num) (by norm_num)

/-- Base conversion: `L` digits in base `b` cover `0, …, N − 1` iff `b ≥ ⌈N^{1/L}⌉`. -/
theorem ceil_rpow_le_iff {N b L : ℕ} (hL : 0 < L) :
    ⌈(N : ℝ) ^ ((L : ℝ)⁻¹)⌉₊ ≤ b ↔ N ≤ b ^ L := by
  rw [Nat.ceil_le, Real.rpow_inv_le_iff_of_pos (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    (by exact_mod_cast hL), Real.rpow_natCast]
  norm_cast

example : ⌈((2000 : ℕ) : ℝ) ^ (((3 : ℕ) : ℝ)⁻¹)⌉₊ ≤ 13 :=
  (ceil_rpow_le_iff (by norm_num)).mpr (by norm_num)

/-- Base conversion: "1749 becomes `[6, 13, 5]`", three hexadecimal digits cover all `n < 2000`,
and `⌈∛2000⌉ = 13` is the least base whose three digits cover them. -/
theorem ceil_cbrt_two_thousand :
    Nat.digits 16 1749 = [5, 13, 6] ∧ (∀ n < 2000, (Nat.digits 16 n).length ≤ 3) ∧
      ⌈(2000 : ℝ) ^ ((3 : ℝ)⁻¹)⌉₊ = 13 ∧ IsLeast {b : ℕ | 2000 ≤ b ^ 3} 13 := by
  have h := fun b => ceil_rpow_le_iff (N := 2000) (b := b) (L := 3) (by norm_num)
  push_cast at h
  have h13 : ⌈(2000 : ℝ) ^ ((3 : ℝ)⁻¹)⌉₊ = 13 := by
    refine le_antisymm ((h 13).mpr (by norm_num)) ?_
    by_contra h12
    exact absurd ((h 12).mp (by omega)) (by norm_num)
  refine ⟨by norm_num, fun n hn => ?_, h13, by norm_num, fun b hb => h13 ▸ (h b).mpr hb⟩
  rw [Nat.digits_length_le_iff (by norm_num)]
  omega

end Transformer.RoPE.BetaBase
