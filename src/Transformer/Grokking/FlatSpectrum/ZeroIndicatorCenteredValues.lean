module

public import Transformer.Grokking.FlatSpectrum.ZeroIndicatorCentered

/-!
# The centered indicator of answer `0`: values

Source: `review/flat_spectrum.md`, Torch check: "31.5 at p=97, 60.0 at p=199 | These are the
centered values", and "r_eff = 0.359p, 0.345p, 0.325p, 0.302p at p = 47, 61, 97, 199". The
closed form of `effectiveRank_centeredZeroIndicatorKernel` is `(c √2 + √(A + B √(p + 1)))² / D`
with `c = p(p − 3)/2`, `A = p³ − 2p² + 4p − 1`, `B = 2(p − 1)²` and `D = (2p − 1)(p − 1)²`. Its
values are `31.4984` at `p = 97` and `60.0249` at `p = 199`, and `r_eff/p` is `0.35866`,
`0.34496`, `0.32473` and `0.30163` at `p = 47, 61, 97, 199`. Each proof brackets `√2`,
`√(p + 1)` and the outer square root by rationals, then squares the bounds of the sum.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

/-- `review/flat_spectrum.md`, Torch check: the centered value is `31.5` at `p = 97`, as
`31.498 < r_eff < 31.499`. -/
theorem effectiveRank_centeredZeroIndicatorKernel_97 [Fact (Nat.Prime 97)] :
    31.498 < effectiveRank (centeredZeroIndicatorKernel 97) ∧
      effectiveRank (centeredZeroIndicatorKernel 97) < 31.499 := by
  have h2 : (97 : ℝ) * (97 - 3) / √2 = 4559 * √2 := by
    rw [div_eq_iff (by positivity), mul_assoc, Real.mul_self_sqrt (by norm_num)]
    norm_num
  rw [effectiveRank_centeredZeroIndicatorKernel (by norm_num),
    show ((97 : ℕ) : ℝ) = 97 by norm_num, h2,
    show (97 : ℝ) ^ 3 - 2 * 97 ^ 2 + 4 * 97 - 1 = 894242 by norm_num,
    show (2 : ℝ) * (97 - 1) ^ 2 = 18432 by norm_num, show (97 : ℝ) + 1 = 98 by norm_num,
    show ((2 : ℝ) * 97 - 1) * (97 - 1) ^ 2 = 1778688 by norm_num]
  have hq1 : (9.899494 : ℝ) < √98 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hq2 : √98 < (9.899495 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ht1 : (1.4142135 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have ht2 : √2 < (1.4142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hn1 : (1037.646 : ℝ) < √(894242 + 18432 * √98) := by
    rw [Real.lt_sqrt (by norm_num)]
    linarith
  have hn2 : √(894242 + 18432 * √98) < (1037.647 : ℝ) := by
    rw [Real.sqrt_lt' (by norm_num)]
    linarith
  have hlo := mul_self_lt_mul_self (by norm_num : (0 : ℝ) ≤ 7485.0453)
    (show (7485.0453 : ℝ) < 4559 * √2 + √(894242 + 18432 * √98) by linarith)
  have hhi := mul_self_lt_mul_self
    (by positivity : (0 : ℝ) ≤ 4559 * √2 + √(894242 + 18432 * √98))
    (show 4559 * √2 + √(894242 + 18432 * √98) < (7485.0469 : ℝ) by linarith)
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  · rw [div_lt_iff₀ (by norm_num)]
    nlinarith

example := @effectiveRank_centeredZeroIndicatorKernel_97 ⟨by norm_num⟩

/-- `review/flat_spectrum.md`, Torch check: the centered value is `60.0` at `p = 199`, as
`60.024 < r_eff < 60.025`. -/
theorem effectiveRank_centeredZeroIndicatorKernel_199 [Fact (Nat.Prime 199)] :
    60.024 < effectiveRank (centeredZeroIndicatorKernel 199) ∧
      effectiveRank (centeredZeroIndicatorKernel 199) < 60.025 := by
  have h2 : (199 : ℝ) * (199 - 3) / √2 = 19502 * √2 := by
    rw [div_eq_iff (by positivity), mul_assoc, Real.mul_self_sqrt (by norm_num)]
    norm_num
  rw [effectiveRank_centeredZeroIndicatorKernel (by norm_num),
    show ((199 : ℕ) : ℝ) = 199 by norm_num, h2,
    show (199 : ℝ) ^ 3 - 2 * 199 ^ 2 + 4 * 199 - 1 = 7802192 by norm_num,
    show (2 : ℝ) * (199 - 1) ^ 2 = 78408 by norm_num, show (199 : ℝ) + 1 = 200 by norm_num,
    show ((2 : ℝ) * 199 - 1) * (199 - 1) ^ 2 = 15563988 by norm_num]
  have hq1 : (14.142135 : ℝ) < √200 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hq2 : √200 < (14.142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ht1 : (1.4142135 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have ht2 : √2 < (1.4142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hn1 : (2985.137 : ℝ) < √(7802192 + 78408 * √200) := by
    rw [Real.lt_sqrt (by norm_num)]
    linarith
  have hn2 : √(7802192 + 78408 * √200) < (2985.138 : ℝ) := by
    rw [Real.sqrt_lt' (by norm_num)]
    linarith
  have hlo := mul_self_lt_mul_self (by norm_num : (0 : ℝ) ≤ 30565.1286)
    (show (30565.1286 : ℝ) < 19502 * √2 + √(7802192 + 78408 * √200) by linarith)
  have hhi := mul_self_lt_mul_self
    (by positivity : (0 : ℝ) ≤ 19502 * √2 + √(7802192 + 78408 * √200))
    (show 19502 * √2 + √(7802192 + 78408 * √200) < (30565.1317 : ℝ) by linarith)
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  · rw [div_lt_iff₀ (by norm_num)]
    nlinarith

example := @effectiveRank_centeredZeroIndicatorKernel_199 ⟨by norm_num⟩

/-- `review/flat_spectrum.md`, Torch check: "r_eff = 0.359p" at `p = 47`, as
`0.3586 < r_eff/47 < 0.3587`. -/
theorem effectiveRank_centeredZeroIndicatorKernel_div_47 [Fact (Nat.Prime 47)] :
    0.3586 < effectiveRank (centeredZeroIndicatorKernel 47) / 47 ∧
      effectiveRank (centeredZeroIndicatorKernel 47) / 47 < 0.3587 := by
  have h2 : (47 : ℝ) * (47 - 3) / √2 = 1034 * √2 := by
    rw [div_eq_iff (by positivity), mul_assoc, Real.mul_self_sqrt (by norm_num)]
    norm_num
  rw [effectiveRank_centeredZeroIndicatorKernel (by norm_num),
    show ((47 : ℕ) : ℝ) = 47 by norm_num, h2,
    show (47 : ℝ) ^ 3 - 2 * 47 ^ 2 + 4 * 47 - 1 = 99592 by norm_num,
    show (2 : ℝ) * (47 - 1) ^ 2 = 4232 by norm_num, show (47 : ℝ) + 1 = 48 by norm_num,
    show ((2 : ℝ) * 47 - 1) * (47 - 1) ^ 2 = 196788 by norm_num, div_div]
  have hq1 : (6.928203 : ℝ) < √48 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hq2 : √48 < (6.928204 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ht1 : (1.4142135 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have ht2 : √2 < (1.4142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hn1 : (359.043 : ℝ) < √(99592 + 4232 * √48) := by
    rw [Real.lt_sqrt (by norm_num)]
    linarith
  have hn2 : √(99592 + 4232 * √48) < (359.044 : ℝ) := by
    rw [Real.sqrt_lt' (by norm_num)]
    linarith
  have hlo := mul_self_lt_mul_self (by norm_num : (0 : ℝ) ≤ 1821.3397)
    (show (1821.3397 : ℝ) < 1034 * √2 + √(99592 + 4232 * √48) by linarith)
  have hhi := mul_self_lt_mul_self (by positivity : (0 : ℝ) ≤ 1034 * √2 + √(99592 + 4232 * √48))
    (show 1034 * √2 + √(99592 + 4232 * √48) < (1821.3409 : ℝ) by linarith)
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  · rw [div_lt_iff₀ (by norm_num)]
    nlinarith

example := @effectiveRank_centeredZeroIndicatorKernel_div_47 ⟨by norm_num⟩

/-- `review/flat_spectrum.md`, Torch check: "r_eff = 0.345p" at `p = 61`, as
`0.3449 < r_eff/61 < 0.3450`. -/
theorem effectiveRank_centeredZeroIndicatorKernel_div_61 [Fact (Nat.Prime 61)] :
    0.3449 < effectiveRank (centeredZeroIndicatorKernel 61) / 61 ∧
      effectiveRank (centeredZeroIndicatorKernel 61) / 61 < 0.3450 := by
  have h2 : (61 : ℝ) * (61 - 3) / √2 = 1769 * √2 := by
    rw [div_eq_iff (by positivity), mul_assoc, Real.mul_self_sqrt (by norm_num)]
    norm_num
  rw [effectiveRank_centeredZeroIndicatorKernel (by norm_num),
    show ((61 : ℕ) : ℝ) = 61 by norm_num, h2,
    show (61 : ℝ) ^ 3 - 2 * 61 ^ 2 + 4 * 61 - 1 = 219782 by norm_num,
    show (2 : ℝ) * (61 - 1) ^ 2 = 7200 by norm_num, show (61 : ℝ) + 1 = 62 by norm_num,
    show ((2 : ℝ) * 61 - 1) * (61 - 1) ^ 2 = 435600 by norm_num, div_div]
  have hq1 : (7.874007 : ℝ) < √62 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hq2 : √62 < (7.874008 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have ht1 : (1.4142135 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have ht2 : √2 < (1.4142136 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hn1 : (525.808 : ℝ) < √(219782 + 7200 * √62) := by
    rw [Real.lt_sqrt (by norm_num)]
    linarith
  have hn2 : √(219782 + 7200 * √62) < (525.809 : ℝ) := by
    rw [Real.sqrt_lt' (by norm_num)]
    linarith
  have hlo := mul_self_lt_mul_self (by norm_num : (0 : ℝ) ≤ 3027.5516)
    (show (3027.5516 : ℝ) < 1769 * √2 + √(219782 + 7200 * √62) by linarith)
  have hhi := mul_self_lt_mul_self (by positivity : (0 : ℝ) ≤ 1769 * √2 + √(219782 + 7200 * √62))
    (show 1769 * √2 + √(219782 + 7200 * √62) < (3027.5529 : ℝ) by linarith)
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    nlinarith
  · rw [div_lt_iff₀ (by norm_num)]
    nlinarith

example := @effectiveRank_centeredZeroIndicatorKernel_div_61 ⟨by norm_num⟩

/-- `review/flat_spectrum.md`, Torch check: "r_eff = 0.325p" at `p = 97`, as
`0.3247 < r_eff/97 < 0.3248`. -/
theorem effectiveRank_centeredZeroIndicatorKernel_div_97 [Fact (Nat.Prime 97)] :
    0.3247 < effectiveRank (centeredZeroIndicatorKernel 97) / 97 ∧
      effectiveRank (centeredZeroIndicatorKernel 97) / 97 < 0.3248 := by
  obtain ⟨h1, h2⟩ := effectiveRank_centeredZeroIndicatorKernel_97
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    linarith
  · rw [div_lt_iff₀ (by norm_num)]
    linarith

example := @effectiveRank_centeredZeroIndicatorKernel_div_97 ⟨by norm_num⟩

/-- `review/flat_spectrum.md`, Torch check: "r_eff = 0.302p" at `p = 199`, as
`0.3016 < r_eff/199 < 0.3017`. -/
theorem effectiveRank_centeredZeroIndicatorKernel_div_199 [Fact (Nat.Prime 199)] :
    0.3016 < effectiveRank (centeredZeroIndicatorKernel 199) / 199 ∧
      effectiveRank (centeredZeroIndicatorKernel 199) / 199 < 0.3017 := by
  obtain ⟨h1, h2⟩ := effectiveRank_centeredZeroIndicatorKernel_199
  constructor
  · rw [lt_div_iff₀ (by norm_num)]
    linarith
  · rw [div_lt_iff₀ (by norm_num)]
    linarith

example := @effectiveRank_centeredZeroIndicatorKernel_div_199 ⟨by norm_num⟩

end Transformer.Grokking.FlatSpectrum
