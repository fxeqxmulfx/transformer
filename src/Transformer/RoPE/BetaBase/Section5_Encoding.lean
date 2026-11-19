module

public import Mathlib.Algebra.Order.Floor.Semifield
public import Mathlib.Analysis.SpecialFunctions.Pow.Continuity
public import Transformer.RoPE.BetaBase.Section1_Representation
public import Transformer.RoPE.YaRN.SectionA2_NTK

/-!
# RoPE is a β-base encoding: position encoding and the origin of NTK-aware scaling

Su, *Transformer升级之路：10、RoPE是一种β进制编码*, kexue.fm/archives/9675, sections 位置编码
(Position encoding) and 追根溯源 (Tracing the origin), Eqs. (1)–(3). RoPE's frequencies are
`frequency base d i = base^{−2i/d}` (`RoPE.Basic`).

* `digit_eq_floor`: Eq. (1), the `m`-th digit of `n` from the right in base `β` is
  `⌊n/β^{m−1}⌋ mod β`, here with `i = m − 1`. The next sentence says "first divide by `β^{k−1}`",
  for `β^{m−1}`;
* `floor_mod_add`: "the most important property of the modulo is periodicity": the digit is a
  `β`-periodic function of the quotient `n/β^{m−1}`. The `cos` and `sin` of Eq. (2) are
  `2π`-periodic functions of the same quotient (`Real.cos_periodic`), so "apart from the floor"
  the two agree up to the period;
* `mul_frequency_eq`: Eq. (2), RoPE's angle of `n` at pair `i` is `n/β^i` with `β = 10000^{2/d}`.
  Position Interpolation replaces `n` by `n/k` (`RoPE.interpolated_phase`);
* `range_le_iff`, `range_mul`: "to enlarge the range `k` times, the base `β` must become at least
  `β k^{2/d}` (… `d/2` digits in base `β`, so the `d/2`-th root and not the `d`-th), or
  equivalently the base `10000` becomes `10000k`";
* `eq3_iff`: "high frequencies extrapolate, low frequencies interpolate": Eq. (3),
  `n/(βλ)^{d/2−1} = (n/k)/β^{d/2−1}` iff `λ = k^{2/(d−2)}`. `ntk_base_eq`: the
  base `(βλ)^{d/2}` is then bloc97's `10000·k^{d/(d−2)}`, YaRN's `adjustedBase`;
* `tendsto_lambda`: "since `d` is usually large, `λ` is very close to `1`". The text calls `n/β` the
  highest-frequency term. The highest is `n/β^0 = n`, which no change of base touches
  (`YaRN.highest_frequency_preserved`); `n/β` is the second, divided by `λ`
  (`YaRN.coordinate_frequency_adjusted` at `i = 1`);
* `tendsto_ratio`: "since `d` is large (BERT 64, LLaMA 128), `k^{2/(d−2)}` differs little from
  `k^{2/d}`": their ratio tends to `1`.

The table of 个人测试 (Personal tests) holds measurements and is not formalized.
-/

@[expose] public section

open Filter Topology

namespace Transformer.RoPE.BetaBase

/-- Eq. (1): the digit `i` of `n` in base `β ≥ 2`, counted from `0` on the right, is
`⌊n/β^i⌋ mod β`. -/
theorem digit_eq_floor {β : ℕ} (hβ : 2 ≤ β) (n i : ℕ) :
    (β.digits n).getD i 0 = ⌊(n : ℝ) / (β : ℝ) ^ i⌋₊ % β := by
  rw [Nat.getD_digits n i hβ, ← Nat.cast_pow, Nat.floor_div_eq_div]

example : (Nat.digits 16 1749).getD 1 0 = ⌊(1749 : ℝ) / (16 : ℝ) ^ 1⌋₊ % 16 := by
  exact_mod_cast digit_eq_floor (β := 16) (by norm_num) 1749 1

/-- Position encoding: the digit is a `β`-periodic function of the quotient, `⌊x + β⌋ mod β =
⌊x⌋ mod β` for `x ≥ 0`. -/
theorem floor_mod_add (β : ℕ) {x : ℝ} (hx : 0 ≤ x) : ⌊x + β⌋₊ % β = ⌊x⌋₊ % β := by
  rw [Nat.floor_add_natCast hx, Nat.add_mod_right]

example : ⌊(874.5 : ℝ) + (10 : ℕ)⌋₊ % 10 = ⌊(874.5 : ℝ)⌋₊ % 10 :=
  floor_mod_add 10 (by norm_num)

/-- Eq. (2): RoPE's angle of position `n` at pair `i` is `n/β^i` with `β = base^{2/d}`. -/
theorem mul_frequency_eq {base : ℝ} (hbase : 0 < base) (d i : ℕ) (n : ℝ) :
    n * frequency base d i = n / (base ^ (2 / (d : ℝ))) ^ i := by
  have h : (base ^ (2 / (d : ℝ))) ^ i = base ^ (2 * (i : ℝ) / d) := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul hbase.le]
    congr 1
    ring
  rw [h, frequency, neg_div, Real.rpow_neg hbase.le, ← div_eq_mul_inv]

example (n : ℝ) : n * frequency 10000 128 5 = n / (10000 ^ (2 / ((128 : ℕ) : ℝ))) ^ 5 :=
  mul_frequency_eq (by norm_num) 128 5 n

/-- Base conversion: `d/2` digits in base `β'` reach `k` times as far as in base `β`,
`β'^{d/2} ≥ k β^{d/2}`, iff `β' ≥ β k^{2/d}`. -/
theorem range_le_iff {β β' k d : ℝ} (hβ : 0 ≤ β) (hβ' : 0 ≤ β') (hk : 0 ≤ k) (hd : 0 < d) :
    k * β ^ (d / 2) ≤ β' ^ (d / 2) ↔ β * k ^ (2 / d) ≤ β' := by
  have hd0 : d ≠ 0 := hd.ne'
  have h : (β * k ^ (2 / d)) ^ (d / 2) = k * β ^ (d / 2) := by
    rw [Real.mul_rpow hβ (Real.rpow_nonneg hk _), ← Real.rpow_mul hk,
      show 2 / d * (d / 2) = 1 by field_simp, Real.rpow_one, mul_comm]
  rw [← h, Real.rpow_le_rpow_iff (mul_nonneg hβ (Real.rpow_nonneg hk _)) hβ' (by positivity)]

example : 8 * (10000 : ℝ) ^ ((128 : ℝ) / 2) ≤
    ((10000 : ℝ) * 8 ^ (2 / (128 : ℝ))) ^ ((128 : ℝ) / 2) :=
  (range_le_iff (by norm_num) (by positivity) (by norm_num) (by norm_num)).mpr le_rfl

/-- Base conversion: replacing the base `base` by `base·k` turns `β = base^{2/d}` into
`β k^{2/d}`, and the range `β^{d/2}` into exactly `k` times it. -/
theorem range_mul {base k d : ℝ} (hbase : 0 ≤ base) (hk : 0 ≤ k) (hd : d ≠ 0) :
    (base * k) ^ (2 / d) = base ^ (2 / d) * k ^ (2 / d) ∧
      ((base * k) ^ (2 / d)) ^ (d / 2) = k * (base ^ (2 / d)) ^ (d / 2) := by
  refine ⟨Real.mul_rpow hbase hk, ?_⟩
  rw [← Real.rpow_mul (mul_nonneg hbase hk), ← Real.rpow_mul hbase,
    show 2 / d * (d / 2) = 1 by field_simp, Real.rpow_one, Real.rpow_one, mul_comm]

example : ((10000 * 8 : ℝ) ^ (2 / (128 : ℝ))) ^ ((128 : ℝ) / 2) =
    8 * ((10000 : ℝ) ^ (2 / (128 : ℝ))) ^ ((128 : ℝ) / 2) :=
  (range_mul (by norm_num) (by norm_num) (by norm_num)).2

/-- Eq. (3): interpolating the slowest term, `n/(βλ)^{d/2−1} = (n/k)/β^{d/2−1}`, holds iff
`λ = k^{2/(d−2)}`. -/
theorem eq3_iff {β k lam d n : ℝ} (hβ : 0 < β) (hk : 0 < k) (hlam : 0 < lam) (hd : 2 < d)
    (hn : n ≠ 0) :
    n / (β * lam) ^ (d / 2 - 1) = n / k / β ^ (d / 2 - 1) ↔ lam = k ^ (2 / (d - 2)) := by
  have he : d / 2 - 1 ≠ 0 := by
    intro h
    linarith
  have hinv : 2 / (d - 2) = (d / 2 - 1)⁻¹ := by
    rw [show d / 2 - 1 = (d - 2) / 2 by ring, inv_div]
  have hβe : 0 < β ^ (d / 2 - 1) := Real.rpow_pos_of_pos hβ _
  have hle : 0 < lam ^ (d / 2 - 1) := Real.rpow_pos_of_pos hlam _
  rw [Real.mul_rpow hβ.le hlam.le, div_div, div_eq_div_iff (by positivity) (by positivity)]
  constructor
  · intro h
    have h1 := mul_left_cancel₀ hn h
    have h2 : lam ^ (d / 2 - 1) * β ^ (d / 2 - 1) = k * β ^ (d / 2 - 1) := by linarith
    rw [hinv, ← mul_right_cancel₀ hβe.ne' h2, Real.rpow_rpow_inv hlam.le he]
  · intro h
    rw [h, hinv, Real.rpow_inv_rpow hk.le he]
    ring

example : (4096 : ℝ) / ((10000 : ℝ) ^ (2 / (128 : ℝ)) * 8 ^ (2 / ((128 : ℝ) - 2))) ^
    ((128 : ℝ) / 2 - 1) = 4096 / 8 / ((10000 : ℝ) ^ (2 / (128 : ℝ))) ^ ((128 : ℝ) / 2 - 1) :=
  (eq3_iff (by positivity) (by norm_num) (by positivity) (by norm_num) (by norm_num)).mpr rfl

/-- Tracing the origin: with `β = base^{2/d}` and `λ = k^{2/(d−2)}` the base is
`(βλ)^{d/2} = base·k^{d/(d−2)}`, bloc97's NTK-aware base. -/
theorem ntk_base_eq {base k d : ℝ} (hbase : 0 ≤ base) (hk : 0 ≤ k) (hd : 2 < d) :
    (base ^ (2 / d) * k ^ (2 / (d - 2))) ^ (d / 2) = YaRN.adjustedBase base k d := by
  have hd0 : d ≠ 0 := by positivity
  have hd2 : d - 2 ≠ 0 := by
    intro h
    linarith
  rw [Real.mul_rpow (Real.rpow_nonneg hbase _) (Real.rpow_nonneg hk _), ← Real.rpow_mul hbase,
    ← Real.rpow_mul hk, show 2 / d * (d / 2) = 1 by field_simp, Real.rpow_one, YaRN.adjustedBase]
  congr 2
  field_simp

example : ((10000 : ℝ) ^ (2 / (128 : ℝ)) * 8 ^ (2 / ((128 : ℝ) - 2))) ^ ((128 : ℝ) / 2) =
    YaRN.adjustedBase 10000 8 128 :=
  ntk_base_eq (by norm_num) (by norm_num) (by norm_num)

/-- Tracing the origin: "since `d` is usually large, `λ` is very close to `1`":
`k^{2/(d−2)} → 1` as `d → ∞`. -/
theorem tendsto_lambda {k : ℝ} (hk : 0 < k) :
    Tendsto (fun d : ℝ => k ^ (2 / (d - 2))) atTop (𝓝 1) := by
  have hsub : Tendsto (fun d : ℝ => d - 2) atTop atTop := by
    simpa [sub_eq_add_neg] using tendsto_atTop_add_const_right atTop (-2) tendsto_id
  have h2 : Tendsto (fun d : ℝ => 2 / (d - 2)) atTop (𝓝 0) := tendsto_const_nhds.div_atTop hsub
  have h := (Real.continuousAt_const_rpow (b := 0) hk.ne').tendsto.comp h2
  rw [Real.rpow_zero] at h
  exact h

example : Tendsto (fun d : ℝ => (8 : ℝ) ^ (2 / (d - 2))) atTop (𝓝 1) :=
  tendsto_lambda (by norm_num)

/-- Tracing the origin: `k^{2/(d−2)}` and Su's `k^{2/d}` "differ little" for large `d`: their
ratio tends to `1`. -/
theorem tendsto_ratio {k : ℝ} (hk : 0 < k) :
    Tendsto (fun d : ℝ => k ^ (2 / (d - 2)) / k ^ (2 / d)) atTop (𝓝 1) := by
  have h0 : Tendsto (fun d : ℝ => 2 / d) atTop (𝓝 0) := tendsto_const_nhds.div_atTop tendsto_id
  have h2 := (Real.continuousAt_const_rpow (b := 0) hk.ne').tendsto.comp h0
  rw [Real.rpow_zero] at h2
  have h := (tendsto_lambda hk).div h2 one_ne_zero
  rw [div_one] at h
  exact h

example : Tendsto (fun d : ℝ => (8 : ℝ) ^ (2 / (d - 2)) / 8 ^ (2 / d)) atTop (𝓝 1) :=
  tendsto_ratio (by norm_num)

end Transformer.RoPE.BetaBase
