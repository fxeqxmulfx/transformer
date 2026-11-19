module

public import Mathlib.Analysis.Real.Pi.Bounds
public import Transformer.RoPE.ScalingLaws.Section3_Periods
public import Transformer.RoPE.YaRN.SectionA2_NTK

/-!
# §2.2 Out-of-distribution rotation angles

Men et al., *Base of RoPE Bounds Context Length*, arXiv:2405.14591v1, §2.2. Eq. (6): the
NTK-aware base `b_new = b s^{d/(d−2)}`, where `s = T_new/T_origin`, "is derived from
`T_new b_new^{−(d−2)/d} = T_origin b^{−(d−2)/d}` to ensure that the lowest frequency part being
interpolated". Then: "A recent study proposes to set a much smaller base (e.g. 500), in which case
`θ_i = base^{−2i/d}` is small enough and typical training length (say 4,096) fully covers the
period of `cos(t − s)θ_i`".

* `ntk_identity`: the derivation of Eq. (6), from YaRN's `slow_phase_preserved`;
* `period_lt_of_base_500`: with base `500` every period `2π · 500^{2i/d}` is below `4096`.

Covering every period does not give a long context: by §4.3 (`exists_ropeKernel_neg`) the
context length of base `500`, like that of every base, is finite.
-/

@[expose] public section

namespace Transformer.RoPE.BaseBounds

/-- Eq. (6): the NTK-aware base `b s^{d/(d−2)}` keeps the slowest phase over the stretched
window, `T_new b_new^{−(d−2)/d} = T_origin b^{−(d−2)/d}` with `T_new = s T_origin`. -/
theorem ntk_identity {b s d : ℝ} (hb : 0 < b) (hs : 0 < s) (hd : 2 < d) (T : ℝ) :
    (s * T) * (b * s ^ (d / (d - 2))) ^ (-(d - 2) / d) = T * b ^ (-(d - 2) / d) := by
  rw [mul_comm (s * T), mul_comm T]
  simpa only [slowFrequency, YaRN.adjustedBase] using YaRN.slow_phase_preserved hb hs hd T

example : (8 * 4096) * (10000 * (8 : ℝ) ^ ((128 : ℝ) / (128 - 2))) ^ (-(128 - 2) / (128 : ℝ)) =
    4096 * (10000 : ℝ) ^ (-(128 - 2) / (128 : ℝ)) :=
  ntk_identity (by norm_num) (by norm_num) (by norm_num) 4096

/-- §2.2: with base `500` a training length of `4096` covers every period `2π · 500^{2i/d}`,
`i < d/2`: each is at most `1000π < 4096`. -/
theorem period_lt_of_base_500 {dimension index : ℕ} (hindex : 2 * index ≤ dimension) :
    period (frequency 500 dimension index) < 4096 := by
  rw [ScalingLaws.frequency_period (by norm_num)]
  have hexp : 2 * (index : ℝ) / dimension ≤ 1 := by
    rcases Nat.eq_zero_or_pos dimension with h | h
    · simp [h]
    · rw [div_le_one (by exact_mod_cast h)]
      exact_mod_cast hindex
  have h500 : (500 : ℝ) ^ (2 * (index : ℝ) / dimension) ≤ 500 := by
    simpa using Real.rpow_le_rpow_of_exponent_le (by norm_num : (1 : ℝ) ≤ 500) hexp
  nlinarith [Real.pi_lt_d2, Real.pi_pos]

example : period (frequency 500 128 63) < 4096 := period_lt_of_base_500 (by norm_num)

end Transformer.RoPE.BaseBounds
