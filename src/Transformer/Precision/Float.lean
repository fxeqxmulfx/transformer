/-
# Floating point has a context length

Round-to-nearest in an IEEE format sends everything below half the smallest
subnormal `σ` to `0` (`ieee_eq_zero`): `σ = 2^{-24}` in binary16, `2^{-133}` in
bfloat16, `2^{-149}` in binary32.  That dead zone is the hypothesis of
`qAttn_eq_zero`: softmax weights stored in such a format are all `0`, whatever
the values, once `n > 2 e^D / σ` for scores in a window of width `D`; in
binary16 that is `n > 2^{25} e^D` (`qAttn_f16_eq_zero`), in the 8-bit floats
E5M2 and E4M3 of FP8 attention kernels `n > 2^{17} e^D` and `n > 2^{10} e^D`
(`qAttn_e5m2_eq_zero`, `qAttn_e4m3_eq_zero`).

FP8 is Micikevicius et al., "FP8 formats for deep learning",
arXiv:2209.05433, Table 1.  Its E4M3 gives up the infinities and keeps every
exponent pattern but one mantissa for normal numbers, so its largest number is
`448`, not the `240` of `grid 4 3`; the subnormals, and with them the dead
zone, are the same.

Round-to-nearest is IEEE 754-2008, §4.3.1.
-/

import Transformer.Precision.IEEE
import Transformer.Precision.ContextLength

namespace Transformer
namespace Precision

/-- **The dead zone of an IEEE format.**  Round-to-nearest sends every number
below half the smallest subnormal to `0`. -/
theorem ieee_eq_zero {E M : ℕ} (hE : 1 ≤ E) {Q : ℝ → ℝ} (hQ : IsNearest (grid E M) Q) {x : ℝ}
    (hx0 : 0 ≤ x) (hx : x < minSub E M / 2) : Q x = 0 :=
  hQ.eq_zero (zero_mem_grid hE M) (fun _ ⟨_, _, h⟩ hv => minSub_le_abs h hv) hx0 hx

/-- An IEEE format has a round-to-nearest map. -/
theorem exists_isNearest_grid {E : ℕ} (hE : 1 ≤ E) (M : ℕ) : ∃ Q, IsNearest (grid E M) Q :=
  exists_isNearest (grid_finite E M) ⟨0, zero_mem_grid hE M⟩

variable {n : ℕ} {E' : Type*} [AddCommGroup E'] [Module ℝ E']

/-- **Softmax weights stored in an IEEE format.**  With scores in a window of
width `D`, rounding the weights to nearest in the format zeroes the head's
output on every input once `n · σ/2 > e^D`, `σ` the smallest subnormal. -/
theorem qAttn_ieee_eq_zero {E M : ℕ} (hE : 1 ≤ E) {Q : ℝ → ℝ} (hQ : IsNearest (grid E M) Q) {D : ℝ}
    {s : Idx n → ℝ} (hs : ∀ i j, s i - s j ≤ D) (hn : Real.exp D < n * (minSub E M / 2))
    (v : Idx n → E') : qAttn Q s v = 0 :=
  qAttn_eq_zero (fun _ h0 h => ieee_eq_zero hE hQ h0 h) hs hn v

/-- **binary16 weights.**  Past `2^{25} e^D` tokens, softmax weights rounded
to binary16 are all `0`. -/
theorem qAttn_f16_eq_zero {Q : ℝ → ℝ} (hQ : IsNearest (grid 5 10) Q) {D : ℝ} {s : Idx n → ℝ}
    (hs : ∀ i j, s i - s j ≤ D) (hn : 2 ^ 25 * Real.exp D < n) (v : Idx n → E') :
    qAttn Q s v = 0 := by
  refine qAttn_ieee_eq_zero (by norm_num) hQ hs ?_ v
  rw [minSub_f16]; norm_num; linarith

/-- The hypotheses of `qAttn_f16_eq_zero` are satisfiable: a nearest rounding
to binary16 exists, and `2^{25} + 1` tokens exceed `2^{25} e^0`. -/
example : (∃ Q, IsNearest (grid 5 10) Q) ∧ (2 : ℝ) ^ 25 * Real.exp 0 < ((2 ^ 25 + 1 : ℕ) : ℝ) :=
  ⟨exists_isNearest_grid (by norm_num) 10, by norm_num⟩

/-- **FP8 E4M3 weights.**  Past `2^{10} e^D` tokens, softmax weights rounded
to E4M3 are all `0`: with equal scores, from `1025` tokens on. -/
theorem qAttn_e4m3_eq_zero {Q : ℝ → ℝ} (hQ : IsNearest (grid 4 3) Q) {D : ℝ} {s : Idx n → ℝ}
    (hs : ∀ i j, s i - s j ≤ D) (hn : 2 ^ 10 * Real.exp D < n) (v : Idx n → E') :
    qAttn Q s v = 0 := by
  refine qAttn_ieee_eq_zero (by norm_num) hQ hs ?_ v
  rw [minSub_e4m3]; norm_num; linarith

/-- The hypotheses of `qAttn_e4m3_eq_zero` are satisfiable: `1025` tokens of
equal score. -/
example : (∃ Q, IsNearest (grid 4 3) Q) ∧ (2 : ℝ) ^ 10 * Real.exp 0 < ((1025 : ℕ) : ℝ) :=
  ⟨exists_isNearest_grid (by norm_num) 3, by norm_num⟩

/-- **FP8 E5M2 weights.**  Past `2^{17} e^D` tokens, softmax weights rounded
to E5M2 are all `0`. -/
theorem qAttn_e5m2_eq_zero {Q : ℝ → ℝ} (hQ : IsNearest (grid 5 2) Q) {D : ℝ} {s : Idx n → ℝ}
    (hs : ∀ i j, s i - s j ≤ D) (hn : 2 ^ 17 * Real.exp D < n) (v : Idx n → E') :
    qAttn Q s v = 0 := by
  refine qAttn_ieee_eq_zero (by norm_num) hQ hs ?_ v
  rw [minSub_e5m2]; norm_num; linarith

/-- The hypotheses of `qAttn_e5m2_eq_zero` are satisfiable: `2^{17} + 1`
tokens of equal score. -/
example : (∃ Q, IsNearest (grid 5 2) Q) ∧ (2 : ℝ) ^ 17 * Real.exp 0 < ((2 ^ 17 + 1 : ℕ) : ℝ) :=
  ⟨exists_isNearest_grid (by norm_num) 2, by norm_num⟩

end Precision
end Transformer
