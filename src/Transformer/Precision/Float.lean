/-
# Floating point has a context length

Round-to-nearest in an IEEE format sends everything below half the smallest
subnormal `σ` to `0` (`ieee_eq_zero`): `σ = 2^{-24}` in binary16, `2^{-133}` in
bfloat16, `2^{-149}` in binary32.  That dead zone is the hypothesis of
`qAttn_eq_zero`: softmax weights stored in such a format are all `0`, whatever
the values, once `n > 2 e^D / σ` for scores in a window of width `D`; in
binary16 that is `n > 2^{25} e^D` (`qAttn_f16_eq_zero`).

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

end Precision
end Transformer
