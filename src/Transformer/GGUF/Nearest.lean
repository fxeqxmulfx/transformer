/-
# Floating point has a context length

Any rounding to the nearest point of a grid that contains `0` and no point
closer to `0` than `σ` sends `[0, σ/2)` to `0` (`Precision.IsNearest.eq_zero`).
For an IEEE binary format `σ`
is the smallest subnormal, `2^{-24}` in binary16, `2^{-133}` in bfloat16,
`2^{-149}` in binary32.

That dead zone is exactly the hypothesis of `Precision.qAttn_eq_zero`: softmax
weights stored in such a format are all `0`, whatever the values, once
`n > 2 e^D / σ` for scores in a window of width `D`.  In binary16 that is
`n > 2^{25} e^D`.

Round-to-nearest is IEEE 754-2008, §4.3.1; the grid of an IEEE format is the
image of `ieee` (`Transformer.GGUF.Basic`).
-/

import Transformer.GGUF.Basic
import Transformer.Precision.ContextLength
import Transformer.Precision.Nearest
import Mathlib.Data.Set.Finite.Lemmas

namespace Transformer
namespace GGUF

open Precision

/-- The numbers of the IEEE format with `E` exponent and `M` mantissa bits. -/
def grid (E M : ℕ) : Set ℝ := {v | ∃ b < 2 ^ (1 + E + M), ieee E M b = some v}

theorem zero_mem_grid {E : ℕ} (hE : 1 ≤ E) (M : ℕ) : (0 : ℝ) ∈ grid E M := by
  refine ⟨0, by positivity, ?_⟩
  have : (2 : ℕ) ^ E - 1 ≠ 0 := by
    have := Nat.one_lt_two_pow_iff.2 (by omega : E ≠ 0); omega
  simp [ieee, this.symm]

theorem grid_finite (E M : ℕ) : (grid E M).Finite := by
  refine ((Finset.range (2 ^ (1 + E + M))).image fun b => (ieee E M b).getD 0).finite_toSet.subset ?_
  rintro v ⟨b, hb, h⟩
  exact Finset.mem_coe.2 (Finset.mem_image.2 ⟨b, Finset.mem_range.2 hb, by simp [h]⟩)

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

end GGUF
end Transformer
