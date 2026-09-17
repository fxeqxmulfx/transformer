/-
# The `quantize_k` straight-through estimator

arXiv:1606.06160v3, "DoReFa-Net: Training Low Bitwidth Convolutional Neural
Networks with Low Bitwidth Gradients", §2.2.

`quantize_k` is the one operator the paper builds everything else out of: on
the forward pass it sends `r_i ∈ [0, 1]` to `r_o = round((2^k - 1) r_i) /
(2^k - 1)`, and on the backward pass it is the identity — "an STE can be
thought of as an operator that has arbitrary forward and backward operations".

Only the forward map is a function, and that is what is defined here; the
backward half is a convention about the gradient, not a claim about `r_o`.
What §2.2 does claim about the forward map is that its output is "a real number
representable by `k` bits", which is `quantize_mem_grid` below, and this is
also the affine mapping "between these low bitwidth numbers and fixed-point
integers" that §2.6 needs in order to call the bit kernels of `Section2_BitConv`.

The error bound `abs_quantize_sub_le` is what §2.5 calls "the possible
quantization error", the magnitude its gradient noise is matched to.
-/

import Mathlib.Algebra.Order.Round
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace Transformer
namespace DoReFa

variable {k : ℕ} {r : ℝ}

/-- `quantize_k` on the forward pass (equation 4): a real number in `[0, 1]` is
rounded to one of the `2^k` levels of a uniform `k`-bit grid. -/
noncomputable def quantize (k : ℕ) (r : ℝ) : ℝ :=
  (round ((2 ^ k - 1 : ℝ) * r) : ℝ) / (2 ^ k - 1)

/-- **The output is representable by `k` bits** (§2.2, "It is obvious by
construction that the output `q` of `quantize_k` STE is a real number
representable by `k` bits"): it is one of the `2^k` multiples of `1/(2^k - 1)`.
This is the affine mapping to fixed-point integers that §2.6 relies on. -/
theorem quantize_mem_grid (hk : 0 < k) (hr : r ∈ Set.Icc (0 : ℝ) 1) :
    ∃ m : ℕ, m < 2 ^ k ∧ quantize k r = m / (2 ^ k - 1) :=
  sorry

/-- **And it stays in `[0, 1]`** (§2.2, "quantizes a real number input
`r_i ∈ [0, 1]` to a `k`-bit number output `r_o ∈ [0, 1]`"). -/
theorem quantize_mem_Icc (hk : 0 < k) (hr : r ∈ Set.Icc (0 : ℝ) 1) :
    quantize k r ∈ Set.Icc (0 : ℝ) 1 :=
  sorry

/-- The hypotheses of the two statements above are satisfiable: one bit, and
the midpoint of the range. -/
example : 0 < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := by
  exact ⟨Nat.one_pos, by norm_num⟩

/-- **`quantize_k` is monotone**, the property §2.7 turns into a fusion of the
nonlinearity with the rounding: "when `h` is monotonic, `f_α · h` is also
monotonic". -/
theorem quantize_monotone (hk : 0 < k) : Monotone (quantize k) :=
  sorry

/-- **Its error is at most half a grid step** — §2.5, "the possible
quantization error", against which the gradient noise `N(k) = σ/(2^k - 1)` is
calibrated. -/
theorem abs_quantize_sub_le (hk : 0 < k) (r : ℝ) :
    |quantize k r - r| ≤ 1 / (2 * (2 ^ k - 1)) :=
  sorry

/-- The hypothesis of the two statements above is satisfiable. -/
example : 0 < 1 := Nat.one_pos

end DoReFa
end Transformer
