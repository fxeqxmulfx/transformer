/-
# Low bitwidth weights and activations

arXiv:1606.06160v3, "DoReFa-Net: Training Low Bitwidth Convolutional Neural
Networks with Low Bitwidth Gradients", §2.3 and §2.4.

Weights are squashed by `tanh`, affinely moved into `[0, 1]`, quantized, and
moved back (equation 5):

`f_ω^k(r_i) = 2 quantize_k(tanh(r_i) / (2 max|tanh(r_i)|) + 1/2) - 1`,

the maximum being over the weights of that layer.  The two claims the paper
makes about this display are that the argument of `quantize_k` is in `[0, 1]`
and that the output is in `[-1, 1]`; both are below.

At `k = 1` this is *not* the binarization of equation 4, `sign(r_i) · E(|r_i|)`,
which scales all filters of a layer by one constant rather than by a
per-channel factor as XNOR-Net does — "when `k = 1`, equation 5 is different
from equation 4, providing a different way of binarizing weights.  Nevertheless,
we find this difference insignificant in experiments."  `exists_fw_ne_fwOne`
states the difference; the insignificance is an experiment, not a statement.

Activations are already in `[0, 1]`, since "the output of the previous layer
has passed through a bounded activation function `h`", so quantizing them is
`quantize_k` itself (equation 6).
-/

import Transformer.DoReFa.Section2_Quantize
import Mathlib.Analysis.Complex.Trigonometric

namespace Transformer
namespace DoReFa

variable {n k : ℕ} {w : Fin (n + 1) → ℝ} {i : Fin (n + 1)}

/-- The mean absolute weight of a layer, `E(|r_i|)` of equation 4. -/
noncomputable def meanAbs (w : Fin (n + 1) → ℝ) : ℝ := ((n : ℝ) + 1)⁻¹ * ∑ i, |w i|

/-- **Equation 4**, the binarization DoReFa-Net uses for `1`-bit weights:
`sign(r_i) × E(|r_i|)`, with one constant for the whole layer, because "the
channel-wise scaling factors will make it impossible to exploit bit convolution
kernels when computing the convolution between gradients and the weights during
back propagation". -/
noncomputable def fwOne (w : Fin (n + 1) → ℝ) (i : Fin (n + 1)) : ℝ :=
  (if 0 ≤ w i then 1 else -1) * meanAbs w

/-- `max(|tanh(r_i)|)` over the weights of a layer. -/
noncomputable def maxAbsTanh (w : Fin (n + 1) → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun i => |Real.tanh (w i)|

/-- The argument handed to `quantize_k` in equation 5. -/
noncomputable def fwArg (w : Fin (n + 1) → ℝ) (i : Fin (n + 1)) : ℝ :=
  Real.tanh (w i) / (2 * maxAbsTanh w) + 1 / 2

/-- **Equation 5**, the `k`-bit weight quantizer `f_ω^k`. -/
noncomputable def fw (k : ℕ) (w : Fin (n + 1) → ℝ) (i : Fin (n + 1)) : ℝ :=
  2 * quantize k (fwArg w i) - 1

/-- **The argument is in `[0, 1]`** (§2.3, "By construction,
`tanh(r_i)/(2 max(|tanh(r_i)|)) + 1/2` is a number in `[0, 1]`, where the
maximum is taken over all weights in that layer"), which is what makes
`quantize_k` applicable to it at all. -/
theorem fwArg_mem_Icc (hw : ∃ j, w j ≠ 0) : fwArg w i ∈ Set.Icc (0 : ℝ) 1 :=
  sorry

/-- **And the output is in `[-1, 1]`** (§2.3, "Finally an affine transform will
bring the range of `f_ω^k(r_i)` to `[-1, 1]`"). -/
theorem fw_mem_Icc (hk : 0 < k) (hw : ∃ j, w j ≠ 0) : fw k w i ∈ Set.Icc (-1 : ℝ) 1 :=
  sorry

/-- The hypotheses of the two statements above are satisfiable: one bit, and a
layer whose single weight is `1`. -/
example : 0 < 1 ∧ ∃ j : Fin (0 + 1), (fun _ => (1 : ℝ)) j ≠ 0 :=
  ⟨Nat.one_pos, ⟨0, one_ne_zero⟩⟩

/-- **At `k = 1` equation 5 is not equation 4** (§2.3, "when `k = 1`,
equation 5 is different from equation 4, providing a different way of binarizing
weights"). -/
theorem exists_fw_ne_fwOne :
    ∃ (m : ℕ) (v : Fin (m + 1) → ℝ) (j : Fin (m + 1)), fw 1 v j ≠ fwOne v j :=
  sorry

/-- **Equation 6**, the `k`-bit activation quantizer: the input of a weight
layer has already been through a bounded nonlinearity, so "quantization of
activations `r` to `k`-bit is simply `quantize_k(r)`". -/
noncomputable def fa (k : ℕ) (r : ℝ) : ℝ := quantize k r

/-- Activations therefore land on the same `k`-bit grid as the weights, which
is what lets a whole convolution run through the kernels of
`Section2_BitConv`. -/
theorem fa_mem_grid {r : ℝ} (hk : 0 < k) (hr : r ∈ Set.Icc (0 : ℝ) 1) :
    ∃ m : ℕ, m < 2 ^ k ∧ fa k r = m / (2 ^ k - 1) :=
  quantize_mem_grid hk hr

/-- Its hypotheses are satisfiable. -/
example : 0 < 1 ∧ (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := ⟨Nat.one_pos, by norm_num⟩

end DoReFa
end Transformer
