/-
# Low bitwidth gradients, and the noise that unbiases them

arXiv:1606.06160v3, "DoReFa-Net: Training Low Bitwidth Convolutional Neural
Networks with Low Bitwidth Gradients", §2.5.

Gradients are the contribution of the paper: "no previous work has succeeded in
quantizing gradients to numbers with bitwidth less than 8 during the backward
pass, while still achieving comparable prediction accuracy".  They are
unbounded, so instead of a nonlinearity they get an affine map built from their
own largest absolute value, `max_0(|dr|)`, taken "over all axis of the gradient
tensor `dr` except for the mini-batch axis":

`f̃_γ^k(dr) = 2 max_0(|dr|) [quantize_k(dr / (2 max_0(|dr|)) + 1/2) - 1/2]`.

"We find stochastic quantization is necessary for low bitwidth gradients to be
effective."  The randomness is an additive dither `N(k) = σ/(2^k - 1)` with
`σ ∼ Uniform(-0.5, 0.5)`, one grid step wide — "the noise therefore has the
same magnitude as the possible quantization error" — giving equation 7, and
"we find that the artificial noise to be critical for achieving good
performance".

What that dither buys is stated here as `integral_fg`: averaged over `σ`, the
quantized gradient is the exact gradient.  The paper puts it as compensating
"the potential bias introduced by gradient quantization", which
`exists_fgNoiseless_ne` is the other half of: without the noise the map is
deterministic and simply wrong on almost every input.
-/

import Transformer.DoReFa.Section2_Quantize
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

namespace Transformer
namespace DoReFa

variable {n k : ℕ} {dr : Fin (n + 1) → ℝ} {i : Fin (n + 1)}

/-- `max_0(|dr|)`, the largest absolute value of one instance's gradient
tensor: "each instance in a mini-batch will have its own scaling factor". -/
noncomputable def maxAbs (dr : Fin (n + 1) → ℝ) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty fun i => |dr i|

/-- `f̃_γ^k`, the gradient quantizer of §2.5 before the noise is added: an
affine map into `[0, 1]`, `quantize_k`, and the inverse affine map. -/
noncomputable def fgNoiseless (k : ℕ) (dr : Fin (n + 1) → ℝ) (i : Fin (n + 1)) : ℝ :=
  2 * maxAbs dr * (quantize k (dr i / (2 * maxAbs dr) + 1 / 2) - 1 / 2)

/-- **Equation 7**, `f_γ^k`: the same map with the dither `N(k) = σ/(2^k - 1)`
added before the rounding, `σ` being the draw from `Uniform(-0.5, 0.5)`. -/
noncomputable def fg (k : ℕ) (dr : Fin (n + 1) → ℝ) (i : Fin (n + 1)) (σ : ℝ) : ℝ :=
  2 * maxAbs dr * (quantize k (dr i / (2 * maxAbs dr) + 1 / 2 + σ / (2 ^ k - 1)) - 1 / 2)

/-- **The affine map does land in `[0, 1]`** (§2.5, "The above function first
applies an affine transform on the gradient, to map it into `[0, 1]`, and then
inverts the transform after quantization"), because the scaling factor is the
gradient's own maximum. -/
theorem fg_arg_mem_Icc (hdr : 0 < maxAbs dr) :
    dr i / (2 * maxAbs dr) + 1 / 2 ∈ Set.Icc (0 : ℝ) 1 :=
  sorry

/-- **The quantization error is one grid step of the gradient range** — the
"possible quantization error" of §2.5, which the amplitude of `N(k)` is chosen
to match. -/
theorem abs_fgNoiseless_sub_le (hk : 0 < k) (hdr : 0 < maxAbs dr) :
    |fgNoiseless k dr i - dr i| ≤ maxAbs dr / (2 ^ k - 1) :=
  sorry

/-- **The dither makes the quantized gradient unbiased**: averaging equation 7
over `σ ∼ Uniform(-0.5, 0.5)` returns the gradient itself.  This is what §2.5
means by introducing the noise "to further compensate the potential bias
introduced by gradient quantization", and what the accuracy of a `2`-bit
backward pass rests on. -/
theorem integral_fg (hk : 0 < k) (hdr : 0 < maxAbs dr) :
    ∫ σ in (-(1 : ℝ) / 2)..(1 / 2), fg k dr i σ = dr i :=
  sorry

/-- The hypotheses of the three statements above are satisfiable: one bit, and
a one-entry gradient equal to `1`. -/
example : 0 < 1 ∧ 0 < maxAbs (fun _ => (1 : ℝ) : Fin (0 + 1) → ℝ) := by
  refine ⟨Nat.one_pos, ?_⟩
  unfold maxAbs
  exact lt_of_lt_of_le (by norm_num) (Finset.le_sup' _ (Finset.mem_univ (0 : Fin (0 + 1))))

/-- **Without the noise there is a bias to compensate** (§2.5): the
deterministic `f̃_γ^k` misses the gradient it is given.  Together with
`integral_fg` this is the paper's reason for calling the artificial noise
"critical for achieving good performance". -/
theorem exists_fgNoiseless_ne :
    ∃ (m : ℕ) (g : Fin (m + 1) → ℝ) (j : Fin (m + 1)), fgNoiseless 1 g j ≠ g j :=
  sorry

end DoReFa
end Transformer
