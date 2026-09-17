/-
# Fusing the nonlinearity with the rounding

arXiv:1606.06160v3, "DoReFa-Net: Training Low Bitwidth Convolutional Neural
Networks with Low Bitwidth Gradients", §2.7.

"A naive implementation of Algorithm 1 would store activations `h(a_k)` in
full-precision numbers, consuming much memory during run-time."  §2.7 removes
those intermediates with two order-theoretic observations about `quantize_k`,
both of which come from its being monotone:

* "when `h` is monotonic, `f_α · h` is also monotonic, the few possible values
  of `a_k^b` corresponds to several non-overlapping value ranges of `a_k`,
  hence we can implement computation of `a_k^b = f_α(h(a_k))` by several
  comparisons between fixed point numbers" — `fa_comp_monotone`, and
  `fa_comp_ordConnected` for the value ranges being intervals;
* "`quantize_k` function commutes with `max` function" (equation 8), which is
  what lets the quantization of a gradient be fused through a max-pooling
  layer — `quantize_max`.

Everything here follows from `quantize_monotone` of `Section2_Quantize`, so
these are the paper's consequences rather than new assumptions.
-/

import Transformer.DoReFa.Section2_Weights

namespace Transformer
namespace DoReFa

variable {k : ℕ}

/-- **Equation 8**: `quantize_k(max(a, b)) = max(quantize_k(a), quantize_k(b))`,
the commutation that fuses gradient quantization through max-pooling. -/
theorem quantize_max (hk : 0 < k) (a b : ℝ) :
    quantize k (max a b) = max (quantize k a) (quantize k b) :=
  (quantize_monotone hk).map_max

/-- **`f_α · h` is monotone whenever `h` is** (§2.7), which is why the
activation of a layer can be read off by comparisons instead of being computed
and stored. -/
theorem fa_comp_monotone (hk : 0 < k) {h : ℝ → ℝ} (hh : Monotone h) :
    Monotone (fun x => fa k (h x)) :=
  (quantize_monotone hk).comp hh

/-- **Each of its few values occupies one value range** (§2.7, "the few possible
values of `a_k^b` corresponds to several non-overlapping value ranges of
`a_k`"): if the two ends of an interval quantize to the same level, everything
between them does. -/
theorem fa_comp_ordConnected (hk : 0 < k) {h : ℝ → ℝ} (hh : Monotone h) {x y z : ℝ}
    (hxy : x ≤ y) (hyz : y ≤ z) (hxz : fa k (h x) = fa k (h z)) :
    fa k (h y) = fa k (h x) := by
  have h₁ : fa k (h x) ≤ fa k (h y) := fa_comp_monotone hk hh hxy
  have h₂ : fa k (h y) ≤ fa k (h z) := fa_comp_monotone hk hh hyz
  rw [← hxz] at h₂
  exact le_antisymm h₂ h₁

/-- The hypotheses of the three statements above are satisfiable: one bit, and
a bounded nondecreasing activation of the kind §2.4 assumes in front of the
quantizer, here the clipping to `[0, 1]`. -/
example : 0 < 1 ∧ Monotone fun x : ℝ => max 0 (min 1 x) :=
  ⟨Nat.one_pos, monotone_const.max (monotone_const.min monotone_id)⟩

end DoReFa
end Transformer
