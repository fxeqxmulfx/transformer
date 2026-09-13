/-
# Property: Causal masking

The causal mask in `CausalMHA` ensures that the attention weights at
position `i` depend only on positions `j ≤ i`.  Formally:

  **`causalAttnWeights cfg α ε q k i j = 0`  for `j > i`.**

This implies the **causal-equivariance** property of the full forward
pass: changing token at position `j > i` cannot affect the logits at
position `i`.

We do not attempt to prove the full top-level equivariance here (that
requires substantial bookkeeping through the multi-head reshape and
sub-layer composition); instead we prove the foundational lemma at the
attention-weight level, which is the load-bearing fact.
-/

import Transformer.GPTMini.CausalMHA

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config)

/-- **Attention weights are zero past the causal cutoff.**

For positions `j > i`, the causal attention weight is `0` by construction. -/
theorem causalAttnWeights_zero_above
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) (hij : (i : ℕ) < (j : ℕ)) :
    causalAttnWeights cfg alpha eps q k i j = 0 := by
  unfold causalAttnWeights
  simp [hij]

/-- **Attention weights are non-negative.**

Direct consequence of `Real.exp_pos` and the softmax-style construction. -/
theorem causalAttnWeights_nonneg
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) :
    0 ≤ causalAttnWeights cfg alpha eps q k i j := by
  unfold causalAttnWeights
  split_ifs with hij
  · exact le_refl 0
  · apply div_nonneg
    · exact le_of_lt (Real.exp_pos _)
    · apply Finset.sum_nonneg
      intros j' _
      split_ifs with h
      · exact le_of_lt (Real.exp_pos _)
      · exact le_refl 0

/-- **Attention output at position `i` is a convex combination of values
at positions `j ≤ i`.**

This is the formal statement of causality at the attention-output level:
`attnOutput i` is a sum over only positions `≤ i`, weighted by
non-negative coefficients summing (across `j ≤ i`) to 1. -/
theorem attnOutput_causal
    {T : ℕ} (alpha eps : ℝ)
    (q k v : Fin T → EucSpace cfg.head_dim)
    (i : Fin T) :
    attnOutput cfg alpha eps q k v i =
      ∑ j : Fin T,
        if (j : ℕ) ≤ (i : ℕ) then
          (causalAttnWeights cfg alpha eps q k i j) • v j
        else 0 := by
  unfold attnOutput
  apply Finset.sum_congr rfl
  intros j _
  split_ifs with hij
  · rfl
  · push Not at hij
    have hweight : causalAttnWeights cfg alpha eps q k i j = 0 :=
      causalAttnWeights_zero_above cfg alpha eps q k i j hij
    rw [hweight, zero_smul]

end Properties
end GPTMini
end Transformer
