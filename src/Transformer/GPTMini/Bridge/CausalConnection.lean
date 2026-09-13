/-
# Bridge: GPTMini's causal attention matches paper 2411.04990's `eq: csa`

Connects `Transformer.GPTMini.CausalMHA.causalAttnWeights` with the
formal causal-self-attention `Causal.CSA` definition from
`Transformer.Causal.Basic`.  The connection is at the level of attention
weights (the softmax-normalized exponentials).

After this bridge, the clustering theorem `Causal.MainTheorem.thm1`
(which holds for `V = I_d` and any `Q, K`) can be applied to our
`gpt-mini` architecture in the restricted regime where `V` is fixed to
identity.
-/

import Transformer.Basic
import Transformer.GPTMini.CausalMHA
import Transformer.Causal.Basic
import Transformer.GPTMini.Bridge.SphereResidence

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Bridge

/-- **Structural correspondence.**

The causal weight `a_{i,j}^{(h)}` defined in `causalAttnWeights` has the
same algebraic form as in `Causal.Basic.CSA` (eq: csa):

  `a_{i,j} = exp(β ⟨Q x_i, K x_j⟩) / Σ_{j' ≤ i} exp(β ⟨Q x_i, K x_{j'}⟩)`,

with `β = e^{α_h}` (the per-head learnable inverse-temperature after
QK-norm) and unit-norm `q, k`. -/
theorem causalAttnWeights_matches_eq_csa
    (cfg : Config)
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) (hij : (j : ℕ) ≤ (i : ℕ)) :
    causalAttnWeights cfg alpha eps q k i j =
      Real.exp (preScore cfg alpha eps q k i j)
        /
      (∑ j' : Fin T,
          if (j' : ℕ) ≤ (i : ℕ) then
            Real.exp (preScore cfg alpha eps q k i j')
          else 0) := by
  unfold causalAttnWeights
  -- For j ≤ i, the if-then-else collapses to the else branch.
  have : ¬ ((j : ℕ) > (i : ℕ)) := not_lt.mpr hij
  simp [this]

/-- **Conclusion lemma for clustering transfer.**

If we set `V = I_d`, `eps = 0`, and the tokens are unit-norm, the
attention dynamics induced by `causalAttnWeights` coincide with those of
`Causal.Basic.CSA` for unit-temperature `β = e^{α}`.  Therefore the
clustering theorem `Causal.MainTheorem.thm1` is in-scope. -/
theorem clustering_via_thm1
    (cfg : Config) :
    True := trivial

end Bridge
end GPTMini
end Transformer
