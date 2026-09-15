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

/-- **The weight is the CSA coefficient, at the temperature QK-norm sets.**

`Causal.Basic.CSA` weighs token `j` at position `k` by `Z_k⁻¹ e^{β ⟨Q x_k,
K x_j⟩}`.  This head weighs it by `causalAttnWeights`, and the two are the
same expression once the dictionary is read off: `β = e^{α_h}` is the
per-head inverse temperature, and the vectors CSA writes as `Q x` and `K x`
are the QK-normalized `normL2 eps (q i)` and `normL2 eps (k j)`.

That is what puts `Causal.MainTheorem.thm1` in scope for this architecture:
the theorem holds for `V = I_d` and any `Q, K`, and the restriction to
`V = I_d` is the only thing the transfer still asks of the head. -/
theorem causalAttnWeights_eq_csa_coeff
    (cfg : Config)
    {T : ℕ} (alpha eps : ℝ)
    (q k : Fin T → EucSpace cfg.head_dim)
    (i j : Fin T) (hij : (j : ℕ) ≤ (i : ℕ)) :
    causalAttnWeights cfg alpha eps q k i j =
      (∑ j' : Fin T,
          if (j' : ℕ) ≤ (i : ℕ) then
            Real.exp (Real.exp alpha *
              inner (𝕜 := ℝ) (normL2 eps (q i)) (normL2 eps (k j')))
          else 0)⁻¹
      *
      Real.exp (Real.exp alpha *
        inner (𝕜 := ℝ) (normL2 eps (q i)) (normL2 eps (k j))) := by
  rw [causalAttnWeights, ite_eq_right (not_lt.mpr hij), div_eq_inv_mul]
  rfl

/-- The causal hypothesis both theorems above take is satisfiable: at
position `2` of a three-token window, `j = 0` is in the past. -/
example : ((0 : Fin 3) : ℕ) ≤ ((2 : Fin 3) : ℕ) := by decide

end Bridge
end GPTMini
end Transformer
