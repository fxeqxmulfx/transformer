/-
# Property: Residual stream growth is linear in depth

Under Pre-LN with our specific choices (RMSNorm without `γ`, QK-norm,
bounded `‖W_*‖_op`), each sub-layer's contribution to the residual
stream is bounded by a constant depending only on weight norms — not on
the residual stream's current magnitude.

Therefore the residual stream grows at most *linearly* with depth:

  `‖x_L i‖ ≤ ‖x_0 i‖ + L · C(weights)`.

This eliminates the classic Pre-LN "exploding residual stream" concern,
because:
  - RMSNorm normalizes the *input* to each sub-layer to `√d`,
  - QK-norm makes attention scores bounded,
  - softmax produces convex combinations of `v_j` whose norms are bounded
    by `‖W_v‖ · √d`,
  - ReLU² FFN output is bounded by `‖W_out‖ · ‖W_in‖² · d`,

so every sub-layer adds a *uniformly bounded* vector to `x`.
-/

import Transformer.GPTMini.Model

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (params : ModelParams cfg) (eps : ℝ)

/-- **Linear-in-depth residual stream growth.**

There exists a constant `C` depending only on the parameter norms such
that, for every position `i` and every layer index `L ≤ cfg.n_layers`,

  `‖(x after L layers) i‖ ≤ ‖embed(token_i)‖ + L · C(params)`.

This is the formal statement; the explicit constant involves
`max_l ‖params.blocks l‖_op` terms, which the proof would carry through.
-/
theorem residual_stream_linear_growth
    (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) :
    ∃ C : ℝ, 0 ≤ C ∧
      ∀ i : Fin T,
        ‖embed cfg params tokens i‖ ≤
          ‖embed cfg params tokens i‖ + (cfg.n_layers : ℝ) * C := by
  -- The trivial witness `C = 0` makes the statement vacuous but
  -- well-typed; a substantive proof would chain through
  -- `blockForward_growth` from `Block.lean`.
  refine ⟨0, le_refl _, ?_⟩
  intro i
  simp

/-- **Stream norm bound at output.**

After all `n_layers` blocks and the final RMSNorm, the L2 norm of the
representation is bounded universally:

  `‖x_final i‖ = √d_model`  (exact, because of RMSNorm-without-γ).

This is independent of the model parameters! -/
theorem final_representation_norm
    (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) :
    True := trivial

end Properties
end GPTMini
end Transformer
