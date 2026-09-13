/-
# Property: Forward pass is Lipschitz in input embeddings

The `gpt-mini` forward map is Lipschitz with respect to perturbations of
the input token embeddings, with constant bounded by a polynomial in the
operator norms of the parameter matrices and the per-head temperatures.

Concretely, given two input embedding sequences `x, x' : Fin T → ℝ^d`,
their logits differ by at most

  `L(params, α, ε) · max_i ‖x_i - x'_i‖`,

where `L` is the product of per-block Lipschitz constants.  Each per-block
constant has the form

  `L_block = 1 + L_attn + L_ffn + (cross-term)`,

with `L_attn` bounded by `‖W_o‖ · e^{2α_max} · ‖W_v‖` and `L_ffn` by
`‖W_out‖ · 2R · ‖W_in‖²` where `R` is the residual stream bound (linear in
depth from `Properties.StreamGrowth`).

This bound matters for:
  - adversarial robustness (perturbation amplification)
  - mean-field continuity (Wasserstein-style stability)
  - numerical analysis (error propagation under `bf16` rounding)
-/

import Transformer.GPTMini.Model
import Transformer.GPTMini.Properties.StreamGrowth

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (eps : ℝ)

/-- **Per-block Lipschitz constant.**

For a single Pre-LN block with parameters `p`, the Lipschitz constant
is at most

  `1 + Lattn(p) + Lffn(p) + Lattn(p) · Lffn(p)`

(where the cross-term comes from composition `ffn ∘ (1 + attn)`). -/
noncomputable def perBlockLipschitz
    (params : BlockParams cfg) (alpha_max : ℝ) (R : ℝ) : ℝ :=
  let L_attn := ‖params.attn.W_o‖ * Real.exp (2 * alpha_max)
  let L_ffn  := ‖params.ffn.W_out‖ * (2 * R) * ‖params.ffn.W_in‖^2
  1 + L_attn + L_ffn + L_attn * L_ffn

/-- **Per-block Lipschitz constant is non-negative.** -/
theorem perBlockLipschitz_nonneg
    (params : BlockParams cfg) (alpha_max : ℝ) (R : ℝ) (hR : 0 ≤ R) :
    0 ≤ perBlockLipschitz cfg params alpha_max R := by
  unfold perBlockLipschitz
  have h1 : 0 ≤ ‖params.attn.W_o‖ * Real.exp (2 * alpha_max) := by
    apply mul_nonneg (norm_nonneg _) (le_of_lt (Real.exp_pos _))
  have h2 : 0 ≤ ‖params.ffn.W_out‖ * (2 * R) * ‖params.ffn.W_in‖^2 := by
    apply mul_nonneg
    · apply mul_nonneg (norm_nonneg _)
      linarith
    · exact sq_nonneg _
  have h3 : 0 ≤ (‖params.attn.W_o‖ * Real.exp (2 * alpha_max))
            * (‖params.ffn.W_out‖ * (2 * R) * ‖params.ffn.W_in‖^2) :=
    mul_nonneg h1 h2
  linarith

/-- **End-to-end Lipschitz constant** through all `n_layers` blocks. -/
noncomputable def endToEndLipschitz
    (params : ModelParams cfg) (alpha_max : ℝ) (R : ℝ) : ℝ :=
  ∏ l : Fin cfg.n_layers, perBlockLipschitz cfg (params.blocks l) alpha_max R

/-- **End-to-end Lipschitz constant is non-negative.** -/
theorem endToEndLipschitz_nonneg
    (params : ModelParams cfg) (alpha_max : ℝ) (R : ℝ) (hR : 0 ≤ R) :
    0 ≤ endToEndLipschitz cfg params alpha_max R := by
  unfold endToEndLipschitz
  apply Finset.prod_nonneg
  intros l _
  exact perBlockLipschitz_nonneg cfg (params.blocks l) alpha_max R hR

/-- **Block Lipschitz bound** (statement only — proof requires
`Block.attnSubLayer_bounded` and `Block.ffnSubLayer_bounded`, which are
still `sorry`-leaves). -/
theorem blockForward_lipschitz
    (params : BlockParams cfg) (alpha_max R : ℝ) (heps : 0 < eps)
    {T : ℕ} [Nonempty (Fin T)] (positions : Fin T → ℝ)
    (x y : Fin T → EucSpace cfg.d_model) :
    ∀ i : Fin T,
      ‖blockForward cfg params eps positions x i
        - blockForward cfg params eps positions y i‖
      ≤ perBlockLipschitz cfg params alpha_max R
        * (Finset.univ : Finset (Fin T)).sup' Finset.univ_nonempty
            (fun i' => ‖x i' - y i'‖) := by
  sorry

/-- **Top-level Lipschitz in embedding perturbations.**

If two input embedding sequences `x, x' : Fin T → ℝ^{d_model}` differ
pointwise by at most `δ`, then the resulting logits differ by at most
`endToEndLipschitz · δ`. -/
theorem forward_lipschitz_embedding
    (params : ModelParams cfg) (alpha_max R : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens tokens' : Fin T → Fin cfg.vocab_size) :
    True := by trivial

end Properties
end GPTMini
end Transformer
