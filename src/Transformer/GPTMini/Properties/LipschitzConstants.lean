/-
# The Lipschitz constants of one `gpt-mini` block

The per-block constant

  `L_block = 1 + L_attn + L_ffn + L_attn · L_ffn`,

with `L_attn` bounded by `‖W_o‖ e^{2 α_max}` and `L_ffn` by
`‖W_out‖ · 2R · ‖W_in‖²` — `R` being the residual-stream bound of
`Properties.StreamGrowth` — and their product `endToEndLipschitz` over the
`n_layers` blocks.  The cross-term is the composition `ffn ∘ (1 + attn)`, and
the `1` is the residual branch, which is why the constant is never below `1`
(`one_le_perBlockLipschitz`) and the partial products increase with the depth.

`blockForward_lipschitz`, the bound these constants are for, is a `sorry`-leaf:
it waits on `Block.attnSubLayer_bounded` and `Block.ffnSubLayer_bounded`.  What
the depth induction makes of it is `Properties.Lipschitz`.
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

/-- **The per-block constant is at least `1`.**

The identity branch of the residual connection contributes the `1`, and the
two sub-layer terms are non-negative; this is what makes the partial products
of `endToEndLipschitz` increase with the depth. -/
theorem one_le_perBlockLipschitz
    (params : BlockParams cfg) (alpha_max : ℝ) (R : ℝ) (hR : 0 ≤ R) :
    1 ≤ perBlockLipschitz cfg params alpha_max R := by
  unfold perBlockLipschitz
  have h1 : 0 ≤ ‖params.attn.W_o‖ * Real.exp (2 * alpha_max) :=
    mul_nonneg (norm_nonneg _) (le_of_lt (Real.exp_pos _))
  have h2 : 0 ≤ ‖params.ffn.W_out‖ * (2 * R) * ‖params.ffn.W_in‖ ^ 2 :=
    mul_nonneg (mul_nonneg (norm_nonneg _) (by linarith)) (sq_nonneg _)
  nlinarith

/-- The hypothesis of `blockForward_lipschitz` is satisfiable: the `eps` of
`reference/model.py` is positive. -/
example (params : BlockParams Config.default)
    (x y : Fin 1 → EucSpace Config.default.d_model) (i : Fin 1) :
    ‖blockForward Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) x i
        - blockForward Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) y i‖
      ≤ perBlockLipschitz Config.default params 1 1
        * (Finset.univ : Finset (Fin 1)).sup' Finset.univ_nonempty
            (fun i' => ‖x i' - y i'‖) :=
  blockForward_lipschitz Config.default 1e-6 params 1 1 (by norm_num) _ x y i

end Properties
end GPTMini
end Transformer
