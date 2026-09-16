/-
# The Lipschitz constants of one `gpt-mini` block

A Pre-LN block is `(1 + ffn) ∘ (1 + attn)`, so its constant factors:

  `L_block = (1 + L_attn) (1 + L_ffn) = 1 + L_attn + L_ffn + L_attn · L_ffn`,

the cross-term being the composition and the `1` the residual branch — which
is why the constant is never below `1` (`one_le_perBlockLipschitz`) and the
partial products of `endToEndLipschitz` increase with the depth.

The two sub-layer constants are the ones actually proved upstream:

  - `L_attn = ‖W_o‖ √n_heads · L(α_max, eps, ‖W_qkv‖√d_model) · 2‖W_qkv‖/√eps`
    from `GPTMini.attnSubLayer_dist_le`,
  - `L_ffn = 4 ‖W_out‖ ‖W_in‖² √d_model / √eps` from
    `GPTMini.ffnSubLayer_lipschitz`.

Both carry `‖W_qkv‖` and the normalization floor `eps`, and neither can drop
them: nothing downstream of the QKV projection renormalizes the values, and
`rmsNormEps` has slope `2/√eps` at the origin.  What the depth induction makes
of `blockForward_lipschitz` is `Properties.Lipschitz`.
-/

import Transformer.GPTMini.Model
import Transformer.GPTMini.AttnSubLayerLipschitz
import Transformer.GPTMini.BlockLipschitz
import Transformer.GPTMini.Properties.StreamGrowth

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (eps : ℝ)

/-- The Lipschitz constant of the attention sub-layer of a block, as proved in
`GPTMini.attnSubLayer_dist_le`. -/
noncomputable def attnLipschitz (params : BlockParams cfg) (alpha_max : ℝ) : ℝ :=
  ‖params.attn.W_o‖ * Real.sqrt (cfg.n_heads : ℝ)
    * headLipschitz alpha_max eps (‖params.attn.W_qkv‖ * Real.sqrt (cfg.d_model : ℝ))
    * (2 * ‖params.attn.W_qkv‖ / Real.sqrt eps)

/-- The Lipschitz constant of the FFN sub-layer of a block, as proved in
`GPTMini.ffnSubLayer_lipschitz`. -/
noncomputable def ffnLipschitz (params : BlockParams cfg) : ℝ :=
  4 * ‖params.ffn.W_out‖ * ‖params.ffn.W_in‖ ^ 2
    * Real.sqrt (cfg.d_model : ℝ) / Real.sqrt eps

/-- Both sub-layer constants are non-negative. -/
theorem attnLipschitz_nonneg (params : BlockParams cfg) (alpha_max : ℝ) (heps : 0 < eps) :
    0 ≤ attnLipschitz cfg eps params alpha_max := by
  unfold attnLipschitz
  have hB : (0 : ℝ) ≤ ‖params.attn.W_qkv‖ * Real.sqrt (cfg.d_model : ℝ) := by positivity
  have hL := headLipschitz_nonneg alpha_max eps _ heps hB
  have h1 : (0 : ℝ) ≤ ‖params.attn.W_o‖ * Real.sqrt (cfg.n_heads : ℝ) := by positivity
  have h2 : (0 : ℝ) ≤ 2 * ‖params.attn.W_qkv‖ / Real.sqrt eps := by positivity
  exact mul_nonneg (mul_nonneg h1 hL) h2

theorem ffnLipschitz_nonneg (params : BlockParams cfg) : 0 ≤ ffnLipschitz cfg eps params := by
  unfold ffnLipschitz
  positivity

/-- **Per-block Lipschitz constant.**

`(1 + L_attn) (1 + L_ffn)` expanded: the residual branch, the two sub-layers,
and the composition of the second with the first. -/
noncomputable def perBlockLipschitz (params : BlockParams cfg) (alpha_max : ℝ) : ℝ :=
  1 + attnLipschitz cfg eps params alpha_max + ffnLipschitz cfg eps params
    + attnLipschitz cfg eps params alpha_max * ffnLipschitz cfg eps params

/-- **The per-block constant is at least `1`.**

The identity branch of the residual connection contributes the `1`, and the
two sub-layer terms are non-negative; this is what makes the partial products
of `endToEndLipschitz` increase with the depth. -/
theorem one_le_perBlockLipschitz (params : BlockParams cfg) (alpha_max : ℝ) (heps : 0 < eps) :
    1 ≤ perBlockLipschitz cfg eps params alpha_max := by
  unfold perBlockLipschitz
  have h1 := attnLipschitz_nonneg cfg eps params alpha_max heps
  have h2 := ffnLipschitz_nonneg cfg eps params
  nlinarith

/-- **Per-block Lipschitz constant is non-negative.** -/
theorem perBlockLipschitz_nonneg (params : BlockParams cfg) (alpha_max : ℝ) (heps : 0 < eps) :
    0 ≤ perBlockLipschitz cfg eps params alpha_max :=
  le_trans zero_le_one (one_le_perBlockLipschitz cfg eps params alpha_max heps)

/-- **End-to-end Lipschitz constant** through all `n_layers` blocks. -/
noncomputable def endToEndLipschitz (params : ModelParams cfg) (alpha_max : ℝ) : ℝ :=
  ∏ l : Fin cfg.n_layers, perBlockLipschitz cfg eps (params.blocks l) alpha_max

/-- **End-to-end Lipschitz constant is non-negative.** -/
theorem endToEndLipschitz_nonneg (params : ModelParams cfg) (alpha_max : ℝ) (heps : 0 < eps) :
    0 ≤ endToEndLipschitz cfg eps params alpha_max := by
  unfold endToEndLipschitz
  exact Finset.prod_nonneg fun l _ =>
    perBlockLipschitz_nonneg cfg eps (params.blocks l) alpha_max heps

/-- **One block is Lipschitz.**

Two residual streams no further apart than `Δ = max_i ‖x i - y i‖` at depth
`l` are no further apart than `L_block · Δ` at depth `l + 1`, at every
position.

The estimate has to be uniform in the position on the input side: the
attention sub-layer mixes positions, so what happens at `i` depends on the
whole stream.  On the output side it is pointwise, which is what lets the
depth induction of `Properties.Lipschitz` iterate it.

Source: `reference/model.py` (`Block.forward`), through
`GPTMini.attnSubLayer_dist_le` and `GPTMini.ffnSubLayer_lipschitz`. -/
theorem blockForward_lipschitz
    (params : BlockParams cfg) (alpha_max : ℝ) (heps : 0 < eps)
    (halpha : ∀ h, params.attn.log_alpha h ≤ alpha_max)
    {T : ℕ} [Nonempty (Fin T)] (positions : Fin T → ℝ)
    (x y : Fin T → EucSpace cfg.d_model) :
    ∀ i : Fin T,
      ‖blockForward cfg params eps positions x i
        - blockForward cfg params eps positions y i‖
      ≤ perBlockLipschitz cfg eps params alpha_max
        * (Finset.univ : Finset (Fin T)).sup' Finset.univ_nonempty
            (fun i' => ‖x i' - y i'‖) := by
  intro i
  set Δ : ℝ := (Finset.univ : Finset (Fin T)).sup' Finset.univ_nonempty
    (fun i' => ‖x i' - y i'‖) with hΔdef
  have hD : ∀ j : Fin T, ‖x j - y j‖ ≤ Δ := fun j =>
    Finset.le_sup' (fun i' => ‖x i' - y i'‖) (Finset.mem_univ j)
  have hΔ0 : 0 ≤ Δ := (norm_nonneg _).trans (hD i)
  have hA0 := attnLipschitz_nonneg cfg eps params alpha_max heps
  have hF0 := ffnLipschitz_nonneg cfg eps params
  set x1 := fun j => x j + attnSubLayer cfg params.attn eps positions x j with hx1
  set y1 := fun j => y j + attnSubLayer cfg params.attn eps positions y j with hy1
  -- the first residual branch multiplies the displacement by `1 + L_attn`
  have hstep1 : ∀ j : Fin T,
      ‖x1 j - y1 j‖ ≤ (1 + attnLipschitz cfg eps params alpha_max) * Δ := by
    intro j
    have hattn : ‖attnSubLayer cfg params.attn eps positions x j
        - attnSubLayer cfg params.attn eps positions y j‖
        ≤ attnLipschitz cfg eps params alpha_max * Δ := by
      refine (attnSubLayer_dist_le cfg params.attn eps heps positions alpha_max halpha
        x y Δ hD j).trans ?_
      rw [attnLipschitz]
      exact le_of_eq (by ring)
    have hsplit : x1 j - y1 j = (x j - y j)
        + (attnSubLayer cfg params.attn eps positions x j
          - attnSubLayer cfg params.attn eps positions y j) := by
      simp only [hx1, hy1]; abel
    rw [hsplit]
    exact (norm_add_le _ _).trans (by linarith [hD j, hattn])
  -- the second residual branch multiplies it by `1 + L_ffn`
  have hffn : ‖ffnSubLayer cfg params.ffn eps x1 i - ffnSubLayer cfg params.ffn eps y1 i‖
      ≤ ffnLipschitz cfg eps params * ‖x1 i - y1 i‖ := by
    refine (ffnSubLayer_lipschitz cfg params.ffn eps heps x1 y1 i).trans ?_
    rw [ffnLipschitz]
  have hsplit : blockForward cfg params eps positions x i
      - blockForward cfg params eps positions y i
      = (x1 i - y1 i)
        + (ffnSubLayer cfg params.ffn eps x1 i - ffnSubLayer cfg params.ffn eps y1 i) := by
    show (x1 i + ffnSubLayer cfg params.ffn eps x1 i)
      - (y1 i + ffnSubLayer cfg params.ffn eps y1 i) = _
    abel
  rw [hsplit, perBlockLipschitz]
  refine (norm_add_le _ _).trans ?_
  nlinarith [hstep1 i, hffn, norm_nonneg (x1 i - y1 i)]

/-- The hypotheses are satisfiable: the `eps = 10⁻⁶` of `reference/model.py`,
and any `α_max` above the block's own `log α`. -/
example (params : BlockParams Config.default) (alpha_max : ℝ)
    (halpha : ∀ h, params.attn.log_alpha h ≤ alpha_max)
    (x y : Fin 1 → EucSpace Config.default.d_model) (i : Fin 1) :
    ‖blockForward Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) x i
        - blockForward Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ)) y i‖
      ≤ perBlockLipschitz Config.default 1e-6 params alpha_max
        * (Finset.univ : Finset (Fin 1)).sup' Finset.univ_nonempty
            (fun i' => ‖x i' - y i'‖) :=
  blockForward_lipschitz Config.default 1e-6 params alpha_max (by norm_num) halpha _ x y i

end Properties
end GPTMini
end Transformer
