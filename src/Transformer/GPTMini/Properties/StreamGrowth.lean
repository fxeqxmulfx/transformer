/-
# Property: the residual stream grows at most linearly in depth

Under Pre-LN with the choices of `reference/model.py` (RMSNorm without `γ`,
QK-norm, bounded `‖W_*‖_op`), each sub-layer's contribution to the residual
stream is bounded by a constant depending only on the weight norms — not on
the stream's current magnitude.  Hence the stream grows at most linearly:

  `‖x_L i‖ ≤ ‖x_0 i‖ + L · C(weights)`,

which is `residual_stream_linear_growth` below, proved by induction on the
depth from `Block.blockForward_growth`.  The constant is the largest of the
per-block constants `blockGrowth`, so it does not depend on the depth, and
this is what rules out the Pre-LN "exploding residual stream":

  - RMSNorm normalizes the *input* of each sub-layer to `√d`,
  - QK-norm keeps the attention scores bounded,
  - softmax outputs convex combinations of the `v_j`, of bounded norm,
  - the ReLU² FFN output is bounded by `‖W_out‖ · ‖W_in‖² · d`.

The attention constant carries `‖W_qkv‖`: the values are never renormalized
after the QKV projection, so the sub-layer output scales with it.

What the representation itself is bounded by is a second, unconditional
statement: the final RMSNorm puts it inside the ball of radius `√d_model`
whatever the parameters and the depth (`final_representation_norm_le`).

Both rest on `Transformer.GPTMini.Block` and `Transformer.GPTMini.RMSNorm`,
which is where the per-block bounds live.
-/

import Transformer.GPTMini.Model

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (params : ModelParams cfg) (eps : ℝ)

/-- The per-block growth constant of `Block.blockForward_growth`:

  `C(p) = 2 ‖W_o‖ ‖W_qkv‖ √n_heads √d_model + ‖W_out‖ ‖W_in‖² d_model`.

Source: `reference/model.py` (`Block.forward`), through
`GPTMini.attnSubLayer_bounded` and `GPTMini.ffnSubLayer_bounded`. -/
noncomputable def blockGrowth (p : BlockParams cfg) : ℝ :=
  2 * ‖p.attn.W_o‖ * ‖p.attn.W_qkv‖
      * Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ)
    + ‖p.ffn.W_out‖ * ‖p.ffn.W_in‖ ^ 2 * (cfg.d_model : ℝ)

/-- The per-block growth constant is non-negative. -/
theorem blockGrowth_nonneg (p : BlockParams cfg) : 0 ≤ blockGrowth cfg p := by
  unfold blockGrowth
  have h₁ : 0 ≤ 2 * ‖p.attn.W_o‖ * ‖p.attn.W_qkv‖
      * Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ) := by positivity
  have h₂ : 0 ≤ ‖p.ffn.W_out‖ * ‖p.ffn.W_in‖ ^ 2 * (cfg.d_model : ℝ) :=
    mul_nonneg (mul_nonneg (norm_nonneg _) (sq_nonneg _)) (Nat.cast_nonneg _)
  linarith

/-- **Linear-in-depth growth of the residual stream.**

There is a constant `C ≥ 0`, depending on the parameter norms alone — the
largest of the per-block `blockGrowth` — such that for every position `i` and
every depth `L`,

  `‖hidden L i‖ ≤ ‖embed(tokens) i‖ + L · C`.

Beyond `L = n_layers` the stream is constant, so the bound is only sharp
there; what matters is that `C` does not depend on `L`.

Source: `reference/model.py` (`GPTMini.forward`), by induction from
`GPTMini.blockForward_growth`. -/
theorem residual_stream_linear_growth
    (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) :
    ∃ C : ℝ, 0 ≤ C ∧
      ∀ (L : ℕ) (i : Fin T),
        ‖hidden cfg params eps positions tokens L i‖
          ≤ ‖embed cfg params tokens i‖ + (L : ℝ) * C := by
  have hne : (Finset.univ : Finset (Fin cfg.n_layers)).Nonempty :=
    ⟨⟨0, cfg.n_layers_pos⟩, Finset.mem_univ _⟩
  set C : ℝ := (Finset.univ : Finset (Fin cfg.n_layers)).sup' hne
    (fun l => blockGrowth cfg (params.blocks l)) with hCdef
  have hle : ∀ l : Fin cfg.n_layers, blockGrowth cfg (params.blocks l) ≤ C :=
    fun l => Finset.le_sup' (fun l => blockGrowth cfg (params.blocks l)) (Finset.mem_univ l)
  have hC0 : 0 ≤ C :=
    le_trans (blockGrowth_nonneg cfg (params.blocks ⟨0, cfg.n_layers_pos⟩))
      (hle ⟨0, cfg.n_layers_pos⟩)
  refine ⟨C, hC0, ?_⟩
  intro L
  induction L with
  | zero => intro i; simp [hidden]
  | succ L ih =>
    intro i
    have hcast : ((L + 1 : ℕ) : ℝ) = (L : ℝ) + 1 := by push_cast; ring
    by_cases h : L < cfg.n_layers
    · have hstep :=
        blockForward_growth cfg (params.blocks ⟨L, h⟩) eps heps positions
          (hidden cfg params eps positions tokens L) i
      have hb : blockGrowth cfg (params.blocks ⟨L, h⟩) ≤ C := hle ⟨L, h⟩
      have hunf : hidden cfg params eps positions tokens (L + 1) i
          = blockForward cfg (params.blocks ⟨L, h⟩) eps positions
              (hidden cfg params eps positions tokens L) i := by
        rw [hidden]; simp [h]
      rw [hunf, hcast]
      have hgrowth : blockGrowth cfg (params.blocks ⟨L, h⟩)
          = 2 * ‖(params.blocks ⟨L, h⟩).attn.W_o‖ * ‖(params.blocks ⟨L, h⟩).attn.W_qkv‖
              * Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ)
            + ‖(params.blocks ⟨L, h⟩).ffn.W_out‖ * ‖(params.blocks ⟨L, h⟩).ffn.W_in‖ ^ 2
              * (cfg.d_model : ℝ) := rfl
      have := ih i
      nlinarith [hstep, hb, this, hC0]
    · have hunf : hidden cfg params eps positions tokens (L + 1) i
          = hidden cfg params eps positions tokens L i := by
        rw [hidden]; simp [h]
      rw [hunf, hcast]
      have := ih i
      nlinarith [this, hC0]

/-- **The final representation lies in the ball of radius `√d_model`.**

The last RMSNorm is applied to the stream before the unembedding, and its
output has norm at most `√d_model` whatever the parameters, the tokens and the
depth — `rmsNormEps` scales any vector to that sphere up to the `eps` in the
denominator.  Source: `reference/model.py` (`GPTMini.forward`, `norm_final`),
through `GPTMini.rmsNormEps_norm_le`. -/
theorem final_representation_norm_le
    (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) :
    ‖rmsNormEps eps (hidden cfg params eps positions tokens cfg.n_layers i)‖
      ≤ Real.sqrt (cfg.d_model : ℝ) :=
  rmsNormEps_norm_le eps heps cfg.d_model_pos _

/-- The hypotheses of both theorems are satisfiable: the default config, the
`eps = 10⁻⁶` of `reference/model.py`, and a one-token input. -/
example (params : ModelParams Config.default) :
    (∃ C : ℝ, 0 ≤ C ∧
      ∀ (L : ℕ) (i : Fin 1),
        ‖hidden Config.default params 1e-6 (fun _ => 0)
            (fun _ => ⟨0, Config.default.vocab_pos⟩) L i‖
          ≤ ‖embed Config.default params (fun _ => ⟨0, Config.default.vocab_pos⟩) i‖
            + (L : ℝ) * C)
    ∧ ‖rmsNormEps 1e-6 (hidden Config.default params 1e-6 (fun _ : Fin 1 => (0 : ℝ))
          (fun _ => ⟨0, Config.default.vocab_pos⟩) Config.default.n_layers 0)‖
        ≤ Real.sqrt (Config.default.d_model : ℝ) :=
  ⟨residual_stream_linear_growth Config.default params 1e-6 (by norm_num) _ _,
    final_representation_norm_le Config.default params 1e-6 (by norm_num) _ _ 0⟩

end Properties
end GPTMini
end Transformer
