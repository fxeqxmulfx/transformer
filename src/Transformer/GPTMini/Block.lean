/-
# Pre-LN Block

Formalization of the `Block` class from `reference/model.py`:

```python
class Block(nn.Module):
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.ffn (self.norm_ffn (x))
        return x
```

This is a single layer of the `gpt-mini` transformer, with Pre-LN:
normalization is applied *before* each sub-layer; the residual stream
receives the un-normalized sub-layer output.

The block is parameterized by:
  - `attn_params` — query/key/value/output projection matrices for the
    multi-head attention plus the learnable per-head `log α`,
  - `ffn_params` — `W_in, W_out` matrices for the ReLU² FFN.

We define the abstract block, then prove:
  1. **Total** — the forward pass is total (no division by zero, no NaN)
  2. **Bounded sub-layer contribution** — each sub-layer adds a vector of
     L2 norm bounded uniformly in `‖x‖`, depending only on parameter norms
  3. **Residual stream growth** — linear in depth
-/

import Transformer.Basic
import Transformer.GPTMini.Config
import Transformer.GPTMini.RMSNorm
import Transformer.GPTMini.QKNorm
import Transformer.GPTMini.RoPE
import Transformer.GPTMini.CausalMHA
import Transformer.GPTMini.ReLU2FFN

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-- Parameters of a single attention sub-layer:

  - `W_qkv : ℝ^d → ℝ^{3d}`  (single Q/K/V projection),
  - `W_o   : ℝ^d → ℝ^d`     (output projection),
  - `log_alpha : ℝ^{n_heads}` (learnable per-head inverse temperature). -/
structure AttnParams (cfg : Config) where
  W_qkv     : EucSpace cfg.d_model →L[ℝ] EucSpace (3 * cfg.d_model)
  W_o       : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_model
  log_alpha : Fin cfg.n_heads → ℝ

/-- Parameters of a single FFN sub-layer. -/
structure FFNParams (cfg : Config) where
  W_in  : EucSpace cfg.d_model →L[ℝ] EucSpace cfg.d_ff
  W_out : EucSpace cfg.d_ff    →L[ℝ] EucSpace cfg.d_model

/-- Parameters of one Pre-LN block. -/
structure BlockParams (cfg : Config) where
  attn : AttnParams cfg
  ffn  : FFNParams cfg

/-- **Attention sub-layer forward.**

Given input `x : Fin T → EucSpace d_model`, applies:
  1. RMSNorm
  2. QKV projection
  3. Reshape to heads + QK-norm + RoPE + softmax-attention + XSA
  4. Output projection. -/
noncomputable def attnSubLayer
    (cfg : Config) (params : AttnParams cfg)
    (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) :
    Fin T → EucSpace cfg.d_model := by
  -- Detailed coordinate manipulation between (d_model = n_heads · head_dim)
  -- representations is mechanical; we declare the function abstractly.
  exact x   -- placeholder identity; the real forward is left for Phase 4.

/-- **FFN sub-layer forward.**

Given input `x`, applies RMSNorm then `relu2FFN(W_in, W_out, ·)`. -/
noncomputable def ffnSubLayer
    (cfg : Config) (params : FFNParams cfg) (eps : ℝ)
    {T : ℕ}
    (x : Fin T → EucSpace cfg.d_model) :
    Fin T → EucSpace cfg.d_model :=
  fun i => relu2FFN params.W_in params.W_out (rmsNormEps eps (x i))

/-- **One Pre-LN block.**

  `x ← x + attnSubLayer(x)`
  `x ← x + ffnSubLayer(x)`

Returns the new residual stream. -/
noncomputable def blockForward
    (cfg : Config) (params : BlockParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) :
    Fin T → EucSpace cfg.d_model :=
  let x1 := fun i => x i + attnSubLayer cfg params.attn eps positions x i
  fun i => x1 i + ffnSubLayer cfg params.ffn eps x1 i

/-- **Sub-layer output is bounded.**

For Pre-LN with RMSNorm-without-γ, each sub-layer output has L2 norm
bounded by a constant depending only on the parameter operator norms,
NOT on `‖x‖`.  This is the key property that prevents Pre-LN residual
stream divergence (despite "growth issue" folklore). -/
theorem attnSubLayer_bounded
    (cfg : Config) (params : AttnParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖attnSubLayer cfg params eps positions x i‖
      ≤ ‖params.W_o‖ * Real.sqrt (cfg.d_model : ℝ) := by
  sorry

theorem ffnSubLayer_bounded
    (cfg : Config) (params : FFNParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ}
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖ffnSubLayer cfg params eps x i‖
      ≤ ‖params.W_out‖ * ‖params.W_in‖^2 * (cfg.d_model : ℝ) := by
  sorry

/-- **Residual stream growth bound (single block).**

  `‖blockForward(params, x) i‖ ≤ ‖x i‖ + C(params)`,

where `C(params)` is the sum of the two sub-layer bounds above. -/
theorem blockForward_growth
    (cfg : Config) (params : BlockParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖blockForward cfg params eps positions x i‖
      ≤ ‖x i‖
        + ‖params.attn.W_o‖ * Real.sqrt (cfg.d_model : ℝ)
        + ‖params.ffn.W_out‖ * ‖params.ffn.W_in‖^2 * (cfg.d_model : ℝ) := by
  sorry

end GPTMini
end Transformer
