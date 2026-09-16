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
import Transformer.GPTMini.Reshape
import Transformer.GPTMini.QKNorm
import Transformer.GPTMini.RoPE
import Transformer.GPTMini.AttentionBounds
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
  4. Output projection.

The reshapes of step 3 are `GPTMini.Reshape`: `qkvSlice` is the
`chunk(3, dim=-1)` of `reference/model.py`, `headSlice` its
`view(T, n_heads, head_dim)`, and `headMerge` the transpose-and-view that
puts the heads back together for the output projection. -/
noncomputable def attnSubLayer
    (cfg : Config) (params : AttnParams cfg)
    (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) :
    Fin T → EucSpace cfg.d_model :=
  let qkv := fun i => params.W_qkv (rmsNormEps eps (x i))
  let q := fun (h : Fin cfg.n_heads) i => headSlice cfg (qkvSlice cfg (qkvQ cfg) (qkv i)) h
  let k := fun (h : Fin cfg.n_heads) i => headSlice cfg (qkvSlice cfg (qkvK cfg) (qkv i)) h
  let v := fun (h : Fin cfg.n_heads) i => headSlice cfg (qkvSlice cfg (qkvV cfg) (qkv i)) h
  fun i =>
    params.W_o (headMerge cfg (fun h =>
      attentionHead cfg (params.log_alpha h) eps (q h) (k h) (v h) positions i))

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
stream divergence (despite "growth issue" folklore).

The chain is: RMSNorm puts the sub-layer input inside the ball of radius
`√d_model`; `W_qkv` and the two reshapes bound every value vector by
`‖W_qkv‖ √d_model`; the head is a convex combination of those followed by
XSA, which at most doubles it; `headMerge` collects `n_heads` of them.

`‖W_qkv‖` is part of the constant and cannot be dropped: nothing downstream
of the QKV projection renormalizes the values, so scaling `W_qkv` scales the
whole sub-layer output. -/
theorem attnSubLayer_bounded
    (cfg : Config) (params : AttnParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖attnSubLayer cfg params eps positions x i‖
      ≤ 2 * ‖params.W_o‖ * ‖params.W_qkv‖
          * Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ) := by
  set B := ‖params.W_qkv‖ * Real.sqrt (cfg.d_model : ℝ) with hB
  have hvalue : ∀ (h : Fin cfg.n_heads) (j : Fin T),
      ‖headSlice cfg (qkvSlice cfg (qkvV cfg)
        (params.W_qkv (rmsNormEps eps (x j)))) h‖ ≤ B := by
    intro h j
    refine (headSlice_norm_le cfg _ h).trans ?_
    refine (qkvSlice_norm_le cfg _ (qkvV_injective cfg) _).trans ?_
    refine (params.W_qkv.le_opNorm _).trans ?_
    exact mul_le_mul_of_nonneg_left
      (rmsNormEps_norm_le eps heps cfg.d_model_pos _) (norm_nonneg _)
  have hhead : ∀ h : Fin cfg.n_heads,
      ‖attentionHead cfg (params.log_alpha h) eps
        (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg)
          (params.W_qkv (rmsNormEps eps (x j)))) h)
        (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg)
          (params.W_qkv (rmsNormEps eps (x j)))) h)
        (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg)
          (params.W_qkv (rmsNormEps eps (x j)))) h)
        positions i‖ ≤ 2 * B := fun h =>
    attentionHead_norm_le cfg _ eps heps.le _ _ _ positions i B (hvalue h)
  have hmerge := headMerge_norm_le cfg _ (2 * B) hhead
  calc ‖attnSubLayer cfg params eps positions x i‖
      ≤ ‖params.W_o‖ * ‖headMerge cfg (fun h =>
          attentionHead cfg (params.log_alpha h) eps
            (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg)
              (params.W_qkv (rmsNormEps eps (x j)))) h)
            (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg)
              (params.W_qkv (rmsNormEps eps (x j)))) h)
            (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg)
              (params.W_qkv (rmsNormEps eps (x j)))) h)
            positions i)‖ := params.W_o.le_opNorm _
    _ ≤ ‖params.W_o‖ * (Real.sqrt (cfg.n_heads : ℝ) * (2 * B)) :=
        mul_le_mul_of_nonneg_left hmerge (norm_nonneg _)
    _ = 2 * ‖params.W_o‖ * ‖params.W_qkv‖
          * Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ) := by
        rw [hB]; ring

/-- The FFN sub-layer is bounded by `‖W_out‖ ‖W_in‖² d_model`: RMSNorm puts
its input in the ball of radius `√d_model`, and `relu2Vec` squares norms. -/
theorem ffnSubLayer_bounded
    (cfg : Config) (params : FFNParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ}
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖ffnSubLayer cfg params eps x i‖
      ≤ ‖params.W_out‖ * ‖params.W_in‖^2 * (cfg.d_model : ℝ) := by
  have hz : ‖rmsNormEps eps (x i)‖ ≤ Real.sqrt (cfg.d_model : ℝ) :=
    rmsNormEps_norm_le eps heps cfg.d_model_pos _
  have hd : Real.sqrt (cfg.d_model : ℝ) ^ 2 = (cfg.d_model : ℝ) :=
    Real.sq_sqrt (Nat.cast_nonneg _)
  calc ‖ffnSubLayer cfg params eps x i‖
      = ‖params.W_out (relu2Vec (params.W_in (rmsNormEps eps (x i))))‖ := rfl
    _ ≤ ‖params.W_out‖ * ‖relu2Vec (params.W_in (rmsNormEps eps (x i)))‖ :=
        params.W_out.le_opNorm _
    _ ≤ ‖params.W_out‖ * ‖params.W_in (rmsNormEps eps (x i))‖ ^ 2 := by
        gcongr
        exact relu2Vec_norm_bound _
    _ ≤ ‖params.W_out‖ * (‖params.W_in‖ * Real.sqrt (cfg.d_model : ℝ)) ^ 2 := by
        gcongr
        exact (params.W_in.le_opNorm _).trans
          (mul_le_mul_of_nonneg_left hz (norm_nonneg _))
    _ = ‖params.W_out‖ * ‖params.W_in‖ ^ 2 * (cfg.d_model : ℝ) := by
        rw [mul_pow, hd]; ring

/-- **Residual stream growth bound (single block).**

  `‖blockForward(params, x) i‖ ≤ ‖x i‖ + C(params)`,

where `C(params)` is the sum of the two sub-layer bounds above. -/
theorem blockForward_growth
    (cfg : Config) (params : BlockParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    ‖blockForward cfg params eps positions x i‖
      ≤ ‖x i‖
        + 2 * ‖params.attn.W_o‖ * ‖params.attn.W_qkv‖
            * Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ)
        + ‖params.ffn.W_out‖ * ‖params.ffn.W_in‖^2 * (cfg.d_model : ℝ) := by
  set x1 := fun j => x j + attnSubLayer cfg params.attn eps positions x j with hx1
  have hattn := attnSubLayer_bounded cfg params.attn eps heps positions x i
  have hffn := ffnSubLayer_bounded cfg params.ffn eps heps x1 i
  have hstep1 : ‖x1 i‖ ≤ ‖x i‖
      + 2 * ‖params.attn.W_o‖ * ‖params.attn.W_qkv‖
          * Real.sqrt (cfg.n_heads : ℝ) * Real.sqrt (cfg.d_model : ℝ) :=
    le_trans (norm_add_le _ _) (by linarith)
  calc ‖blockForward cfg params eps positions x i‖
      = ‖x1 i + ffnSubLayer cfg params.ffn eps x1 i‖ := rfl
    _ ≤ ‖x1 i‖ + ‖ffnSubLayer cfg params.ffn eps x1 i‖ := norm_add_le _ _
    _ ≤ _ := by linarith

end GPTMini
end Transformer
