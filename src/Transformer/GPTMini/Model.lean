/-
# `gpt-mini` Top-Level Model

Formalization of the `GPTMini` class from `reference/model.py`:

```python
class GPTMini(nn.Module):
    def forward(self, tokens):
        x = self.embed(tokens)
        cos = self.cos[:T]; sin = self.sin[:T]
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.unembed(self.norm_final(x))
```

We assemble the complete forward pass, and the residual stream `hidden`
after each layer, which the properties of `Transformer.GPTMini.Properties` are
stated about.  Totality needs no theorem: `forward` is a Lean function into
`ℝ`, defined on every input.

The simplex property of the softmax output is in
`Transformer.GPTMini.Properties.OutputSimplex`, and the growth of the stream
in `Transformer.GPTMini.Properties.StreamGrowth`.
-/

import Transformer.Basic
import Transformer.GPTMini.Config
import Transformer.GPTMini.RMSNorm
import Transformer.GPTMini.Block
import Mathlib.Analysis.InnerProductSpace.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini

/-- Parameters of the complete `gpt-mini` model:

  - `embedding : Fin vocab_size → ℝ^{d_model}`
  - `n_layers` copies of `BlockParams` (one per layer)
  - `unembed` is tied to `embedding`, so no separate parameter. -/
structure ModelParams (cfg : Config) where
  embedding : Fin cfg.vocab_size → EucSpace cfg.d_model
  blocks    : Fin cfg.n_layers → BlockParams cfg

/-- The embedding lookup. -/
noncomputable def embed
    (cfg : Config) (params : ModelParams cfg)
    {T : ℕ} (tokens : Fin T → Fin cfg.vocab_size) :
    Fin T → EucSpace cfg.d_model :=
  fun i => params.embedding (tokens i)

/-- The unembedding (tied with `embed`):

  `logits_i v = ⟨x_i, embedding[v]⟩`. -/
noncomputable def unembed
    (cfg : Config) (params : ModelParams cfg)
    {T : ℕ} (x : Fin T → EucSpace cfg.d_model)
    (i : Fin T) (v : Fin cfg.vocab_size) : ℝ :=
  inner (𝕜 := ℝ) (x i) (params.embedding v)

/-- **The residual stream after `L` layers.**

  `x_0 = embed(tokens)`,   `x_{L+1} = blockForward(blocks[L], x_L)`,

the blocks being applied in index order and the stream staying put once the
`n_layers` blocks are exhausted, so that `hidden … L` is the input of block
`L` for every `L ≤ n_layers`.  Mirrors the `for block in self.blocks` loop of
`reference/model.py` (`GPTMini.forward`). -/
noncomputable def hidden
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) :
    ℕ → Fin T → EucSpace cfg.d_model
  | 0 => embed cfg params tokens
  | L + 1 =>
      if h : L < cfg.n_layers then
        blockForward cfg (params.blocks ⟨L, h⟩) eps positions
          (hidden cfg params eps positions tokens L)
      else
        hidden cfg params eps positions tokens L

/-- **Top-level forward.**

Given tokens of any length `T`, applies the stack of `n_layers` Pre-LN blocks
and produces logits over `vocab_size`.  The reference model caps `T` at
`max_seq_len` because its RoPE tables have that many rows; here the rotation
is computed from `positions` directly, so no cap is needed and none is
imposed. -/
noncomputable def forward
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size) : ℝ :=
  unembed cfg params
    (fun i => rmsNormEps eps (hidden cfg params eps positions tokens cfg.n_layers i)) i v

/-- The softmax probability vector at position `i`:

  `prob_i(v) = exp(logits_i(v)) / Σ_w exp(logits_i(w))`. -/
noncomputable def softmaxOutput
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size) : ℝ :=
  Real.exp (forward cfg params eps positions tokens i v)
    /
  (∑ w : Fin cfg.vocab_size,
      Real.exp (forward cfg params eps positions tokens i w))

end GPTMini
end Transformer
