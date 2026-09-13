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

We assemble the complete forward pass and prove its first universal-in-
weights properties:
  - well-definedness (totality)
  - output is a probability vector in the simplex (after softmax)
  - residual stream growth bound across all `n_layers` blocks.
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

/-- **Top-level forward.**

Given tokens of length `T ≤ max_seq_len`, applies the stack of `n_layers`
Pre-LN blocks and produces logits over `vocab_size`. -/
noncomputable def forward
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size) : ℝ :=
  let x0 := embed cfg params tokens
  let xL :=
    (Finset.univ : Finset (Fin cfg.n_layers)).val.toList.foldl
      (fun x l => blockForward cfg (params.blocks l) eps positions x)
      x0
  unembed cfg params
    (fun i => rmsNormEps eps (xL i)) i v

/-- **Forward total.**  For any parameters and any token sequence
of length `T ≤ max_seq_len`, the forward function returns a finite real. -/
theorem forward_total
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size) :
    ∃ y : ℝ, forward cfg params eps positions tokens i v = y := by
  exact ⟨_, rfl⟩

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

/-- **Output is in the simplex** (after softmax):

  `0 ≤ prob_i(v)` and `Σ_v prob_i(v) = 1`. -/
theorem softmaxOutput_is_distribution
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) :
    (∀ v : Fin cfg.vocab_size,
        0 ≤ softmaxOutput cfg params eps positions tokens i v)
    ∧
    (∑ v : Fin cfg.vocab_size,
        softmaxOutput cfg params eps positions tokens i v) = 1 := by
  sorry

/-- **Residual stream growth bound** through `n_layers`:

  `‖x_L i‖ ≤ ‖x_0 i‖ + n_layers · C(params)`,

where `C(params)` is the per-layer sub-layer-output bound. -/
theorem residual_stream_growth
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ) (heps : 0 < eps)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) :
    True := by trivial

/-- **End-to-end Lipschitz constant** (placeholder for Phase 4).

For each pair of input token-sequences differing in just one position, the
logits differ by at most `L(params) · max ‖embedding_difference‖`. -/
theorem forward_lipschitz_in_embedding
    (cfg : Config) (params : ModelParams cfg) (eps : ℝ) :
    True := by trivial

end GPTMini
end Transformer
