/-
# `gpt-mini` Config

Lean record mirroring the Python `Config` dataclass in `reference/model.py`.

All hyperparameters are positive natural numbers (or positive reals).  The
record is parameterized by ordinary natural / real numbers; downstream
specifications take the config as input and derive `Fin`-indexed types from
it.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic

namespace Transformer
namespace GPTMini

/-- All architectural hyperparameters of `gpt-mini`, matching `Config` in
`reference/model.py`. -/
structure Config where
  vocab_size  : ℕ
  n_layers    : ℕ
  n_heads     : ℕ
  d_model     : ℕ
  d_ff        : ℕ
  max_seq_len : ℕ
  rope_theta  : ℝ
  -- well-formedness: divisibility of `d_model` by `n_heads` is required for
  -- well-defined head reshaping; even head dimension is required by RoPE.
  divides     : n_heads ∣ d_model
  head_even   : Even (d_model / n_heads)
  -- positivity:
  n_heads_pos     : 0 < n_heads
  n_layers_pos    : 0 < n_layers
  d_model_pos     : 0 < d_model
  d_ff_pos        : 0 < d_ff
  vocab_pos       : 0 < vocab_size
  max_seq_len_pos : 0 < max_seq_len
  theta_pos       : 0 < rope_theta

namespace Config

/-- Head dimension `d_head = d_model / n_heads`. -/
def head_dim (cfg : Config) : ℕ := cfg.d_model / cfg.n_heads

lemma head_dim_pos (cfg : Config) : 0 < cfg.head_dim := by
  unfold head_dim
  exact Nat.div_pos (Nat.le_of_dvd cfg.d_model_pos cfg.divides) cfg.n_heads_pos

/-- The default `gpt-mini` config from `reference/model.py`:

  `vocab_size=50257, n_layers=12, n_heads=12, d_model=768, d_ff=3072,
   max_seq_len=2048, rope_theta=10000.0`. -/
noncomputable def default : Config where
  vocab_size      := 50_257
  n_layers        := 12
  n_heads         := 12
  d_model         := 768
  d_ff            := 3_072
  max_seq_len     := 2_048
  rope_theta      := 10_000
  divides         := by decide
  head_even       := by decide
  n_heads_pos     := by decide
  n_layers_pos    := by decide
  d_model_pos     := by decide
  d_ff_pos        := by decide
  vocab_pos       := by decide
  max_seq_len_pos := by decide
  theta_pos       := by norm_num

end Config

end GPTMini
end Transformer
