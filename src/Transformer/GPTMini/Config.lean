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
import Mathlib.Data.Fintype.Pigeonhole

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

/-! ### Grouped-query attention

`train_gpt.py` (`openai/parameter-golf`, 2026-09-13) attends with `n_heads`
query heads against `n_kv_heads < n_heads` key/value heads, each key/value head
shared by a contiguous group of `n_heads / n_kv_heads` query heads
(`repeat_interleave` on the head axis).  Divisibility is what makes the groups
equal, and it is the only well-formedness the construction needs beyond
`Config`. -/

/-- A `Config` together with a grouped-query head split: `n_kv_heads` key/value
heads, dividing the `n_heads` query heads.

`n_kv_heads = n_heads` is ordinary multi-head attention and `n_kv_heads = 1` is
multi-query attention, so this is a generalization of `Config` rather than a
different architecture.

Source: `openai/parameter-golf`, `train_gpt.py` (`CausalSelfAttention`). -/
structure GQAConfig extends Config where
  n_kv_heads     : ℕ
  kv_divides     : n_kv_heads ∣ n_heads
  n_kv_heads_pos : 0 < n_kv_heads

namespace GQAConfig

variable (cfg : GQAConfig)

/-- Query heads per key/value head: `n_heads / n_kv_heads`, the argument of
`repeat_interleave`. -/
def group_size : ℕ := cfg.n_heads / cfg.n_kv_heads

theorem group_size_pos : 0 < cfg.group_size :=
  Nat.div_pos (Nat.le_of_dvd cfg.n_heads_pos cfg.kv_divides) cfg.n_kv_heads_pos

/-- The groups tile the query heads: `n_kv_heads · group_size = n_heads`. -/
theorem n_kv_heads_mul_group_size : cfg.n_kv_heads * cfg.group_size = cfg.n_heads :=
  Nat.mul_div_cancel' cfg.kv_divides

/-- The key/value head that query head `h` reads: contiguous grouping, as
`repeat_interleave` produces. -/
def kvHead (h : Fin cfg.n_heads) : Fin cfg.n_kv_heads :=
  ⟨h / cfg.group_size, by
    rw [Nat.div_lt_iff_lt_mul cfg.group_size_pos, cfg.n_kv_heads_mul_group_size]
    exact h.isLt⟩

/-- Every key/value head is read by some query head: no key/value head is
computed and then wasted. -/
theorem kvHead_surjective : Function.Surjective cfg.kvHead := by
  intro j
  refine ⟨⟨j * cfg.group_size, ?_⟩, ?_⟩
  · rw [← cfg.n_kv_heads_mul_group_size]
    exact Nat.mul_lt_mul_of_lt_of_le j.isLt le_rfl cfg.group_size_pos
  · exact Fin.ext (Nat.mul_div_cancel _ cfg.group_size_pos)

/-- **Multi-head attention is the full case.**  With one key/value head per
query head, `kvHead` is the identity: `Config` is the `n_kv_heads = n_heads`
instance of `GQAConfig`, and every theorem stated for the latter covers it. -/
theorem kvHead_val_eq (hfull : cfg.n_kv_heads = cfg.n_heads) (h : Fin cfg.n_heads) :
    (cfg.kvHead h : ℕ) = (h : ℕ) := by
  have hg : cfg.group_size = 1 := by
    rw [group_size, hfull, Nat.div_self cfg.n_heads_pos]
  simp [kvHead, hg]

/-- **Multi-query attention is the other end.**  One key/value head, read by
every query head. -/
theorem kvHead_val_eq_zero (hmq : cfg.n_kv_heads = 1) (h : Fin cfg.n_heads) :
    (cfg.kvHead h : ℕ) = 0 :=
  Nat.lt_one_iff.mp (hmq ▸ (cfg.kvHead h).isLt)

/-- **Sharing is what fewer key/value heads means.**  Below the full count two
distinct query heads read the same key/value head, so the key/value projections
of a `GQAConfig` are not those of any `Config` with `n_heads` heads: the
architecture is genuinely coarser, not a reparameterization. -/
theorem exists_shared_kvHead (hlt : cfg.n_kv_heads < cfg.n_heads) :
    ∃ h₁ h₂ : Fin cfg.n_heads, h₁ ≠ h₂ ∧ cfg.kvHead h₁ = cfg.kvHead h₂ :=
  Fintype.exists_ne_map_eq_of_card_lt cfg.kvHead (by simpa using hlt)

/-- **And what it buys.**  The per-token key/value cache of the full head split
is `group_size` times the cache of the grouped one: `n_heads · d_head` entries
against `n_kv_heads · d_head`. -/
theorem cache_eq_group_size_mul :
    cfg.n_heads * cfg.head_dim = cfg.group_size * (cfg.n_kv_heads * cfg.head_dim) := by
  rw [← cfg.n_kv_heads_mul_group_size, mul_comm cfg.n_kv_heads cfg.group_size, mul_assoc]

/-- Any divisor of the head count gives a grouped-query split of a `Config`. -/
noncomputable def ofDivisor (c : Config) (m : ℕ) (hm : m ∣ c.n_heads) (hpos : 0 < m) :
    GQAConfig where
  toConfig       := c
  n_kv_heads     := m
  kv_divides     := hm
  n_kv_heads_pos := hpos

/-- The hypotheses of the theorems above are satisfiable, and not vacuously:
the default config carries the full split (`12` key/value heads, groups of
`1`), a proper grouped split (`4` key/value heads, groups of `3`) and the
multi-query split (one key/value head), and the last two are below the full
head count, where `exists_shared_kvHead` applies. -/
example :
    (ofDivisor Config.default 12 dvd_rfl (by decide)).group_size = 1 ∧
      (ofDivisor Config.default 4 (by decide) (by decide)).group_size = 3 ∧
      (ofDivisor Config.default 1 (one_dvd _) (by decide)).n_kv_heads = 1 ∧
      (ofDivisor Config.default 4 (by decide) (by decide)).n_kv_heads
        < Config.default.n_heads := by
  refine ⟨rfl, rfl, rfl, by decide⟩

end GQAConfig

end GPTMini
end Transformer
