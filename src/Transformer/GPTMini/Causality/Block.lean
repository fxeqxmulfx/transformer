import Transformer.GPTMini.Causality.Basic
import Transformer.GPTMini.Block

/-!
# Prefix preservation through the actual pre-norm block

Source: Block.forward in archived gpt_mini.py at f11b6e2, and
infrastructure/nn/transformer.py at cbafbe9. Prefix agreement survives
RMSNorm, all QKV/head reshapes, the output projection, both residual
additions and the ReLU-squared FFN. Parameters are arbitrary and shared
between the shorter and longer evaluations, as in the actual model.
-/

namespace Transformer.GPTMini.Causality

noncomputable section

/-- A shared pointwise operation cannot introduce dependence on future rows.
Source: the pointwise normalization, projection and activation in Block.forward. -/
theorem map_prefix {A B : Type*} {S R : ℕ} (f : A → B)
    (short : Fin S → A) (long : Fin (S + R) → A) (h : PrefixEq short long) :
    PrefixEq (fun j => f (short j)) (fun j => f (long j)) := by
  intro j
  exact congrArg f (h j)

example : PrefixEq (fun j : Fin 2 => j.val)
    (fun j : Fin (2 + 1) => if j.val < 2 then j.val else 99) := by
  intro j
  dsimp only
  have hj : (j.castAdd 1).val < 2 := j.isLt
  rw [ite_eq_left hj]
  rfl

/-- Adding two prefix-preserving streams preserves their common prefix.
Source: both residual additions in the reference pre-norm Block.forward. -/
theorem add_prefix {A : Type*} [Add A] {S R : ℕ}
    (x y : Fin S → A) (xl yl : Fin (S + R) → A)
    (hx : PrefixEq x xl) (hy : PrefixEq y yl) :
    PrefixEq (fun j => x j + y j) (fun j => xl j + yl j) := by
  intro j
  dsimp only
  rw [hx j, hy j]

example : PrefixEq (fun j : Fin 2 => j.val) (fun j : Fin (2 + 1) => j.val) ∧
    PrefixEq (fun j : Fin 2 => j.val + 3) (fun j : Fin (2 + 1) => j.val + 3) :=
  ⟨fun _ => rfl, fun _ => rfl⟩

/-- The full multi-head attention sublayer preserves every old row.
Source: CausalMHA.forward, including the actual fused QKV and headMerge coordinate layout. -/
theorem attention_prefix (cfg : Config) (params : AttnParams cfg) (eps : ℝ) {S R : ℕ}
    (positions : Fin S → ℝ) (pl : Fin (S + R) → ℝ)
    (x : Fin S → EucSpace cfg.d_model) (xl : Fin (S + R) → EucSpace cfg.d_model)
    (hp : PrefixEq positions pl) (hx : PrefixEq x xl) :
    PrefixEq (attnSubLayer cfg params eps positions x)
      (attnSubLayer cfg params eps pl xl) := by
  have hqkv : PrefixEq
      (fun j => params.W_qkv (rmsNormEps eps (x j)))
      (fun j => params.W_qkv (rmsNormEps eps (xl j))) :=
    map_prefix (fun z => params.W_qkv (rmsNormEps eps z)) x xl hx
  have hq : ∀ h : Fin cfg.n_heads, PrefixEq
      (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg)
        (params.W_qkv (rmsNormEps eps (x j)))) h)
      (fun j => headSlice cfg (qkvSlice cfg (qkvQ cfg)
        (params.W_qkv (rmsNormEps eps (xl j)))) h) :=
    fun h => map_prefix (fun z => headSlice cfg (qkvSlice cfg (qkvQ cfg) z) h) _ _ hqkv
  have hk : ∀ h : Fin cfg.n_heads, PrefixEq
      (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg)
        (params.W_qkv (rmsNormEps eps (x j)))) h)
      (fun j => headSlice cfg (qkvSlice cfg (qkvK cfg)
        (params.W_qkv (rmsNormEps eps (xl j)))) h) :=
    fun h => map_prefix (fun z => headSlice cfg (qkvSlice cfg (qkvK cfg) z) h) _ _ hqkv
  have hv : ∀ h : Fin cfg.n_heads, PrefixEq
      (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg)
        (params.W_qkv (rmsNormEps eps (x j)))) h)
      (fun j => headSlice cfg (qkvSlice cfg (qkvV cfg)
        (params.W_qkv (rmsNormEps eps (xl j)))) h) :=
    fun h => map_prefix (fun z => headSlice cfg (qkvSlice cfg (qkvV cfg) z) h) _ _ hqkv
  intro i
  dsimp only [attnSubLayer]
  apply congrArg params.W_o
  apply congrArg (headMerge cfg)
  funext h
  apply head_prefix
  · exact hq h
  · exact hk h
  · exact hv h
  · exact hp

example (cfg : Config) :
    PrefixEq (fun j : Fin 2 => (j.val : ℝ))
      (fun j : Fin (2 + 1) => (j.val : ℝ)) ∧
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.d_model))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.d_model)) :=
  ⟨fun _ => rfl, fun _ => rfl⟩

/-- The FFN and its preceding RMSNorm are pointwise in sequence position.
Source: FeedForward.forward and Block.forward at cbafbe9, with the GPTMini ReLU-squared recipe. -/
theorem ffn_prefix (cfg : Config) (params : FFNParams cfg) (eps : ℝ) {S R : ℕ}
    (x : Fin S → EucSpace cfg.d_model) (xl : Fin (S + R) → EucSpace cfg.d_model)
    (hx : PrefixEq x xl) :
    PrefixEq (ffnSubLayer cfg params eps x) (ffnSubLayer cfg params eps xl) := by
  exact map_prefix (fun z => relu2FFN params.W_in params.W_out (rmsNormEps eps z)) x xl hx

example (cfg : Config) :
    PrefixEq (fun _ : Fin 2 => (0 : EucSpace cfg.d_model))
      (fun _ : Fin (2 + 1) => (0 : EucSpace cfg.d_model)) := fun _ => rfl

/-- One actual transformer block preserves the entire old prefix, for arbitrary parameters.
Source: Block.forward; the FFN and residual connections are included rather than omitted. -/
theorem block_prefix (cfg : Config) (params : BlockParams cfg) (eps : ℝ) {S R : ℕ}
    (positions : Fin S → ℝ) (pl : Fin (S + R) → ℝ)
    (x : Fin S → EucSpace cfg.d_model) (xl : Fin (S + R) → EucSpace cfg.d_model)
    (hp : PrefixEq positions pl) (hx : PrefixEq x xl) :
    PrefixEq (blockForward cfg params eps positions x) (blockForward cfg params eps pl xl) := by
  have hx1 := add_prefix x (attnSubLayer cfg params.attn eps positions x)
    xl (attnSubLayer cfg params.attn eps pl xl) hx
    (attention_prefix cfg params.attn eps positions pl x xl hp hx)
  exact add_prefix _ _ _ _ hx1 (ffn_prefix cfg params.ffn eps _ _ hx1)

example (cfg : Config) :
    PrefixEq (fun j : Fin 1 => (j.val : ℝ))
      (fun j : Fin (1 + 3) => (j.val : ℝ)) ∧
    PrefixEq (fun _ : Fin 1 => (0 : EucSpace cfg.d_model))
      (fun _ : Fin (1 + 3) => (0 : EucSpace cfg.d_model)) :=
  ⟨fun _ => rfl, fun _ => rfl⟩

/-- Pointwise FFN outputs at equal rows agree even when all other rows differ.
Source: the actual ffnSubLayer definition, including both learned projections. -/
theorem ffn_row (cfg : Config) (params : FFNParams cfg) (eps : ℝ) {S T : ℕ}
    (x : Fin S → EucSpace cfg.d_model) (y : Fin T → EucSpace cfg.d_model)
    (i : Fin S) (j : Fin T) (h : x i = y j) :
    ffnSubLayer cfg params eps x i = ffnSubLayer cfg params eps y j := by
  unfold ffnSubLayer
  rw [h]

example (cfg : Config) :
    (fun _ : Fin 2 => (0 : EucSpace cfg.d_model)) ⟨0, by decide⟩ =
      (fun _ : Fin 3 => (0 : EucSpace cfg.d_model)) ⟨0, by decide⟩ := rfl

/-- Equal input embeddings remain equal after all pointwise QKV operations at a row.
Source: the actual pre-normalized fused QKV projection used in attnSubLayer. -/
theorem qkv_row (cfg : Config) (params : AttnParams cfg) (eps : ℝ) {S T : ℕ}
    (x : Fin S → EucSpace cfg.d_model) (y : Fin T → EucSpace cfg.d_model)
    (i : Fin S) (j : Fin T) (h : x i = y j) :
    params.W_qkv (rmsNormEps eps (x i)) = params.W_qkv (rmsNormEps eps (y j)) := by
  rw [h]

example (cfg : Config) :
    (fun _ : Fin 2 => (0 : EucSpace cfg.d_model)) ⟨1, by decide⟩ =
      (fun _ : Fin 3 => (0 : EucSpace cfg.d_model)) ⟨1, by decide⟩ := rfl

end
end Transformer.GPTMini.Causality
