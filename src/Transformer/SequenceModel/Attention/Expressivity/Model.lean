module

public import Transformer.SequenceModel.Attention.Expressivity.Block
public import Transformer.SequenceModel.Attention.Stationary

/-!
# Exact finite-task classification by the complete tied Transformer

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E8.
One active attention block computes an arbitrary binary task. Additional
source blocks have zero output projections, hence identity residuals.
Embedding, final RMSNorm and tied readout give a strictly positive margin.
The single weight vector depends on the task, never on its queried operands.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini
open scoped RealInnerProductSpace

/-- One shared input/output embedding, with exact RMSNorm compensation. -/
def taskEmbedding (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (a : Fin cfg.vocab_size) : EucSpace cfg.d_model :=
  amplitude cfg eps • merge cfg h1 (unit (e (.inl a)))

/-- Full finite-depth model: one active block followed by genuine identity residual blocks. -/
def taskModel (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {Z : Type*} (layout : Fin cfg.d_ff ≃ (Features cfg.vocab_size × Fin 2) ⊕ Z)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size) : ModelParams cfg where
  embedding := taskEmbedding cfg h1 eps e
  blocks d := if d.val = 0 then taskBlock cfg h1 eps e layout f else (decode cfg 0).blocks d

/-- E8: the actual embedding normalizer produces exactly the unit token code. -/
theorem embedding_normalize (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps : ℝ} (heps : 0 < eps)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) (a : Fin cfg.vocab_size) :
    rmsNormEps eps (taskEmbedding cfg h1 eps e a) = merge cfg h1 (unit (e (.inl a))) := by
  have hn : ‖merge cfg h1 (unit (e (.inl a)))‖ ^ 2 = 1 := by
    rw [LinearIsometry.norm_map, norm_unit, one_pow]
  have hpre : taskEmbedding cfg h1 eps e a =
      unnormalize cfg.d_model eps (merge cfg h1 (unit (e (.inl a)))) := by
    simp only [taskEmbedding, unnormalize, amplitude, hn]
  rw [hpre]
  exact rms_unnormalize _ cfg.d_model_pos eps heps _
    (by rw [hn]; exact_mod_cast (show 1 < cfg.d_model by omega))

/-- E8: after the active block the answer code survives any finite number
of additional real normalized residual blocks. -/
theorem task_hidden (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {Z : Type*} (layout : Fin cfg.d_ff ≃ (Features cfg.vocab_size × Fin 2) ⊕ Z)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size)
    (angle : Fin cfg.n_layers → Fin 2 → Fin (cfg.head_dim / 2) → ℝ)
    (a b : Fin cfg.vocab_size) (n : ℕ) :
    hidden cfg (taskModel cfg h1 eps e layout f) eps xeps angle ![a, b] (n + 1) 1 =
      merge cfg h1 (unit (e (.inl (f a b)))) := by
  induction n with
  | zero =>
    simp only [hidden, cfg.n_layers_pos, ↓reduceDIte]
    change block cfg (taskBlock cfg h1 eps e layout f) eps xeps (angle ⟨0, cfg.n_layers_pos⟩)
      (fun j => rmsNormEps eps (taskEmbedding cfg h1 eps e (![a, b] j))) 1 = _
    simp only [embedding_normalize cfg h1 hw heps]
    exact task_block cfg h1 hw heps hx e layout f _ a b
  | succ n ih =>
    rw [hidden]
    split_ifs with hn
    · have hz : n + 1 ≠ 0 := by omega
      simp only [taskModel, hz, ↓reduceIte, block_zero]
      exact ih
    · exact ih

/-- E8: the positive tied-logit margin includes the final RMSNorm and raw embedding scale. -/
theorem amplitude_pos (cfg : Config) (hw : 3 < cfg.d_model) {eps : ℝ} (heps : 0 < eps) :
    0 < amplitude cfg eps := by
  apply Real.sqrt_pos.mpr
  apply div_pos (mul_pos (Nat.cast_pos.mpr cfg.d_model_pos) heps)
  have h : (1 : ℝ) < cfg.d_model := by exact_mod_cast (show 1 < cfg.d_model by omega)
  linarith

/-- E8: every actual logit is exactly a positive scalar times the correct-label indicator. -/
theorem task_score (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {Z : Type*} (layout : Fin cfg.d_ff ≃ (Features cfg.vocab_size × Fin 2) ⊕ Z)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size)
    (angle : Fin cfg.n_layers → Fin 2 → Fin (cfg.head_dim / 2) → ℝ)
    (a b q : Fin cfg.vocab_size) :
    score cfg eps xeps angle 1 (encode cfg (taskModel cfg h1 eps e layout f)) ![a, b] q =
      (normScale cfg.d_model eps 1 * amplitude cfg eps) * (if q = f a b then 1 else 0) := by
  have hn : cfg.n_layers - 1 + 1 = cfg.n_layers := by have := cfg.n_layers_pos; omega
  have hh := task_hidden cfg h1 hw heps hx e layout f angle a b (cfg.n_layers - 1)
  rw [hn] at hh
  rw [score, decode_encode, hh, rms_merge_unit]
  change ⟪normScale cfg.d_model eps 1 • merge cfg h1 (unit (e (.inl (f a b)))),
    amplitude cfg eps • merge cfg h1 (unit (e (.inl q)))⟫ = _
  rw [real_inner_smul_left, real_inner_smul_right, LinearIsometry.inner_map_map, inner_unit]
  simp only [e.injective.eq_iff, Sum.inl.injEq]
  by_cases hq : q = f a b
  · simp [hq]
  · simp [hq, Ne.symm hq]

end Transformer.SequenceModel.Attention.Expressivity
