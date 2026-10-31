module

public import Transformer.SequenceModel.Attention.Expressivity.Context
public import Transformer.SequenceModel.Attention.Expressivity.Dimensions

/-!
# One actual normalized attention block computes any binary finite task

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, E7.
The attention branch supplies a lossless ordered-pair code. Fixed SwiGLU
matrices interpolate the inverse-normalized displacement to the requested
answer vector. The target table enters fixed output weights only.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Expressivity

open GPTMini
open scoped BigOperators

/-- Raw branch outputs that land exactly on the answer code after RMSNorm. -/
def targets (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size)
    (a b : Fin cfg.vocab_size) : EucSpace cfg.d_model :=
  unnormalize cfg.d_model eps
    (merge cfg h1 (unit (e (.inl (f a b)))) - merge cfg h1 (pairCode e a b))

/-- The complete attention and feed-forward matrices, fixed for a given task. -/
def taskBlock (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {Z : Type*} (layout : Fin cfg.d_ff ≃ (Features cfg.vocab_size × Fin 2) ⊕ Z)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size) : BlockParams cfg :=
  withProducts cfg (contextBlock cfg h1 eps e) layout
    (fun r => (interpGate e r).comp (slice cfg))
    (fun r => (interpValue e r).comp (slice cfg))
    (interpOut (normScale cfg.d_model eps 1) (normScale cfg.d_model eps 2) (targets cfg h1 eps e f))

/-- E7: pre-FFN RMSNorm retains the two-scale code through the actual head reshape. -/
theorem rms_merge_pair (cfg : Config) (h1 : cfg.n_heads = 1) (eps : ℝ)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim) (a b : Fin cfg.vocab_size) :
    rmsNormEps eps (merge cfg h1 (pairCode e a b)) =
      merge cfg h1 ((if a = b then normScale cfg.d_model eps 1 else normScale cfg.d_model eps 2) •
        pairCode e a b) := by
  rw [rms_of_normSq cfg.d_model eps _ _ (by rw [LinearIsometry.norm_map, pairCode_normSq])]
  by_cases hab : a = b <;> simp only [hab, ↓reduceIte, map_smul]

/-- E7: the genuine fixed SwiGLU branch interpolates every raw target exactly. -/
theorem task_feedForward (cfg : Config) (h1 : cfg.n_heads = 1) {eps : ℝ} (heps : 0 < eps)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {Z : Type*} (layout : Fin cfg.d_ff ≃ (Features cfg.vocab_size × Fin 2) ⊕ Z)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size) (a b : Fin cfg.vocab_size) :
    feedForward cfg (taskBlock cfg h1 eps e layout f)
      (rmsNormEps eps (merge cfg h1 (pairCode e a b))) = targets cfg h1 eps e f a b := by
  rw [rms_merge_pair, taskBlock, feedForward_withProducts]
  simp only [ContinuousLinearMap.comp_apply, slice_merge]
  exact interpolate_pair e _ _
    (normScale_pos _ cfg.d_model_pos eps heps 1 (by norm_num)).ne'
    (normScale_pos _ cfg.d_model_pos eps heps 2 (by norm_num)).ne' _ a b

/-- E7: every answer displacement lies within the actual normalized branch's image. -/
theorem targets_normalize (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps : ℝ} (heps : 0 < eps)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size) (a b : Fin cfg.vocab_size) :
    rmsNormEps eps (targets cfg h1 eps e f a b) =
      merge cfg h1 (unit (e (.inl (f a b)))) - merge cfg h1 (pairCode e a b) := by
  apply rms_unnormalize _ cfg.d_model_pos eps heps
  rw [← map_sub, LinearIsometry.norm_map]
  exact lt_of_le_of_lt (displacement_normSq_le e a b (f a b)) (by exact_mod_cast hw)

/-- E7: the complete block, with all four branch normalizers and active XSA,
computes any finite binary task using one fixed set of matrices. -/
theorem task_block (cfg : Config) (h1 : cfg.n_heads = 1) (hw : 3 < cfg.d_model)
    {eps xeps : ℝ} (heps : 0 < eps) (hx : xeps ≤ 1)
    (e : (Fin cfg.vocab_size ⊕ Fin cfg.vocab_size) ↪ Fin cfg.head_dim)
    {Z : Type*} (layout : Fin cfg.d_ff ≃ (Features cfg.vocab_size × Fin 2) ⊕ Z)
    (f : Fin cfg.vocab_size → Fin cfg.vocab_size → Fin cfg.vocab_size)
    (angle : Fin 2 → Fin (cfg.head_dim / 2) → ℝ) (a b : Fin cfg.vocab_size) :
    block cfg (taskBlock cfg h1 eps e layout f) eps xeps angle
      (fun j => merge cfg h1 (unit (e (.inl (![a, b] j))))) 1 =
      merge cfg h1 (unit (e (.inl (f a b)))) := by
  let x := fun j : Fin 2 => merge cfg h1 (unit (e (.inl (![a, b] j))))
  have ha : attention cfg (taskBlock cfg h1 eps e layout f) eps xeps angle
      (fun j => rmsNormEps eps (x j)) 1 =
      attention cfg (contextBlock cfg h1 eps e) eps xeps angle (fun j => rmsNormEps eps (x j)) 1 := rfl
  change (x 1 + rmsNormEps eps _) + rmsNormEps eps
    (feedForward cfg (taskBlock cfg h1 eps e layout f) (rmsNormEps eps (x 1 + rmsNormEps eps _))) = _
  rw [ha]
  dsimp only [x, Matrix.cons_val_one, Matrix.cons_val_fin_one, Matrix.cons_val_zero]
  simp only [context_residual cfg h1 hw heps hx]
  rw [task_feedForward cfg h1 heps, targets_normalize cfg h1 hw heps]
  abel

private abbrev cfg := config 2 (by decide) 0 0
private abbrev e := banks 2 (by decide) 0 0
private abbrev layout := hiddenLayout 2 (by decide) 0 0

example := interpolate_pair e 2 3 (by norm_num) (by norm_num)
  (fun a b => WithLp.toLp 2 ![(a.val : ℝ), (b.val : ℝ), 1]) 0 1

example := rms_merge_pair cfg rfl (1 / 8388608) e 0 1
example := rms_merge_unit cfg rfl (1 / 8388608) ⟨0, by decide⟩
example := context_qkv cfg rfl (eps := 1 / 8388608) (by norm_num) e 0
example := context_attention cfg rfl (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) e (fun j k => j.val * ((k.val : ℝ) + 1) / 7) 0 1
example := context_residual cfg rfl (by decide) (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) e (fun j k => j.val * ((k.val : ℝ) + 1) / 7) 1 1
example := task_feedForward cfg rfl (eps := 1 / 8388608) (by norm_num) e layout
  (fun a b => a + b) 0 1
example := targets_normalize cfg rfl (by decide) (eps := 1 / 8388608) (by norm_num) e
  (fun a b => a + b) 0 1
example := task_block cfg rfl (by decide) (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) e layout (fun a b => a + b)
  (fun j k => j.val * ((k.val : ℝ) + 1) / 7) 0 1

end Transformer.SequenceModel.Attention.Expressivity
