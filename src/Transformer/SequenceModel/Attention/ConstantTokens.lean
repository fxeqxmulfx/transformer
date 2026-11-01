module

public import Transformer.SequenceModel.Attention.CenteredXSA

/-!
# Repeated-token readouts have a stable common path above XSA clipping

Source: `review/residual_connections.md`, *Repeated-input numerical audit*.
Centering fixes a cancellation-prone forward but does not by itself stabilize
backpropagation through independent position coordinates. For repeated tokens,
the actual parameterized source function can instead read the first position,
provided every layer's value heads remain above the L2 clipping threshold.
The guard below inspects those actual values, not labels or a presumed output.
Clipped heads retain the original readout. No floating-point guard certificate
or unconditional gradient accuracy theorem is asserted here.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini

/-- Actual value head of a block after its input RMSNorm. -/
def normalizedValueHead (cfg : Config) (B : BlockParams cfg) (eps : ℝ)
    (x : EucSpace cfg.d_model) (h : Fin cfg.n_heads) : EucSpace cfg.head_dim :=
  headSlice cfg (qkvSlice cfg (qkvV cfg) (B.qkv (rmsNormEps eps x))) h

/-- C6: constant input states have zero contextual attention when all their
value heads reach the clipping threshold, for arbitrary rotary phases. -/
theorem attention_constant_unclipped (cfg : Config) (B : BlockParams cfg)
    (eps xeps : ℝ) {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : EucSpace cfg.d_model)
    (hclip : ∀ h, xeps ≤ ‖normalizedValueHead cfg B eps x h‖) (i : Fin T) :
    attention cfg B eps xeps angle (fun _ => rmsNormEps eps x) i = 0 := by
  unfold attention
  have hz (h : Fin cfg.n_heads) := headOutput_constant_values eps xeps angle
    (fun _ => headSlice cfg (qkvSlice cfg (qkvQ cfg) (B.qkv (rmsNormEps eps x))) h)
    (fun _ => headSlice cfg (qkvSlice cfg (qkvK cfg) (B.qkv (rmsNormEps eps x))) h)
    (normalizedValueHead cfg B eps x h) i (hclip h)
  simp only [normalizedValueHead] at hz
  simp_rw [hz]
  have hm : headMerge cfg (fun _ => 0) = 0 := by ext j; simp
  rw [hm, map_zero]

/-- C7: the actual Peri-RMSNorm block then applies only its common FFN update.
The output projection and both residual additions are still those of source. -/
theorem block_constant_unclipped (cfg : Config) (B : BlockParams cfg)
    (eps xeps : ℝ) {T : ℕ} (angle : Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (x : EucSpace cfg.d_model)
    (hclip : ∀ h, xeps ≤ ‖normalizedValueHead cfg B eps x h‖) (i : Fin T) :
    block cfg B eps xeps angle (fun _ => x) i =
      x + rmsNormEps eps (feedForward cfg B (rmsNormEps eps x)) := by
  unfold block
  dsimp only
  rw [attention_constant_unclipped cfg B eps xeps angle x hclip i]
  have hz : rmsNormEps eps (0 : EucSpace cfg.d_model) = 0 := by
    simp [rmsNormEps]
  simp only [hz, add_zero]

/-- A finite layer/head check on the source's actual first-position states.
This is a predicate of model parameters and inputs, not an assumed conclusion. -/
def constantValueGuard (cfg : Config) (p : ModelParams cfg) (eps xeps : ℝ)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (token : Fin cfg.vocab_size) (first : Fin T) : Prop :=
  ∀ (n : ℕ) (hn : n < cfg.n_layers) (h : Fin cfg.n_heads),
    xeps ≤ ‖normalizedValueHead cfg (p.blocks ⟨n, hn⟩) eps
      (hidden cfg p eps xeps angle (fun _ => token) n first) h‖

/-- C8: with the explicit guard, every layer of a repeated-token sequence
agrees with its first position. Arbitrary depth and arbitrary RoPE remain. -/
theorem hidden_constant_unclipped (cfg : Config) (p : ModelParams cfg)
    (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (token : Fin cfg.vocab_size) (first : Fin T)
    (hclip : constantValueGuard cfg p eps xeps angle token first) (n : ℕ) :
    ∀ i, hidden cfg p eps xeps angle (fun _ => token) n i =
      hidden cfg p eps xeps angle (fun _ => token) n first := by
  induction n with
  | zero => intro i; rfl
  | succ n ih =>
    intro i
    simp only [hidden]
    split_ifs with hn
    · have hx := funext ih
      rw [hx]
      rw [block_constant_unclipped cfg (p.blocks ⟨n, hn⟩) eps xeps (angle ⟨n, hn⟩)
        _ (hclip n hn) i,
        block_constant_unclipped cfg (p.blocks ⟨n, hn⟩) eps xeps (angle ⟨n, hn⟩)
        _ (hclip n hn) first]
    · exact ih i

/-- C9: the tied readout may select the first position under the same guard. -/
theorem score_constant_unclipped (cfg : Config) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (token : Fin cfg.vocab_size) (first readout : Fin T) (w : Parameters cfg)
    (hclip : constantValueGuard cfg (decode cfg w) eps xeps angle token first)
    (symbol : Fin cfg.vocab_size) :
    score cfg eps xeps angle readout w (fun _ => token) symbol =
      score cfg eps xeps angle first w (fun _ => token) symbol := by
  unfold score
  dsimp only
  rw [hidden_constant_unclipped cfg (decode cfg w) eps xeps angle token first hclip
    cfg.n_layers readout]

/-- C10: clipping zero witnesses a nonvacuous guard for any model, including
nonzero attention/FFN parameters and nonzero embeddings. -/
theorem constantValueGuard_zero (cfg : Config) (p : ModelParams cfg) (eps : ℝ)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (token : Fin cfg.vocab_size) (first : Fin T) :
    constantValueGuard cfg p eps 0 angle token first := by
  intro n hn h
  exact norm_nonneg _

example (B : BlockParams witnessConfig) (x : EucSpace 4) :=
  attention_constant_unclipped witnessConfig B (1 / 8388608) 0
    (T := 3) (fun j k => (j.val * (k.val + 1) : ℝ) / 7) x
    (fun _ => norm_nonneg _) 2

example (B : BlockParams witnessConfig) (x : EucSpace 4) :=
  block_constant_unclipped witnessConfig B (1 / 8388608) 0
    (T := 3) (fun j k => (j.val * (k.val + 1) : ℝ) / 7) x
    (fun _ => norm_nonneg _) 2

example (p : ModelParams witnessConfig) :=
  hidden_constant_unclipped witnessConfig p (1 / 8388608) 0
    (T := 3) (fun n j k => (n.val + j.val * (k.val + 1) : ℝ) / 7) 1 0
    (constantValueGuard_zero _ _ _ _ _ _) 2

example (w : Parameters witnessConfig) :=
  score_constant_unclipped witnessConfig (1 / 8388608) 0
    (T := 3) (fun n j k => (n.val + j.val * (k.val + 1) : ℝ) / 7) 1 0 2 w
    (constantValueGuard_zero _ _ _ _ _ _) 2

end Transformer.SequenceModel.Attention
