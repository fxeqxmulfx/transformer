module

public import Transformer.SequenceModel.Attention.Execution.Tables

/-!
# Finite tensor traces at the source operations

Source: `review/flat_spectrum.md`, *Primitive execution certificates*, C3;
diff_gpt `b98c665b`, `mha.py`, `feed_forward.py`, `block.py`.
Structures contain numerical data only. Accuracy predicates check each
operation against its own recorded inputs, not against a desired final
answer. A backend must supply finite tensors satisfying those predicates.
The SDPA average is an explicit backend-kernel contract, not an assumed
IEEE error theorem for softmax, fused reductions or transcendental calls.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Execution

open GPTMini Landscape.Neural
open scoped BigOperators

/-- Recorded inputs/outputs of the attention branch, preserving the source layout. -/
structure AttentionTrace (cfg : Config) (T : ℕ) where
  pre : Fin T → EucSpace cfg.d_model
  fused : Fin T → EucSpace (3 * cfg.d_model)
  qnorm : Fin cfg.n_heads → Fin T → EucSpace cfg.head_dim
  knorm : Fin cfg.n_heads → Fin T → EucSpace cfg.head_dim
  cosTable : Fin T → Fin (cfg.head_dim / 2) → ℝ
  sinTable : Fin T → Fin (cfg.head_dim / 2) → ℝ
  query : Fin cfg.n_heads → Fin T → EucSpace cfg.head_dim
  key : Fin cfg.n_heads → Fin T → EucSpace cfg.head_dim
  scores : Fin cfg.n_heads → Fin T → Fin T → ℝ
  average : Fin T → Fin cfg.n_heads → EucSpace cfg.head_dim
  direction : Fin T → Fin cfg.n_heads → EucSpace cfg.head_dim
  heads : Fin T → Fin cfg.n_heads → EucSpace cfg.head_dim
  projected : Fin T → EucSpace cfg.d_model
  output : Fin T → EucSpace cfg.d_model

/-- Q/K/V selectors and reshape are exact index operations on the recorded fused tensor. -/
def AttentionTrace.component {cfg : Config} {T : ℕ} (t : AttentionTrace cfg T)
    (part : Fin cfg.d_model → Fin (3 * cfg.d_model)) (i : Fin T) (h : Fin cfg.n_heads) :
    EucSpace cfg.head_dim := headSlice cfg (qkvSlice cfg part (t.fused i)) h

/-- Each norm error is measured at the operation's own inputs. The uniform
delta is a convenient upper bound; a backend may establish smaller site bounds.
Table errors tau are separate from arithmetic applying those tables. -/
def AttentionAccurate (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model)
    (t : AttentionTrace cfg T) (δ τ : ℝ) : Prop :=
  (∀ i, ‖t.pre i - rmsNormEps eps (x i)‖ ≤ δ) ∧
  (∀ i, ‖t.fused i - B.qkv (t.pre i)‖ ≤ δ) ∧
  (∀ h i, ‖t.qnorm h i - rmsNormEps eps (t.component (qkvQ cfg) i h)‖ ≤ δ) ∧
  (∀ h i, ‖t.knorm h i - rmsNormEps eps (t.component (qkvK cfg) i h)‖ ≤ δ) ∧
  (∀ i c, |t.cosTable i c - Real.cos (a i c)| ≤ τ) ∧
  (∀ i c, |t.sinTable i c - Real.sin (a i c)| ≤ τ) ∧
  (∀ h i, ‖t.query h i - tableRotate cfg.head_dim (t.cosTable i) (t.sinTable i) (t.qnorm h i)‖ ≤ δ) ∧
  (∀ h i, ‖t.key h i - tableRotate cfg.head_dim (t.cosTable i) (t.sinTable i) (t.knorm h i)‖ ≤ δ) ∧
  (∀ h i j, |t.scores h i j - inner ℝ (t.query h i) (t.key h j) / Real.sqrt (cfg.head_dim : ℝ)| ≤ δ) ∧
  (∀ i h, ‖t.average i h - ∑ j, causalWeight (t.scores h) i j • t.component (qkvV cfg) j h‖ ≤ δ) ∧
  (∀ i h, ‖t.direction i h - normL2 xeps (t.component (qkvV cfg) i h)‖ ≤ δ) ∧
  (∀ i h, ‖t.heads i h - (if i.val = 0 then 0 else
    t.average i h - inner ℝ (t.average i h) (t.direction i h) • t.direction i h)‖ ≤ δ) ∧
  (∀ i, ‖t.projected i - B.projection (headMerge cfg (t.heads i))‖ ≤ δ) ∧
  (∀ i, ‖t.output i - rmsNormEps eps (t.projected i)‖ ≤ δ)

/-- Recorded SwiGLU branch, keeping activation and product as separate operations. -/
structure FFNTrace (cfg : Config) (T : ℕ) where
  pre : Fin T → EucSpace cfg.d_model
  gate : Fin T → EucSpace cfg.d_ff
  value : Fin T → EucSpace cfg.d_ff
  activated : Fin T → EucSpace cfg.d_ff
  product : Fin T → EucSpace cfg.d_ff
  projected : Fin T → EucSpace cfg.d_model
  output : Fin T → EucSpace cfg.d_model

/-- RMS, three independent learned matrices, SiLU, multiplication and post-RMS
are all checked; no complete-FFN or complete-block error is assumed. -/
def FFNAccurate (cfg : Config) (B : BlockParams cfg) (eps : ℝ) {T : ℕ}
    (x : Fin T → EucSpace cfg.d_model) (t : FFNTrace cfg T) (δ : ℝ) : Prop :=
  (∀ i, ‖t.pre i - rmsNormEps eps (x i)‖ ≤ δ) ∧
  (∀ i, ‖t.gate i - B.gate (t.pre i)‖ ≤ δ) ∧
  (∀ i, ‖t.value i - B.value (t.pre i)‖ ≤ δ) ∧
  (∀ i, ‖t.activated i - WithLp.toLp 2 (fun c => silu (t.gate i c))‖ ≤ δ) ∧
  (∀ i, ‖t.product i - WithLp.toLp 2 (fun c => t.activated i c * t.value i c)‖ ≤ δ) ∧
  (∀ i, ‖t.projected i - B.output (t.product i)‖ ≤ δ) ∧
  (∀ i, ‖t.output i - rmsNormEps eps (t.projected i)‖ ≤ δ)

/-- The two branches are linked by the first residual addition. -/
structure BlockTrace (cfg : Config) (T : ℕ) where
  attn : AttentionTrace cfg T
  mid : Fin T → EucSpace cfg.d_model
  ffn : FFNTrace cfg T
  output : Fin T → EucSpace cfg.d_model

/-- Both residual additions have their own error check, at their actual inputs. -/
def BlockAccurate (cfg : Config) (B : BlockParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin T → Fin (cfg.head_dim / 2) → ℝ) (x : Fin T → EucSpace cfg.d_model)
    (t : BlockTrace cfg T) (δ τ : ℝ) : Prop :=
  AttentionAccurate cfg B eps xeps a x t.attn δ τ ∧
  (∀ i, ‖t.mid i - (x i + t.attn.output i)‖ ≤ δ) ∧
  FFNAccurate cfg B eps t.mid t.ffn δ ∧
  (∀ i, ‖t.output i - (t.mid i + t.ffn.output i)‖ ≤ δ)

/-- Whole finite execution data, including the shared tied embedding table. -/
structure ModelTrace (cfg : Config) (T : ℕ) where
  embedding : Fin cfg.vocab_size → EucSpace cfg.d_model
  initial : Fin T → EucSpace cfg.d_model
  blocks : Fin cfg.n_layers → BlockTrace cfg T
  normalized : Fin T → EucSpace cfg.d_model
  logits : Fin T → Fin cfg.vocab_size → ℝ

/-- Read each recorded residual at its actual depth; depth zero is the input. -/
def ModelTrace.state {cfg : Config} {T : ℕ} (t : ModelTrace cfg T) : ℕ → Fin T → EucSpace cfg.d_model
  | 0 => t.initial
  | n + 1 => if h : n < cfg.n_layers then (t.blocks ⟨n, h⟩).output else t.state n

/-- The complete operation contract, including embedding quantization,
normalized input, all blocks, final RMSNorm and the raw tied readout.
Nonfinite backend values must be excluded before such a real trace exists. -/
def ModelAccurate (cfg : Config) (θ : ModelParams cfg) (eps xeps : ℝ) {T : ℕ}
    (a : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (t : ModelTrace cfg T) (δ τ : ℝ) : Prop :=
  (∀ v, ‖t.embedding v - θ.embedding v‖ ≤ δ) ∧
  (∀ i, ‖t.initial i - rmsNormEps eps (t.embedding (tokens i))‖ ≤ δ) ∧
  (∀ l, BlockAccurate cfg (θ.blocks l) eps xeps (a l) (t.state l.val) (t.blocks l) δ τ) ∧
  (∀ i, ‖t.normalized i - rmsNormEps eps (t.state cfg.n_layers i)‖ ≤ δ) ∧
  (∀ i v, |t.logits i v - inner ℝ (t.normalized i) (t.embedding v)| ≤ δ)

end Transformer.SequenceModel.Attention.Execution
