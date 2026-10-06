import Transformer.GPTMini.Semantics.CountEmbedding

/-!
# Preserve a semantic phase feature through every original residual layer

Source: Block.forward/GPTMini.forward at f11b6e2 and Parity.solve at
cbafbe9. A linear feature survives a residual block when both output
projections have image in its kernel. These are local matrix equations;
no correct task prediction or already preserved hidden feature is assumed.
Induction proves preservation through the entire actual stack, including
RMSNorm, attention, XSA and ReLU². Distinct input-token feature values
then force distinct hidden states, even across different context lengths.

The explicit parity control uses an independent phase coordinate:
SEP has feature +1 and EVEN/ODD have feature -1. Attention may write
nonzero updates into the ONE coordinate, but cannot overwrite the phase.
It separates the two legal equal-count prefixes from Counting. The
final parity/EOS readout is a separate obligation; retained phase does
not imply that the tied unembedding has already decoded the correct label.
-/

namespace Transformer.GPTMini.Semantics

/-- Local kernel equations preserve a semantic probe through the complete residual block.
Source: the original two residual additions at f11b6e2, with unconstrained internal Q/K/values and FFN input. -/
theorem block_probe_preserved (cfg : Config) (params : BlockParams cfg) (eps : ℝ)
    (probe : EucSpace cfg.d_model →L[ℝ] ℝ)
    (hattn : probe.comp params.attn.W_o = 0) (hffn : probe.comp params.ffn.W_out = 0)
    {T : ℕ} (positions : Fin T → ℝ) (x : Fin T → EucSpace cfg.d_model) (i : Fin T) :
    probe (blockForward cfg params eps positions x i) = probe (x i) := by
  have ha (z : EucSpace cfg.d_model) : probe (params.attn.W_o z) = 0 := by
    have h := congrArg (fun f : EucSpace cfg.d_model →L[ℝ] ℝ => f z) hattn
    simpa only [ContinuousLinearMap.comp_apply, zero_apply] using h
  have hf (z : EucSpace cfg.d_ff) : probe (params.ffn.W_out z) = 0 := by
    have h := congrArg (fun f : EucSpace cfg.d_ff →L[ℝ] ℝ => f z) hffn
    simpa only [ContinuousLinearMap.comp_apply, zero_apply] using h
  unfold blockForward
  rw [probe.map_add, probe.map_add]
  simp only [attnSubLayer, ffnSubLayer, relu2FFN, ha, hf, add_zero]

example (probe : EucSpace countConfig.d_model →L[ℝ] ℝ) :
    probe.comp ({ W_qkv := countQKV, W_o := 0, log_alpha := fun _ => 0 } :
      AttnParams countConfig).W_o = 0 ∧
      probe.comp ({ W_in := 0, W_out := 0 } : FFNParams countConfig).W_out = 0 := by
  simp

/-- The matrix constraints propagate the input embedding's semantic feature through all actual layers.
Source: GPTMini.hidden's original block loop at f11b6e2; the result is universal over the other weights. -/
theorem hidden_probe_preserved (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (probe : EucSpace cfg.d_model →L[ℝ] ℝ)
    (hattn : ∀ b, probe.comp (params.blocks b).attn.W_o = 0)
    (hffn : ∀ b, probe.comp (params.blocks b).ffn.W_out = 0)
    {T : ℕ} (positions : Fin T → ℝ) (tokens : Fin T → Fin cfg.vocab_size) (L : ℕ) (i : Fin T) :
    probe (hidden cfg params eps positions tokens L i) = probe (params.embedding (tokens i)) := by
  induction L with
  | zero => rfl
  | succ L ih =>
      rw [hidden]
      split
      · rw [block_probe_preserved cfg _ eps probe (hattn _) (hffn _)]
        exact ih
      · exact ih

example (probe : EucSpace TokenInterface.controlConfig.d_model →L[ℝ] ℝ) :
    (∀ b, probe.comp (TokenInterface.controlParams.blocks b).attn.W_o = 0) ∧
      (∀ b, probe.comp (TokenInterface.controlParams.blocks b).ffn.W_out = 0) := by
  simp [TokenInterface.controlParams]

/-- A retained semantic embedding distinction cannot collapse anywhere in the full stack.
Source: the original residual architecture at f11b6e2; the premise concerns only embeddings and output matrices. -/
theorem hidden_probe_distinct (cfg : Config) (params : ModelParams cfg) (eps : ℝ)
    (probe : EucSpace cfg.d_model →L[ℝ] ℝ)
    (hattn : ∀ b, probe.comp (params.blocks b).attn.W_o = 0)
    (hffn : ∀ b, probe.comp (params.blocks b).ffn.W_out = 0)
    {T S : ℕ} (p : Fin T → ℝ) (r : Fin S → ℝ)
    (left : Fin T → Fin cfg.vocab_size) (right : Fin S → Fin cfg.vocab_size)
    (L : ℕ) (i : Fin T) (j : Fin S)
    (hdifferent : probe (params.embedding (left i)) ≠ probe (params.embedding (right j))) :
    hidden cfg params eps p left L i ≠ hidden cfg params eps r right L j := by
  intro heq
  apply hdifferent
  have h := congrArg probe heq
  rw [hidden_probe_preserved cfg params eps probe hattn hffn,
    hidden_probe_preserved cfg params eps probe hattn hffn] at h
  exact h

/-- An independent phase coordinate; Source: the new semantic control for parity's answer/EOS positions. -/
noncomputable def phaseUnit : EucSpace 64 := EuclideanSpace.single 1 1

/-- The phase probe reads that coordinate without using a desired parity answer.
Source: the original residual stream's ordinary Euclidean coordinate representation. -/
noncomputable def phaseProbe : EucSpace 64 →L[ℝ] ℝ := innerSL ℝ phaseUnit

/-- ONE has its own coordinate, while SEP and either supplied label retain different phase codes.
Source: vocabulary.py at cbafbe9, with a new explicit embedding assignment. -/
noncomputable def phaseEmbedding (token : Fin countConfig.vocab_size) : EucSpace 64 :=
  if token.val = 22 then TokenInterface.controlUnit else
    if token.val = 18 then phaseUnit else if token.val = 24 ∨ token.val = 25 then -phaseUnit else 0

/-- A nonzero attention output projection writes into the ONE coordinate while protecting phase.
Source: the unchanged original W_o operator type at f11b6e2. -/
noncomputable def phaseOutput : EucSpace 64 →L[ℝ] EucSpace 64 :=
  (innerSL ℝ TokenInterface.controlUnit).smulRight TokenInterface.controlUnit

/-- Concrete original-model parameters with a protected phase and an active ONE channel.
Source: the small Basis parity shape; these weights preserve phase, not a claimed full parity decoder. -/
noncomputable def phaseParams : ModelParams countConfig where
  embedding := phaseEmbedding
  blocks := fun _ =>
    { attn := { W_qkv := countQKV, W_o := phaseOutput, log_alpha := fun _ => 0 }
      ffn := { W_in := 0, W_out := 0 } }

/-- The concrete active attention updates lie in the phase probe's kernel at every layer.
Source: the independent first and second embedding coordinates, with the actual output matrix. -/
theorem phase_output_kernel :
    (∀ b, phaseProbe.comp (phaseParams.blocks b).attn.W_o = 0) ∧
      (∀ b, phaseProbe.comp (phaseParams.blocks b).ffn.W_out = 0) := by
  constructor
  · intro b
    ext x
    simp [phaseParams, phaseProbe, phaseOutput, ContinuousLinearMap.comp_apply,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
      phaseUnit, TokenInterface.controlUnit, EuclideanSpace.inner_single_left]
  · intro b
    simp [phaseParams]

/-- The explicit embeddings distinguish SEP from the correct supplied EVEN label by +1 versus -1.
Source: the actual raw phase pair, independently of logits or trained correctness. -/
theorem phase_embedding_separated :
    phaseProbe (phaseParams.embedding ⟨18, by decide⟩) ≠
      phaseProbe (phaseParams.embedding ⟨24, by decide⟩) := by
  have hu : ‖phaseUnit‖ = 1 := by simp [phaseUnit, PiLp.norm_single]
  simp [phaseParams, phaseEmbedding, phaseProbe, innerSL_apply_apply, hu]
  norm_num

example : (∀ b, phaseProbe.comp (phaseParams.blocks b).attn.W_o = 0) ∧
    (∀ b, phaseProbe.comp (phaseParams.blocks b).ffn.W_out = 0) ∧
    phaseProbe (phaseParams.embedding ⟨18, by decide⟩) ≠
      phaseProbe (phaseParams.embedding ⟨24, by decide⟩) :=
  ⟨phase_output_kernel.1, phase_output_kernel.2, phase_embedding_separated⟩

/-- Every layer of this actual model distinguishes two legal equal-count prefixes requiring EVEN and EOS.
Source: Counting.parity_phase_count_collision; the residual phase feature prevents its collapse. -/
theorem phase_prefix_hidden_distinct (eps : ℝ) (L : ℕ) :
    hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
          ⟨21, by decide⟩, ⟨18, by decide⟩] : List (Fin countConfig.vocab_size)).get L ⟨5, by decide⟩ ≠
      hidden countConfig phaseParams eps (fun j => (j.val : ℝ))
        ([⟨1, by decide⟩, ⟨22, by decide⟩, ⟨22, by decide⟩, ⟨21, by decide⟩,
          ⟨18, by decide⟩, ⟨24, by decide⟩] : List (Fin countConfig.vocab_size)).get L ⟨5, by decide⟩ := by
  exact hidden_probe_distinct countConfig phaseParams eps phaseProbe
    phase_output_kernel.1 phase_output_kernel.2 _ _ _ _ L _ _ phase_embedding_separated

end Transformer.GPTMini.Semantics
