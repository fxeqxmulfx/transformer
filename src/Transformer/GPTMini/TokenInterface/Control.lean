import Transformer.GPTMini.TokenInterface.Basic
import Transformer.Basis.Depth

/-!
# Concrete actual-model controls for the integer interface

Source: experiments/basis/experiment.py at cbafbe9 (the small 64-wide,
two-layer, four-head GPTMini), and the new checked List Int adapter.
These explicit weights leave both residual blocks inactive and use the
tied embedding readout to reject one depth prefix. A second valid prefix
fails, demonstrating that token-function agreement is a real condition.

This is a satisfiable logit-margin witness, not a construction solving
depth on all prefixes. Head epsilon is immaterial here because the output
projections are zero; the final RMSNorm uses Python's 1e-5 epsilon.
-/

namespace Transformer.GPTMini.TokenInterface

open scoped Classical
noncomputable section

/-- The small Basis depth architecture, including its 128-token OOD context.
Source: experiments/basis/experiment.py and Synthetic.context at cbafbe9. -/
abbrev controlConfig : Config where
  vocab_size := 36
  n_layers := 2
  n_heads := 4
  d_model := 64
  d_ff := 256
  max_seq_len := 128
  rope_theta := 10000
  divides := by decide
  head_even := by decide
  n_heads_pos := by decide
  n_layers_pos := by decide
  d_model_pos := by decide
  d_ff_pos := by decide
  vocab_pos := by decide
  max_seq_len_pos := by decide
  theta_pos := by norm_num

/-- A unit embedding direction; Source: the new constructive tied-readout control. -/
def controlUnit : EucSpace 64 := EuclideanSpace.single 0 1

/-- The depth letter a has unit embedding, REJECT has twice that embedding.
Source: the new explicit weight assignment; every other token has zero embedding. -/
def controlEmbedding (v : Fin controlConfig.vocab_size) : EucSpace controlConfig.d_model :=
  if v.val = 15 then (2 : ℝ) • controlUnit else if v.val = 9 then controlUnit else 0

/-- An actual parameter assignment in the original GPTMini architecture.
Source: the new readout control; no oracle, alternate head, bias or optimizer is introduced. -/
def controlParams : ModelParams controlConfig where
  embedding := controlEmbedding
  blocks := fun _ =>
    { attn := { W_qkv := 0, W_o := 0, log_alpha := fun _ => 0 }
      ffn := { W_in := 0, W_out := 0 } }

/-- The active embedding direction really has unit Euclidean norm.
Source: the first standard basis vector used by the constructive control. -/
theorem controlUnit_norm : ‖controlUnit‖ = 1 := by
  simp [controlUnit, PiLp.norm_single]

/-- All residual layers preserve the input embeddings at these concrete zero projections.
Source: GPTMini.hidden/blockForward; this is a zero-weight control, not a general model claim. -/
theorem hidden_control {T : ℕ} (eps : ℝ) (positions : Fin T → ℝ)
    (tokens : Fin T → Fin controlConfig.vocab_size) (L : ℕ) (i : Fin T) :
    hidden controlConfig controlParams eps positions tokens L i = controlEmbedding (tokens i) := by
  induction L with
  | zero => rfl
  | succ L ih =>
      simp only [controlParams] at ih
      simp only [hidden]
      split
      · simp [controlParams, blockForward, attnSubLayer, ffnSubLayer, relu2FFN, ih]
      · exact ih

/-- The exact final-normalization multiplier on the unit embedding is positive.
Source: GPTMini.rmsNormEps with Python's final-norm epsilon 1e-5. -/
theorem control_scale_pos :
    0 < Real.sqrt (64 : ℝ) / Real.sqrt (1 + 64 * (1 / 100000 : ℝ)) := by
  positivity

/-- The complete forward logits on BOS,a have an explicit finite strict maximum.
Source: the actual stack, final RMSNorm and tied embedding readout under controlParams. -/
theorem lastLogits_control (v : Fin controlConfig.vocab_size) :
    lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩ [⟨9, by decide⟩] v =
      (Real.sqrt (64 : ℝ) / Real.sqrt (1 + 64 * (1 / 100000 : ℝ))) *
        (if v.val = 15 then 2 else if v.val = 9 then 1 else 0) := by
  simp only [lastLogits, forward, unembed]
  rw [hidden_control]
  change inner (𝕜 := ℝ) (rmsNormEps (1 / 100000) (controlEmbedding ⟨9, by decide⟩))
    (controlEmbedding v) = _
  have ha : controlEmbedding ⟨9, by decide⟩ = controlUnit := by
    simp [controlEmbedding]
  rw [ha, rmsNormEps, controlUnit_norm]
  simp only [controlConfig, Nat.cast_ofNat, one_pow, real_inner_smul_left]
  unfold controlEmbedding
  split_ifs
  · rw [real_inner_smul_right, real_inner_self_eq_norm_mul_norm, controlUnit_norm]
    ring
  · rw [real_inner_self_eq_norm_mul_norm, controlUnit_norm]
    ring
  · simp

/-- A strict actual-model margin is attained over all 35 competing vocabulary tokens.
Source: the concrete readout construction; this supplies a genuine model certificate witness. -/
theorem control_depth_margin :
    ∀ v : Fin controlConfig.vocab_size, v ≠ ⟨15, by decide⟩ →
      lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩ [⟨9, by decide⟩] v <
        lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩ [⟨9, by decide⟩]
          ⟨15, by decide⟩ := by
  intro v hv
  have hv15 : v.val ≠ 15 := fun h => hv (Fin.ext h)
  rw [lastLogits_control, lastLogits_control]
  simp only [ite_eq_right hv15, ↓reduceIte]
  split_ifs <;> nlinarith [control_scale_pos]

/-- The real model's integer continuation agrees with depth E_2 on this one valid prefix.
Source: the new concrete margin witness and the raw depth oracle; no global capacity claim is made. -/
theorem control_depth_one_prefix :
    tokenFunction controlConfig controlParams (1 / 100000) [1, 9] =
      Transformer.Basis.depthFunction 2 [1, 9] := by
  have hm := tokenFunction_decode_cons controlConfig controlParams (1 / 100000)
    ⟨1, by decide⟩ [⟨9, by decide⟩] (by decide)
  rw [bestToken_of_strict controlConfig.vocab_pos _ ⟨15, by decide⟩ control_depth_margin] at hm
  have ho : Transformer.Basis.depthFunction 2 [1, 9] = [1, 9, 15] := by decide
  simpa [Transformer.Basis.decodeTokens, ho] using hm

/-- On BOS,a,b the same weights give a zero final stream and tied zero logits.
Source: the actual forward's inactive blocks and the zero embedding of b. -/
theorem lastLogits_control_wrong :
    lastLogits controlConfig controlParams (1 / 100000) ⟨1, by decide⟩
      [⟨9, by decide⟩, ⟨10, by decide⟩] = fun _ => 0 := by
  funext v
  simp only [lastLogits, forward, unembed]
  rw [hidden_control]
  change inner (𝕜 := ℝ) (rmsNormEps (1 / 100000) (controlEmbedding ⟨10, by decide⟩))
    (controlEmbedding v) = 0
  simp [controlEmbedding, rmsNormEps]

/-- One passing example does not make these concrete weights solve the task.
Source: depth E_2 accepts a,b, whereas this actual model greedily returns PAD. -/
theorem control_depth_failure :
    tokenFunction controlConfig controlParams (1 / 100000) [1, 9, 10] ≠
      Transformer.Basis.depthFunction 2 [1, 9, 10] := by
  have hm := tokenFunction_decode_cons controlConfig controlParams (1 / 100000)
    ⟨1, by decide⟩ [⟨9, by decide⟩, ⟨10, by decide⟩] (by decide)
  rw [lastLogits_control_wrong, bestToken_const] at hm
  have hmraw : tokenFunction controlConfig controlParams (1 / 100000) [1, 9, 10] =
      [1, 9, 10, 0] := by
    simpa [Transformer.Basis.decodeTokens] using hm
  have ho : Transformer.Basis.depthFunction 2 [1, 9, 10] = [1, 9, 10, 16] := by decide
  rw [hmraw, ho]
  decide

end
end Transformer.GPTMini.TokenInterface
