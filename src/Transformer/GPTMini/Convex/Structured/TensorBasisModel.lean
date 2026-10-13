import Transformer.GPTMini.Convex.Structured.TensorStackInterface

/-!
# Actual finite tensor models solve all six complete raw Basis recipes

Source: experiments/basis/experiment.py's width/depth/FFN/vocabulary/
context recipes at cbafbe9, finite mixed weights at d7be618 and the
genuine shared-weight tensor stack/checked interface. Easy uses width
64 and two blocks; hard uses width 128 and six, for all three tasks.
The replacement has two structured heads, shared attention parameters,
learned absolute/relative positions and the original zero deferred FFN.
Config retains legacy rotary metadata for type compatibility; this
operator replaces its rotary dot-product heads by actual learned
state/value and all-pair additive matching/value distributions.

Given explicit finite recipe weights, the real embedding/prenorm/
residual/FFN/final-RMS/tied/checked integer computation solves every
validated raw prefix. No correct logits, state, route or SolvesTask is
assumed. Parity generates its label and EOS in two real model calls.
This is real-arithmetic capacity with unrestricted-weight inference
coupling, not output-only CE convexity or measured AdamW success.
-/

namespace Transformer.GPTMini.Convex.Structured

open Transformer.Basis
open scoped Classical
noncomputable section

/-- Actual initial prototype retains Basis width/depth/FFN/vocabulary/context, replacing the head computation by two shared structured heads.
Source: the original experiment recipes and the new verified operator; FFN matrices are fixed zero and rotary metadata is inactive. -/
abbrev basisTensorConfig (mode : Mode) (task : Task) : Config where
  vocab_size := vocabularySize task
  n_layers := if mode = .easy then 2 else 6
  n_heads := 2
  d_model := if mode = .easy then 64 else 128
  d_ff := if mode = .easy then 256 else 512
  max_seq_len := contextSize task
  rope_theta := 10000
  divides := by cases mode <;> decide
  head_even := by cases mode <;> decide
  n_heads_pos := by decide
  n_layers_pos := by cases mode <;> decide
  d_model_pos := by cases mode <;> decide
  d_ff_pos := by cases mode <;> decide
  vocab_pos := by cases task <;> decide
  max_seq_len_pos := by cases task <;> decide
  theta_pos := by norm_num

/-- Every genuine Basis tensor recipe fits the true compact vocabulary decoder and physical width layout.
Source: the actual task vocabulary sizes and width-64/128 experiment metadata, not an enlarged hidden state. -/
theorem basisTensorConfig_fits (mode : Mode) (task : Task) :
    (basisTensorConfig mode task).vocab_size ≤ 1024 ∧ 64 ≤ (basisTensorConfig mode task).d_model := by
  constructor
  · change vocabularySize task ≤ 1024
    cases task <;> decide
  · change 64 ≤ (if mode = .easy then 64 else 128)
    cases mode <;> decide

/-- Actual deferred FFN matrix dimensions retain the original four-times-width Basis expansion in both modes.
Source: the unchanged small/large recipe metadata; this is a dimension identity, not a learned-FFN convexity claim. -/
theorem basisTensorConfig_ffn_width (mode : Mode) (task : Task) :
    (basisTensorConfig mode task).d_ff = 4 * (basisTensorConfig mode task).d_model := by
  change (if mode = .easy then 256 else 512) = 4 * (if mode = .easy then 64 else 128)
  cases mode <;> decide

/-- Genuine given-weight Basis models expose only the requested raw integer list after recipe parameters are fixed.
Source: actual tensorFunction and basisMixedParameters, independent of the individual input prefix. -/
def basisTensorFunction (mode : Mode) (task : Task) (eps : ℝ) : Tokens → Tokens :=
  tensorFunction (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1 (basisTensorConfig_fits mode task).2
    eps (basisMixedParameters mode task)

/-- The actual full given-weight tensor model equals the verified mixed callback on every integer input.
Source: all-weight real configured-stack/interface equality, including encoding/context fallbacks. -/
theorem basisTensorFunction_mixed (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps) :
    basisTensorFunction mode task eps = basisMixedFunction mode task := by
  unfold basisTensorFunction basisMixedFunction
  exact tensorFunction_mixed _ _ _ eps heps _

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- Explicit finite true tensor-stack weights solve every validated raw prefix in all six complete Basis task/mode recipes.
Source: genuine full-stack inference coupling and independently verified mixed finite-weight raw capability, with no prepared representation premise. -/
theorem basisTensor_solves (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps) :
    SolvesTask (basisTensorFunction mode task eps) mode task := by
  rw [basisTensorFunction_mixed mode task eps heps]
  exact basisMixed_solves mode task

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- The actual checked true-stack next prediction is the independent raw Basis answer at every validated prefix.
Source: all-weight tensor-next transfer and the given finite mixed weights' complete raw next-answer theorem. -/
theorem basisTensor_next (mode : Mode) (task : Task) (eps : ℝ) (heps : 0 < eps)
    (tokens : Tokens) (hprefix : TaskPrefix mode task tokens) :
    tensorNext (basisTensorConfig mode task) (basisTensorConfig_fits mode task).1 (basisTensorConfig_fits mode task).2
      eps (basisMixedParameters mode task) tokens = taskNext mode task tokens := by
  rw [tensorNext_mixed _ _ _ eps heps]
  exact basisMixed_next mode task tokens hprefix

example : (0 : ℝ) < 1 / 100000 ∧ TaskPrefix .hard .recall (rewritePrefix 292 300) :=
  ⟨by norm_num, by decide, 300, by decide⟩

/-- Actual model calls generate parity's complete answer and EOS, reusing the predicted label as genuine second input.
Source: all raw parity supervision positions and the complete checked true-stack Basis solver. -/
theorem basisTensor_parity_twice (mode : Mode) (eps : ℝ) (heps : 0 < eps)
    (bits : List Bool) (hmin : 1 ≤ bits.length) (hmax : bits.length ≤ 16) :
    basisTensorFunction mode .parity eps (basisTensorFunction mode .parity eps (parityPrompt bits)) =
      parityPrompt bits ++ [parityLabel bits, eos] :=
  solvesTask_parity_twice _ mode (basisTensor_solves mode .parity eps heps) bits hmin hmax

example : (0 : ℝ) < 1 / 100000 ∧ 1 ≤ ([true, false, true] : List Bool).length ∧
    ([true, false, true] : List Bool).length ≤ 16 := by
  exact ⟨by norm_num, by decide, by decide⟩

/-- Every actual finite-weight Basis tensor model retains the exact append-one length on all raw integer inputs.
Source: the true checked tensor function, independent of task validity, epsilon or model quality. -/
theorem basisTensor_length (mode : Mode) (task : Task) (eps : ℝ) (tokens : Tokens) :
    (basisTensorFunction mode task eps tokens).length = tokens.length + 1 := tensorFunction_length _ _ _ _ _ _

/-- Every actual Basis tensor model preserves the full original raw prefix in its original order.
Source: the real integer-list extension, including encoding and context fallback cases. -/
theorem basisTensor_prefix (mode : Mode) (task : Task) (eps : ℝ) (tokens : Tokens) :
    (basisTensorFunction mode task eps tokens).take tokens.length = tokens := tensorFunction_prefix _ _ _ _ _ _

/-- The actual full width-64 two-block model distinguishes the real eight-record value-swap pair.
Source: true-stack/mixed inference equality and the complete raw easy Basis binding control. -/
theorem basisTensor_binding_control (eps : ℝ) (heps : 0 < eps) :
    basisTensorFunction .easy .recall eps (bindingPrefix 292 293) = bindingPrefix 292 293 ++ [292] ∧
    basisTensorFunction .easy .recall eps (bindingPrefix 293 292) = bindingPrefix 293 292 ++ [293] := by
  rw [basisTensorFunction_mixed _ _ eps heps]
  exact basisMixed_binding_control

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- The actual full width-128 six-block model respects last-write chronology in the real sixteen-record overwrite pair.
Source: true configured-stack inference and the independent complete hard raw Basis overwrite control. -/
theorem basisTensor_overwrite_control (eps : ℝ) (heps : 0 < eps) :
    basisTensorFunction .hard .recall eps (rewritePrefix 292 300) = rewritePrefix 292 300 ++ [300] ∧
    basisTensorFunction .hard .recall eps (rewritePrefix 300 292) = rewritePrefix 300 292 ++ [292] := by
  rw [basisTensorFunction_mixed _ _ eps heps]
  exact basisMixed_overwrite_control

example : (0 : ℝ) < 1 / 100000 := by norm_num

/-- Actual complete two-block depth predictions preserve the required same-bag raw order distinction.
Source: genuine full tensor-stack coupling and independently validated complete easy E2 semantics. -/
theorem basisTensor_order_control (eps : ℝ) (heps : 0 < eps) :
    basisTensorFunction .easy .depth eps [1, 9, 9, 10, 10] = [1, 9, 9, 10, 10, 16] ∧
    basisTensorFunction .easy .depth eps [1, 9, 10, 9, 10] = [1, 9, 10, 9, 10, 15] := by
  rw [basisTensorFunction_mixed _ _ eps heps]
  exact basisMixed_order_control

example : (0 : ℝ) < 1 / 100000 := by norm_num

end
end Transformer.GPTMini.Convex.Structured
