module

public import Transformer.SequenceModel.Attention.Backend.Complete.NormInputs
public import Transformer.SequenceModel.Attention.Backend.Complete.NormOutputs
public import Transformer.SequenceModel.Attention.Backend.Complete.Linear
public import Transformer.SequenceModel.Attention.Backend.Complete.Arithmetic
public import Transformer.SequenceModel.Attention.Backend.Complete.Scalars
public import Transformer.SequenceModel.Attention.Backend.Complete.Activations
public import Transformer.SequenceModel.Attention.Backend.Complete.Softmax
public import Transformer.SequenceModel.Attention.Execution.Readout

/-!
# A complete source execution with every local premise discharged

Source: `review/flat_spectrum.md`, *Complete captured execution*, L3--L4.
The captured invocation uses the real source GPT.forward, one active block,
two tokens, exact source epsilons and stored geometric RoPE. The resulting
ModelAccurate proof is assembled from independently checked primitive sites.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend.Complete

open GPTMini Stability Execution

/-- L3: every attention operation of the captured invocation is certified. -/
theorem attention_accurate : AttentionAccurate cfg blockWeights (1 / 8388608)
    (1 / 1000000000000) (angles 0) raw_initial attnTrace (1 / 10000) (2 / 1000000) := by
  refine ⟨checked_pre, checked_fused, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    checked_projected, checked_attn_output⟩
  · intro h i; fin_cases h; exact checked_qnorm i
  · intro h i; fin_cases h; exact checked_knorm i
  · intro i c; exact (checked_tables i c).1
  · intro i c; exact (checked_tables i c).2
  · intro h i; fin_cases h; exact checked_query i
  · intro h i; fin_cases h; exact checked_key i
  · intro h i j; fin_cases h; exact checked_scores i j
  · intro i h; fin_cases h; fin_cases i
    · exact checked_average_0
    · exact checked_average_1
  · intro i h; fin_cases h; exact checked_direction i
  · intro i h; fin_cases h; exact checked_heads i

/-- L3: the width-eleven SwiGLU branch has all seven operation checks. -/
theorem ffn_accurate : FFNAccurate cfg blockWeights (1 / 8388608) raw_mid ffnTrace
    (1 / 10000) := by
  refine ⟨checked_ffn_pre, checked_gate, checked_value, ?_, checked_product,
    checked_ffn_projected, checked_ffn_output⟩
  intro i
  fin_cases i
  · exact checked_activated_0
  · exact checked_activated_1

/-- L3: both branch certificates are linked by the actual recorded residual sums. -/
theorem block_accurate : BlockAccurate cfg blockWeights (1 / 8388608)
    (1 / 1000000000000) (angles 0) raw_initial blockTrace (1 / 10000) (2 / 1000000) :=
  ⟨attention_accurate, checked_mid, ffn_accurate, checked_output⟩

/-- L3: a complete captured compiled source ModelAccurate certificate.
No operation-error premise remains: every field was proved from saved binary values. -/
theorem captured_accurate : ModelAccurate cfg weights (1 / 8388608)
    (1 / 1000000000000) angles tokens captured (1 / 10000) (2 / 1000000) := by
  refine ⟨?_, checked_initial, ?_, ?_, checked_logits⟩
  · intro v; norm_num [captured]
  · intro l; fin_cases l; exact block_accurate
  · intro i
    simpa [captured, ModelTrace.state, blockTrace, cfg] using checked_normalized i

/-- L4: the existing composition theorem now applies to actual saved CUDA logits. -/
theorem captured_logit_error (i : Fin 2) (v : Fin cfg.vocab_size) :
    |captured.logits i v - inner ℝ
      (rmsNormEps (1 / 8388608)
        (hidden cfg weights (1 / 8388608) (1 / 1000000000000) angles tokens cfg.n_layers i))
      (weights.embedding v)| ≤
        logitRadius cfg (1 / 8388608) (weights.embedding v)
          (modelBudget cfg weights (1 / 8388608) (1 / 1000000000000)
            (1 / 10000) (2 / 1000000) cfg.n_layers) (1 / 10000) :=
  logit_error cfg weights (by norm_num) (by norm_num) angles tokens captured
    (1 / 10000) (2 / 1000000) (by norm_num) (by norm_num) captured_accurate i v

/-- L4: the captured two-token classifier has two different strict winners. -/
theorem captured_decisions : captured.predict 0 0 = true ∧ captured.predict 1 2 = true := by
  constructor <;> apply decide_eq_true <;> intro v hv <;>
    fin_cases v <;> norm_num [captured, raw_logits] at *
  exact hv rfl

/-- L4: both branches actively change the second residual; this is not an identity stack. -/
theorem active_branches : raw_attn_output 1 ≠ 0 ∧ raw_ffn_output 1 ≠ 0 := by
  constructor
  · intro h
    have hc := congrArg (fun x : EucSpace 4 => x 0) h
    norm_num [raw_attn_output] at hc
  · intro h
    have hc := congrArg (fun x : EucSpace 4 => x 0) h
    norm_num [raw_ffn_output] at hc

end Transformer.SequenceModel.Attention.Backend.Complete
