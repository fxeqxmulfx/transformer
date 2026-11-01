module

public import Transformer.SequenceModel.Attention.Backend.Refined.Bounds

/-!
# Exact source intermediates for the captured invocation

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M4;
diff_gpt b98c665b, gpt.py and block.py. These are exact-real operations with
the stored checkpoint and exact geometric phases, not floating references.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Execution Complete

/-- The actual normalized tied embedding at the captured tokens. -/
def sourceInitial (i : Fin 2) : EucSpace 4 :=
  rmsNormEps (1 / 8388608) (weights.embedding (tokens i))

/-- Exact source attention, including pre/post RMS, SDPA, XSA and RoPE. -/
def sourceAttention : AttentionTrace cfg 2 :=
  exactAttention cfg blockWeights (1 / 8388608) (1 / 1000000000000) (angles 0) sourceInitial

/-- Exact width-four source Q slice after fused projection and reshape. -/
def sourceQ (i : Fin 2) : EucSpace 4 := sourceAttention.component (qkvQ cfg) i 0

/-- Exact width-four source K slice after fused projection and reshape. -/
def sourceK (i : Fin 2) : EucSpace 4 := sourceAttention.component (qkvK cfg) i 0

/-- Exact width-four source V slice after fused projection and reshape. -/
def sourceV (i : Fin 2) : EucSpace 4 := sourceAttention.component (qkvV cfg) i 0

/-- Exact width-four source qnorm operation. -/
def sourceQNorm (i : Fin 2) : EucSpace 4 := sourceAttention.qnorm 0 i

/-- Exact width-four source knorm operation. -/
def sourceKNorm (i : Fin 2) : EucSpace 4 := sourceAttention.knorm 0 i

/-- Exact width-four source query operation. -/
def sourceQuery (i : Fin 2) : EucSpace 4 := sourceAttention.query 0 i

/-- Exact width-four source key operation. -/
def sourceKey (i : Fin 2) : EucSpace 4 := sourceAttention.key 0 i

/-- Exact width-four source average operation. -/
def sourceAverage (i : Fin 2) : EucSpace 4 := sourceAttention.average i 0

/-- Exact width-four source direction operation. -/
def sourceDirection (i : Fin 2) : EucSpace 4 := sourceAttention.direction i 0

/-- Exact width-four source heads operation. -/
def sourceHeads (i : Fin 2) : EucSpace 4 := sourceAttention.heads i 0

/-- First source residual addition. -/
def sourceMid (i : Fin 2) : EucSpace 4 := sourceInitial i + sourceAttention.output i

/-- Exact source SwiGLU branch, with both normalizers. -/
def sourceFFN : FFNTrace cfg 2 := exactFFN cfg blockWeights (1 / 8388608) sourceMid

/-- Exact width-eleven source FFN gate operation. -/
def sourceGate (i : Fin 2) : EucSpace 11 := sourceFFN.gate i

/-- Exact width-eleven source FFN value operation. -/
def sourceValue (i : Fin 2) : EucSpace 11 := sourceFFN.value i

/-- Exact width-eleven source FFN activated operation. -/
def sourceActivated (i : Fin 2) : EucSpace 11 := sourceFFN.activated i

/-- Exact width-eleven source FFN product operation. -/
def sourceProduct (i : Fin 2) : EucSpace 11 := sourceFFN.product i

/-- Second source residual addition. -/
def sourceOutput (i : Fin 2) : EucSpace 4 := sourceMid i + sourceFFN.output i

/-- M4: the exposed operation graph is exactly the source hidden state. -/
theorem source_hidden (i : Fin 2) : sourceOutput i =
    hidden cfg weights (1 / 8388608) (1 / 1000000000000) angles tokens cfg.n_layers i := rfl

/-- The source raw tied readout after final RMS. -/
def sourceLogits (i : Fin 2) (v : Fin 3) : ℝ :=
  inner ℝ (rmsNormEps (1 / 8388608) (sourceOutput i)) (weights.embedding v)

end Transformer.SequenceModel.Attention.Backend.Refined
