module

public import Transformer.SequenceModel.Attention.Backend.Refined.Attention
public import Transformer.SequenceModel.Attention.Backend.Refined.Activations

/-!
# Completing the local source trace bound through SwiGLU

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M4.
The independent gate/value maps, SiLU, product, output map, peri-RMS and
second residual are all retained. The bounds depend on the stored weights.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution Complete Adaptive

/-- M4: FFN input RMS at the recorded first residual. -/
theorem error_ffn_pre (i : Fin 2) : ‖raw_ffn_pre i - sourceFFN.pre i‖ ≤ 31 / 100000 := by
  have h := rms_step (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    (raw_mid i) (sourceMid i) (raw_ffn_pre i) (21 / 10) (147 / 1000000)
    (11 / 100000000) (by norm_num) (gain_ffn_pre i) (error_mid i) (local_ffn_pre i)
  exact h.trans (by norm_num)

/-- M4: the learned gate map keeps its own local residual. -/
theorem error_gate (i : Fin 2) :
    ‖raw_gate i - sourceGate i‖ ≤ 131 / 1000000 := by
  have h := linear_step blockWeights.gate (raw_ffn_pre i) (sourceFFN.pre i) (raw_gate i)
    (21 / 50) (31 / 100000) (8 / 100000000) gate_norm (error_ffn_pre i) (local_gate i)
  exact h.trans (by norm_num)

/-- M4: the independent learned value map has the same verified norm bound. -/
theorem error_value (i : Fin 2) :
    ‖raw_value i - sourceValue i‖ ≤ 131 / 1000000 := by
  have h := linear_step blockWeights.value (raw_ffn_pre i) (sourceFFN.pre i) (raw_value i)
    (21 / 50) (31 / 100000) (8 / 100000000) value_norm (error_ffn_pre i) (local_value i)
  exact h.trans (by norm_num)

/-- M4: exact SiLU propagation plus the polynomially certified captured activation. -/
theorem error_activated (i : Fin 2) :
    ‖raw_activated i - sourceActivated i‖ ≤ 164 / 1000000 := by
  have hl : ‖raw_activated i - WithLp.toLp 2 (fun c => Landscape.Neural.silu (raw_gate i c))‖ ≤
      12 / 100000000 := by fin_cases i; exact local_activated_0; exact local_activated_1
  have h := silu_step (raw_gate i) (sourceGate i) (raw_activated i)
    (1 / 4) (131 / 1000000) (12 / 100000000) (by norm_num)
    (coordinates_gate i) (error_gate i) hl
  exact h.trans (by norm_num)

/-- M4: the product uses recorded coordinate multipliers and retains the cross term. -/
theorem error_product (i : Fin 2) :
    ‖raw_product i - sourceProduct i‖ ≤ 575 / 10000000 := by
  have h := product_step (raw_activated i) (raw_value i) (sourceActivated i)
    (sourceValue i) (raw_product i) (1 / 8) (1 / 4) (164 / 1000000)
    (131 / 1000000) (4 / 1000000000) (by norm_num) (by norm_num)
    (coordinates_activated i) (coordinates_value i) (error_activated i) (error_value i) (local_product i)
  exact h.trans (by norm_num)

/-- M4: the third FFN matrix is the actual stored output projection. -/
theorem error_ffn_projected (i : Fin 2) :
    ‖raw_ffn_projected i - sourceFFN.projected i‖ ≤ 242 / 10000000 := by
  have h := linear_step blockWeights.output (raw_product i) (sourceProduct i) (raw_ffn_projected i)
    (21 / 50) (575 / 10000000) (4 / 10000000000) output_norm (error_product i) (local_ffn_projected i)
  exact h.trans (by norm_num)

/-- M4: FFN post-RMS uses the verified recorded denominators at both positions. -/
theorem error_ffn_output (i : Fin 2) :
    ‖raw_ffn_output i - sourceFFN.output i‖ ≤ 364 / 10000 := by
  have h := rms_step (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    (raw_ffn_projected i) (sourceFFN.projected i) (raw_ffn_output i)
    1500 (242 / 10000000) (21 / 100000000) (by norm_num) (gain_ffn_out i)
    (error_ffn_projected i) (local_ffn_output i)
  exact h.trans (by norm_num)

/-- M4: all branch and residual errors give the actual final hidden-state bound. -/
theorem error_output (i : Fin 2) : ‖raw_output i - sourceOutput i‖ ≤ 366 / 10000 := by
  have h := add_step (raw_mid i) (raw_ffn_output i) (sourceMid i) (sourceFFN.output i) (raw_output i)
    (147 / 1000000) (364 / 10000) (12 / 100000000) (error_mid i) (error_ffn_output i) (local_output i)
  exact h.trans (by norm_num)

/-- M4: the final RMS completes the operation-by-operation propagation. -/
theorem error_normalized (i : Fin 2) :
    ‖raw_normalized i - rmsNormEps (1 / 8388608) (sourceOutput i)‖ ≤ 55 / 1000 := by
  have h := rms_step (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    (raw_output i) (sourceOutput i) (raw_normalized i) (3 / 2) (366 / 10000)
    (11 / 100000000) (by norm_num) (gain_final i) (error_output i) (local_normalized i)
  exact h.trans (by norm_num)

end Transformer.SequenceModel.Attention.Backend.Refined
