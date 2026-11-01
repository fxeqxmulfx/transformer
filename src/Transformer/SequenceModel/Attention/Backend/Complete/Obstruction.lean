module

public import Transformer.SequenceModel.Attention.Backend.Complete.Certificate
public import Transformer.SequenceModel.Attention.Backend.Complete.BudgetFloor
public import Transformer.SequenceModel.Attention.Spectral.Numerics

/-!
# An unavoidable failure of the current global margin certificate

Source: `review/flat_spectrum.md`, *Complete captured execution*, L4--L5.
One actual residual addition has error 2^-24. Every valid uniform allowance
therefore makes the existing global class radii exceed the largest possible
source margin. This refutes applicability of that sufficient bound to this
trace; it does not refute the model, the source decision or the error theorem.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend.Complete

open GPTMini Stability Execution

/-- L4: exact nonzero rounding error in a captured residual-add coordinate. -/
theorem residual_error_exact :
    |raw_output 1 2 - (raw_mid 1 2 + raw_ffn_output 1 2)| = 1 / 16777216 := by
  norm_num [raw_output, raw_mid, raw_ffn_output]

/-- L4: every accurate uniform certificate for this same data must allow at least 2^-24. -/
theorem allowance_floor (δ τ : ℝ)
    (ht : ModelAccurate cfg weights (1 / 8388608) (1 / 1000000000000)
      angles tokens captured δ τ) : (1 / 16777216 : ℝ) ≤ δ := by
  have h := (ht.2.2.1 0).2.2.2 1
  change ‖raw_output 1 - (raw_mid 1 + raw_ffn_output 1)‖ ≤ δ at h
  have hc := PiLp.norm_apply_le (raw_output 1 - (raw_mid 1 + raw_ffn_output 1)) (2 : Fin 4)
  change |raw_output 1 2 - (raw_mid 1 2 + raw_ffn_output 1 2)| ≤
    ‖raw_output 1 - (raw_mid 1 + raw_ffn_output 1)‖ at hc
  rw [residual_error_exact] at hc
  exact hc.trans h

/-- L4: even an optimally tightened uniform allowance cannot satisfy the old margin test.
The conclusion holds at every residual vector and every target class. -/
theorem no_global_margin (δ τ : ℝ) (hτ : 0 ≤ τ)
    (ht : ModelAccurate cfg weights (1 / 8388608) (1 / 1000000000000)
      angles tokens captured δ τ) (x : EucSpace 4) (target : Fin cfg.vocab_size) :
    ¬ MarginAbove cfg weights (1 / 8388608) x
      (modelBudget cfg weights (1 / 8388608) (1 / 1000000000000) δ τ cfg.n_layers) δ target := by
  intro hm
  let v : Fin cfg.vocab_size := if target = 0 then 1 else 0
  have hv : v ≠ target := by fin_cases target <;> norm_num [v]
  have hδ := allowance_floor δ τ ht
  have h1 := radius_gt_two δ τ hδ hτ target
  have h2 := radius_gt_two δ τ hδ hτ v
  have h3 := source_margin_le_four x target v
  have h4 := hm v hv
  linarith

/-- L5: the current spectral/counting bridge's ExecutionBelow premise is impossible
for this trace at any attainable source margin, regardless of counting precision. -/
theorem no_execution_below (δ τ γ : ℝ) (hτ : 0 ≤ τ) (hγ : γ ≤ 4)
    (ht : ModelAccurate cfg weights (1 / 8388608) (1 / 1000000000000)
      angles tokens captured δ τ) :
    ¬ Spectral.ExecutionBelow cfg weights (1 / 8388608) (1 / 1000000000000) δ τ γ := by
  intro h
  have hδ := allowance_floor δ τ ht
  have h0 := radius_gt_two δ τ hδ hτ 0
  have h1 := radius_gt_two δ τ hδ hτ 1
  have h2 := h 0 1
  linarith

example := allowance_floor (1 / 10000) (2 / 1000000) captured_accurate
example := no_global_margin (1 / 10000) (2 / 1000000) (by norm_num)
  captured_accurate (raw_output 1) 2
example := no_execution_below (1 / 10000) (2 / 1000000) 4
  (by norm_num) (by norm_num) captured_accurate

end Transformer.SequenceModel.Attention.Backend.Complete
