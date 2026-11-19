module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Guarantees
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Witness

/-!
# Nonzero content change and a strict inexact-cache witness

Source: `review/convex/moments.md`, CM106. Changing the actual learned frequency
changes the ordered state. A separate complete retained-signal example has
strict corrected descent with nonzero signal error, not a supplied exact gradient.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

open Pair Optimality Estimate

/-- Historical content with a different learned frequency, CM106. -/
noncomputable def unrotated : Parameters Bool := ⟨(0, 0), positionContent⟩

/-- The nonzero historical model obeys the same capacity, CM106. -/
theorem unrotated_unit : Unit unrotated := position_unit

/-- Its actual ordered state differs from the quarter-turn state, CM106. -/
theorem unrotated_execution : runFrom unrotated 0 unrotated.content.initial [true, false] =
    (0, 1) := by
  rw [unrotated, run_zero]
  simp [LearnedRecurrent.runFrom, positionContent, Pair.parameters,
    positionTransition, Pair.operator_apply]

/-- Changing actual learned frequency creates a strictly nonzero state
discrepancy, CM106, despite both models respecting the same norm budget. -/
theorem changed_state_nonzero : 0 < ‖runFrom positionWitness 0
    positionWitness.content.initial [true, false] -
      runFrom unrotated 0 unrotated.content.initial [true, false]‖ := by
  have he := position_execution true
  simp only [Bool.not_true, ite_true] at he
  rw [he, unrotated_execution]
  apply norm_pos_iff.mpr
  intro h
  have hi := congrArg (fun z : State => z.2.im) h
  norm_num at hi

/-- A retained signal differs from the actual zero-logit CE residual, CM106. -/
noncomputable def witnessSignal (c : Bool) : ℝ := if c then -(3 / 8) else 3 / 8

/-- Retained complete coefficients execute the actual ordered state, CM106. -/
theorem witness_cached : cached 0 (fun (_ : Fin 1) (_ : Fin 1) => positionWitness)
    (fun _ => [true, false]) (fun _ => witnessSignal) = fun _ => approximateRow := by
  have he := position_execution true
  simp only [Bool.not_true, ite_true] at he
  funext j c
  cases c <;> norm_num [cached, mean, historical, he, witnessSignal,
    approximateRow, imaginaryTarget]

/-- The complete signal/content error is one quarter here; the signal error
is nonzero and the historical/current content happens to agree, CM106. -/
theorem witness_current_error : currentError 0 (fun _ : Fin 1 => positionWitness)
    (fun _ _ => positionWitness) (fun _ => [true, false]) (fun _ => true)
      (fun _ => witnessRows) (fun _ => witnessSignal) = 1 / 4 := by
  norm_num [currentError, errorBudget, contentDrift, state_drift_self,
    Optimality.residual, witness_scores, Geometry.probability, witnessSignal]

/-- The strict corrected premise is inhabited with an actual inexact cache,
CM106; all original classes and complex coordinates remain in the calculation. -/
theorem witness_strict : 2 * 16 * currentError 0 (fun _ : Fin 1 => positionWitness)
    (fun _ _ => positionWitness) (fun _ => [true, false]) (fun _ => true)
      (fun _ => witnessRows) (fun _ => witnessSignal) <
        estimatedGap 16 (fun _ : Fin 1 => witnessRows)
          (cached 0 (fun _ _ => positionWitness) (fun _ => [true, false])
            (fun _ => witnessSignal)) := by
  rw [witness_current_error, witness_cached, witness_estimated_gap]
  norm_num

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained
