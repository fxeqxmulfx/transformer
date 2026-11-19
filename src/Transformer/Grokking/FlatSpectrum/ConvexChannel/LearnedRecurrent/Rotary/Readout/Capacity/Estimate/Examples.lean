module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Counterexample

/-!
# Inhabited estimated-gap hypotheses with actual nonzero error

Source: `review/convex/moments.md`, CM105. All estimated-gap premises are
instantiated at the nonzero complex ordered witness. Its estimated gap is
twelve and its actual coefficient-error correction is eight, rather than zero.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate

open Pair Optimality

example : 0 ≤ estimatedGap 16 (fun _ : Fin 1 => witnessRows) (fun _ => approximateRow) :=
  estimated_nonneg _ _ _ witness_mem

example : |estimatedGap 16 (fun _ : Fin 1 => witnessRows) (fun _ => exactRow) -
    estimatedGap 16 (fun _ : Fin 1 => witnessRows) (fun _ => approximateRow)| ≤
      2 * 16 * peak ((fun _ : Fin 1 => exactRow) - (fun _ => approximateRow)) :=
  gap_error _ _ _ _ witness_mem

example : jointLoss 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
    (fun _ => true) (fun _ => witnessRows) -
      estimatedGap 16 (fun _ : Fin 1 => witnessRows) (fun _ => approximateRow) -
      2 * 16 * peak (jointCoefficient 0 (fun _ : Fin 1 => positionWitness)
        (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) -
          (fun _ => approximateRow)) ≤
    jointLoss 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
      (fun _ => true) (fun _ => imaginaryTarget) :=
  estimated_loss_lower _ _ _ _ _ _ _ _ witness_mem target_mem

example : jointLoss 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
    (fun _ => true) (fun _ => witnessRows) -
      jointLoss 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
        (fun _ => true) (fun _ => imaginaryTarget) ≤
    estimatedGap 16 (fun _ : Fin 1 => witnessRows) (fun _ => approximateRow) +
      2 * 16 * peak (jointCoefficient 0 (fun _ : Fin 1 => positionWitness)
        (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) -
          (fun _ => approximateRow)) :=
  estimated_improvement_le _ _ _ _ _ _ _ _ witness_mem target_mem

example : linearCost (proposal 16 (fun _ : Fin 1 => approximateRow)
    (selected (fun _ : Fin 1 => approximateRow)) - (fun _ => witnessRows))
      (jointCoefficient 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
        (fun _ => true) (fun _ => witnessRows)) ≤
    -estimatedGap 16 (fun _ : Fin 1 => witnessRows) (fun _ => approximateRow) +
      2 * 16 * peak (jointCoefficient 0 (fun _ : Fin 1 => positionWitness)
        (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) -
          (fun _ => approximateRow)) :=
  actual_direction_upper _ _ _ _ _ _ _ witness_mem

example : ∃ v ∈ jointDomain 16, linearCost (v - fun _ : Fin 1 => witnessRows)
    (jointCoefficient 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
      (fun _ => true) (fun _ => witnessRows)) < 0 :=
  negative_if_error_small _ _ _ _ _ _ _ witness_mem witness_error_small

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate
