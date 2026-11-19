module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Gap

/-!
# A strict actual descent condition with nonzero coefficient error

Source: `review/convex/moments.md`, CM105. The actual ordered CM98 execution and
CM103 imaginary readouts give A, while H=3A/4 has nonzero discrepancy. The
estimated gap twelve exceeds correction eight at the original budget sixteen.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate

open Pair Optimality

/-- Imaginary target contrasts for the explicit CM105 proof witness. -/
noncomputable def imaginaryTarget (c : Bool) : State := (0, if c then Complex.I else -Complex.I)

/-- Actual coefficient rows of the displayed zero-logit single record, CM105. -/
noncomputable def exactRow (c : Bool) : State := (1 / 2 : ℝ) • imaginaryTarget c

/-- A complete inexact estimate, three quarters of the actual coefficient, CM105. -/
noncomputable def approximateRow (c : Bool) : State := (3 / 8 : ℝ) • imaginaryTarget c

/-- The original nonzero full imaginary current rows obey shared capacity, CM105. -/
theorem witness_mem : (fun _ : Fin 1 => witnessRows) ∈ jointDomain 16 := by
  have hb := block_le witnessRows 1 (witness_rows 0)
  change size _ ≤ 16
  simpa only [size, Fin.sum_univ_one] using hb.trans (by norm_num : (1 : ℝ) ≤ 16)

/-- The nonzero imaginary competitor also obeys the complete shared capacity, CM105. -/
theorem target_mem : (fun _ : Fin 1 => imaginaryTarget) ∈ jointDomain 16 := by
  have hb : blockSize imaginaryTarget ≤ 1 := block_le _ _ (by
    intro c
    cases c <;> simp [imaginaryTarget, rowSize])
  change size _ ≤ 16
  simpa only [size, Fin.sum_univ_one] using hb.trans (by norm_num : (1 : ℝ) ≤ 16)

/-- Actual zero logits at the full imaginary current table, CM105. -/
theorem witness_scores (c : Bool) : jointScore (fun _ : Fin 1 => positionWitness)
    (fun _ => witnessRows) [true, false] c = 0 := by
  have he := position_execution true
  simp only [Bool.not_true, ite_true] at he
  rw [jointScore, scores_pairing]
  cases c <;> simp [he, pairing, witnessRows, Pair.row_apply]

/-- The displayed A is derived from actual executed state and full-class CE
residuals, CM105; it is not a supplied gradient premise. -/
theorem witness_coefficient : jointCoefficient 0 (fun _ : Fin 1 => positionWitness)
    (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) = fun _ => exactRow := by
  have he := position_execution true
  simp only [Bool.not_true, ite_true] at he
  funext j c
  rw [joint_coefficient_apply]
  cases c <;> norm_num [Optimality.residual, witness_scores, Geometry.probability, he,
    exactRow, imaginaryTarget]

/-- The complete estimated gap is twelve despite nonzero coefficient error, CM105. -/
theorem witness_estimated_gap : estimatedGap 16 (fun _ : Fin 1 => witnessRows)
    (fun _ => approximateRow) = 12 := by
  norm_num [estimatedGap, linearCost, peak, blockSupport, pairing, Pair.row_apply,
    witnessRows, approximateRow, imaginaryTarget, Prod.norm_def]

/-- All original classes contribute to the actual nonzero discrepancy, CM105. -/
theorem witness_error : peak (jointCoefficient 0 (fun _ : Fin 1 => positionWitness)
    (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) -
      (fun _ => approximateRow)) = 1 / 4 := by
  rw [witness_coefficient]
  have he : (fun _ : Fin 1 => exactRow) - (fun _ => approximateRow) =
      (fun _ c => (1 / 8 : ℝ) • imaginaryTarget c) := by
    funext j c
    change (1 / 2 : ℝ) • imaginaryTarget c - (3 / 8 : ℝ) • imaginaryTarget c = _
    rw [← sub_smul]
    norm_num
  rw [he]
  norm_num [peak, blockSupport, imaginaryTarget, Prod.norm_def]

/-- The actual error correction is strictly smaller than the estimated gap,
inhabiting CM105's strict premise without an exact-gradient assumption. -/
theorem witness_error_small : 2 * 16 * peak
    (jointCoefficient 0 (fun _ : Fin 1 => positionWitness) (fun _ => [true, false])
      (fun _ => true) (fun _ => witnessRows) - (fun _ => approximateRow)) <
      estimatedGap 16 (fun _ : Fin 1 => witnessRows) (fun _ => approximateRow) := by
  rw [witness_error, witness_estimated_gap]
  norm_num

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate
