module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality.Gap

/-!
# Nonzero complex examples of conditional CE certificates

Source: `review/flat_spectrum.md`, CM103. The existing nonconstant ordered
content, capacity sixteen and nonzero imaginary readouts inhabit the gap
hypotheses. Opposite labels on a repeated prefix give a zero gap; a single
label gives a positive gap. These are proof witnesses, never benchmark inputs.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality

open Pair
open scoped BigOperators

/-- Nonzero full complex rows for proof examples only, CM103. -/
noncomputable def witnessRows (c : Bool) : State :=
  if c then (Complex.I, 0) else (-Complex.I, 0)

/-- Both imaginary class rows satisfy the original capacity, CM103. -/
theorem witness_rows : (fun _ : Fin 1 => witnessRows) ∈ rowDomain (Fin 1) Bool := by
  intro j c
  cases c <;> simp [witnessRows, rowSize]

/-- Actual nonconstant learned content produces equal logits on this prefix
with the displayed nonzero readouts, CM103. -/
theorem witness_logits (c : Bool) :
    scoreMap 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
      (fun _ => witnessRows) [true, false] c = 0 := by
  have h := position_execution true
  simp only [Bool.not_true, ite_true] at h
  rw [scores_pairing]
  cases c <;> simp [h, pairing, witnessRows, Pair.row_apply]

/-- Opposite targets on the same indexed prefix cancel the complete actual
CE coefficient, with nonzero capacity and learned states, CM103. -/
theorem balanced_coefficient :
    coefficient 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
      (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) = 0 := by
  have h := position_execution true
  simp only [Bool.not_true, ite_true] at h
  funext j c
  dsimp only [coefficient]
  rw [Fin.sum_univ_two]
  cases c <;> norm_num [residual, witness_logits, Geometry.probability, h]

/-- This explicit balanced training set has zero gap, CM103. It makes no
unseen-answer claim and does not discard the conflicting indexed records. -/
theorem balanced_gap :
    gap 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
      (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) = 0 := by
  unfold gap
  rw [balanced_coefficient]
  simp [linearCost, supportSize]

/-- On one original target the same complete imaginary rows have gap sixteen,
inhabiting the strict-direction premise without an optimum assumption, CM103. -/
theorem witness_positive_gap :
    gap 0 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
      (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows) = 16 := by
  have h := position_execution true
  simp only [Bool.not_true, ite_true] at h
  norm_num [gap, coefficient, linearCost, supportSize, residual, witness_logits,
    Geometry.probability, h, pairing, witnessRows, Pair.row_apply, Prod.norm_def]

example : 0 ≤ gap 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) :=
  gap_nonneg _ _ _ _ _ _ _ witness_rows

example : meanLoss 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) -
      gap 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
        (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) ≤
      meanLoss 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
        (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) :=
  loss_lower_bound _ _ _ _ _ _ _ _ witness_rows

example : meanLoss 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) -
      meanLoss 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
        (fun _ => [true, false]) (fun i => i.val == 0) (fun _ _ => (0, Complex.I)) ≤
      gap 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
        (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) :=
  improvement_le_gap _ _ _ _ _ _ _ _ (by intro j c; simp [rowSize])

example : meanLoss 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
    (fun _ => [true, false]) (fun i => i.val == 0) (fun _ => witnessRows) ≤
      meanLoss 1 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
        (fun _ => [true, false]) (fun i => i.val == 0) (fun _ _ => (0, Complex.I)) :=
  optimal_of_gap_zero _ _ _ _ _ _ _ balanced_gap _ (by intro j c; simp [rowSize])

example : ∃ v ∈ rowDomain (Fin 1) Bool,
    linearCost (v - fun _ => witnessRows)
      (coefficient 0 16 (fun _ : Fin 1 => 1) (fun _ => positionWitness)
        (fun _ => [true, false]) (fun _ => true) (fun _ => witnessRows)) < 0 :=
  negative_direction _ _ _ _ _ _ _ (by rw [witness_positive_gap]; norm_num)

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Optimality
