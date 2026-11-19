module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Counterexample

/-!
# Inhabited enlarged-gap hypotheses with full imaginary readouts

Source: `review/convex/moments.md`, CM104. Actual distinct ordered prefixes
inhabit the strict-direction premises. Opposite labels on a repeated prefix
inhabit the zero-gap premises without assuming an optimum or separability.
The duplicate records are retained and serve only as formal proof witnesses.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair Optimality

/-- Indexed opposite targets on the same prefix cancel the actual complete
coefficient of both retained components, CM104, even with imaginary readouts. -/
theorem repeated_coefficient :
    jointCoefficient 1 allocationModel (fun _ => [true, false]) allocationTarget parked = 0 := by
  have hs (j : Bool) : runFrom (allocationModel j) 0 (allocationModel j).content.initial
      [true, false] = if j then (0, -Complex.I) else (1, 0) := by
    cases j
    · exact allocation_constant _
    · simpa [allocationModel] using position_execution true
  funext j c
  rw [joint_coefficient_apply, Fin.sum_univ_two]
  cases j <;> cases c <;>
    norm_num [Optimality.residual, parked_scores, Geometry.probability, allocationTarget, hs]

/-- This complete duplicated-prefix problem has zero enlarged gap, CM104.
The conflicting records remain indexed, and no unseen-label claim follows. -/
theorem repeated_gap :
    jointGap 1 16 allocationModel (fun _ => [true, false]) allocationTarget parked = 0 := by
  unfold jointGap
  rw [repeated_coefficient]
  simp [linearCost, peak, blockSupport]

example : 0 ≤ jointGap 1 16 allocationModel allocationPrefix allocationTarget parked :=
  joint_gap_nonneg _ _ _ _ _ _ parked_mem

example : jointLoss 1 allocationModel allocationPrefix allocationTarget parked -
    jointGap 1 16 allocationModel allocationPrefix allocationTarget parked ≤
      jointLoss 1 allocationModel allocationPrefix allocationTarget moved :=
  joint_gap_lower _ _ _ _ _ _ _ moved_mem

example : jointLoss 1 allocationModel allocationPrefix allocationTarget parked -
    jointLoss 1 allocationModel allocationPrefix allocationTarget moved ≤
      jointGap 1 16 allocationModel allocationPrefix allocationTarget parked :=
  joint_improvement_le _ _ _ _ _ _ _ moved_mem

example : jointLoss 1 allocationModel (fun _ => [true, false]) allocationTarget parked ≤
    jointLoss 1 allocationModel (fun _ => [true, false]) allocationTarget moved :=
  joint_optimal_of_zero _ _ _ _ _ _ repeated_gap _ moved_mem

example : ∃ v ∈ jointDomain 16, linearCost (v - parked)
    (jointCoefficient 1 allocationModel allocationPrefix allocationTarget parked) < 0 :=
  joint_negative_direction _ _ _ _ _ _ parked_mem parked_joint_gap_pos

end Transformer.Grokking.Convex.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
