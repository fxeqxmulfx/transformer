module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Gap
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Fixed

/-!
# Two actual learned components and a complete capacity reallocation

Source: `review/convex/moments.md`, CM104. A constant-state component and the
ordered CM98 witness both remain in the model. Nonzero imaginary rows first
place all capacity on the former, then move it to the latter. These are proof
witnesses, not benchmark initialization or prescribed training features.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity

open Pair Optimality
open scoped BigOperators

/-- Two retained complete components for the CM104 counterexample. -/
noncomputable def allocationModel (j : Bool) : Parameters Bool :=
  if j then positionWitness else Fixed.bias (0, 0) true 0

/-- Nonzero complex class rows, including the initially inactive component, CM104. -/
noncomputable def allocationRows (j c : Bool) : State :=
  if j then (if c then (0, Complex.I) else (0, 1)) else (0, Complex.I)

/-- Both learned components satisfy the original full capacity bounds, CM104. -/
theorem allocation_unit (j : Bool) : Unit (allocationModel j) := by
  cases j
  · exact Fixed.bias_unit _ _ _ (by norm_num)
  · exact position_unit

/-- All full imaginary readouts satisfy unit row capacity, CM104. -/
theorem allocation_rows : allocationRows ∈ rowDomain Bool Bool := by
  intro j c
  cases j <;> cases c <;> simp [allocationRows, rowSize]

/-- Park all native capacity on the constant-state component, CM104. -/
noncomputable def parked (j c : Bool) : State :=
  (if j then (0 : ℝ) else 16) • allocationRows j c

/-- Move the same native capacity to the ordered component, CM104. -/
noncomputable def moved (j c : Bool) : State :=
  (if j then (16 : ℝ) else 0) • allocationRows j c

/-- The initial complete table is feasible at capacity sixteen, CM104. -/
theorem parked_mem : parked ∈ jointDomain 16 :=
  weighted_mem _ _ _ (by intro j; cases j <;> norm_num)
    (by norm_num) allocation_rows

/-- Moving all capacity preserves the same complete feasible family, CM104. -/
theorem moved_mem : moved ∈ jointDomain 16 :=
  weighted_mem _ _ _ (by intro j; cases j <;> norm_num)
    (by norm_num) allocation_rows

/-- Actual execution of the retained constant-state component, CM104. -/
theorem allocation_constant (xs : List Bool) :
    runFrom (allocationModel false) 0 (allocationModel false).content.initial xs = (1, 0) := by
  change runFrom (Fixed.bias (0, 0) true 0) 0 _ xs = _
  rw [Fixed.bias_run]
  rfl

/-- Actual logits are zero before reallocation despite nonzero imaginary rows, CM104. -/
theorem parked_scores (xs : List Bool) (c : Bool) :
    jointScore allocationModel parked xs c = 0 := by
  rw [jointScore, scores_pairing, Fintype.sum_bool]
  simp [parked, allocationRows, allocation_constant, pairing, Pair.row_apply]

/-- Reallocation yields the correct ordered margin sixteen on both prefixes,
using the very same retained learned states, CM104. -/
theorem moved_scores (b c : Bool) :
    jointScore allocationModel moved [b, !b] c = if c = b then 16 else 0 := by
  rw [jointScore, scores_pairing, Fintype.sum_bool]
  have h := position_execution b
  cases b <;> cases c <;>
    simp only [Bool.not_false, Bool.not_true, ite_true] at h <;>
    norm_num [moved, allocationRows, allocationModel, h, pairing, Pair.row_apply]

/-- The two indexed targets in the CM104 witness. -/
def allocationTarget (i : Fin 2) : Bool := i.val == 0

/-- Distinct opposite object orders; neither indexed record is dropped, CM104. -/
def allocationPrefix (i : Fin 2) : List Bool := [allocationTarget i, !allocationTarget i]

/-- The old normalized weights give exactly the parked actual scores, CM104. -/
theorem fixed_scores (xs : List Bool) (c : Bool) :
    scoreMap 16 (fun j : Bool => if j then 0 else 1) allocationModel allocationRows xs c = 0 := by
  rw [scores_pairing, Fintype.sum_bool]
  simp [allocationRows, allocation_constant, pairing, Pair.row_apply]

/-- Actual balanced residuals cancel on the constant state, while its zero
weight suppresses every coefficient of the ordered component, CM104. -/
theorem fixed_coefficient :
    coefficient 1 16 (fun j : Bool => if j then 0 else 1) allocationModel
      allocationPrefix allocationTarget allocationRows = 0 := by
  funext j c
  dsimp only [coefficient]
  rw [Fin.sum_univ_two]
  cases j <;> cases c <;>
    norm_num [Optimality.residual, fixed_scores, Geometry.probability, allocation_constant,
      allocationTarget]

/-- Fixed-weight complete readout optimization has zero remaining CE gap
at this actual feasible model, CM104. Joint reallocation will improve it. -/
theorem fixed_gap_zero :
    gap 1 16 (fun j : Bool => if j then 0 else 1) allocationModel
      allocationPrefix allocationTarget allocationRows = 0 := by
  unfold gap
  rw [fixed_coefficient]
  simp [linearCost, supportSize]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity
