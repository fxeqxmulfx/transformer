import Transformer.GPTMini.Sparsemax.SharedRowDerivative

/-!
# Counterexample to an unconditional summed-loss transfer

The claim being refuted is the unconditional extension of the derived
single-row no-wrong-minimum results motivated by arXiv:1602.02068v2,
§2.5: individually wrong rows need not make a shared summed loss improvable.
The manuscript itself does not assert that extension. Here two identical
actual observations demand ordinary scalar outputs zero and one. The
shared Q/K matrices and frozen values at `73f8a0b` give them the same
output for every parameter point. Its optimal value is one half.

That output is attained by actual sparsemax with two active anchors and
zero ordinary weights. Both rows are wrong, their total error is one half,
and the matrix point is a global and local minimum. No common target fit
exists. This is a shared-target conflict, not a one-hot saturated route.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Actual output of the repeated observation through shared matrices.
Source: projections and value sum at `73f8a0b`, after §2.2's sparsemax. -/
def conflictingReadout (projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) : ℝ :=
  sharedRowReadouts (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
    (fun _ : Fin 2 => (2 : Fin 4)) projections 0

/-- Actual two-row squared error with ordinary target values zero and one.
Source: the counterexample to an unconditional §2.5 summed-loss extension;
the operator uses the real projections and sparsemax at `73f8a0b`. -/
def conflictingSquaredLoss
    (projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) : ℝ :=
  sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
    (fun _ : Fin 2 => (2 : Fin 4)) (fun r => (r.val : ℝ)) projections

/-- The two observations have equal actual outputs for every shared parameter point.
Source: the repeated-input counterexample to the unconditional §2.5 extension. -/
theorem conflictingRowReadouts_eq
    (projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) (r : Fin 2) :
    sharedRowReadouts (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (fun _ : Fin 2 => (2 : Fin 4)) projections r = conflictingReadout projections := by
  unfold conflictingReadout sharedRowReadouts
  rfl

/-- The real summed objective is the sum of two incompatible ordinary errors.
Source: the actual shared-output counterexample to the unconditional §2.5 extension. -/
theorem conflictingSquaredLoss_eq
    (projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) :
    conflictingSquaredLoss projections = conflictingReadout projections ^ 2 +
      (conflictingReadout projections - 1) ^ 2 := by
  unfold conflictingSquaredLoss sharedSquaredLoss
  rw [Fin.sum_univ_two]
  simp only [conflictingRowReadouts_eq]
  norm_num [squaredReadoutLoss, Real.norm_eq_abs, sq_abs]

/-- Completing the square identifies the positive optimal shared-output loss.
Source: the derived algebra in the unconditional §2.5 transfer counterexample. -/
theorem conflictingSquaredLoss_completed_square
    (projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) :
    conflictingSquaredLoss projections = 1 / 2 + 2 * (conflictingReadout projections - 1 / 2) ^ 2 := by
  rw [conflictingSquaredLoss_eq]
  ring

/-- Every actual joint Q/K parameter point has loss at least one half.
Source: the repeated-output obstruction to the unconditional §2.5 extension. -/
theorem conflictingSquaredLoss_lower_bound
    (projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) :
    1 / 2 ≤ conflictingSquaredLoss projections := by
  rw [conflictingSquaredLoss_completed_square]
  nlinarith [sq_nonneg (conflictingReadout projections - 1 / 2)]

/-- The actual sparse shared matrix attains the optimal output one half.
Source: §2.2's certified example weights, through actual projections at `73f8a0b`. -/
theorem conflictingReadout_start :
    conflictingReadout (sharedExampleQueries, sharedExampleStart) = 1 / 2 := by
  exact sharedExampleReadout_start 0

/-- The certified sparse example attains the positive global loss lower bound.
Source: the actual shared-matrix counterexample to the unconditional §2.5 extension. -/
theorem conflictingSquaredLoss_start :
    conflictingSquaredLoss (sharedExampleQueries, sharedExampleStart) = 1 / 2 := by
  rw [conflictingSquaredLoss_completed_square, conflictingReadout_start]
  norm_num

/-- Both ordinary rows are wrong at the globally optimal shared matrix.
Source: the repeated-observation counterexample to the unconditional §2.5 extension. -/
theorem conflictingRows_wrong (r : Fin 2) :
    sharedRowReadouts (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (fun _ : Fin 2 => (2 : Fin 4)) (sharedExampleQueries, sharedExampleStart) r ≠ (r.val : ℝ) := by
  rw [conflictingRowReadouts_eq, conflictingReadout_start]
  fin_cases r <;> norm_num

/-- Actual shared parameters cannot simultaneously attain the two conflicting targets.
Source: the necessary joint-attainability obstruction in the derived §2.5 transfer. -/
theorem conflictingTargets_not_attainable :
    ¬ ∃ projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2), ∀ r : Fin 2,
      sharedRowReadouts (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
        (fun _ : Fin 2 => (2 : Fin 4)) projections r = (r.val : ℝ) := by
  rintro ⟨projections, hfit⟩
  have h0 := hfit 0
  have h1 := hfit 1
  rw [conflictingRowReadouts_eq] at h0 h1
  norm_num at h0 h1
  linarith

/-- The positive-loss matrix point is a global minimum of the actual joint objective.
Source: the counterexample to unconditional shared-row transfer of the §2.5 result. -/
theorem conflictingSquaredLoss_globalMin :
    IsMinOn conflictingSquaredLoss Set.univ (sharedExampleQueries, sharedExampleStart) := by
  apply isMinOn_univ_iff.mpr
  intro projections
  rw [conflictingSquaredLoss_start]
  exact conflictingSquaredLoss_lower_bound projections

/-- Both wrong rows coexist at an actual positive-loss local minimum in joint Q/K.
Source: the counterexample to unconditional shared-row transfer of the §2.5 result. -/
theorem conflictingSquaredLoss_isLocalMin :
    IsLocalMin conflictingSquaredLoss (sharedExampleQueries, sharedExampleStart) := by
  exact conflictingSquaredLoss_globalMin.isLocalMin Filter.univ_mem

/-- The positive optimal objective refutes unconditional no-wrong-row local minima.
Source: the actual shared sparse counterexample to the unconditional §2.5 extension. -/
theorem conflictingSquaredLoss_positive_localMin :
    0 < conflictingSquaredLoss (sharedExampleQueries, sharedExampleStart) ∧
      IsLocalMin conflictingSquaredLoss (sharedExampleQueries, sharedExampleStart) := by
  constructor
  · rw [conflictingSquaredLoss_start]
    norm_num
  · exact conflictingSquaredLoss_isLocalMin

/-- The positive optimum is attained exactly when the common output is one half.
Source: the completed-square obstruction to the unconditional §2.5 extension. -/
theorem conflictingSquaredLoss_min_iff
    (projections : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) :
    conflictingSquaredLoss projections = 1 / 2 ↔ conflictingReadout projections = 1 / 2 := by
  rw [conflictingSquaredLoss_completed_square]
  constructor
  · intro hm
    have hz : (conflictingReadout projections - 1 / 2) ^ 2 = 0 := by linarith
    exact sub_eq_zero.mp (sq_eq_zero_iff.mp hz)
  · intro hm
    rw [hm]
    ring

/-- This positive-loss minimum has two active positions and exact ordinary zeros.
Source: §2.2's actual projection, distinguishing the counterexample from saturation. -/
theorem conflictingRow_sparse :
    sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
      basis 2) 2 0 = 1 / 2 ∧
    sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
      basis 2) 2 1 = 1 / 2 ∧
    sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
      basis 2) 2 2 = 0 ∧
    sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
      basis 2) 2 3 = 0 := by
  have hw := sharedExampleWeights_start 0
  change sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
    basis 2) 2 = _ at hw
  rw [hw]
  norm_num

end Transformer.GPTMini.Sparsemax
