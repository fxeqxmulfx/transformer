import Transformer.GPTMini.Sparsemax.SharedRowsExample

/-!
# Actual readouts and summed loss of shared projection matrices

Derived multi-row formulation for arXiv:1602.02068v2, §2.2 and §2.5,
and the shared Q/K projections and frozen value sum at `73f8a0b`.
Every row, including rows from different examples, uses the same pair
of matrices. Ordinary output targets may differ. A common matrix segment
transports the exact sparsemax readouts when projected endpoint keys are
epsilon-clipped and endpoint zero patterns agree for each row.

Neither independent queries nor private row parameters are assumed.
The finite example has two distinct targets and opposite initial errors;
one actual shared key change fits both while ordinary weights stay zero.
Readouts are unrotated and before XSA/output projection, with fixed inputs,
values, epsilon and gain. This is a restricted shared-parameter result.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- All actual sparse attention readouts using one pair of shared Q/K matrices.
Source: projections and value sum at `73f8a0b`, after §2.2's sparsemax.
Inputs and frozen values may vary between examples; matrices do not. -/
def sharedRowReadouts {R F T h : ℕ} (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (projections : (Fin F → EucSpace h) × (Fin F → EucSpace h)) (r : Fin R) : E :=
  frozenValueReadout (values r) (sparseWeights
    (projectedQKScores alpha eps projections.1 projections.2 (inputs r) (rows r)) (rows r))

/-- Sum of ordinary squared output losses, with no attention routing labels.
Source: the derived outer-loss formulation for §2.5 of arXiv:1602.02068v2,
on the actual shared matrices at `73f8a0b`. -/
def sharedSquaredLoss {R F T h : ℕ} (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T) (targets : Fin R → E)
    (projections : (Fin F → EucSpace h) × (Fin F → EucSpace h)) : ℝ :=
  ∑ r, squaredReadoutLoss (targets r) (sharedRowReadouts values alpha eps inputs rows projections r)

/-- The example's two endpoint matrices have the same actual zero pattern in every row.
Source: §2.2's projection and the normalized thresholds in `SharedRowsExample`. -/
theorem sharedExample_support (r : Fin 2) (n : Fin 4) :
    sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStart
      basis (Fin.natAdd 2 r)) (Fin.natAdd 2 r) n = 0 ↔
    sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries sharedExampleStop
      basis (Fin.natAdd 2 r)) (Fin.natAdd 2 r) n = 0 := by
  rw [sharedExampleWeights_start, sharedExampleWeights_stop]
  fin_cases r <;> fin_cases n <;> norm_num [sharedExampleTargets]

/-- Both actual initial readouts are one half, despite distinct queries and targets.
Source: the shared frozen-value operator at `73f8a0b`, after §2.2's projection. -/
theorem sharedExampleReadout_start (r : Fin 2) :
    sharedRowReadouts (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) (sharedExampleQueries, sharedExampleStart) r = (1 / 2 : ℝ) := by
  unfold sharedRowReadouts
  rw [sharedExampleWeights_start, frozenValueReadout_apply]
  norm_num [Fin.sum_univ_four, sharedExampleValues]

/-- One shared key matrix attains both ordinary task outputs simultaneously.
Source: the derived §2.5 instance through the actual projections at `73f8a0b`. -/
theorem sharedExampleReadout_stop (r : Fin 2) :
    sharedRowReadouts (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) (sharedExampleQueries, sharedExampleStop) r = sharedExampleTargets r := by
  unfold sharedRowReadouts
  rw [sharedExampleWeights_stop, frozenValueReadout_apply]
  fin_cases r <;> norm_num [Fin.sum_univ_four, sharedExampleValues, sharedExampleTargets]

/-- The actual shared segment gives an affine readout for every row at once.
Source: §2.2's threshold formula, with clipped projected keys at `73f8a0b`.
The zero-pattern condition is imposed on actual endpoint projections,
and the norm conditions precede actual QKNorm; neither is an oracle. -/
theorem sharedRowReadouts_segment {R F T h : ℕ} (values : Fin R → Fin T → E)
    (alpha eps : ℝ) (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (queries start stop : Fin F → EucSpace h) (t : ℝ)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (ht : 0 ≤ t) (hu : t ≤ 1) (r : Fin R) :
    sharedRowReadouts values alpha eps inputs rows (keyMatrixSegment queries start stop t) r =
      (1 - t) • sharedRowReadouts values alpha eps inputs rows (queries, start) r +
        t • sharedRowReadouts values alpha eps inputs rows (queries, stop) r := by
  simp only [sharedRowReadouts, keyMatrixSegment]
  rw [projectedQKScores_key_segment alpha eps queries start stop (inputs r) (rows r) t
    (hs r) (he r) ht hu]
  exact frozenReadout_segment (values r) _ _ (rows r) t ht hu (hz r)

/-- Two different target rows inhabit the simultaneous affine-readout premises.
Source context: the derived shared-matrix example for §2.5, epsilon one. -/
example (r : Fin 2) :
    sharedRowReadouts (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) (keyMatrixSegment sharedExampleQueries sharedExampleStart
        sharedExampleStop (1 / 2)) r =
      (1 - (1 / 2 : ℝ)) • (1 / 2 : ℝ) + (1 / 2 : ℝ) • sharedExampleTargets r := by
  rw [sharedRowReadouts_segment, sharedExampleReadout_start, sharedExampleReadout_stop]
  · intro s n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStart_bound n
  · intro s n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStop_bound n
  · exact sharedExample_support
  · norm_num
  · norm_num

/-- The summed actual initial error is positive, with opposite errors in the two rows.
Source: the derived ordinary squared-loss instance for §2.5 at `73f8a0b`. -/
theorem sharedExampleLoss_start :
    sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets (sharedExampleQueries, sharedExampleStart) = 9 / 128 := by
  unfold sharedSquaredLoss
  simp only [sharedExampleReadout_start]
  norm_num [Fin.sum_univ_two, squaredReadoutLoss, sharedExampleTargets, Real.norm_eq_abs]

/-- Both task errors vanish at the common corrected matrix.
Source: the derived simultaneous output correction for §2.5 at `73f8a0b`. -/
theorem sharedExampleLoss_stop :
    sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets (sharedExampleQueries, sharedExampleStop) = 0 := by
  unfold sharedSquaredLoss
  simp only [sharedExampleReadout_stop, squaredReadoutLoss, sub_self, norm_zero, zero_pow
    (by decide : 2 ≠ 0), Finset.sum_const_zero]

/-- Every initially inactive actual weight stays exactly zero on the shared matrix path.
Source: §2.2's threshold formula under matching endpoint supports,
transported through actual clipped-key QKNorm at `73f8a0b`. -/
theorem sharedRowWeights_segment_zero {R F T h : ℕ} (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (queries start stop : Fin F → EucSpace h) (t : ℝ) (r : Fin R) (n : Fin T)
    (hs : ∀ s j, ‖projectionEvaluation (inputs s) start j‖ ≤ eps)
    (he : ∀ s j, ‖projectionEvaluation (inputs s) stop j‖ ≤ eps)
    (hz : ∀ s j, sparseWeights (projectedQKScores alpha eps queries start (inputs s) (rows s))
      (rows s) j = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs s) (rows s)) (rows s) j = 0)
    (ht : 0 ≤ t) (hu : t ≤ 1)
    (hj : sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0) :
    sparseWeights (projectedQKScores alpha eps queries ((1 - t) • start + t • stop)
      (inputs r) (rows r)) (rows r) n = 0 := by
  rw [projectedQKScores_key_segment alpha eps queries start stop (inputs r) (rows r) t
    (hs r) (he r) ht hu]
  exact sparseWeights_segment_zero _ _ (rows r) n t ht hu (hz r) hj

/-- Ordinary weights stay exactly zero for both real example rows at an interior point.
Source context: the actual shared sparse construction derived for §2.5. -/
example (r : Fin 2) : sparseWeights (projectedQKScores (Real.log 3) 1 sharedExampleQueries
    ((1 - (1 / 2 : ℝ)) • sharedExampleStart + (1 / 2 : ℝ) • sharedExampleStop)
    basis (Fin.natAdd 2 r)) (Fin.natAdd 2 r) 2 = 0 := by
  apply sharedRowWeights_segment_zero (Real.log 3) 1 (fun _ : Fin 2 => basis) (Fin.natAdd 2)
    sharedExampleQueries sharedExampleStart sharedExampleStop (1 / 2) r 2
  · intro s n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStart_bound n
  · intro s n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStop_bound n
  · exact sharedExample_support
  · norm_num
  · norm_num
  · rw [sharedExampleWeights_start]
    norm_num

end Transformer.GPTMini.Sparsemax
