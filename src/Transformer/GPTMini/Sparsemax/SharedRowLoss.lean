import Transformer.GPTMini.Sparsemax.SquaredSegment

/-!
# Summed ordinary loss along the actual shared matrix segment

Derived transfer of arXiv:1602.02068v2, §2.5, through the actual shared
Q/K matrices, epsilon clipping and frozen values at `73f8a0b`.
The same one-dimensional matrix path affects every row simultaneously.
Endpoint projected keys must lie within the epsilon ball, and each row
must have the same actual sparsemax zero pattern at both endpoints.

Under these explicit restrictions the sum is convex along that path.
If the common endpoint realizes all ordinary output targets, its exact
loss is `(1-t)^2` times the initial summed loss. Targets are readout
values and provide no desired sparse attention distributions.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The summed actual task loss is nonnegative at every joint matrix point.
Source: the derived ordinary squared-loss objective for §2.5 at `73f8a0b`. -/
theorem sharedSquaredLoss_nonneg {R F T h : ℕ} (values : Fin R → Fin T → E)
    (alpha eps : ℝ) (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (targets : Fin R → E) (projections : (Fin F → EucSpace h) × (Fin F → EucSpace h)) :
    0 ≤ sharedSquaredLoss values alpha eps inputs rows targets projections := by
  unfold sharedSquaredLoss squaredReadoutLoss
  exact Finset.sum_nonneg fun _ _ => sq_nonneg _

/-- Convexity along a single admissible shared matrix segment.
Source: the derived summed-loss extension of §2.5 of arXiv:1602.02068v2;
`sharedRowReadouts_segment` connects the analytic inequality to actual Q/K. -/
theorem sharedSquaredLoss_segment_le {R F T h : ℕ} (values : Fin R → Fin T → E)
    (alpha eps : ℝ) (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (targets : Fin R → E) (queries start stop : Fin F → EucSpace h) (t : ℝ)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (ht : 0 ≤ t) (hu : t ≤ 1) :
    sharedSquaredLoss values alpha eps inputs rows targets (keyMatrixSegment queries start stop t) ≤
      (1 - t) * sharedSquaredLoss values alpha eps inputs rows targets (queries, start) +
        t * sharedSquaredLoss values alpha eps inputs rows targets (queries, stop) := by
  unfold sharedSquaredLoss
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro r _
  rw [sharedRowReadouts_segment values alpha eps inputs rows queries start stop t hs he hz ht hu r]
  exact squaredReadoutLoss_segment_le _ _ _ t ht hu

/-- Opposite errors in two real rows inhabit the simultaneous loss-convexity premises.
Source context: the derived shared-matrix example for §2.5 at `73f8a0b`. -/
example : sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
    (Fin.natAdd 2) sharedExampleTargets (keyMatrixSegment sharedExampleQueries
      sharedExampleStart sharedExampleStop (1 / 2)) ≤
    (1 - (1 / 2 : ℝ)) * (9 / 128 : ℝ) + (1 / 2 : ℝ) * 0 := by
  rw [← sharedExampleLoss_start, ← sharedExampleLoss_stop]
  apply sharedSquaredLoss_segment_le
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStart_bound n
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStop_bound n
  · exact sharedExample_support
  · norm_num
  · norm_num

/-- Exact quadratic decay when one shared admissible endpoint fits every task target.
Source: the derived ordinary-loss transfer of §2.5 of arXiv:1602.02068v2.
Joint attainability is a condition on actual output values, not attention routes. -/
theorem sharedSquaredLoss_segment_fit {R F T h : ℕ} (values : Fin R → Fin T → E)
    (alpha eps : ℝ) (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (targets : Fin R → E) (queries start stop : Fin F → EucSpace h) (t : ℝ)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (hfit : ∀ r, sharedRowReadouts values alpha eps inputs rows (queries, stop) r = targets r)
    (ht : 0 ≤ t) (hu : t ≤ 1) :
    sharedSquaredLoss values alpha eps inputs rows targets (keyMatrixSegment queries start stop t) =
      (1 - t) ^ 2 * sharedSquaredLoss values alpha eps inputs rows targets (queries, start) := by
  unfold sharedSquaredLoss
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro r _
  rw [sharedRowReadouts_segment values alpha eps inputs rows queries start stop t hs he hz ht hu r,
    hfit r]
  exact squaredReadoutLoss_segment_target _ _ t

/-- The actual two-row loss drops by a factor four halfway to the shared correction.
Source context: the derived exact §2.5 path at epsilon one, with fixed zero weights. -/
example : sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
    (Fin.natAdd 2) sharedExampleTargets (keyMatrixSegment sharedExampleQueries
      sharedExampleStart sharedExampleStop (1 / 2)) = 9 / 512 := by
  have hd := sharedSquaredLoss_segment_fit (fun _ : Fin 2 => sharedExampleValues)
    (Real.log 3) 1 (fun _ => basis) (Fin.natAdd 2) sharedExampleTargets sharedExampleQueries
    sharedExampleStart sharedExampleStop (1 / 2)
    (fun r n => by
      rw [projectionEvaluation_apply, projection_basis_column]
      exact sharedExampleStart_bound n)
    (fun r n => by
      rw [projectionEvaluation_apply, projection_basis_column]
      exact sharedExampleStop_bound n)
    sharedExample_support sharedExampleReadout_stop (by norm_num) (by norm_num)
  rw [sharedExampleLoss_start] at hd
  norm_num at hd
  exact hd

/-- Fitting every ordinary target at one common matrix makes the summed loss zero.
Source: the derived shared-output condition for §2.5, before XSA at `73f8a0b`. -/
theorem sharedSquaredLoss_eq_zero_of_fit {R F T h : ℕ} (values : Fin R → Fin T → E)
    (alpha eps : ℝ) (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (targets : Fin R → E) (projections : (Fin F → EucSpace h) × (Fin F → EucSpace h))
    (hfit : ∀ r, sharedRowReadouts values alpha eps inputs rows projections r = targets r) :
    sharedSquaredLoss values alpha eps inputs rows targets projections = 0 := by
  unfold sharedSquaredLoss
  simp only [hfit, squaredReadoutLoss, sub_self, norm_zero, zero_pow
    (by decide : 2 ≠ 0), Finset.sum_const_zero]

/-- The target-fit zero-loss hypothesis is realized by the common example matrix.
Source context: the derived ordinary targets for §2.5, with no routing supervision. -/
example : sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
    (Fin.natAdd 2) sharedExampleTargets (sharedExampleQueries, sharedExampleStop) = 0 :=
  sharedSquaredLoss_eq_zero_of_fit _ _ _ _ _ _ _ sharedExampleReadout_stop

/-- Any wrong ordinary row makes the actual summed squared loss positive.
Source: the ordinary-output objective derived for §2.5; cancellations of
gradients do not cancel nonnegative squared error terms. -/
theorem sharedSquaredLoss_pos_of_wrong_row {R F T h : ℕ} (values : Fin R → Fin T → E)
    (alpha eps : ℝ) (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T)
    (targets : Fin R → E) (projections : (Fin F → EucSpace h) × (Fin F → EucSpace h))
    (r : Fin R) (hbad : sharedRowReadouts values alpha eps inputs rows projections r ≠ targets r) :
    0 < sharedSquaredLoss values alpha eps inputs rows targets projections := by
  have hp := sq_pos_of_pos (norm_pos_iff.mpr (sub_ne_zero.mpr hbad))
  have hb := Finset.single_le_sum
    (f := fun s => squaredReadoutLoss (targets s)
      (sharedRowReadouts values alpha eps inputs rows projections s))
    (fun _ _ => sq_nonneg _) (Finset.mem_univ r)
  exact lt_of_lt_of_le hp hb

/-- The initially incorrect first row inhabits the positive summed-loss premise.
Source context: the actual two-target shared example derived for §2.5. -/
example : 0 < sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
    (Fin.natAdd 2) sharedExampleTargets (sharedExampleQueries, sharedExampleStart) := by
  apply sharedSquaredLoss_pos_of_wrong_row _ _ _ _ _ _ _ (0 : Fin 2)
  rw [sharedExampleReadout_start]
  norm_num [sharedExampleTargets]

end Transformer.GPTMini.Sparsemax
