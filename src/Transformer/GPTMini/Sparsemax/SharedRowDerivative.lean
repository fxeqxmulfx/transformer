import Transformer.GPTMini.Sparsemax.SharedRowMinimum

/-!
# A nonzero total-loss direction in the actual shared matrices

Derived extension of arXiv:1602.02068v2, §2.5, through the actual shared
Q/K projections at `73f8a0b`. If one admissible common key matrix realizes
all ordinary output targets, the summed loss has right derivative
`-2 * initialLoss` along the shared affine key segment. At positive loss
this contradicts a zero full derivative in the jointly learned Q/K.

Only the actual matrix segment is globally differentiable here. Sparsemax
can have inactive threshold ties at the initial point; the loss identity
and derivative along `[0,1]` suffice. Joint output attainability, clipped
endpoint projected keys and equal endpoint supports remain explicit.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The actual shared-matrix path has right derivative minus twice total initial error.
Source: the derived simultaneous task-loss consequence of §2.5 of
arXiv:1602.02068v2, after exact QKNorm and projection at `73f8a0b`. -/
theorem sharedSquaredLoss_hasDerivWithinAt_fit {R F T h : ℕ}
    (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T) (targets : Fin R → E)
    (queries start stop : Fin F → EucSpace h)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (hfit : ∀ r, sharedRowReadouts values alpha eps inputs rows (queries, stop) r = targets r) :
    HasDerivWithinAt (fun t => sharedSquaredLoss values alpha eps inputs rows targets
      (keyMatrixSegment queries start stop t))
      (-2 * sharedSquaredLoss values alpha eps inputs rows targets (queries, start))
      (Set.Icc 0 1) 0 := by
  have hd := (targetSegment_hasDerivAt
    (sharedSquaredLoss values alpha eps inputs rows targets (queries, start))).hasDerivWithinAt
    (s := Set.Icc 0 1)
  apply hd.congr
  · intro t ht
    exact sharedSquaredLoss_segment_fit values alpha eps inputs rows targets queries start stop t
      hs he hz hfit ht.1 ht.2
  · rw [keyMatrixSegment_zero]
    norm_num

/-- The opposite-error example has an actual total-loss path derivative of minus 9/64.
Source context: the derived simultaneous sparse correction for §2.5 at epsilon one. -/
example : HasDerivWithinAt (fun t => sharedSquaredLoss (fun _ => sharedExampleValues)
    (Real.log 3) 1 (fun _ => basis) (Fin.natAdd 2) sharedExampleTargets
    (keyMatrixSegment sharedExampleQueries sharedExampleStart sharedExampleStop t))
    (-(9 / 64)) (Set.Icc 0 1) 0 := by
  have hd := sharedSquaredLoss_hasDerivWithinAt_fit (fun _ : Fin 2 => sharedExampleValues)
    (Real.log 3) 1 (fun _ => basis) (Fin.natAdd 2) sharedExampleTargets
    sharedExampleQueries sharedExampleStart sharedExampleStop
    (fun r n => by
      rw [projectionEvaluation_apply, projection_basis_column]
      exact sharedExampleStart_bound n)
    (fun r n => by
      rw [projectionEvaluation_apply, projection_basis_column]
      exact sharedExampleStop_bound n)
    sharedExample_support sharedExampleReadout_stop
  rw [sharedExampleLoss_start] at hd
  norm_num at hd
  exact hd

/-- Positive total error excludes a zero full derivative of jointly learned Q/K.
Source: the derived shared-objective transfer of §2.5 of arXiv:1602.02068v2,
using one actual shared key perturbation at `73f8a0b`. No per-row gradient
is substituted for the derivative of the sum. -/
theorem sharedSquaredLoss_no_zero_projection_derivative_of_fit {R F T h : ℕ}
    (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T) (targets : Fin R → E)
    (queries start stop : Fin F → EucSpace h)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (hfit : ∀ r, sharedRowReadouts values alpha eps inputs rows (queries, stop) r = targets r)
    (hpos : 0 < sharedSquaredLoss values alpha eps inputs rows targets (queries, start)) :
    ¬ HasFDerivAt (𝕜 := ℝ) (sharedSquaredLoss values alpha eps inputs rows targets) 0
      (queries, start) := by
  intro hz0
  have hz' : HasFDerivAt (𝕜 := ℝ) (sharedSquaredLoss values alpha eps inputs rows targets) 0
      (keyMatrixSegment queries start stop 0) := by
    rw [keyMatrixSegment_zero]
    exact hz0
  have hc := hz'.comp (0 : ℝ)
    (keyMatrixSegment_differentiableAt queries start stop 0).hasFDerivAt
  have hzpath : HasDerivAt (fun t => sharedSquaredLoss values alpha eps inputs rows targets
      (keyMatrixSegment queries start stop t)) 0 0 := by
    simpa only [ContinuousLinearMap.zero_comp, zero_apply, Function.comp_def] using hc.hasDerivAt
  exact no_zero_derivative_of_segment _ _ (by nlinarith)
    (sharedSquaredLoss_hasDerivWithinAt_fit values alpha eps inputs rows targets
      queries start stop hs he hz hfit) hzpath

/-- The actual two-row example inhabits every premise excluding a zero joint derivative.
Source context: §2.5's derived shared sparse correction, with ordinary task targets. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets) 0 (sharedExampleQueries, sharedExampleStart) := by
  apply sharedSquaredLoss_no_zero_projection_derivative_of_fit _ _ _ _ _ _ _ _ sharedExampleStop
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStart_bound n
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStop_bound n
  · exact sharedExample_support
  · exact sharedExampleReadout_stop
  · rw [sharedExampleLoss_start]
    norm_num

/-- The one-sided derivative value of the actual summed loss is exact, even at ties.
Source: the derived §2.5 common-path identity at `73f8a0b`; uniqueness
is used only within the real interval, not across ambient support cells. -/
theorem sharedSquaredLoss_derivWithin_fit {R F T h : ℕ}
    (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T) (targets : Fin R → E)
    (queries start stop : Fin F → EucSpace h)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (hfit : ∀ r, sharedRowReadouts values alpha eps inputs rows (queries, stop) r = targets r) :
    derivWithin (fun t => sharedSquaredLoss values alpha eps inputs rows targets
      (keyMatrixSegment queries start stop t)) (Set.Icc 0 1) 0 =
      -2 * sharedSquaredLoss values alpha eps inputs rows targets (queries, start) := by
  exact (sharedSquaredLoss_hasDerivWithinAt_fit values alpha eps inputs rows targets queries
    start stop hs he hz hfit).derivWithin
      (uniqueDiffOn_Icc_zero_one 0 ⟨by norm_num, by norm_num⟩)

/-- The exact actual derivative in the finite shared example is minus 9/64.
Source context: the derived two-target ordinary-loss instance for §2.5. -/
example : derivWithin (fun t => sharedSquaredLoss (fun _ => sharedExampleValues)
    (Real.log 3) 1 (fun _ => basis) (Fin.natAdd 2) sharedExampleTargets
    (keyMatrixSegment sharedExampleQueries sharedExampleStart sharedExampleStop t))
    (Set.Icc 0 1) 0 = -(9 / 64) := by
  rw [sharedSquaredLoss_derivWithin_fit, sharedExampleLoss_start]
  · norm_num
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStart_bound n
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStop_bound n
  · exact sharedExample_support
  · exact sharedExampleReadout_stop

end Transformer.GPTMini.Sparsemax
