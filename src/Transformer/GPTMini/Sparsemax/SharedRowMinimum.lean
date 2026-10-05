import Transformer.GPTMini.Sparsemax.SharedRowLoss

/-!
# Local minima of the summed loss in actual shared Q/K matrices

Derived transfer of arXiv:1602.02068v2, §2.5, through the actual shared
projection parameters at `73f8a0b`. A better endpoint with fixed Q,
epsilon-clipped projected keys and matching actual row supports excludes
a local minimum of the total objective in the jointly learned matrices.
No row independence, attention routing labels or ambient differentiability
is required. In particular, a jointly attainable ordinary target tuple
excludes positive-loss local minima under these explicit restrictions.

The existence of an admissible improving endpoint is essential: arbitrary
row targets can conflict through shared parameters. Convexity of the
projection does not itself resolve that obstruction.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A better compatible common key matrix excludes a local minimum of the full sum.
Source: the derived shared-objective extension of §2.5 of arXiv:1602.02068v2,
via actual epsilon-clipped projections at `73f8a0b`. The same perturbation
affects every row, so summation occurs before testing improvement. -/
theorem sharedSquaredLoss_not_isLocalMin_of_better {R F T h : ℕ}
    (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T) (targets : Fin R → E)
    (queries start stop : Fin F → EucSpace h)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (hbetter : sharedSquaredLoss values alpha eps inputs rows targets (queries, stop) <
      sharedSquaredLoss values alpha eps inputs rows targets (queries, start)) :
    ¬ IsLocalMin (sharedSquaredLoss values alpha eps inputs rows targets) (queries, start) := by
  intro hm
  have hm0 : IsLocalMin (sharedSquaredLoss values alpha eps inputs rows targets)
      (keyMatrixSegment queries start stop 0) := by
    rw [keyMatrixSegment_zero]
    exact hm
  have hm' := hm0.comp_continuous
    (keyMatrixSegment_differentiableAt queries start stop 0).continuousAt
  apply not_isLocalMin_of_segment_decrease
    (fun t => sharedSquaredLoss values alpha eps inputs rows targets
      (keyMatrixSegment queries start stop t)) ?_ hm'
  intro t ht hu
  rw [keyMatrixSegment_zero]
  have hc := sharedSquaredLoss_segment_le values alpha eps inputs rows targets queries start stop t
    hs he hz ht.le hu
  exact lt_of_le_of_lt hc (lossSegment_strict_decrease _ _ t hbetter ht)

/-- One shared improvement realizes all the better-endpoint hypotheses for two rows.
Source context: §2.5's derived actual shared-matrix example at epsilon one. -/
example : ¬ IsLocalMin
    (sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets) (sharedExampleQueries, sharedExampleStart) := by
  apply sharedSquaredLoss_not_isLocalMin_of_better _ _ _ _ _ _ _ _ sharedExampleStop
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStart_bound n
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStop_bound n
  · exact sharedExample_support
  · rw [sharedExampleLoss_stop, sharedExampleLoss_start]
    norm_num

/-- A common admissible task fit excludes positive-loss local minima in joint Q/K.
Source: the derived ordinary-output version of §2.5 of arXiv:1602.02068v2,
with the actual shared projections at `73f8a0b`. Joint attainability of the
ordinary targets is explicit; per-row fit directions alone do not suffice. -/
theorem sharedSquaredLoss_not_isLocalMin_of_fit {R F T h : ℕ}
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
    ¬ IsLocalMin (sharedSquaredLoss values alpha eps inputs rows targets) (queries, start) := by
  apply sharedSquaredLoss_not_isLocalMin_of_better values alpha eps inputs rows targets
    queries start stop hs he hz
  rw [sharedSquaredLoss_eq_zero_of_fit values alpha eps inputs rows targets (queries, stop) hfit]
  exact hpos

/-- The positive-error and simultaneous-fit premises have an actual sparse instance.
Source context: the derived §2.5 shared Q/K example with distinct ordinary targets. -/
example : ¬ IsLocalMin
    (sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets) (sharedExampleQueries, sharedExampleStart) := by
  apply sharedSquaredLoss_not_isLocalMin_of_fit _ _ _ _ _ _ _ _ sharedExampleStop
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

/-- Every fitted common matrix is a global minimizer of the actual summed loss.
Source: the derived attainable-output setting for §2.5, using ordinary loss
nonnegativity rather than any assertion about all training trajectories. -/
theorem sharedSquaredLoss_globalMin_of_fit {R F T h : ℕ}
    (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T) (targets : Fin R → E)
    (projections : (Fin F → EucSpace h) × (Fin F → EucSpace h))
    (hfit : ∀ r, sharedRowReadouts values alpha eps inputs rows projections r = targets r) :
    ∀ candidate : (Fin F → EucSpace h) × (Fin F → EucSpace h),
      sharedSquaredLoss values alpha eps inputs rows targets projections ≤
      sharedSquaredLoss values alpha eps inputs rows targets candidate := by
  intro candidate
  rw [sharedSquaredLoss_eq_zero_of_fit values alpha eps inputs rows targets projections hfit]
  exact sharedSquaredLoss_nonneg values alpha eps inputs rows targets candidate

/-- The example's common fitted matrix inhabits the global-minimum premise.
Source context: the derived two-target shared sparsemax construction for §2.5. -/
example (candidate : (Fin 4 → EucSpace 2) × (Fin 4 → EucSpace 2)) :
    sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets (sharedExampleQueries, sharedExampleStop) ≤
    sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets candidate :=
  sharedSquaredLoss_globalMin_of_fit _ _ _ _ _ _ _ sharedExampleReadout_stop candidate

/-- A joint local minimum is optimal against every compatible clipped-key endpoint.
Source: the derived common-support transfer of §2.5 at `73f8a0b`.
The comparison fixes Q and the endpoint supports, not the full model. -/
theorem sharedSquaredLoss_localMin_le_endpoint {R F T h : ℕ}
    (values : Fin R → Fin T → E) (alpha eps : ℝ)
    (inputs : Fin R → Fin T → (Fin F → ℝ)) (rows : Fin R → Fin T) (targets : Fin R → E)
    (queries start stop : Fin F → EucSpace h)
    (hs : ∀ r n, ‖projectionEvaluation (inputs r) start n‖ ≤ eps)
    (he : ∀ r n, ‖projectionEvaluation (inputs r) stop n‖ ≤ eps)
    (hz : ∀ r n, sparseWeights (projectedQKScores alpha eps queries start (inputs r) (rows r))
      (rows r) n = 0 ↔ sparseWeights
        (projectedQKScores alpha eps queries stop (inputs r) (rows r)) (rows r) n = 0)
    (hm : IsLocalMin (sharedSquaredLoss values alpha eps inputs rows targets) (queries, start)) :
    sharedSquaredLoss values alpha eps inputs rows targets (queries, start) ≤
      sharedSquaredLoss values alpha eps inputs rows targets (queries, stop) := by
  by_contra hn
  exact sharedSquaredLoss_not_isLocalMin_of_better values alpha eps inputs rows targets queries
    start stop hs he hz (lt_of_not_ge hn) hm

/-- A fitted common matrix is an inhabited local minimum for the endpoint comparison.
Source context: the actual shared sparse example derived for §2.5. -/
example : sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
    (Fin.natAdd 2) sharedExampleTargets (sharedExampleQueries, sharedExampleStop) ≤
    sharedSquaredLoss (fun _ => sharedExampleValues) (Real.log 3) 1 (fun _ => basis)
      (Fin.natAdd 2) sharedExampleTargets (sharedExampleQueries, sharedExampleStart) := by
  apply sharedSquaredLoss_localMin_le_endpoint
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStop_bound n
  · intro r n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact sharedExampleStart_bound n
  · intro r n
    exact (sharedExample_support r n).symm
  · exact Filter.Eventually.of_forall
      (sharedSquaredLoss_globalMin_of_fit _ _ _ _ _ _ _ sharedExampleReadout_stop)

end Transformer.GPTMini.Sparsemax
