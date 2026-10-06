import Transformer.GPTMini.Sparsemax.OutputTiedMemory

/-!
# Nontrivial ordinary-answer fits under affine parameter sharing

New restricted examples following arXiv:1602.02068v2, Eq. (1). Use the
same three-slot energy domain as the previous regression witnesses, with
fixed offset and gain both 1/24. Adjacent ordinary output sums determine
the two learned edge weights. Answers `(1,1,-2)` and `(-2,1,1)` both fit,
select opposite nonidentity attention, and change both physical Q and K.

All common values remain learned and nonconstant. The two endpoints and
their midpoint inhabit the same convex architectural domain. Thus the
new injectivity restriction is not a singleton domain or a requirement
that every output equal a prescribed attention-label table.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- The same fixed affine rule recovers opposite edge allocations from the ordinary answers.
Source: new affine-sharing witnesses after arXiv:1602.02068v2, Eq. (1). -/
theorem taskOutputTied_edges (edge : Fin 2) :
    outputTiedMemoryEdges (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1)
      (taskEnergyTarget edge) = (taskEnergyParameters edge).1 := by
  funext e
  fin_cases edge <;> fin_cases e <;>
    norm_num [outputTiedMemoryEdges, taskEnergyTarget, taskEnergyParameters, Matrix.of_apply]

/-- Both genuine nonidentity/nonconstant fits satisfy the entire tied energy domain.
Source: new restricted regressions after arXiv:1602.02068v2, Eq. (1). -/
theorem taskOutputTiedPair_mem (edge : Fin 2) :
    ((taskEnergyParameters edge).1, taskEnergyTarget edge) ∈
      outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
        (fun _ => (1 / 24 : ℝ)) (1 / 24) 0 :=
  ⟨taskPeriodicEnergyPair_mem edge, (taskOutputTied_edges edge).symm⟩

/-- Actual width-three sparsemax with all common values learned fits each ordinary answer pattern.
Source: genuine constant-width inverse decoding after arXiv:1602.02068v2, Eq. (1). -/
theorem taskOutputTied_forward_exact (edge : Fin 2) :
    periodicEnergyForward (fun j : Fin 3 => j)
      ((taskEnergyParameters edge).1, taskEnergyTarget edge) = taskEnergyTarget edge :=
  periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _ (by norm_num) (taskOutputTiedPair_mem edge).1

/-- The recovered original common values are the genuinely nonconstant answer tables.
Source: the original inverse value decoder after arXiv:1602.02068v2, Eq. (1). -/
theorem taskOutputTied_values_recovered (edge : Fin 2) :
    periodicMemoryValues (taskEnergyParameters edge).1 (taskEnergyTarget edge) =
      taskEnergyTarget edge := by
  have h := taskEnergyValues_recovered edge
  unfold periodicMemoryValues
  rw [periodicMemoryAttention_normalized (3 / 4) _ (by norm_num)
    (taskEnergyParameters_mem edge).1]
  unfold recoverMemoryValues at h
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num)
    (taskEnergyParameters_mem edge)] at h
  exact h

/-- The two ordinary answer patterns change actual width-three sparsemax attention.
Source: nontrivial affine-sharing instances following sparsemax Eq. (1). -/
theorem taskOutputTied_attention_ne :
    periodicMemoryAttention (taskEnergyParameters 0).1 ≠
      periodicMemoryAttention (taskEnergyParameters 1).1 := by
  have h := taskEnergySelected_attention_ne
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem 0),
    incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) (taskEnergyParameters_mem 1)] at h
  rw [periodicMemoryAttention_normalized (3 / 4) _ (by norm_num) (taskEnergyParameters_mem 0).1,
    periodicMemoryAttention_normalized (3 / 4) _ (by norm_num) (taskEnergyParameters_mem 1).1]
  exact h

/-- Both attention choices retain positive genuine mixing rather than an identity bypass.
Source: the two restricted regression witnesses following sparsemax Eq. (1). -/
theorem taskOutputTied_attention_ne_one (edge : Fin 2) :
    periodicMemoryAttention (taskEnergyParameters edge).1 ≠ 1 := by
  have h := taskEnergySelected_attention_ne_one edge
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num)
    (taskEnergyParameters_mem edge)] at h
  rw [periodicMemoryAttention_normalized (3 / 4) _ (by norm_num) (taskEnergyParameters_mem edge).1]
  exact h

/-- The physical query table really changes between the two ordinary-answer fits.
Source: genuine parameter-sharing Q/K examples before sparsemax Eq. (1). -/
theorem taskOutputTied_query_ne :
    periodicMemoryQuery (taskEnergyParameters 0).1 (0 : Fin 3) ≠
      periodicMemoryQuery (taskEnergyParameters 1).1 (0 : Fin 3) := by
  intro h
  have he := congrFun h 1
  norm_num [periodicMemoryQuery, localMemoryNeighbours, localSlotClamp, periodicMemoryClass,
    periodicMemoryScale, localIncidentWeight, taskEnergyParameters, Fin.sum_univ_succ,
    localMemoryCore_scores_apply, Fintype.sum_option, localMemoryWeights,
    localMemoryPermutation, Equiv.swap_apply_def] at he

/-- The same ordinary answer change also changes physical keys; keys are not frozen anchors.
Source: genuine bounded parameter-sharing Q/K examples before sparsemax Eq. (1). -/
theorem taskOutputTied_key_ne :
    periodicMemoryKey (taskEnergyParameters 0).1 (0 : Fin 3) ≠
      periodicMemoryKey (taskEnergyParameters 1).1 (0 : Fin 3) := by
  intro h
  have he := congrFun h 0
  norm_num [periodicMemoryKey, periodicMemoryScale, periodicMemoryClass,
    localIncidentWeight, taskEnergyParameters, Fin.sum_univ_succ] at he

/-- Both genuinely different trained points attain zero ordinary answer error.
Source: exact sparsemax/common-value fitting under the affine tie after §2.5. -/
theorem taskOutputTied_loss_zero (edge : Fin 2) :
    periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget edge)
      ((taskEnergyParameters edge).1, taskEnergyTarget edge) = 0 := by
  unfold periodicEnergySquaredError
  rw [taskOutputTied_forward_exact edge]
  exact (matrixOutputError_eq_zero _ _).mpr rfl

/-- Each answer-fit point is an attained ordinary-error minimum on the same tied domain.
Source: zero ordinary error under the new sharing restriction after sparsemax §2.5. -/
theorem taskOutputTied_isMinOn (edge : Fin 2) :
    IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget edge))
      (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
        (fun _ => (1 / 24 : ℝ)) (1 / 24) 0)
      ((taskEnergyParameters edge).1, taskEnergyTarget edge) := by
  intro x hx
  change periodicEnergySquaredError _ _ _ ≤ periodicEnergySquaredError _ _ x
  rw [taskOutputTied_loss_zero edge]
  exact periodicEnergySquaredError_nonneg _ _ x

/-- A support-changing midpoint also inhabits the full tied energy domain.
Source: convex feasibility in the new restricted chart after sparsemax Eq. (1). -/
theorem taskOutputTied_midpoint_mem :
    (1 / 2 : ℝ) • ((taskEnergyParameters 0).1, taskEnergyTarget 0) +
      (1 / 2 : ℝ) • ((taskEnergyParameters 1).1, taskEnergyTarget 1) ∈
      outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
        (fun _ => (1 / 24 : ℝ)) (1 / 24) 0 :=
  outputTiedEnergyDomain_convex _ _ _ _ _ _ _ _
    (taskOutputTiedPair_mem 0) (taskOutputTiedPair_mem 1)
    (by norm_num) (by norm_num) (by norm_num)

/-- At the feasible midpoint both learned local edges are positive.
Source: a concrete support change in the new tied sparsemax chart after Eq. (1). -/
theorem taskOutputTied_midpoint_edges (e : Fin 2) :
    ((1 / 2 : ℝ) • (taskEnergyParameters 0).1 +
      (1 / 2 : ℝ) • (taskEnergyParameters 1).1) e = 1 / 16 := by
  fin_cases e <;> norm_num [taskEnergyParameters]

/-- The tied energy domain contains genuinely different learned answer/geometry points.
Source: the new nontrivial architectural instances following sparsemax Eq. (1). -/
theorem taskOutputTied_domain_nonsingleton :
    ∃ x ∈ outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
      (fun _ => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1),
    ∃ y ∈ outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
      (fun _ => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1), x ≠ y := by
  refine ⟨((taskEnergyParameters 0).1, taskEnergyTarget 0),
    taskOutputTiedPair_mem 0, ((taskEnergyParameters 1).1, taskEnergyTarget 1),
    taskOutputTiedPair_mem 1, ?_⟩
  intro h
  have he := congrArg (fun p : (Fin 2 → ℝ) × Matrix (Fin 3) (Fin 1) ℝ => p.2 0 0) h
  norm_num [taskEnergyTarget, Matrix.of_apply] at he

end Transformer.GPTMini.Sparsemax
