module

public import Transformer.Grokking.Convex.ConvexDepth.CoordinatePrograms.Witness

/-!
# Uncovered directions and inhabited complete-head guarantees

Source: `review/convex/moments.md`, CM57. Omitting role observations permits
opposite feasible raw winners. Nonzero examples instantiate the identification,
complete budget and ordinary full/minibatch results, with repeated record IDs.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.CoordinatePrograms

open scoped BigOperators

/-- Quantity-only observations leave the first role direction unobserved,
CM57. The two rows have distinct known coordinate answers. -/
noncomputable def holeRows (a : Fin 2) : Fin 5 → ℝ := exampleFeatures 0 (![0, 1] a)

/-- Only the two quantity coordinates are observed in this control, CM57. -/
noncomputable def holeValues : Fin 2 → ℝ := ![1 / 4, 1 / 2]

/-- A freely varying unseen role coefficient while quantity and copy
coefficients stay fixed, CM57. -/
noncomputable def completion (r : ℝ) : Fin 6 → ℝ := ![2, 1, r, 0, 4, 4]

/-- The unseen role is outside the observed-plus-copy span, CM57. This is an
input-feature fact, independent of any target assigned to that role. -/
theorem hole_not_covered : ¬ ∃ weights : Option (Fin 2) → ℝ,
    ∀ j, exampleFeatures 0 3 j = ∑ a, weights a * augmentedRows holeRows a j := by
  rintro ⟨weights, h⟩
  have hj := h 2
  norm_num [exampleFeatures, holeRows, exampleQuantity, augmentedRows, copyRow,
    Fintype.sum_option, Fin.sum_univ_succ] at hj

/-- Both completions obey the same observations, copy prior and complete
budgets. A convex feasible domain alone does not identify an uncovered role,
CM57. -/
theorem opposite_completions_feasible :
    (fun p : Unit × Fin 6 => completion 0 p.2) ∈
      domain (fun _ => holeRows) (fun _ => holeValues) (fun _ => 32) 128 1 ∧
    (fun p : Unit × Fin 6 => completion 4 p.2) ∈
      domain (fun _ => holeRows) (fun _ => holeValues) (fun _ => 32) 128 1 := by
  norm_num [domain, ParameterSharing.weightedDomain, Fintype.sum_prod_type,
    Fin.sum_univ_succ, costs, charge, completion, center, holeRows, holeValues,
    exampleFeatures, exampleQuantity, Fin.forall_fin_succ]

/-- The actual complex heads give opposite raw decisions for the uncovered
role. No target-identifying conclusion follows merely from flat features and
exact observed fit, CM57. -/
theorem opposite_raw_winners :
    headLogits (features (fun _ : Unit => exampleFeatures 0))
      (head exampleCoordinates (fun p => completion 0 p.2)) 3 4 <
    headLogits (features (fun _ : Unit => exampleFeatures 0))
      (head exampleCoordinates (fun p => completion 0 p.2)) 3 0 ∧
    headLogits (features (fun _ : Unit => exampleFeatures 0))
      (head exampleCoordinates (fun p => completion 4 p.2)) 3 0 <
    headLogits (features (fun _ : Unit => exampleFeatures 0))
      (head exampleCoordinates (fun p => completion 4 p.2)) 3 4 := by
  simp only [head_logits]
  norm_num [score, center, exampleFeatures, completion, exampleCoordinates,
    ClassGeometry.exampleCoordinates, SignedContext.normalizer, SignedContext.mass,
    bankRow, row, Fintype.sum_prod_type, Fin.sum_univ_succ]

/-- Another feasible mixture exchanges the two active masses while preserving
the distinct per-program decoders, CM57. -/
noncomputable def alternativeParameters (p : Fin 3 × Fin 6) : ℝ :=
  ![![1, 1 / 2, 0, 2, 2, 2], ![1, 2, 0, 4, 4, 4], ![0, 0, 0, 0, 0, 0]] p.1 p.2

/-- The domain admits genuine variation of program allocation, CM57. -/
theorem alternative_feasible : alternativeParameters ∈
    domain exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1 := by
  norm_num [domain, ParameterSharing.weightedDomain, Fintype.sum_prod_type,
    Fin.sum_univ_succ, costs, charge, alternativeParameters, center,
    exampleRows, exampleValues, exampleFeatures, exampleQuantity,
    exampleCoordinates, ClassGeometry.exampleCoordinates, exampleLabels,
    exampleTarget, exampleInputs, Fin.forall_fin_succ]

example : exampleParameters ≠ alternativeParameters := by
  intro h
  have hg := congrFun h (0, 5)
  norm_num [exampleParameters, alternativeParameters] at hg

example := example_unseen_prediction alternative_feasible 0 (by decide)
example := example_copy_prediction alternative_feasible 1 (by decide)

example := charge_combination (fun j => exampleParameters (0, j))
  (fun j => exampleParameters (1, j)) (1 / 2) (1 / 2) (by norm_num) (by norm_num)

example := zero_mass (fun j => exampleParameters (2, j)) 16
  (example_feasible.2.2.2.1 2) (by norm_num [exampleParameters]) 0

example := domain_convex exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1

example := dot_combination (exampleDecoder 0) (augmentedRows (exampleRows 0)) exampleWeights

example := augmented_anchors (fun j => exampleParameters (0, j)) (exampleRows 0) exampleValues
  (example_feasible.2.2.2.2.1 0) (example_feasible.2.2.2.2.2 0) none

example := augmented_realizable (exampleDecoder 0) (exampleRows 0) exampleValues
  (by norm_num [exampleDecoder])
  (fun a => example_realizable 0 (by decide) (exampleInputs a)) none

example := query_identified (fun j => exampleParameters (0, j)) (exampleRows 0) exampleValues
  (exampleDecoder 0) (exampleFeatures 0 2) (exampleCoordinates 3) exampleWeights
  (example_feasible.2.2.2.2.1 0) (example_feasible.2.2.2.2.2 0)
  (by norm_num [exampleDecoder])
  (fun a => example_realizable 0 (by decide) (exampleInputs a))
  (example_realizable 0 (by decide) 2) (example_span 0)

example := all_centers_identified exampleRows (fun _ => exampleValues) (fun _ => 16)
  example_feasible (fun p => exampleFeatures p 2) (exampleCoordinates 3)
  (example_query_coverage example_feasible)

example := score_difference exampleCoordinates (fun j => exampleParameters (0, j))
  (exampleFeatures 0 2) 3 4
  (all_centers_identified exampleRows (fun _ => exampleValues) (fun _ => 16)
    example_feasible (fun p => exampleFeatures p 2) (exampleCoordinates 3)
    (example_query_coverage example_feasible) 0)

example := mixture_strict exampleCoordinates ClassGeometry.exampleCoordinates_injective
  exampleParameters (fun p => exampleFeatures p 2) 3 4
  (by norm_num [exampleParameters, Fin.sum_univ_succ]) (by decide)
  (all_centers_identified exampleRows (fun _ => exampleValues) (fun _ => 16)
    example_feasible (fun p => exampleFeatures p 2) (exampleCoordinates 3)
    (example_query_coverage example_feasible))

example (p : Fin 6) (c : Fin 5) : ∑ k, |localExpansion exampleCoordinates p k c| ≤ costs p :=
  local_expansion_bound exampleCoordinates ClassGeometry.exampleCoordinates_bound p c

example (p : Fin 3 × Fin 6) (c : Fin 5) :
    ∑ k, |expansion exampleCoordinates p k c| ≤ costs p.2 :=
  expansion_bound exampleCoordinates ClassGeometry.exampleCoordinates_bound p c

example : head exampleCoordinates exampleParameters ∈ headBudget 128 :=
  head_budget exampleCoordinates ClassGeometry.exampleCoordinates_bound example_feasible.1

-- A genuinely imaginary coefficient of the actual complete head is nonzero.
example : head exampleCoordinates exampleParameters (0, 0, 1, true) 4 = 3 := by
  rw [head_block]
  norm_num [localHead, ParameterSharing.parameterMap, ParameterSharing.coefficientMap,
    localExpansion, source, complexFeatures, fourier4, exampleParameters,
    exampleCoordinates, ClassGeometry.exampleCoordinates, Fin.sum_univ_succ]

example : head exampleCoordinates alternativeParameters (0, 0, 1, true) 4 = 3 / 2 := by
  rw [head_block]
  norm_num [localHead, ParameterSharing.parameterMap, ParameterSharing.coefficientMap,
    localExpansion, source, complexFeatures, fourier4, alternativeParameters,
    exampleCoordinates, ClassGeometry.exampleCoordinates, Fin.sum_univ_succ]

example : (∑ p : Fin 3 × Fin 6, costs p.2 * |exampleParameters p|) = 78 ∧
    (∑ p : Fin 3 × Fin 6, costs p.2 * |alternativeParameters p|) = 78 := by
  norm_num [Fintype.sum_prod_type, Fin.sum_univ_succ, costs,
    exampleParameters, alternativeParameters]

example := loss_convex exampleCoordinates ClassGeometry.exampleCoordinates_bound
  exampleFeatures exampleInputs exampleLabels 128
example := batchLoss_convex exampleCoordinates ClassGeometry.exampleCoordinates_bound
  exampleFeatures exampleInputs exampleLabels ({0, 1, 2, 3} : Finset (Fin 5)) 128
example := batchLoss_convex exampleCoordinates ClassGeometry.exampleCoordinates_bound
  exampleFeatures exampleInputs exampleLabels ({4} : Finset (Fin 5)) 128
example := domain_loss_convex exampleCoordinates ClassGeometry.exampleCoordinates_bound
  exampleFeatures exampleInputs exampleLabels exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1
example := domain_batchLoss_convex exampleCoordinates ClassGeometry.exampleCoordinates_bound
  exampleFeatures exampleInputs exampleLabels ({4} : Finset (Fin 5))
  exampleRows (fun _ => exampleValues) (fun _ => 16) 128 1

example := identified_prediction exampleCoordinates ClassGeometry.exampleCoordinates_injective
  exampleFeatures exampleRows (fun _ => exampleValues) (fun _ => 16)
  (by norm_num : (0 : ℝ) < 1) example_feasible 2 3 4 (by decide)
  (example_query_coverage example_feasible)
example := identified_gap exampleCoordinates exampleFeatures exampleRows
  (fun _ => exampleValues) (fun _ => 16) example_feasible 2 3 4
  (example_query_coverage example_feasible)

end Transformer.Grokking.Convex.ConvexDepth.CoordinatePrograms
