import Transformer.GPTMini.Sparsemax.ObservedMemoryPreferences

/-!
# Observation changes produce different selected attention and embedding norms

Derived witnesses for the data criterion before arXiv:1602.02068v2, Eq. (1).
Two real prototype tables, `(0,1)` and `(0,3)`, use the same metric, order,
scale and independent nonconstant feature maps. Their references are both
strictly inside the incident budget and norm cap. Unique projection therefore
recovers them exactly. Actual sparsemax edge weights change from 1/16 to
1/32, and both genuine query and key squared norms change.

These examples verify dependence on observations after constrained selection,
not just dependence of an unprojected preference formula. They do not claim
semantic usefulness of the metric or identification by output-only targets.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- The same data criterion applied to two nearby registered observations.
Source: a concrete observation-derived preference for arXiv:1602.02068v2, Eq. (1). -/
def observedMemoryNearReference : LocalMemoryParameters 1 :=
  observedMemoryReference (fun x y : ℝ => |x - y|) (fun j : Fin 2 => (j.val : ℝ))
    (fun x => fun _ : Fin 1 => x / 4) (fun x => fun _ : Fin 1 => x / 8) (1 / 8)

/-- Change the second observation while keeping every criterion choice fixed.
Source: the same derived data criterion before arXiv:1602.02068v2, Eq. (1). -/
def observedMemoryFarReference : LocalMemoryParameters 1 :=
  observedMemoryReference (fun x y : ℝ => |x - y|) (fun j : Fin 2 => (3 * j.val : ℝ))
    (fun x => fun _ : Fin 1 => x / 4) (fun x => fun _ : Fin 1 => x / 8) (1 / 8)

/-- The nearby-data preferences are genuine nonzero edge and separate norm coordinates.
Source: actual observed distances and feature energies for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryNearReference_coordinates :
    observedMemoryNearReference.1 0 = 1 / 16 ∧
    observedMemoryNearReference.2 (Sum.inl 1) = 1 / 16 ∧
    observedMemoryNearReference.2 (Sum.inr 1) = 1 / 64 := by
  norm_num [observedMemoryNearReference, observedMemoryReference, Fin.sum_univ_one]

/-- Changed observations change all three preferred coordinates at the second slot.
Source: the computed data criterion for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryFarReference_coordinates :
    observedMemoryFarReference.1 0 = 1 / 32 ∧
    observedMemoryFarReference.2 (Sum.inl 1) = 9 / 16 ∧
    observedMemoryFarReference.2 (Sum.inr 1) = 9 / 64 := by
  norm_num [observedMemoryFarReference, observedMemoryReference, Fin.sum_univ_one]

/-- The two data tables give distinct complete geometry references.
Source: computed observation dependence before arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryExamples_references_ne : observedMemoryNearReference ≠ observedMemoryFarReference := by
  intro h
  have he := congrArg (fun p : LocalMemoryParameters 1 => p.1 0) h
  rw [observedMemoryNearReference_coordinates.1, observedMemoryFarReference_coordinates.1] at he
  norm_num at he

/-- The nearby-data reference satisfies the entire actual structural domain.
Source: a nonconstant data-derived feasible witness for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryNearReference_mem :
    observedMemoryNearReference ∈ incidentMemoryParameterDomain 1 4 (3 / 4) := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro e
    fin_cases e
    norm_num [observedMemoryNearReference, observedMemoryReference]
  · intro i
    fin_cases i <;> norm_num [localIncidentWeight, observedMemoryNearReference,
      observedMemoryReference, Fin.sum_univ_one]
  · intro x
    rcases x with i | i <;> fin_cases i <;>
      norm_num [observedMemoryNearReference, observedMemoryReference, Fin.sum_univ_one]

/-- Changed data still satisfy every incident budget and independent norm cap.
Source: the second observation-derived feasible witness for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryFarReference_mem :
    observedMemoryFarReference ∈ incidentMemoryParameterDomain 1 4 (3 / 4) := by
  refine ⟨⟨?_, ?_⟩, ?_⟩
  · intro e
    fin_cases e
    norm_num [observedMemoryFarReference, observedMemoryReference]
  · intro i
    fin_cases i <;> norm_num [localIncidentWeight, observedMemoryFarReference,
      observedMemoryReference, Fin.sum_univ_one]
  · intro x
    rcases x with i | i <;> fin_cases i <;>
      norm_num [observedMemoryFarReference, observedMemoryReference, Fin.sum_univ_one]

/-- Constrained selection recovers both different feasible observation-derived references.
Source: the unique complete-coordinate minimum for arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryExamples_selected :
    incidentMemorySelectedParameters 4 (3 / 4) observedMemoryNearReference (by norm_num) (by norm_num) =
      observedMemoryNearReference ∧
    incidentMemorySelectedParameters 4 (3 / 4) observedMemoryFarReference (by norm_num) (by norm_num) =
      observedMemoryFarReference :=
  ⟨incidentMemorySelectedParameters_eq_reference _ _ _ (by norm_num) (by norm_num)
      observedMemoryNearReference_mem,
    incidentMemorySelectedParameters_eq_reference _ _ _ (by norm_num) (by norm_num)
      observedMemoryFarReference_mem⟩

/-- Actual sparsemax edge weights differ after observation-derived constrained selection.
Source: variational arXiv:1602.02068v2, Eq. (1), on both proved data-derived Grams. -/
theorem observedMemoryExamples_selected_attention :
    memoryGramAttention (localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4)
      observedMemoryNearReference (by norm_num) (by norm_num))) 0 1 = 1 / 16 ∧
    memoryGramAttention (localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4)
      observedMemoryFarReference (by norm_num) (by norm_num))) 0 1 = 1 / 32 := by
  rw [observedMemoryExamples_selected.1, observedMemoryExamples_selected.2]
  rw [incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) observedMemoryNearReference_mem,
    incidentMemoryAttention_normalized 4 (3 / 4) _ (by norm_num) observedMemoryFarReference_mem]
  rw [localMemoryCore_scores_apply, localMemoryCore_scores_apply,
    Fintype.sum_option, Fintype.sum_option]
  norm_num [localMemoryWeights, observedMemoryNearReference, observedMemoryFarReference,
    observedMemoryReference, Fin.sum_univ_one, localMemoryPermutation, Equiv.swap_apply_def]

/-- Actual attention, after constrained selection, changes when the observed distance changes.
Source: the data-dependent actual arXiv:1602.02068v2, Eq. (1) witness. -/
theorem observedMemoryExamples_attention_ne :
    memoryGramAttention (localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4)
      observedMemoryNearReference (by norm_num) (by norm_num))) 0 1 ≠
    memoryGramAttention (localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4)
      observedMemoryFarReference (by norm_num) (by norm_num))) 0 1 := by
  rw [observedMemoryExamples_selected_attention.1, observedMemoryExamples_selected_attention.2]
  norm_num

/-- Both genuine query and key squared norms change under the observation-derived selected geometry.
Source: actual embedding Gram diagonals before arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryExamples_selected_norms :
    localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4) observedMemoryNearReference
      (by norm_num) (by norm_num)) (Sum.inl 1) (Sum.inl 1) ≠
    localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4) observedMemoryFarReference
      (by norm_num) (by norm_num)) (Sum.inl 1) (Sum.inl 1) ∧
    localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4) observedMemoryNearReference
      (by norm_num) (by norm_num)) (Sum.inr 1) (Sum.inr 1) ≠
    localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4) observedMemoryFarReference
      (by norm_num) (by norm_num)) (Sum.inr 1) (Sum.inr 1) := by
  rw [observedMemoryExamples_selected.1, observedMemoryExamples_selected.2,
    localMemoryGram_diagonal, localMemoryGram_diagonal,
    localMemoryGram_diagonal, localMemoryGram_diagonal]
  rw [observedMemoryNearReference_coordinates.2.1, observedMemoryFarReference_coordinates.2.1,
    observedMemoryNearReference_coordinates.2.2, observedMemoryFarReference_coordinates.2.2]
  norm_num

/-- Data-selected attention has a genuine inverse without supplying original Q/K coordinates.
Source: the proved local-budget inverse for actual arXiv:1602.02068v2, Eq. (1). -/
theorem observedMemoryExamples_inverse :
    IsUnit (memoryGramAttention (localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4)
      observedMemoryNearReference (by norm_num) (by norm_num)))).det ∧
    IsUnit (memoryGramAttention (localMemoryGram (incidentMemorySelectedParameters 4 (3 / 4)
      observedMemoryFarReference (by norm_num) (by norm_num)))).det := by
  rw [observedMemoryExamples_selected.1, observedMemoryExamples_selected.2]
  exact ⟨incidentMemoryAttention_det_unit _ _ _ (by norm_num) observedMemoryNearReference_mem,
    incidentMemoryAttention_det_unit _ _ _ (by norm_num) observedMemoryFarReference_mem⟩

end Transformer.GPTMini.Sparsemax
