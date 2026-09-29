/-
# Explicit attention projection matrices

Arora et al., arXiv:2312.04927v1, Appendix Proposition
`prop: app-attention`, matrices `W¹_V`, `W²_Q`, `W²_K`, `W²_V`.
Feature coordinates are represented as a three-slot index and a vocabulary
coordinate, equivalent targetSlot `3c` scalar coordinates. The second-layer display
interchanges the `Q` and `K` matrix names once; the matrices themselves
select the query and key slots as stated here.
-/

import Transformer.Zoology.Appendix_TwoLayerAttention

open scoped BigOperators

namespace Transformer.Zoology

/-- A feature coordinate in one of the three `c`-dimensional slots.
Source: Appendix `rem: attn-input`, `d=3c`. -/
abbrev TripleFeature (c : ℕ) := Fin 3 × Fin c

/-- A sparse projection matrix that copies the `sourceSlot` slot to the `targetSlot`
slot, preserving vocabulary coordinates. Source: Appendix Proposition
`prop: app-attention`, displayed projection matrices. -/
def slotSelectionWeight {c : ℕ} (sourceSlot targetSlot : Fin 3)
    (source target : TripleFeature c) : ℝ :=
  if source.1 = sourceSlot ∧ target.1 = targetSlot ∧ source.2 = target.2 then 1 else 0

/-- Ordinary row-vector by matrix multiplication in the `3c` feature
space. Source: Appendix Proposition `prop: app-attention`, `uW`. -/
def applyFeatureWeight {c : ℕ} (u : TripleFeature c → ℝ)
    (W : TripleFeature c → TripleFeature c → ℝ)
    (target : TripleFeature c) : ℝ :=
  ∑ source, u source * W source target

/-- A slot-selection matrix copies exactly one input slot into one output
slot. Source: Appendix Proposition `prop: app-attention`, all displayed
projection matrices. -/
theorem apply_slotSelectionWeight {c : ℕ} (u : TripleFeature c → ℝ)
    (sourceSlot targetSlot : Fin 3) (target : TripleFeature c) :
    applyFeatureWeight u (slotSelectionWeight sourceSlot targetSlot) target =
      if target.1 = targetSlot then u (sourceSlot, target.2) else 0 := by
  classical
  unfold applyFeatureWeight
  rw [Finset.sum_eq_single (sourceSlot, target.2)]
  · by_cases ht : target.1 = targetSlot <;>
      simp [slotSelectionWeight, ht]
  · intro source _ hne
    have hnot : ¬(source.1 = sourceSlot ∧ source.2 = target.2) := by
      rintro ⟨hfrom, hcoord⟩
      exact hne (Prod.ext hfrom hcoord)
    by_cases ht : target.1 = targetSlot <;>
      simp [slotSelectionWeight, ht, hnot]
  · intro h
    exact (h (Finset.mem_univ (sourceSlot, target.2))).elim

/-- The raw `key,value,query` one-hot encoding at each triple position.
Source: Appendix `rem: attn-input`, input display. -/
def rawTripleFeature {n c : ℕ} (x : MQARInstance n c)
    (position : TriplePosition n) (feature : TripleFeature c) : ℝ :=
  if position.2 = 0 then
    if feature.1 = 0 then oneHot (x.key position.1) feature.2 else 0
  else if position.2 = 1 then
    if feature.1 = 1 then oneHot (x.value position.1) feature.2 else 0
  else if feature.1 = 2 then oneHot (x.query position.1) feature.2 else 0

/-- The first layer's `W¹_V` indeed isolates the value embedding.
Source: Appendix Proposition `prop: app-attention`, first-layer projection. -/
theorem first_value_matrix_correct {n c : ℕ} (x : MQARInstance n c)
    (position : TriplePosition n) (v : Fin c) :
    applyFeatureWeight (rawTripleFeature x position)
      (slotSelectionWeight 1 1) (1, v) = rawValueProjection x position v := by
  rw [apply_slotSelectionWeight]
  rcases position with ⟨j, slot⟩
  fin_cases slot <;> simp [rawTripleFeature, rawValueProjection]

/-- The residual input after the first layer's value shift.
Source: Appendix Proposition `prop: app-attention`, displayed `u²`. -/
def mergedTripleFeature {n c : ℕ} (x : MQARInstance n c)
    (position : TriplePosition n) (feature : TripleFeature c) : ℝ :=
  rawTripleFeature x position feature +
    if feature.1 = 1 then shiftedValueOutput x position feature.2 else 0

/-- The second layer's `W²_Q` copies a query into the score slot.
Source: Appendix Proposition `prop: app-attention`, second-layer matrices. -/
theorem second_query_matrix_correct {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) :
    applyFeatureWeight (mergedTripleFeature x (i, 2))
      (slotSelectionWeight 2 0) (0, v) = oneHot (x.query i) v := by
  rw [apply_slotSelectionWeight]
  simp [mergedTripleFeature, rawTripleFeature]

/-- The second layer's `W²_K` copies a key into the score slot.
Source: Appendix Proposition `prop: app-attention`, second-layer matrices. -/
theorem second_key_matrix_correct {n c : ℕ} (x : MQARInstance n c)
    (j : Fin n) (v : Fin c) :
    applyFeatureWeight (mergedTripleFeature x (j, 0))
      (slotSelectionWeight 0 0) (0, v) = oneHot (x.key j) v := by
  rw [apply_slotSelectionWeight]
  simp [mergedTripleFeature, rawTripleFeature]

/-- The second layer's `W²_V` reads the value aligned with a key.
Source: Appendix Proposition `prop: app-attention`, second-layer matrices. -/
theorem second_value_matrix_correct {n c : ℕ} (x : MQARInstance n c)
    (j : Fin n) (v : Fin c) :
    applyFeatureWeight (mergedTripleFeature x (j, 0))
      (slotSelectionWeight 1 0) (0, v) = oneHot (x.value j) v := by
  rw [apply_slotSelectionWeight]
  simp [mergedTripleFeature, rawTripleFeature, shiftedValueOutput_key]

/-- Storing three dense `3c × 3c` projection matrices per layer uses at
most `54c²` scalar slots over the two layers. The fixed bias is excluded,
as in the paper's count. Source: Appendix Proposition `prop: app-attention`,
parameter-count paragraph. -/
theorem attention_projection_slots (c : ℕ) :
    2 * 3 * (Fintype.card (TripleFeature c)) ^ 2 = 54 * c ^ 2 := by
  simp [TripleFeature, Fintype.card_prod]
  ring

/-- A direct implementation budget: six dense projections over `N` tokens,
and score/output matrix products in two layers. This is a scalar-operation
accounting model for the algorithm, separate from the functional theorem.
Source: Appendix Proposition `prop: app-attention`, runtime paragraph. -/
def naiveAttentionWork (N c : ℕ) : ℕ :=
  6 * N * (3 * c) ^ 2 + 4 * N ^ 2 * (3 * c)

/-- The direct two-layer accounting has the paper's claimed polynomial
terms, `N c²` for projections and `N² c` for attention products.
Source: Appendix Proposition `prop: app-attention`, runtime paragraph. -/
theorem naiveAttentionWork_eq (N c : ℕ) :
    naiveAttentionWork N c = 54 * N * c ^ 2 + 12 * N ^ 2 * c := by
  unfold naiveAttentionWork
  ring

/-- The second attention layer as a matrix expression, with the strict
prior-pair mask needed for the task. Its query comes from the query slot,
its key comes from the key slot, and its value comes from the residual
input after the first-layer shift. Source: Appendix Proposition
`prop: app-attention`, corrected second-layer calculation. -/
def projectedSecondLayerOutput {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) : ℝ :=
  ∑ position : TriplePosition n,
    (if position.2 = 0 ∧ position.1 < i then
      ∑ t : Fin c,
        applyFeatureWeight (mergedTripleFeature x (i, 2))
          (slotSelectionWeight 2 0) (0, t) *
        applyFeatureWeight (mergedTripleFeature x position)
          (slotSelectionWeight 0 0) (0, t)
    else 0) *
      applyFeatureWeight (mergedTripleFeature x position)
        (slotSelectionWeight 1 0) (0, v)

/-- The matrix expression reduces exactly to the corrected two-stage
attention formula, including the first-layer value shift. Source: Appendix
Proposition `prop: app-attention`, with the explicit prior-pair mask. -/
theorem projectedSecondLayerOutput_eq {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) :
    projectedSecondLayerOutput x i v = twoLayerAttentionOutput x i v := by
  classical
  unfold projectedSecondLayerOutput twoLayerAttentionOutput
  apply Finset.sum_congr rfl
  intro position _
  by_cases h : position.2 = 0 ∧ position.1 < i
  · rcases position with ⟨j, slot⟩
    have hslot : slot = 0 := h.1
    subst slot
    simp only [secondLayerScore, h]
    rw [second_value_matrix_correct, alignedValue_key]
    congr 1
    apply Finset.sum_congr rfl
    intro t _
    rw [second_query_matrix_correct, second_key_matrix_correct]
  · simp [secondLayerScore, h]

/-- The explicitly projected and causally masked two-layer model solves
MQAR for unique keys. Source: Appendix Proposition `prop: app-attention`,
corrected functional part; resource counts are separate above. -/
theorem projectedSecondLayer_solves_mqar {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (i : Fin n) (v : Fin c) :
    projectedSecondLayerOutput x i v = expectedPairedAnswer x i v := by
  rw [projectedSecondLayerOutput_eq]
  exact twoLayerAttention_solves_mqar x hx i v

/-- The unique-key condition in the corrected matrix theorem is
satisfiable. -/
example : ∃ x : MQARInstance 2 2, UniqueKeys x := by
  refine ⟨{key := id, value := id, query := id}, ?_⟩
  intro a b h
  exact h

end Transformer.Zoology
