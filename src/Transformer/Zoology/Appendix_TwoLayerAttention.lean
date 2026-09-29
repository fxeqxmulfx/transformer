/-
# The two attention stages for one-hot MQAR triples

Arora et al., arXiv:2312.04927v1, Appendix Proposition
`prop: app-attention`. This module calculates the first layer's fixed
value-to-key shift from a bias matrix and then the second layer's exact
one-hot score over all triple positions. A strict prior-pair mask is needed:
the unmasked display in the paper also reads current and future keys.
-/

import Transformer.Zoology.Section4_PairedRecall

open scoped BigOperators

namespace Transformer.Zoology

/-- A position in the key-value-query triple sequence. The second component
is zero for a key, one for a value, and two for a query. Source: Appendix
`rem: attn-input`, equation `eq: kqv-indices`. -/
abbrev TriplePosition (n : ℕ) := Fin n × Fin 3

/-- The first layer's fixed attention bias moves a value from its slot to
the immediately preceding key slot. Source: Appendix Proposition
`prop: app-attention`, first-layer matrix `B¹`. -/
def valueToKeyBias {n : ℕ} (target source : TriplePosition n) : ℝ :=
  if target.2 = 0 ∧ source = (target.1, 1) then 1 else 0

/-- Projection of raw one-hot triples onto the value channel.
Source: Appendix Proposition `prop: app-attention`, matrix `W¹_V`. -/
def rawValueProjection {n c : ℕ} (x : MQARInstance n c)
    (source : TriplePosition n) (v : Fin c) : ℝ :=
  if source.2 = 1 then oneHot (x.value source.1) v else 0

/-- First-layer output obtained by multiplying the bias matrix with the
value projection. Source: Appendix Proposition `prop: app-attention`, first
layer calculation `O¹ = B¹V¹`. -/
def shiftedValueOutput {n c : ℕ} (x : MQARInstance n c)
    (target : TriplePosition n) (v : Fin c) : ℝ :=
  ∑ source : TriplePosition n,
    valueToKeyBias target source * rawValueProjection x source v

/-- Every key slot receives its following value by the fixed bias shift.
Source: Appendix Proposition `prop: app-attention`, first-layer output. -/
theorem shiftedValueOutput_key {n c : ℕ} (x : MQARInstance n c)
    (j : Fin n) (v : Fin c) :
    shiftedValueOutput x (j, 0) v = oneHot (x.value j) v := by
  classical
  unfold shiftedValueOutput
  rw [Finset.sum_eq_single (j, 1)]
  · simp [valueToKeyBias, rawValueProjection]
  · intro source _ hsource
    simp [valueToKeyBias, hsource]
  · intro h
    exact (h (Finset.mem_univ (j, 1))).elim

/-- The first layer outputs zero outside the key slots.
Source: Appendix Proposition `prop: app-attention`, first-layer output. -/
theorem shiftedValueOutput_not_key {n c : ℕ} (x : MQARInstance n c)
    (target : TriplePosition n) (h : target.2 ≠ 0) (v : Fin c) :
    shiftedValueOutput x target v = 0 := by
  simp [shiftedValueOutput, valueToKeyBias, h]

/-- The value channel after adding the first layer's output to the raw
input. Source: Appendix Proposition `prop: app-attention`, residual input
`u²`. -/
def alignedValue {n c : ℕ} (x : MQARInstance n c)
    (position : TriplePosition n) (v : Fin c) : ℝ :=
  rawValueProjection x position v + shiftedValueOutput x position v

/-- At a key slot the residual contains the associated value.
Source: Appendix Proposition `prop: app-attention`, displayed `u²`. -/
theorem alignedValue_key {n c : ℕ} (x : MQARInstance n c)
    (j : Fin n) (v : Fin c) :
    alignedValue x (j, 0) v = oneHot (x.value j) v := by
  simp [alignedValue, rawValueProjection, shiftedValueOutput_key]

/-- A second-layer score from query `i` to a raw triple position. The key
score is a one-hot inner product. The strict prior-pair indicator corrects
the appendix's unmasked `QKᵀV` expression. Source: Appendix Proposition
`prop: app-attention`, second-layer score with explicit correction. -/
def secondLayerScore {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (position : TriplePosition n) : ℝ :=
  if position.2 = 0 ∧ position.1 < i then
    ∑ t : Fin c, oneHot (x.query i) t * oneHot (x.key position.1) t
  else 0

/-- The corrected second attention layer sums over all raw triple positions.
Source: Appendix Proposition `prop: app-attention`, second-layer output. -/
def twoLayerAttentionOutput {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) : ℝ :=
  ∑ position : TriplePosition n,
    secondLayerScore x i position * alignedValue x position v

/-- The full two-stage expression agrees with attention over aligned
key-value pairs. Source: Appendix Proposition `prop: app-attention`, with
the strict prior-pair mask and explicit first-layer value shift. -/
theorem twoLayerAttention_eq_paired {n c : ℕ} (x : MQARInstance n c)
    (i : Fin n) (v : Fin c) :
    twoLayerAttentionOutput x i v = pairedAttentionOutput x i v := by
  classical
  have huniv : (Finset.univ : Finset (TriplePosition n)) =
      (Finset.univ : Finset (Fin n)) ×ˢ (Finset.univ : Finset (Fin 3)) := by
    ext p
    simp
  unfold twoLayerAttentionOutput pairedAttentionOutput
  rw [huniv, Finset.sum_product]
  apply Finset.sum_congr rfl
  intro j _
  rw [Fin.sum_univ_three]
  simp [secondLayerScore, alignedValue_key, oneHot,
    eq_comm]

/-- Consequently every query in a unique-key instance receives the
indicator of its correct prior value. Source: Appendix Proposition
`prop: app-attention`, corrected mask and unique-key hypothesis. -/
theorem twoLayerAttention_solves_mqar {n c : ℕ} (x : MQARInstance n c)
    (hx : UniqueKeys x) (i : Fin n) (v : Fin c) :
    twoLayerAttentionOutput x i v = expectedPairedAnswer x i v := by
  rw [twoLayerAttention_eq_paired]
  exact paired_attention_solves_mqar x hx i v

/-- Distinct keys and a valid prior query can occur in the two-layer model. -/
example : ∃ x : MQARInstance 2 2, UniqueKeys x ∧
    PriorAnswer x 1 0 := by
  let x : MQARInstance 2 2 := {
    key := id
    value := id
    query := fun _ => 0
  }
  refine ⟨x, ?_, ?_⟩
  · intro a b h
    exact h
  · exact ⟨0, by decide, rfl, rfl⟩

end Transformer.Zoology
