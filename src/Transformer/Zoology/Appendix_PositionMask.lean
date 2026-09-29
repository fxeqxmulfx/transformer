/-
# A missing position mask in the Coyote MQAR construction

Arora et al., arXiv:2312.04927v1, Appendix `sec: data-dep-ar`, equation
`eq: kqv-projections`, and §4 equation `eq: coyote-recursion`. The appendix
sets `Linear_K(u)=K`, where `K` keeps key positions and zeros query positions.
For the appendix's untagged one-hot input of width `c`, a shared rowwise
matrix `W` cannot perform that projection when a query repeats a key.
Position-dependent gating or tagged channels could repair this local step;
the result below does not refute the Coyote existence theorem.
-/

import Transformer.Zoology.Appendix_TwoLayerAttention

open scoped BigOperators

namespace Transformer.Zoology

/-- The raw triple sequence, with all three token roles embedded in the
same `c` feature channels. Source: Appendix `sec: data-dep-ar`, Setup and
equation `eq: kqv-projections`. -/
def untaggedTripleInput {n c : ℕ} (x : MQARInstance n c)
    (position : TriplePosition n) (q : Fin c) : ℝ :=
  if position.2 = 0 then oneHot (x.key position.1) q
  else if position.2 = 1 then oneHot (x.value position.1) q
  else oneHot (x.query position.1) q

/-- The key-only row mask used by the appendix's first Coyote stage.
Source: Appendix equation `eq: kqv-projections`. -/
def untaggedKeyMask {n c : ℕ} (x : MQARInstance n c)
    (position : TriplePosition n) (q : Fin c) : ℝ :=
  if position.2 = 0 then oneHot (x.key position.1) q else 0

/-- A position-dependent gate repairs the key projection at the level of
the triple sequence. Source: Appendix equation `eq: kqv-projections`, with
the required position mask made explicit. -/
theorem untaggedKeyMask_eq_positionGate {n c : ℕ}
    (x : MQARInstance n c) (position : TriplePosition n) (q : Fin c) :
    untaggedKeyMask x position q =
      (if position.2 = 0 then (1 : ℝ) else 0) *
        untaggedTripleInput x position q := by
  by_cases h : position.2 = 0 <;>
    simp [untaggedKeyMask, untaggedTripleInput, h]

/-- Application of a Coyote-style shared feature projection to every raw
triple position. Source: §4 equation `eq: coyote-recursion`, `u W`. -/
def sharedTripleProjection {n c : ℕ} (x : MQARInstance n c)
    (W : Fin c → Fin c → ℝ) (position : TriplePosition n)
    (q : Fin c) : ℝ :=
  ∑ t : Fin c, untaggedTripleInput x position t * W t q

/-- A repeated key and query have identical untagged rows, so no shared
rowwise linear projection can keep the key row and zero the query row.
This refutes the local equality `Linear_K(u)=K` as an implementation by
the shared `W` from the Coyote definition. Source: Appendix equation
`eq: kqv-projections` versus §4 equation `eq: coyote-recursion`. -/
theorem no_shared_untagged_key_projection {n c : ℕ}
    (x : MQARInstance n c) (j i : Fin n)
    (hmatch : x.key j = x.query i) :
    ¬ ∃ W : Fin c → Fin c → ℝ,
      ∀ position : TriplePosition n, ∀ q : Fin c,
        sharedTripleProjection x W position q =
          untaggedKeyMask x position q := by
  rintro ⟨W, hW⟩
  have hsame : sharedTripleProjection x W (j, 0) (x.key j) =
      sharedTripleProjection x W (i, 2) (x.key j) := by
    unfold sharedTripleProjection
    apply Finset.sum_congr rfl
    intro t _
    simp [untaggedTripleInput, hmatch]
  have hkey := hW (j, 0) (x.key j)
  have hquery := hW (i, 2) (x.key j)
  rw [hkey, hquery] at hsame
  simp [untaggedKeyMask, oneHot] at hsame

/-- The obstruction occurs on a valid two-pair MQAR input with distinct
keys and a query recalling the first pair. -/
example : ∃ x : MQARInstance 2 2,
    UniqueKeys x ∧ PriorAnswer x 1 0 ∧
      ¬ ∃ W : Fin 2 → Fin 2 → ℝ,
        ∀ position : TriplePosition 2, ∀ q : Fin 2,
          sharedTripleProjection x W position q =
            untaggedKeyMask x position q := by
  let x : MQARInstance 2 2 := {
    key := id
    value := id
    query := fun _ => 0
  }
  refine ⟨x, ?_, ?_, ?_⟩
  · intro a b h
    exact h
  · exact ⟨0, by decide, rfl, rfl⟩
  · exact no_shared_untagged_key_projection x 0 1 rfl

end Transformer.Zoology
