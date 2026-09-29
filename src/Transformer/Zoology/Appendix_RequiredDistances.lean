/-
# The exact interaction-distance set for MQAR

Arora et al., arXiv:2312.04927v1, Appendix `sec: data-dep-ar`, Setup
and Theorem `thm: input-dep-genar`. The set below is defined from the
actual earlier query-key matches. It supplies exactly the coverage assumed
by the functional parallel-shift proof. This is an ideal data-dependent
selector, not the paper's proposed autocorrelation `Top` computation.
-/

import Transformer.Zoology.Appendix_DependentDistanceMany

namespace Transformer.Zoology

/-- Every interaction distance between a query and an earlier matching key.
Source: Appendix `sec: data-dep-ar`, Setup, distance `i-j`. -/
def requiredDistances {n c : ℕ} (x : MQARInstance n c) : Finset (Fin n) := by
  exact Finset.univ.filter fun s =>
    ∃ i j : Fin n, j < i ∧ x.key j = x.query i ∧ s = i - j

/-- Membership in the required-distance set is exactly the existence of a
matching earlier key at that distance. Source: Appendix `sec: data-dep-ar`,
Setup. -/
theorem mem_requiredDistances_iff {n c : ℕ} (x : MQARInstance n c)
    (s : Fin n) :
    s ∈ requiredDistances x ↔
      ∃ i j : Fin n, j < i ∧ x.key j = x.query i ∧ s = i - j := by
  simp [requiredDistances]

/-- All required interaction distances are positive, since each matching
key precedes the query. Source: Appendix `sec: data-dep-ar`, Setup. -/
theorem requiredDistances_positive {n c : ℕ} (x : MQARInstance n c) :
    PositiveDistances (requiredDistances x) := by
  intro s hs
  obtain ⟨i, j, hj, _, rfl⟩ := (mem_requiredDistances_iff x s).mp hs
  rw [Fin.sub_val_of_le (le_of_lt hj)]
  have hv : j.val < i.val := Fin.lt_def.mp hj
  omega

/-- Every earlier query-key match contributes its distance to the set.
Source: Appendix `sec: data-dep-ar`, Setup. -/
theorem requiredDistances_cover {n c : ℕ} (x : MQARInstance n c) :
    CoversPriorDistances x (requiredDistances x) := by
  intro i j hj hkey
  exact (mem_requiredDistances_iff x (i - j)).mpr
    ⟨i, j, hj, hkey, rfl⟩

/-- With unique keys, the ideal data-dependent distance set lets parallel
shift-and-gate lookup solve every query exactly. Source: Appendix Theorem
`thm: input-dep-genar`, functional output only. The Coyote realization,
autocorrelation selector, and complexity bound are separate claims. -/
theorem requiredDistanceLookup_solves_mqar {n c : ℕ}
    (x : MQARInstance n c) (hx : UniqueKeys x)
    (i : Fin n) (q : Fin c) :
    distanceLookupMany x (requiredDistances x) i q =
      expectedPairedAnswer x i q :=
  distanceLookupMany_solves_mqar x hx (requiredDistances x)
    (requiredDistances_positive x) (requiredDistances_cover x) i q

/-- The appendix's bounded-number-of-distances input class, expressed by
the cardinality of the exact interaction-distance set. Source: Appendix
Theorem `thm: input-dep-genar`, hypothesis “at most `t` distinct interaction
distances.” -/
def BoundedInteractionDistances {n c : ℕ}
    (x : MQARInstance n c) (t : ℕ) : Prop :=
  (requiredDistances x).card ≤ t

/-- If at most `t` interaction distances occur, there exists a set of at
most `t` positive shifts whose parallel functional lookup gives every MQAR
answer. Source: Appendix Theorem `thm: input-dep-genar`, functional and
cardinality components, with unique keys made explicit. -/
theorem exists_exact_distance_lookup {n c : ℕ}
    (x : MQARInstance n c) (hx : UniqueKeys x) (t : ℕ)
    (ht : BoundedInteractionDistances x t) :
    ∃ shifts : Finset (Fin n), shifts.card ≤ t ∧
      PositiveDistances shifts ∧
      ∀ i : Fin n, ∀ q : Fin c,
        distanceLookupMany x shifts i q = expectedPairedAnswer x i q := by
  refine ⟨requiredDistances x, ht, requiredDistances_positive x, ?_⟩
  intro i q
  exact requiredDistanceLookup_solves_mqar x hx i q

/-- The bounded-distance and unique-key hypotheses hold for a query that
actually recalls the preceding pair. -/
example : ∃ x : MQARInstance 2 2,
    UniqueKeys x ∧ BoundedInteractionDistances x 1 ∧
      PriorAnswer x 1 0 := by
  let x : MQARInstance 2 2 := {
    key := id
    value := id
    query := fun _ => 0
  }
  refine ⟨x, ?_, ?_, ?_⟩
  · intro i j h
    exact h
  · dsimp [BoundedInteractionDistances, requiredDistances, x]
    decide
  · exact ⟨0, by decide, rfl, rfl⟩

end Transformer.Zoology
