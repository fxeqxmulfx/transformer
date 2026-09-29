/-
# One input-dependent distance for associative recall

Arora et al., arXiv:2312.04927v1, Appendix `sec: data-dep-ar`,
equations `eq: y-data-ind-t1` through `eq: output-t1`. Given one selected
interaction distance, shifted one-hot keys are compared with queries and
the same shift retrieves values only where that comparison succeeds.
-/

import Transformer.Zoology.Appendix_Attention
import Transformer.Zoology.Appendix_Shift

open scoped BigOperators

namespace Transformer.Zoology

/-- One-hot key sequence for the triple representation of MQAR.
Source: Appendix `sec: data-dep-ar`, projection `K`. -/
def keySequence {n c : ℕ} (x : MQARInstance n c) : RealSequence n c :=
  fun i q => oneHot (x.key i) q

/-- One-hot value sequence for the triple representation of MQAR.
Source: Appendix `sec: data-dep-ar`, projection `V`. -/
def valueSequence {n c : ℕ} (x : MQARInstance n c) : RealSequence n c :=
  fun i q => oneHot (x.value i) q

/-- One-hot query sequence for the triple representation of MQAR.
Source: Appendix `sec: data-dep-ar`, projection `Q`. -/
def querySequence {n c : ℕ} (x : MQARInstance n c) : RealSequence n c :=
  fun i q => oneHot (x.query i) q

/-- Coordinatewise match after shifting the keys down by one selected
interaction distance. Source: Appendix equation `eq: y-data-ind-t1`. -/
def distanceMatchChannel {n c : ℕ} (x : MQARInstance n c)
    (s : Fin n) : RealSequence n c :=
  fun i q => querySequence x i q * shiftDown (keySequence x) s i q

/-- The all-ones linear projection broadcasts a one-hot match to every
value coordinate. Source: Appendix equation `eq: E-def`. -/
def distanceMatchBroadcast {n c : ℕ} (x : MQARInstance n c)
    (s : Fin n) : RealSequence n c :=
  linearProjection (distanceMatchChannel x s) (fun _ _ => 1)

/-- The selected-distance lookup gates a shifted value by the broadcast
query-key match. Source: Appendix equation `eq: z-data-ind-t1`. -/
def distanceLookup {n c : ℕ} (x : MQARInstance n c)
    (s : Fin n) : RealSequence n c :=
  fun i q => distanceMatchBroadcast x s i q *
    shiftDown (valueSequence x) s i q

/-- A selected distance retrieves the associated value exactly when the
key at that distance matches the query, and returns zero otherwise.
Source: Appendix equation `eq: output-t1`, at tuple-level distance `s`.
The raw-token construction in the paper shifts values by `s-1` because
values follow keys immediately; this aligned-stream model needs shift `s`. -/
theorem distanceLookup_eq {n c : ℕ} (x : MQARInstance n c)
    (s i : Fin n) (q : Fin c) :
    distanceLookup x s i q =
      if s ≤ i then
        (if x.key (i - s) = x.query i then
          oneHot (x.value (i - s)) q else 0)
      else 0 := by
  by_cases hs : s ≤ i
  · by_cases hk : x.key (i - s) = x.query i
    · simp [distanceLookup, distanceMatchBroadcast,
        distanceMatchChannel, linearProjection, querySequence,
        keySequence, valueSequence, shiftDown, hs, oneHot, hk]
    · simp [distanceLookup, distanceMatchBroadcast,
        distanceMatchChannel, linearProjection, querySequence,
        keySequence, valueSequence, shiftDown, hs, oneHot, hk]
  · simp [distanceLookup, distanceMatchBroadcast,
      distanceMatchChannel, linearProjection, querySequence,
      shiftDown, hs]

/-- A matching prior key at exactly this shift yields the expected value.
Source: Appendix equation `eq: output-t1`, matching case. -/
theorem distanceLookup_match {n c : ℕ} (x : MQARInstance n c)
    (s i : Fin n) (hs : s ≤ i)
    (hmatch : x.key (i - s) = x.query i) (q : Fin c) :
    distanceLookup x s i q = oneHot (x.value (i - s)) q := by
  simp [distanceLookup_eq, hs, hmatch]

/-- A selected distance with a different key contributes zero.
Source: Appendix equation `eq: output-t1`, nonmatching case. -/
theorem distanceLookup_no_match {n c : ℕ} (x : MQARInstance n c)
    (s i : Fin n) (hmatch : s ≤ i → x.key (i - s) ≠ x.query i)
    (q : Fin c) : distanceLookup x s i q = 0 := by
  rw [distanceLookup_eq]
  by_cases hs : s ≤ i
  · simp [hs, hmatch hs]
  · simp [hs]

/-- The matching and nonmatching hypotheses can both occur on concrete
unique-key instances. -/
example : ∃ x : MQARInstance 2 2,
    x.key ((1 : Fin 2) - 1) = x.query 1 ∧
      x.key ((1 : Fin 2) - 0) ≠ x.query 1 := by
  refine ⟨{key := id, value := id, query := fun _ => 0}, ?_⟩
  decide

end Transformer.Zoology
