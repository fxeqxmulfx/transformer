module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Defs

/-!
# Pruning preserves every compatible phase assignment

Source: `review/flat_spectrum.md`, CM19. The invariant concerns every fitting
assignment, including the unknown target. A support witness is its actual
neighbor phase, so larger or asymmetric decoder fibers require no new axiom.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Finset DecodedRecovery

variable {q : ℕ} {n S C : Type*} [DecidableEq n] [DecidableEq C]
  [Fintype S] {d : ZMod q → C} {cell : S → n × n} {labels : S → C}

/-- Every compatible assignment retained before a sweep is retained afterwards. -/
theorem step_preserves {D : Domains n q} {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) (hD : Contains D r s) :
    Contains (step d cell labels D) r s := by
  constructor
  · intro i
    refine mem_filter.mpr ⟨hD.1 i, ?_⟩
    intro t hi
    refine ⟨s (cell t).2, hD.2 _, ?_⟩
    simpa only [hi] using hfit t
  · intro j
    refine mem_filter.mpr ⟨hD.2 j, ?_⟩
    intro t hj
    refine ⟨r (cell t).1, hD.1 _, ?_⟩
    simpa only [hj] using hfit t

/-- A sweep can only remove phase values. -/
theorem step_subset (D : Domains n q) :
    (∀ i, (step d cell labels D).row i ⊆ D.row i) ∧
      ∀ j, (step d cell labels D).col j ⊆ D.col j := by
  constructor
  · intro i a ha
    exact (mem_filter.mp ha).1
  · intro j b hb
    exact (mem_filter.mp hb).1

/-- The preservation invariant holds at every finite iteration budget. -/
theorem run_preserves {D : Domains n q} {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) (hD : Contains D r s) (steps : ℕ) :
    Contains (run d cell labels steps D) r s := by
  induction steps with
  | zero => exact hD
  | succ k ih =>
    simpa only [run, Function.iterate_succ_apply'] using step_preserves hfit ih

/-- No finite run creates a phase outside its initial domains. -/
theorem run_subset (D : Domains n q) (steps : ℕ) :
    (∀ i, (run d cell labels steps D).row i ⊆ D.row i) ∧
      ∀ j, (run d cell labels steps D).col j ⊆ D.col j := by
  induction steps with
  | zero => exact ⟨fun _ _ h => h, fun _ _ h => h⟩
  | succ k ih =>
    have hs := step_subset (d := d) (cell := cell) (labels := labels)
      (run d cell labels k D)
    constructor
    · intro i a ha
      apply ih.1 i
      apply hs.1 i
      simpa only [run, Function.iterate_succ_apply'] using ha
    · intro j b hb
      apply ih.2 j
      apply hs.2 j
      simpa only [run, Function.iterate_succ_apply'] using hb

omit [DecidableEq n] in
/-- Empty-domain rejection cannot discard a retained assignment. -/
theorem live_of_contains {D : Domains n q} {r s : n → ZMod q}
    (hD : Contains D r s) : Live D :=
  ⟨fun i => ⟨r i, hD.1 i⟩, fun j => ⟨s j, hD.2 j⟩⟩

omit [DecidableEq n] in
/-- Normalize at any row without changing a single decoded answer. -/
theorem normalize_sum (r s : n → ZMod q) (a i j : n) :
    (r i - r a) + (r a + s j) = r i + s j := by
  abel

omit [DecidableEq n] [DecidableEq C] [Fintype S] in
/-- Normalization also preserves every observed constraint. -/
theorem normalize_fits {r s : n → ZMod q} (hfit : Fits d cell labels r s) (a : n) :
    Fits d cell labels (fun i => r i - r a) (fun j => r a + s j) := by
  intro t
  rw [normalize_sum]
  exact hfit t

variable [NeZero q]

/-- The normalized assignment is contained in the branch of its anchor phase. -/
theorem initial_contains (r s : n → ZMod q) (a b : n) :
    Contains (initial a b (r a + s b)) (fun i => r i - r a) (fun j => r a + s j) := by
  constructor
  · intro i
    by_cases hi : i = a
    · simp [initial, hi]
    · simp [initial, hi]
  · intro j
    by_cases hj : j = b
    · simp [initial, hj]
    · simp [initial, hj]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
