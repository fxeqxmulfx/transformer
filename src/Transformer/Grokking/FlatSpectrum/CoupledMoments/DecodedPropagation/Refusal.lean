module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Witness

/-!
# Explicit ambiguous and inconsistent training sets

Source: `review/flat_spectrum.md`, CM19. The same three-state-fiber decoder has
two compatible completions of three `false` labels with opposite answers at
the unseen cell. Every iteration budget refuses a singleton there. Distinct
decoders in a bank can also disagree. Contradictory labels on the same cell
give an empty answer set, a sound proof of inconsistency.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Schatten DecodedRecovery

/-- Three equal observations with no revealed lower-right label. -/
def ambiguousLabels : Fin 3 → Bool := ![false, false, false]

/-- The completion with a false lower-right answer. -/
theorem ambiguity_fit_false :
    Fits wideDecoder threeCells ambiguousLabels ![0, 0] ![0, 0] := by unfold Fits; decide

/-- Another completion has a true lower-right answer. -/
theorem ambiguity_fit_true :
    Fits wideDecoder threeCells ambiguousLabels ![0, 1] ![0, 2] := by unfold Fits; decide

/-- Real ambiguity cannot be eliminated by running more pruning sweeps. -/
theorem wide_ambiguity_refused (steps : ℕ) (label : Bool) :
    answers wideDecoder threeCells ambiguousLabels 0 steps 1 1 ≠ {label} :=
  not_singleton_of_disagree ambiguity_fit_false ambiguity_fit_true 0 steps (by decide) label

/-- Two different categorical maps, without selecting a winner by its fit. -/
def refusalBank : Fin 2 → ZMod 4 → Bool :=
  ![wideDecoder, fun z => decide (z = 1 ∨ z = 3)]

/-- The second decoder fits the same data with a false answer. -/
theorem bank_ambiguity_fit_false :
    Fits (refusalBank 1) threeCells ambiguousLabels ![0, 0] ![0, 0] := by unfold Fits; decide

/-- The first decoder fits with a true answer. -/
theorem bank_ambiguity_fit_true :
    Fits (refusalBank 0) threeCells ambiguousLabels ![0, 1] ![0, 2] := by unfold Fits; decide

/-- A disagreement across compatible decoder choices also survives every
budget and forbids certification by the union. -/
theorem bank_ambiguity_refused (steps : ℕ) (label : Bool) :
    bankAnswers refusalBank threeCells ambiguousLabels 0 steps 1 1 ≠ {label} :=
  bank_not_singleton_of_disagree 1 0 bank_ambiguity_fit_false bank_ambiguity_fit_true
    0 steps (by decide) label

/-- The same single cell appears twice. -/
def conflictCells : Fin 2 → Fin 1 × Fin 1 := ![(0, 0), (0, 0)]

/-- Opposite labels on that single cell. -/
def conflictLabels : Fin 2 → Bool := ![true, false]

/-- One sweep detects the contradiction for the fixed decoder. -/
theorem conflict_empty : answers wideDecoder conflictCells conflictLabels 0 1 0 0 = ∅ := by
  decide

/-- Every branch of both bank members is rejected by the same contradiction. -/
theorem conflict_bank_empty :
    bankAnswers wideBank conflictCells conflictLabels 0 1 0 0 = ∅ := by
  decide

/-! Inhabited hypotheses for both empty-set refutations and both ambiguity rules. -/

example := empty_refutes 0 1 conflict_empty
example := empty_bank_refutes 0 1 conflict_bank_empty
example := not_singleton_of_disagree (i := 1) (j := 1) ambiguity_fit_false ambiguity_fit_true
  0 2 (by decide) false
example := bank_not_singleton_of_disagree (i := 1) (j := 1) 1 0
  bank_ambiguity_fit_false bank_ambiguity_fit_true
  0 2 (by decide) true

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
