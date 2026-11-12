module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.CrossEntropy

/-!
# An unseen answer with a three-state fiber and an even latent group

Source: `review/flat_spectrum.md`, CM19. The decoder on `ZMod 4` sends `3` to
`true` and `0,1,2` to `false`. Only three cells of a two-by-two table are
observed: `true,true,false`. The last label is forced to be `false`, but its
latent state can be any of `0,1,2`. Anchor at the `false` observation and retain
all three branches. Two simultaneous sweeps certify the withheld answer.
This lies outside the opposite-pair decoder class of CM18.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Finset Schatten DecodedRecovery

/-- A categorical decoder with a singleton and a three-state fiber. -/
def wideDecoder (z : ZMod 4) : Bool := decide (z = 3)

/-- Phases used only to prove realizability of the witness. -/
def wideRow : Fin 2 → ZMod 4 := ![0, 1]

/-- Two equal unknown column phases; their equality is not given to pruning. -/
def wideColumn : Fin 2 → ZMod 4 := ![3, 3]

/-- The three original observed labels; the missing label is not an input. -/
def wideLabels : Fin 3 → Bool := ![true, true, false]

/-- The explicit target fits the three observations. -/
theorem wide_fits : Fits wideDecoder threeCells wideLabels wideRow wideColumn := by
  unfold Fits
  decide

/-- Three distinct anchor states are retained, not one chosen representative. -/
theorem wide_fiber : anchors wideDecoder false = {0, 1, 2} := by
  decide

/-- This decoder is outside the opposite-pair class, even aside from its even
number of states: `0` and `1` collide without being opposite. -/
theorem wide_not_paired : ¬ PairedDecoder wideDecoder := by
  intro h
  have h01 := (h 0 1).mp (by decide)
  exact (by decide : ¬ ((1 : ZMod 4) = 0 ∨ 1 = - (0 : ZMod 4))) h01

/-- The finite, training-only computation identifies the unobserved label. -/
theorem wide_answer : answers wideDecoder threeCells wideLabels 2 2 1 1 = {false} := by
  decide

/-- **A nonvacuous unseen answer beyond paired fibers** (CM19). The cell is not
observed, the computation accepts, and every latent completion has its label. -/
theorem wide_recovery :
    (1, 1) ∉ Set.range threeCells ∧
      answers wideDecoder threeCells wideLabels 2 2 1 1 = {false} ∧
      ∀ r s : Fin 2 → ZMod 4, Fits wideDecoder threeCells wideLabels r s →
        wideDecoder (r 1 + s 1) = false := by
  refine ⟨?_, wide_answer, fun r s hfit => identifies hfit 2 2 wide_answer⟩
  rintro ⟨t, ht⟩
  revert t
  decide

/-- All three states of the false fiber really are possible at the missing
cell. The certified decoded answer does not reconstruct the latent phase. -/
theorem wide_latent_undetermined :
    (∃ r s, Fits wideDecoder threeCells wideLabels r s ∧ r 1 + s 1 = 0) ∧
    (∃ r s, Fits wideDecoder threeCells wideLabels r s ∧ r 1 + s 1 = 1) ∧
    (∃ r s, Fits wideDecoder threeCells wideLabels r s ∧ r 1 + s 1 = 2) := by
  exact ⟨⟨![3, 0], ![0, 0], by unfold Fits; decide, by decide⟩,
    ⟨![2, 0], ![1, 1], by unfold Fits; decide, by decide⟩,
    ⟨![1, 0], ![2, 2], by unfold Fits; decide, by decide⟩⟩

/-- A fixed two-member bank with different decoder maps. Both must be checked. -/
def wideBank : Fin 2 → ZMod 4 → Bool := ![wideDecoder, fun z => wideDecoder (z + 1)]

/-- The first bank member realizes the observed table. -/
theorem wide_bank_fits : Fits (wideBank 0) threeCells wideLabels wideRow wideColumn := by
  unfold Fits
  decide

/-- The other decoder is also compatible, with different column phases. -/
theorem wide_shifted_fits : Fits (wideBank 1) threeCells wideLabels wideRow ![2, 2] := by
  unfold Fits
  decide

/-- The union over both decoders still certifies the missing label. -/
theorem wide_bank_answer : bankAnswers wideBank threeCells wideLabels 2 2 1 1 = {false} := by
  decide

/-! Concrete hypotheses for the preservation and identification statements. -/

example := step_preserves (normalize_fits wide_fits 1)
  (initial_contains wideRow wideColumn 1 0)
example := run_preserves (normalize_fits wide_fits 1)
  (initial_contains wideRow wideColumn 1 0) 2
example := live_of_contains (initial_contains wideRow wideColumn 1 0)
example := normalize_fits wide_fits 1
example := mem_outputs (d := wideDecoder) (initial_contains wideRow wideColumn 1 0) 1 1
example := mem_answers wide_fits 2 2 1 1
example := identifies wide_fits 2 2 wide_answer
example := mem_bankAnswers 0 wide_bank_fits 2 2 1 1
example := bank_identifies 0 wide_bank_fits 2 2 wide_bank_answer
example := bank_identifies 1 wide_shifted_fits 2 2 wide_bank_answer
example := certified_witness 0 wide_bank_fits 2 2
example := certified_optimum 0 wide_bank_fits 2 2
example := certifiedFeasible_convex wideBank threeCells wideLabels 2 2
example := certified_loss_convex wideBank threeCells wideLabels 2 2
example := certified_restriction 0 wide_bank_fits 2 2
  (certified_witness 0 wide_bank_fits 2 2) (certified_optimum 0 wide_bank_fits 2 2)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
