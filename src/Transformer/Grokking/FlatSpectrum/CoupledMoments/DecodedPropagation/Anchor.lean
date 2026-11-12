module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Identification

/-!
# Anchor-dependent coverage of finite support propagation

Source: `review/flat_spectrum.md`, CM19, anchor coverage. Local support can stall
at a genuine fixed point despite a unique decoded completion. Another observed
anchor recovers the answer without adding observations or selecting a branch.
These finite examples do not assert completeness of any anchor policy.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Finset DecodedRecovery

/-- Three states decode to equality versus inequality. -/
def triangleDecoder (z : ZMod 3) : Bool := decide (z ≠ 0)

/-- The eight observed cells, withholding only `(0,0)`. -/
def triangleCells : Fin 8 → Fin 3 × Fin 3 :=
  ![(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

/-- The observed equality table, without the withheld answer. -/
def triangleLabels : Fin 8 → Bool := ![true, true, true, false, true, true, true, false]

/-- The domains reached from the observed diagonal anchor `(1,1)`. -/
def triangleStable : Domains (Fin 3) 3 where
  row := ![{1, 2}, {0}, {1, 2}]
  col := ![{1, 2}, {0}, {1, 2}]

/-- A genuine compatible completion witnesses realizability. -/
theorem triangle_fits :
    Fits triangleDecoder triangleCells triangleLabels ![0, 1, 2] ![0, 2, 1] := by
  unfold Fits
  decide

/-- One sweep from the diagonal anchor reaches the displayed domains. -/
theorem triangle_first : step triangleDecoder triangleCells triangleLabels
    (initial 1 1 0) = triangleStable := by
  change Domains.mk _ _ = Domains.mk _ _
  congr 1 <;> funext i <;> fin_cases i <;> decide

/-- Every retained phase has local support, so no further sweep changes it. -/
theorem triangle_fixed :
    step triangleDecoder triangleCells triangleLabels triangleStable = triangleStable := by
  change Domains.mk _ _ = Domains.mk _ _
  congr 1 <;> funext i <;> fin_cases i <;> decide

/-- The stall persists for every positive iteration budget. -/
theorem triangle_stalls (steps : ℕ) :
    run triangleDecoder triangleCells triangleLabels (steps + 1) (initial 1 1 0) =
      triangleStable := by
  rw [run, Function.iterate_succ_apply, triangle_first]
  exact Function.iterate_fixed triangle_fixed steps

/-- The diagonal anchor never certifies the withheld label at positive budgets,
even though the observed labels determine it, as proved below. -/
theorem triangle_poor_answer (steps : ℕ) :
    answers triangleDecoder triangleCells triangleLabels 3 (steps + 1) 0 0 =
      {false, true} := by
  have ha : anchors triangleDecoder (triangleLabels 3) = {0} := by decide
  simp only [answers, ha, singleton_biUnion]
  change outputs triangleDecoder
    (run triangleDecoder triangleCells triangleLabels (steps + 1) (initial 1 1 0)) 0 0 = _
  rw [triangle_stalls]
  decide

/-- An off-diagonal observed anchor identifies the same missing answer in three
sweeps, taking the union over both possible anchor phases. -/
theorem triangle_good_answer :
    answers triangleDecoder triangleCells triangleLabels 0 3 0 0 = {false} := by
  decide

/-- Every fitting completion agrees on the withheld answer. The poor anchor's
two-element output set is therefore not evidence of actual ambiguity. -/
theorem triangle_identified {r s : Fin 3 → ZMod 3}
    (hfit : Fits triangleDecoder triangleCells triangleLabels r s) :
    triangleDecoder (r 0 + s 0) = false :=
  identifies hfit 0 3 triangle_good_answer

example := triangle_identified triangle_fits

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
