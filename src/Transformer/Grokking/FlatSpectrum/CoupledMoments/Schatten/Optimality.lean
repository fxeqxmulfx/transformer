module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Recovery

/-!
# From ordinary empirical CE to missing-entry recovery

Source: `review/flat_spectrum.md`, CM6. Because every observed CE term is at least its
one-hot value, attaining the summed lower bound forces each target coordinate to one.
In the three-observation rank-one example, the nuclear budget then determines the missing
label. Correct observed answers are conclusions of the CE inequality, not hypotheses.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

open Finset ConvexRecall
open Landscape.Neural (crossEntropy)

variable {m n C S : Type*} [Fintype C] [Fintype S] [DecidableEq C] [Nontrivial C]

/-- Attaining the ideal summed CE forces every observed target mass to one (CM6). -/
theorem target_mass_of_loss_le (cell : S → m × n) (labels : S → C) {P : Table m n C}
    (hP : Rows P) (hL : loss cell labels P ≤ idealLoss labels) (s : S) :
    P (cell s).1 (cell s).2 (labels s) = 1 := by
  have hl (t : S) : crossEntropy (targetLogits (labels t)) (labels t) ≤
      crossEntropy (P (cell t).1 (cell t).2) (labels t) :=
    target_crossEntropy_le _ (hP.1 _ _) (coordinate_le_one hP _ _ _)
  have he : idealLoss labels = loss cell labels P := (idealLoss_le cell labels hP).antisymm hL
  have hp := (Finset.sum_eq_sum_iff_of_le (s := Finset.univ) (fun t _ => hl t)).1 he s
    (Finset.mem_univ s)
  exact target_mass_eq_one _ (hP.1 _ _) (coordinate_le_one hP _ _ _) hp.ge

/-- Three distinct observed cells, with the lower-right entry withheld (CM6). -/
def threeCells : Fin 3 → Fin 2 × Fin 2 := ![(0, 0), (0, 1), (1, 0)]

/-- Any probability table attaining the observed CE lower bound under the nuclear
budget recovers the unseen rank-one label (CM6). -/
theorem rankOne_missing_of_loss_le {P : Table (Fin 2) (Fin 2) Bool} (hP : Rows P)
    (hB : nuclearNorm (moment signCode P) ≤ 2)
    (hL : loss threeCells (fun _ => true) P ≤ idealLoss (fun (_ : Fin 3) => true)) :
    P 1 1 true = 1 := by
  apply rankOne_missing_probability hP
  · simpa [threeCells] using target_mass_of_loss_le threeCells (fun _ => true) hP hL 0
  · simpa [threeCells] using target_mass_of_loss_le threeCells (fun _ => true) hP hL 1
  · simpa [threeCells] using target_mass_of_loss_le threeCells (fun _ => true) hP hL 2
  · exact hB

private def witness : Table (Fin 2) (Fin 2) Bool := pointMass (fun _ _ => true)

private theorem witness_loss : loss threeCells (fun _ => true) witness =
    idealLoss (fun (_ : Fin 3) => true) := by
  apply loss_eq_ideal_of_fit
  intro s
  funext c
  simp [witness, pointMass, targetLogits]

private theorem witness_nuclear : nuclearNorm (moment signCode witness) ≤ 2 := by
  rw [witness, moment_pointMass]
  have he : (fun (_ : Fin 2) (_ : Fin 2) => signCode true) = rankOnePatch 1 := by
    ext i j
    fin_cases i <;> fin_cases j <;> norm_num [rankOnePatch, signCode]
  rw [he]
  exact (rankOnePatch_budget_iff _).2 rfl

example := target_mass_of_loss_le threeCells (fun _ => true)
  (rows_pointMass (fun (_ : Fin 2) (_ : Fin 2) => true)) witness_loss.le 0

example := rankOne_missing_of_loss_le
  (rows_pointMass (fun (_ : Fin 2) (_ : Fin 2) => true)) witness_nuclear witness_loss.le

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
