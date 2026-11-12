module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.CrossEntropy
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Domain
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.ZeroMass

/-!
# The spectral relaxation has a wrong minimum at a certified cell

Source: `review/flat_spectrum.md`, CM18. Three latent states, two classes, the decoder that
merges the two nonzero states; a three by three table whose label says whether the row differs
from the column, with the cell `(0, 0)` withheld. The certificate of the true phases is
accepted and covers the withheld cell, so every completion of the training samples has the
label `false` there.

The symmetric latent table that agrees with the target at the eight training cells and is
uniform at the withheld cell has nonconstant moments of nuclear norm at most `3`, so it lies
in the domains of CM15 and CM16 at the budgets `3`, and it fits the training samples. Its
decoded table is a global minimizer of the training cross-entropy over both domains that
gives the wrong label `true` the probability `2 / 3` (`budget_wrong_minimum`). It is outside
the hull of decoded phase tables (`wrong_not_in_hull`). The identification of CM18 is
therefore a statement about the hull and about the certified restriction, not about the
minimizers of the spectral relaxation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Matrix Finset Schatten

/-- The mass at zero of the wrong table: the target's, with a uniform row at `(0, 0)`. -/
noncomputable def wrongMass : Fin 3 → Fin 3 → ℝ := ![![1 / 3, 0, 0], ![0, 1, 0], ![0, 0, 1]]

private lemma wrong_spectral :
    ((fun i j => (((3 * wrongMass i j - 1) / 2 : ℝ) : ℂ)) : Matrix (Fin 3) (Fin 3) ℂ) =
      (3 / 4 : ℂ) • vecMulVec ![0, 1, -1] ![0, 1, -1] +
        ((1 / 3 : ℂ) • vecMulVec ![1, -1, -1] ![1, -1, -1] +
          (-(1 / 12) : ℂ) • vecMulVec ![2, 1, 1] ![2, 1, 1]) := by
  ext i j
  change _ = (3 / 4 : ℂ) * (![0, 1, -1] i * ![0, 1, -1] j) +
    ((1 / 3 : ℂ) * (![1, -1, -1] i * ![1, -1, -1] j) +
      (-(1 / 12) : ℂ) * (![2, 1, 1] i * ![2, 1, 1] j))
  fin_cases i <;> fin_cases j <;> norm_num [wrongMass]

/-- The triangle inequality on three weighted rank-one matrices. -/
private lemma nuclear_three (a b c : ℂ) (u v w : Fin 3 → ℂ) :
    nuclearNorm (a • vecMulVec u u + (b • vecMulVec v v + c • vecMulVec w w)) ≤
      ‖a‖ * ∑ i, ‖u i‖ ^ 2 + (‖b‖ * ∑ i, ‖v i‖ ^ 2 + ‖c‖ * ∑ i, ‖w i‖ ^ 2) := by
  have hv (v : Fin 3 → ℂ) : nuclearNorm (vecMulVec v v) = ∑ i, ‖v i‖ ^ 2 := by
    rw [nuclearNorm_vecMulVec, Real.mul_self_sqrt (sum_nonneg fun _ _ => sq_nonneg _)]
  calc nuclearNorm (a • vecMulVec u u + (b • vecMulVec v v + c • vecMulVec w w))
      ≤ nuclearNorm (a • vecMulVec u u) +
          nuclearNorm (b • vecMulVec v v + c • vecMulVec w w) := nuclearNorm_add_le _ _
    _ ≤ nuclearNorm (a • vecMulVec u u) +
          (nuclearNorm (b • vecMulVec v v) + nuclearNorm (c • vecMulVec w w)) :=
      add_le_add le_rfl (nuclearNorm_add_le _ _)
    _ = _ := by rw [nuclearNorm_smul, nuclearNorm_smul, nuclearNorm_smul, hv, hv, hv]

/-- Its nonconstant moments are a combination of three rank-one matrices, which bounds their
nuclear norm by `3 / 2 + 1 + 1 / 2`. -/
theorem wrong_nuclear {f : ZMod 3} (hf : f ≠ 0) :
    nuclearNorm (moment (phaseCode (Equiv.refl (ZMod 3)) f) (zeroMass wrongMass)) ≤ 3 := by
  rw [moment_zeroMass _ hf, wrong_spectral]
  refine (nuclear_three _ _ _ _ _ _).trans ?_
  simp only [Fin.sum_univ_three]
  norm_num

/-- All cells but `(0, 0)`. -/
def offCells : Fin 8 → Fin 3 × Fin 3 :=
  ![(0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

/-- The label says whether the row differs from the column. -/
def offLabels (t : Fin 8) : Bool := decide ((offCells t).1 ≠ (offCells t).2)

/-- Row phases `i`, column phases `-j`, the rectangle of the rows `1`, `2` and columns `0`, `1`. -/
def offCert : Certificate (Fin 3) 3 :=
  ⟨fun i => (i.val : ZMod 3), fun j => -(j.val : ZMod 3), 1, 2, 0, 1⟩

/-- The phases of the certificate fit the eight training samples. -/
lemma off_fits : Fits pairedBoolDecoder offCells offLabels offCert.row offCert.col := by
  intro t
  fin_cases t <;> decide

/-- The certificate is accepted (CM18). -/
theorem offCert_accepts : Accepts pairedBoolDecoder offCells offLabels offCert :=
  ⟨fun t _ _ => off_fits t, ⟨⟨2, rfl⟩, ⟨5, rfl⟩⟩, ⟨⟨3, rfl⟩, ⟨6, rfl⟩⟩, by decide, by decide,
    by decide⟩

/-- It reaches the row of the withheld cell through the columns `1` and `2`. -/
lemma off_row : Pinned offCells offCert (.inl 0) :=
  .row (j := 1) (j' := 2) (.anchored ⟨3, rfl⟩ ⟨6, rfl⟩) (.anchored ⟨4, rfl⟩ ⟨7, rfl⟩)
    (by decide) ⟨0, rfl⟩ ⟨1, rfl⟩

/-- It reaches the column of the withheld cell, observed in both rows of the rectangle. -/
lemma off_col : Pinned offCells offCert (.inr 0) := .anchored ⟨2, rfl⟩ ⟨5, rfl⟩

/-- The decoded wrong table. -/
noncomputable def wrongTable : Table (Fin 3) (Fin 3) Bool :=
  decode pairedBoolDecoder (zeroMass wrongMass)

/-- The masses at zero of the wrong table are probabilities. -/
lemma wrong_range (i j : Fin 3) : 0 ≤ wrongMass i j ∧ wrongMass i j ≤ 1 := by
  fin_cases i <;> fin_cases j <;> norm_num [wrongMass]

/-- The wrong latent table satisfies both budgets `3` at every frequency (CM18). -/
lemma wrong_budget : BudgetFeasible (phaseCode (Equiv.refl (ZMod 3))) (fun _ => 3)
    (fun _ => 3) (zeroMass wrongMass) := by
  have h := (budget_iff_nuclear (Equiv.refl (ZMod 3)) (zeroMass wrongMass)).2
    ⟨rows_zeroMass wrong_range, fun f hf => by simpa using wrong_nuclear hf⟩
  simpa only [Fintype.card_fin, Nat.cast_ofNat] using h

/-- The wrong table has the one-hot row of the label at every training cell. -/
lemma wrong_fit (t : Fin 8) :
    wrongTable (offCells t).1 (offCells t).2 = ConvexRecall.targetLogits (offLabels t) := by
  rw [wrongTable, decode_zeroMass wrong_range]
  funext b
  fin_cases t <;> cases b <;>
    simp [wrongMass, offCells, offLabels, ConvexRecall.targetLogits]

/-- At the withheld cell it gives `true` the probability `2 / 3`. -/
lemma wrong_value : wrongTable 0 0 true = 2 / 3 := by
  rw [wrongTable, decode_zeroMass wrong_range]
  norm_num [wrongMass]

/-- **The spectral relaxation has a wrong minimum at a certified cell** (CM18). The
certificate is accepted and covers the withheld cell `(0, 0)`, where it proposes `false`; the
table `wrongTable` lies in the symmetric domain of CM16, minimizes the training cross-entropy
over the whole domain of CM15 at the budgets `3`, and gives `true` the probability `2 / 3`. -/
theorem budget_wrong_minimum :
    Accepts pairedBoolDecoder offCells offLabels offCert ∧
      Pinned offCells offCert (.inl 0) ∧ Pinned offCells offCert (.inr 0) ∧
      (0, 0) ∉ Set.range offCells ∧ offCert.decoded pairedBoolDecoder 0 0 = false ∧
      PairedFeasible pairedBoolDecoder 3 3 wrongTable ∧
      IsMinOn (loss offCells offLabels)
        {Q | DecodedFeasible pairedBoolDecoder (phaseCode (Equiv.refl (ZMod 3)))
          (fun _ => 3) (fun _ => 3) Q} wrongTable ∧
      wrongTable 0 0 true = 2 / 3 := by
  refine ⟨offCert_accepts, off_row, off_col, ?_, by decide,
    ⟨zeroMass wrongMass, wrong_budget, zeroMass_even _, rfl⟩,
    (decoded_fit_minimum pairedBoolDecoder _ _ _ offCells offLabels wrong_budget wrong_fit).2,
    wrong_value⟩
  rintro ⟨t, ht⟩
  revert t
  decide

/-- The hull of decoded phase tables excludes the wrong minimum (CM18). -/
theorem wrong_not_in_hull : wrongTable ∉ phaseHull pairedBoolDecoder (Fin 3) := by
  intro h
  have hrow := hull_fit_decoding (by decide) pairedDecoder_bool offCert_accepts off_row off_col
    h (loss_eq_ideal_of_fit offCells offLabels wrong_fit).le
  have h1 := congrFun hrow true
  rw [wrong_value, show offCert.decoded pairedBoolDecoder 0 0 = false by decide] at h1
  norm_num [ConvexRecall.targetLogits] at h1

example := wrong_nuclear (f := 2) (by decide)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
