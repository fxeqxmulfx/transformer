module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Learner
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.CrossEntropy

/-!
# A recovered unseen cell behind a many-to-one decoder, and the refusals

Source: `review/flat_spectrum.md`, CM18.

`held_recovery`: the folded table of CM16 (three rows, five latent states, three classes) with
the cell `(2, 1)` withheld. The training labels meet the label conditions; the procedure
answers the target's label at the withheld cell, every completion of the eight training
samples has that label there, and two training cells with the distinct latent states `1` and
`4` share the label `1`. `held_cross_entropy` is the same example in the convex programs.

The refusals. On the three cells of a two by two table, the samples of
`corner_decoded_ambiguity`, `corner_decoderHull_ambiguity` and `paired_unseen_ambiguity`, no
certificate is accepted for any decoder and any labels, because there is no rectangle of
training cells (`threeCells_no_certificate`), although the observed graph is connected with a
perfect matching (`corner_decoded_graph_coverage`). The procedure answers `uncertified` on
the paired example and `unsupported` for the corner decoder, which is not even; conflicting
samples are `inconsistent`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

/-- **A recovered unseen cell behind a many-to-one decoder** (CM18). The cell `(2, 1)` is not
a training cell; the procedure answers its label; every completion of the training samples
has that label; the training cells `(1, 0)` and `(0, 1)` have different latent states and one
label; the training labels are not constant. -/
theorem held_recovery :
    (2, 1) ∉ Set.range heldCells ∧
      recover (foldDecoder 2) heldCells heldLabels 2 1 = .accept (foldedLabels 2 1) ∧
      (∀ r s : Fin 3 → ZMod 5, Fits (foldDecoder 2) heldCells heldLabels r s →
        foldDecoder 2 (r 2 + s 1) = foldedLabels 2 1) ∧
      foldedRow 1 + foldedColumn 0 ≠ foldedRow 0 + foldedColumn 1 ∧
      foldDecoder 2 (foldedRow 1 + foldedColumn 0) =
        foldDecoder 2 (foldedRow 0 + foldedColumn 1) ∧
      heldLabels 0 ≠ heldLabels 1 := by
  have hrec : recover (foldDecoder 2) heldCells heldLabels 2 1 = .accept (foldedLabels 2 1) := by
    have h := recover_of_labels (by decide) (pairedDecoder_fold 2) held_fits held_anchor
      held_reached_row held_reached_col
    rwa [show foldDecoder 2 (foldedRow 2 + foldedColumn 1) = foldedLabels 2 1 by decide] at h
  refine ⟨?_, hrec, fun r s hfit => recover_sound hrec hfit, by decide, by decide, by decide⟩
  rintro ⟨t, ht⟩
  revert t
  decide

/-- **The same example in the convex programs** (CM18). The folded table lies in the symmetric
domain and minimizes the training cross-entropy over the CM15 domain at the budgets `3`. Every
minimizer over the hull of decoded phase tables, and every minimizer of the program restricted
by the certificate, has the one-hot row of the withheld label at `(2, 1)`. -/
theorem held_cross_entropy :
    PairedFeasible (foldDecoder 2) 3 3 (pointMass foldedLabels) ∧
      IsMinOn (loss heldCells heldLabels)
        {Q | DecodedFeasible (foldDecoder 2) (phaseCode (Equiv.refl (ZMod 5)))
          (fun _ => 3) (fun _ => 3) Q} (pointMass foldedLabels) ∧
      (∀ Q ∈ phaseHull (foldDecoder 2) (Fin 3),
        IsMinOn (loss heldCells heldLabels) (phaseHull (foldDecoder 2) (Fin 3)) Q →
        Q 2 1 = ConvexRecall.targetLogits (foldedLabels 2 1)) ∧
      ∀ Q, CertifiedFeasible (foldDecoder 2) heldCells heldCert Q →
        IsMinOn (loss heldCells heldLabels)
          {Q | CertifiedFeasible (foldDecoder 2) heldCells heldCert Q} Q →
        Q 2 1 = ConvexRecall.targetLogits (foldedLabels 2 1) := by
  have hlab : foldDecoder 2 (foldedRow 2 + foldedColumn 1) = foldedLabels 2 1 := by decide
  have hmin := certified_minimum (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits
    held_fits
  simp only [folded_decoding, Fintype.card_fin, Nat.cast_ofNat] at hmin
  refine ⟨hmin.1, hmin.2.1, fun Q hQ hQmin => ?_, fun Q hQ hQmin => ?_⟩
  · rw [← hlab]
    exact hull_minimizer_decoding (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits
      held_row held_col hQ hQmin
  · rw [← hlab]
    exact ((certified_restriction (by decide) (pairedDecoder_fold 2) heldCert_accepts
      held_fits).2 Q hQ hQmin).2 2 1 held_row held_col

/-- **No certificate on three cells of a two by two table** (CM18): whatever the decoder and
the labels, the conditions of acceptance fail, because no rectangle of training cells exists.
These are the cells of the ambiguous examples of CM15 and CM16. -/
theorem threeCells_no_certificate {q : ℕ} {C : Type*} (d : ZMod q → C) (labels : Fin 3 → C)
    (c : Certificate (Fin 2) q) : ¬ Accepts d threeCells labels c := by
  rintro ⟨-, ⟨hotl, hobl⟩, ⟨hotr, hobr⟩, hrow, hcol, -⟩
  have no11 : ¬ Observed threeCells (1, 1) := by
    rintro ⟨t, ht⟩
    revert t
    decide
  have h1 : ∀ a b : Fin 2, a ≠ b → a = 1 ∨ b = 1 := by decide
  rcases h1 c.top c.bot (fun h => hrow (by rw [h])) with ht | hb <;>
    rcases h1 c.left c.right (fun h => hcol (by rw [h])) with hl | hr
  · rw [ht, hl] at hotl
    exact no11 hotl
  · rw [ht, hr] at hotr
    exact no11 hotr
  · rw [hb, hl] at hobl
    exact no11 hobl
  · rw [hb, hr] at hobr
    exact no11 hobr

/-- **Refusal of the paired ambiguity** (CM16, CM18): on the samples of
`paired_unseen_ambiguity`, whose two completions are both cross-entropy minima, the procedure
answers `uncertified` at the unseen cell and no accepted certificate covers it. -/
theorem paired_ambiguity_refused :
    recover pairedBoolDecoder threeCells pairedThreeLabels 1 1 = .uncertified ∧
      ∀ c : Certificate (Fin 2) 3, ¬ Covers pairedBoolDecoder threeCells pairedThreeLabels 1 1 c :=
  ⟨recover_uncertified (by decide) pairedDecoder_bool ambiguous_fit ambiguous_fit_neg
      (by decide),
    not_covers_of_disagree (by decide) pairedDecoder_bool ambiguous_fit ambiguous_fit_neg
      (by decide)⟩

/-- **Refusal of the corner decoder** (CM15, CM18): the decoder of `corner_decoded_ambiguity`
separates a latent state from its negative, so it is outside the class. -/
theorem corner_unsupported (i j : Fin 2) :
    recover cornerDecoder threeCells (fun _ => true) i j = .unsupported :=
  recover_unsupported (fun h => absurd (h.2.even 1) (by decide)) _ _ i j

/-- **Refusal of conflicting samples** (CM18): one cell with two labels. -/
theorem conflict_inconsistent :
    recover (foldDecoder 1) (fun _ : Fin 2 => ((0 : Fin 1), (0 : Fin 1))) ![0, 1] 0 0 =
      .inconsistent :=
  recover_inconsistent (by decide) (pairedDecoder_fold 1)
    (fun ⟨_, _, h⟩ => absurd ((h 0).symm.trans (h 1)) (by decide)) 0 0

example := threeCells_no_certificate cornerDecoder (fun _ => true)
example := threeCells_no_certificate pairedBoolDecoder pairedThreeLabels

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
