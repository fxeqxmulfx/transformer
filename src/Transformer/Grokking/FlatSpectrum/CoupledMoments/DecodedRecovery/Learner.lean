module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Coverage

/-!
# A training-only procedure with explicit refusals

Source: `review/flat_spectrum.md`, CM18. `recover` reads the decoder, the training cells and
their labels, and answers for one cell. It refuses a decoder outside the class
(`unsupported`), training samples that no phases fit (`inconsistent`), and a cell that no
accepted certificate covers (`uncertified`); otherwise it returns the label that a covering
certificate proposes.

The definition is a specification by search over phases and certificates, written with
classical choice: it is not executable code and no running time is claimed for it.

* `recover_sound`: an accepted label is the label of every completion of the training
  samples, in particular of the unknown target, whatever its phases;
* `recover_accept`, `recover_of_labels`: for a realizable table the cell is accepted as soon
  as some accepted certificate covers it, which the label conditions `Anchor` and `Reached`
  guarantee;
* `recover_uncertified`: if two completions of the training samples disagree at the cell, no
  accepted certificate covers it and the answer is the refusal `uncertified`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

variable {q : ℕ} {C n S : Type*} {d : ZMod q → C} {cell : S → n × n} {labels : S → C}

/-- The answers of the procedure (CM18). -/
inductive Verdict (C : Type*)
  /-- The decoder is outside the class, or the number of latent states is even. -/
  | unsupported
  /-- No phases fit the training samples. -/
  | inconsistent
  /-- No accepted certificate covers the cell. -/
  | uncertified
  /-- The label of every completion of the training samples at the cell. -/
  | accept (label : C)

open Classical in
/-- **The procedure** (CM18): its inputs are the decoder, the training cells, their labels and
the cell in question. -/
noncomputable def recover (d : ZMod q → C) (cell : S → n × n) (labels : S → C) (i j : n) :
    Verdict C :=
  if Odd q ∧ PairedDecoder d then
    if ∃ r s : n → ZMod q, Fits d cell labels r s then
      if h : ∃ c : Certificate n q, Covers d cell labels i j c then
        .accept (h.choose.decoded d i j)
      else .uncertified
    else .inconsistent
  else .unsupported

/-- **Soundness** (CM18): an accepted label is the decoded label of every pair of phases that
fits the training samples. Nothing is assumed about the phases beyond the fit. -/
theorem recover_sound {i j : n} {y : C} (h : recover d cell labels i j = .accept y)
    {r s : n → ZMod q} (hfit : Fits d cell labels r s) : d (r i + s j) = y := by
  unfold recover at h
  split_ifs at h with hcls hex hcov
  · obtain ⟨hc, hi, hj⟩ := hcov.choose_spec
    rw [accepted_decoding hcls.1 hcls.2 hc hfit hi hj]
    exact Verdict.accept.inj h

/-- **Acceptance** (CM18): for a realizable table, a cell covered by some accepted
certificate is answered with the label of the realizing phases. -/
theorem recover_accept (hq : Odd q) (hd : PairedDecoder d) {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) {i j : n} {c : Certificate n q}
    (hc : Covers d cell labels i j c) :
    recover d cell labels i j = .accept (d (r i + s j)) := by
  have hex : ∃ c : Certificate n q, Covers d cell labels i j c := ⟨c, hc⟩
  unfold recover
  rw [ite_eq_left ⟨hq, hd⟩, ite_eq_left ⟨r, s, hfit⟩, dite_eq_left hex]
  obtain ⟨hc', hi, hj⟩ := hex.choose_spec
  rw [accepted_decoding hq hd hc' hfit hi hj]

/-- **Acceptance under conditions on training labels** (CM18): the unknown target has the
phases `r`, `s`; the remaining hypotheses mention training cells and training labels only. -/
theorem recover_of_labels (hq : Odd q) (hd : PairedDecoder d) {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) {tl bl tr br ts bs : S}
    (ha : Anchor cell labels tl bl tr br ts bs) {i j : n}
    (hi : Reached cell labels (cell tl).1 (cell bl).1 (.inl i))
    (hj : Reached cell labels (cell tl).1 (cell bl).1 (.inr j)) :
    recover d cell labels i j = .accept (d (r i + s j)) :=
  recover_accept hq hd hfit (covers_of_labels hd hfit ha hi hj)

/-- Where two completions of the training samples disagree, no accepted certificate covers. -/
theorem not_covers_of_disagree (hq : Odd q) (hd : PairedDecoder d) {r s r' s' : n → ZMod q}
    (hfit : Fits d cell labels r s) (hfit' : Fits d cell labels r' s') {i j : n}
    (hne : d (r i + s j) ≠ d (r' i + s' j)) (c : Certificate n q) :
    ¬ Covers d cell labels i j c :=
  fun ⟨hc, hi, hj⟩ => hne (fits_agree hq hd hc hfit hfit' hi hj)

/-- **Refusal of an ambiguous cell** (CM18): where two completions of the training samples
disagree, the answer is `uncertified`. -/
theorem recover_uncertified (hq : Odd q) (hd : PairedDecoder d) {r s r' s' : n → ZMod q}
    (hfit : Fits d cell labels r s) (hfit' : Fits d cell labels r' s') {i j : n}
    (hne : d (r i + s j) ≠ d (r' i + s' j)) :
    recover d cell labels i j = .uncertified := by
  unfold recover
  rw [ite_eq_left ⟨hq, hd⟩, ite_eq_left ⟨r, s, hfit⟩, dite_eq_right]
  rintro ⟨c, hc⟩
  exact not_covers_of_disagree hq hd hfit hfit' hne c hc

/-- **Refusal of a decoder outside the class** (CM18). -/
theorem recover_unsupported (h : ¬ (Odd q ∧ PairedDecoder d)) (cell : S → n × n)
    (labels : S → C) (i j : n) : recover d cell labels i j = .unsupported := by
  unfold recover
  rw [ite_eq_right h]

/-- **Refusal of samples that no phases fit** (CM18). -/
theorem recover_inconsistent (hq : Odd q) (hd : PairedDecoder d)
    (h : ¬ ∃ r s : n → ZMod q, Fits d cell labels r s) (i j : n) :
    recover d cell labels i j = .inconsistent := by
  unfold recover
  rw [ite_eq_left ⟨hq, hd⟩, ite_eq_right h]

/-! The folded table with a withheld cell is accepted; the three cells of
`paired_unseen_ambiguity` have two completions with different labels at the fourth. -/

/-- The phases `i + j` fit the three samples of the CM16 limit. -/
lemma ambiguous_fit : Fits pairedBoolDecoder threeCells pairedThreeLabels cornerPhase
    cornerPhase := by
  intro t
  fin_cases t <;> decide

/-- The phases `i - j` fit them as well. -/
lemma ambiguous_fit_neg : Fits pairedBoolDecoder threeCells pairedThreeLabels cornerPhase
    (fun j => -cornerPhase j) := by
  intro t
  fin_cases t <;> decide

example := recover_sound (recover_of_labels (by decide) (pairedDecoder_fold 2) held_fits
  held_anchor held_reached_row held_reached_col) held_fits_reflected
example := recover_of_labels (by decide) (pairedDecoder_fold 2) held_fits held_anchor
  held_reached_row held_reached_col
example := recover_accept (by decide) (pairedDecoder_fold 2) held_fits
  (covers_of_labels (pairedDecoder_fold 2) held_fits held_anchor held_reached_row
    held_reached_col)
example := not_covers_of_disagree (by decide) pairedDecoder_bool ambiguous_fit ambiguous_fit_neg
  (i := 1) (j := 1) (by decide)
example := recover_uncertified (by decide) pairedDecoder_bool ambiguous_fit ambiguous_fit_neg
  (i := 1) (j := 1) (by decide)
example := recover_unsupported (d := cornerDecoder)
  (fun h => absurd (h.2.even 1) (by decide)) threeCells (fun _ => true) 1 1
example := recover_inconsistent (d := foldDecoder 1)
  (cell := fun _ : Fin 2 => ((0 : Fin 1), (0 : Fin 1))) (labels := ![0, 1]) (by decide)
  (pairedDecoder_fold 1) (fun ⟨_, _, h⟩ => absurd ((h 0).symm.trans (h 1)) (by decide)) 0 0

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
