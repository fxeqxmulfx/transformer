module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Rigidity

/-!
# Coverage conditions on training labels alone

Source: `review/flat_spectrum.md`, CM18. The conditions of `Accepts` and `Pinned` speak of the
proposed phases. For phases that fit the training samples they follow from conditions on the
training labels alone, which can be checked before any phase is known:

* `Anchor`: a rectangle of training cells whose two columns carry pairs of labels that are
  neither equal nor exchanged, and a training column where the two rows have different labels;
* `Reached`: the closure of `Pinned` in which "different phases" is witnessed by two training
  cells of one row, or of one column, with different labels.

`decoding_of_labels`: under these conditions any two completions of the training samples have
the same label at the cell. The conditions are sufficient; they are not claimed necessary.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

variable {q : ℕ} {C n S : Type*} {d : ZMod q → C} {cell : S → n × n} {labels : S → C}

/-- Two training cells of one row with different labels lie in columns of different phases. -/
lemma Fits.col_ne {r s : n → ZMod q} (hfit : Fits d cell labels r s) {t t' : S}
    (hrow : (cell t).1 = (cell t').1) (hne : labels t ≠ labels t') :
    s (cell t).2 ≠ s (cell t').2 := by
  intro h
  apply hne
  rw [← hfit t, ← hfit t', hrow, h]

/-- Two training cells of one column with different labels lie in rows of different phases. -/
lemma Fits.row_ne {r s : n → ZMod q} (hfit : Fits d cell labels r s) {t t' : S}
    (hcol : (cell t).2 = (cell t').2) (hne : labels t ≠ labels t') :
    r (cell t).1 ≠ r (cell t').1 := by
  intro h
  apply hne
  rw [← hfit t, ← hfit t', hcol, h]

/-- **The label conditions of a rectangle** (CM18). The samples `tl`, `bl`, `tr`, `br` are the
corners of a rectangle; `ts` and `bs` lie in its two rows and in one column and have different
labels; the pair of labels of the left column is neither the pair of the right column nor that
pair exchanged. -/
def Anchor (cell : S → n × n) (labels : S → C) (tl bl tr br ts bs : S) : Prop :=
  ((cell bl).2 = (cell tl).2 ∧ (cell br).2 = (cell tr).2) ∧
    ((cell tr).1 = (cell tl).1 ∧ (cell br).1 = (cell bl).1) ∧
    ((cell ts).1 = (cell tl).1 ∧ (cell bs).1 = (cell bl).1 ∧ (cell ts).2 = (cell bs).2) ∧
    labels ts ≠ labels bs ∧ (labels tl, labels bl) ≠ (labels tr, labels br) ∧
    (labels tl, labels bl) ≠ (labels br, labels tr)

/-- Phases with the rectangle of the samples `tl`, `bl`, `tr`. -/
def anchorCert (cell : S → n × n) (r s : n → ZMod q) (tl bl tr : S) : Certificate n q :=
  ⟨r, s, (cell tl).1, (cell bl).1, (cell tl).2, (cell tr).2⟩

/-- **Acceptance of the rectangle from training labels** (CM18): any phases that fit the
training samples are accepted with a rectangle that meets the label conditions. -/
theorem Anchor.accepts (hd : PairedDecoder d) {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) {tl bl tr br ts bs : S}
    (ha : Anchor cell labels tl bl tr br ts bs) :
    Accepts d cell labels (anchorCert cell r s tl bl tr) := by
  obtain ⟨⟨hleft, hright⟩, ⟨htop, hbot⟩, ⟨hst, hsb, hsc⟩, hsplit, hpair, hswap⟩ := ha
  refine ⟨fun t _ _ => hfit t, ⟨⟨tl, rfl⟩, ⟨bl, Prod.ext rfl hleft⟩⟩,
    ⟨⟨tr, Prod.ext htop rfl⟩, ⟨br, Prod.ext hbot hright⟩⟩, ?_, ?_, ?_⟩
  · intro h
    apply hsplit
    change r (cell tl).1 = r (cell bl).1 at h
    rw [← hfit ts, ← hfit bs, hst, hsb, hsc, h]
  · intro h
    apply hpair
    change s (cell tl).2 = s (cell tr).2 at h
    rw [← hfit tl, ← hfit bl, ← hfit tr, ← hfit br, hleft, hright, htop, hbot, h]
  · intro h
    apply hswap
    change r (cell tl).1 + r (cell bl).1 + s (cell tl).2 + s (cell tr).2 = 0 at h
    rw [← hfit tl, ← hfit bl, ← hfit tr, ← hfit br, hleft, hright, htop, hbot,
      show r (cell tl).1 + s (cell tl).2 = -(r (cell bl).1 + s (cell tr).2) by
        linear_combination h,
      show r (cell bl).1 + s (cell tl).2 = -(r (cell tl).1 + s (cell tr).2) by
        linear_combination h, hd.even, hd.even]

/-- **Coverage in training labels alone** (CM18): what the rows `top`, `bot` reach when two
columns count as different once some row has different training labels in them, and two rows
once some column has. -/
inductive Reached (cell : S → n × n) (labels : S → C) (top bot : n) : n ⊕ n → Prop
  | top : Reached cell labels top bot (.inl top)
  | bot : Reached cell labels top bot (.inl bot)
  | anchored {j : n} : Observed cell (top, j) → Observed cell (bot, j) →
      Reached cell labels top bot (.inr j)
  | row {i : n} {t t' : S} : Reached cell labels top bot (.inr (cell t).2) →
      Reached cell labels top bot (.inr (cell t').2) → (cell t).1 = (cell t').1 →
      labels t ≠ labels t' → Observed cell (i, (cell t).2) → Observed cell (i, (cell t').2) →
      Reached cell labels top bot (.inl i)
  | col {j : n} {t t' : S} : Reached cell labels top bot (.inl (cell t).1) →
      Reached cell labels top bot (.inl (cell t').1) → (cell t).2 = (cell t').2 →
      labels t ≠ labels t' → Observed cell ((cell t).1, j) → Observed cell ((cell t').1, j) →
      Reached cell labels top bot (.inr j)

/-- What the labels reach, every certificate with fitting phases and these two rows reaches. -/
theorem Reached.pinned {c : Certificate n q} (hfit : Fits d cell labels c.row c.col)
    {v : n ⊕ n} (h : Reached cell labels c.top c.bot v) : Pinned cell c v := by
  induction h with
  | top => exact .top
  | bot => exact .bot
  | anchored ht hb => exact .anchored ht hb
  | row _ _ hrow hne ho ho' ih ih' => exact .row ih ih' (hfit.col_ne hrow hne) ho ho'
  | col _ _ hcol hne ho ho' ih ih' => exact .col ih ih' (hfit.row_ne hcol hne) ho ho'

/-- An accepted certificate that reaches the row and the column of the cell. -/
def Covers (d : ZMod q → C) (cell : S → n × n) (labels : S → C) (i j : n)
    (c : Certificate n q) : Prop :=
  Accepts d cell labels c ∧ Pinned cell c (.inl i) ∧ Pinned cell c (.inr j)

/-- **Acceptance under conditions on training labels** (CM18): if the table is realizable,
with whatever phases, then those phases form a certificate that covers every cell whose row
and column the labels reach from a rectangle with the label conditions. -/
theorem covers_of_labels (hd : PairedDecoder d) {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) {tl bl tr br ts bs : S}
    (ha : Anchor cell labels tl bl tr br ts bs) {i j : n}
    (hi : Reached cell labels (cell tl).1 (cell bl).1 (.inl i))
    (hj : Reached cell labels (cell tl).1 (cell bl).1 (.inr j)) :
    Covers d cell labels i j (anchorCert cell r s tl bl tr) :=
  ⟨ha.accepts hd hfit, Reached.pinned (c := anchorCert cell r s tl bl tr) hfit hi,
    Reached.pinned (c := anchorCert cell r s tl bl tr) hfit hj⟩

/-- **Identification from training labels** (CM18): under the label conditions, any two
completions of the training samples have the same label at the cell. With `r`, `s` the phases
of the unknown target, every fitted completion predicts the target's label there. -/
theorem decoding_of_labels (hq : Odd q) (hd : PairedDecoder d) {r s r' s' : n → ZMod q}
    (hfit : Fits d cell labels r s) (hfit' : Fits d cell labels r' s')
    {tl bl tr br ts bs : S} (ha : Anchor cell labels tl bl tr br ts bs) {i j : n}
    (hi : Reached cell labels (cell tl).1 (cell bl).1 (.inl i))
    (hj : Reached cell labels (cell tl).1 (cell bl).1 (.inr j)) :
    d (r' i + s' j) = d (r i + s j) := by
  obtain ⟨hc, hpi, hpj⟩ := covers_of_labels hd hfit ha hi hj
  exact fits_agree hq hd hc hfit' hfit hpi hpj

/-! The folded table with the withheld cell `(2, 1)` meets the label conditions. -/

/-- The rows `0`, `1` and the columns `0`, `2` of the folded table meet the label conditions. -/
lemma held_anchor : Anchor heldCells heldLabels 0 3 2 5 0 3 :=
  ⟨⟨rfl, rfl⟩, ⟨rfl, rfl⟩, ⟨rfl, rfl, rfl⟩, by decide, by decide, by decide⟩

/-- The labels reach the row of the withheld cell through the columns `0` and `2`. -/
lemma held_reached_row : Reached heldCells heldLabels 0 1 (.inl 2) :=
  .row (t := 0) (t' := 2) (.anchored ⟨0, rfl⟩ ⟨3, rfl⟩) (.anchored ⟨2, rfl⟩ ⟨5, rfl⟩) rfl
    (by decide) ⟨6, rfl⟩ ⟨7, rfl⟩

/-- The labels reach the column of the withheld cell: it is observed in both rows. -/
lemma held_reached_col : Reached heldCells heldLabels 0 1 (.inr 1) :=
  .anchored ⟨1, rfl⟩ ⟨4, rfl⟩

example := held_fits.col_ne (t := 0) (t' := 2) rfl (by decide)
example := held_fits.row_ne (t := 0) (t' := 3) rfl (by decide)
example := held_anchor.accepts (pairedDecoder_fold 2) held_fits
example := Reached.pinned (c := heldCert) held_fits held_reached_row
example := covers_of_labels (pairedDecoder_fold 2) held_fits held_anchor held_reached_row
  held_reached_col
example : foldDecoder 2 ((fun i => -foldedRow i + 1) 2 + (fun j => -foldedColumn j - 1) 1) =
    foldDecoder 2 (foldedRow 2 + foldedColumn 1) :=
  decoding_of_labels (by decide) (pairedDecoder_fold 2) held_fits held_fits_reflected
    held_anchor held_reached_row held_reached_col

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
