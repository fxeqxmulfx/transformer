module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Signs

/-!
# A checkable certificate for decoded answers

Source: `review/flat_spectrum.md`, CM18. The unknown table is `y i j = d (r i + s j)` with a
fixed known decoder `d` of the class `PairedDecoder`. A training sample says only that
`r i + s j` lies in the fiber of its label. A certificate proposes phases and names a rectangle
of four training cells: two rows `top`, `bot` and two columns `left`, `right`. From the
rectangle it reaches further rows and columns (`Pinned`): the columns observed in both rows,
then a row with two training cells in reached columns of different proposed phases, and a
column likewise.

`Accepts` is what a checker verifies. The proposed phases have the training labels at the
training cells of reached rows and columns, and nothing is asked of them elsewhere; the rows
of the rectangle have different phases, its columns have different phases, and the four
phases do not sum to zero. Every condition is a finite statement about the certificate, the
decoder and the training samples; none mentions a label that was not observed.

`anchor_rigid`: any phases that fit all training samples are, on the two rows and on every
column observed in both, the proposed ones up to one sign and one shift.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

variable {q : ℕ} {C n S : Type*} {d : ZMod q → C} {cell : S → n × n} {labels : S → C}

/-- The cell is a training cell. -/
def Observed (cell : S → n × n) (ij : n × n) : Prop := ∃ t, cell t = ij

/-- Row and column phases whose decoded sums are the training labels (CM18). -/
def Fits (d : ZMod q → C) (cell : S → n × n) (labels : S → C) (r s : n → ZMod q) : Prop :=
  ∀ t, d (r (cell t).1 + s (cell t).2) = labels t

/-- **The certificate** (CM18): proposed phases and the rectangle `{top, bot} × {left, right}`. -/
structure Certificate (n : Type*) (q : ℕ) where
  /-- The proposed phase of each row. -/
  row : n → ZMod q
  /-- The proposed phase of each column. -/
  col : n → ZMod q
  /-- The first row of the rectangle. -/
  top : n
  /-- The second row of the rectangle. -/
  bot : n
  /-- The first column of the rectangle. -/
  left : n
  /-- The second column of the rectangle. -/
  right : n

/-- The label that the certificate proposes for a cell. -/
def Certificate.decoded (d : ZMod q → C) (c : Certificate n q) (i j : n) : C :=
  d (c.row i + c.col j)

/-- **The rows and columns that the certificate reaches** (CM18): the rows of the rectangle,
the columns observed in both, then any row with two training cells in reached columns of
different proposed phases, and any column with two training cells in reached rows of
different proposed phases. Rows are `Sum.inl`, columns `Sum.inr`. -/
inductive Pinned (cell : S → n × n) (c : Certificate n q) : n ⊕ n → Prop
  | top : Pinned cell c (.inl c.top)
  | bot : Pinned cell c (.inl c.bot)
  | anchored {j : n} : Observed cell (c.top, j) → Observed cell (c.bot, j) →
      Pinned cell c (.inr j)
  | row {i j j' : n} : Pinned cell c (.inr j) → Pinned cell c (.inr j') →
      c.col j ≠ c.col j' → Observed cell (i, j) → Observed cell (i, j') → Pinned cell c (.inl i)
  | col {j i i' : n} : Pinned cell c (.inl i) → Pinned cell c (.inl i') →
      c.row i ≠ c.row i' → Observed cell (i, j) → Observed cell (i', j) → Pinned cell c (.inr j)

/-- **What a checker verifies** (CM18): the proposed phases have the training label at every
training cell of a reached row and a reached column, the four cells of the rectangle are
training cells, its rows have different phases, its columns have different phases, and the
four phases do not sum to zero. -/
def Accepts (d : ZMod q → C) (cell : S → n × n) (labels : S → C) (c : Certificate n q) : Prop :=
  (∀ t, Pinned cell c (.inl (cell t).1) → Pinned cell c (.inr (cell t).2) →
      d (c.row (cell t).1 + c.col (cell t).2) = labels t) ∧
    (Observed cell (c.top, c.left) ∧ Observed cell (c.bot, c.left)) ∧
    (Observed cell (c.top, c.right) ∧ Observed cell (c.bot, c.right)) ∧
    c.row c.top ≠ c.row c.bot ∧ c.col c.left ≠ c.col c.right ∧
    c.row c.top + c.row c.bot + c.col c.left + c.col c.right ≠ 0

/-- At a training cell of a reached row and a reached column, the proposed phases and any
phases that fit the training samples have the same label. -/
lemma Accepts.agree {c : Certificate n q} (hc : Accepts d cell labels c) {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) {i j : n} (hi : Pinned cell c (.inl i))
    (hj : Pinned cell c (.inr j)) (ho : Observed cell (i, j)) :
    d (c.row i + c.col j) = d (r i + s j) := by
  obtain ⟨t, ht⟩ := ho
  have h1 := hc.1 t (by rw [ht]; exact hi) (by rw [ht]; exact hj)
  have h2 := hfit t
  rw [ht] at h1 h2
  exact h1.trans h2.symm

/-- **The rectangle fixes its rows and their shared columns up to a sign and a shift** (CM18).
The phases `r`, `s` are any that fit the training samples; nothing is assumed about them at
other cells. -/
theorem anchor_rigid (hq : Odd q) (hd : PairedDecoder d) {c : Certificate n q}
    (hc : Accepts d cell labels c) {r s : n → ZMod q} (hfit : Fits d cell labels r s) :
    ∃ ε g : ZMod q, (ε = 1 ∨ ε = -1) ∧ r c.top = ε * c.row c.top + g ∧
      r c.bot = ε * c.row c.bot + g ∧
      ∀ j, Observed cell (c.top, j) → Observed cell (c.bot, j) → s j = ε * c.col j - g := by
  have key (j : n) (ht : Observed cell (c.top, j)) (hb : Observed cell (c.bot, j)) :
      (r c.top + s j = c.row c.top + c.col j ∨ r c.top + s j = -(c.row c.top + c.col j)) ∧
      (r c.top + s j + (r c.bot - r c.top) =
          c.row c.top + c.col j + (c.row c.bot - c.row c.top) ∨
        r c.top + s j + (r c.bot - r c.top) =
          -(c.row c.top + c.col j + (c.row c.bot - c.row c.top))) := by
    refine ⟨(hd _ _).1 (hc.agree hfit .top (.anchored ht hb) ht), ?_⟩
    rcases (hd _ _).1 (hc.agree hfit .bot (.anchored ht hb) hb) with h | h
    · exact Or.inl (by linear_combination h)
    · exact Or.inr (by linear_combination h)
  obtain ⟨-, ⟨hotl, hobl⟩, ⟨hotr, hobr⟩, hrow, hcol, hsum⟩ := hc
  obtain ⟨hl, hlδ⟩ := key c.left hotl hobl
  obtain ⟨hr, hrδ⟩ := key c.right hotr hobr
  have hδ := delta_sign hq hl hlδ hr hrδ (fun h => hcol (add_left_cancel h))
    (fun h => hsum (by linear_combination h))
  obtain ⟨ε, hε, hεδ⟩ : ∃ ε : ZMod q, (ε = 1 ∨ ε = -1) ∧
      r c.bot - r c.top = ε * (c.row c.bot - c.row c.top) := by
    rcases hδ with h | h
    · exact ⟨1, Or.inl rfl, by rw [one_mul]; exact h⟩
    · exact ⟨-1, Or.inr rfl, by rw [neg_one_mul]; exact h⟩
  refine ⟨ε, r c.top - ε * c.row c.top, hε, by ring, by linear_combination hεδ, ?_⟩
  intro j ht hb
  obtain ⟨hj, hjδ⟩ := key j ht hb
  rw [hεδ] at hjδ
  linear_combination column_sign hq hε (sub_ne_zero.2 (Ne.symm hrow)) hj hjδ

/-! The folded table of CM16 with the cell `(2, 1)` withheld: three rows, five latent states,
three classes; the latent states `1` and `4` at the cells `(1, 0)` and `(0, 1)` share a label. -/

/-- The eight training cells: all but `(2, 1)`. -/
def heldCells : Fin 8 → Fin 3 × Fin 3 :=
  ![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 2)]

/-- Their labels in the folded table. -/
def heldLabels (t : Fin 8) : Fin 3 := foldedLabels (heldCells t).1 (heldCells t).2

/-- The phases of CM16 with the rectangle of the rows `0`, `1` and the columns `0`, `2`. -/
def heldCert : Certificate (Fin 3) 5 := ⟨foldedRow, foldedColumn, 0, 1, 0, 2⟩

/-- The phases of CM16 fit the eight training samples. -/
lemma held_fits : Fits (foldDecoder 2) heldCells heldLabels foldedRow foldedColumn := fun t =>
  congrFun (congrFun folded_decoding (heldCells t).1) (heldCells t).2

/-- The certificate of the folded table is accepted (CM18). -/
theorem heldCert_accepts : Accepts (foldDecoder 2) heldCells heldLabels heldCert :=
  ⟨fun t _ _ => held_fits t, ⟨⟨0, rfl⟩, ⟨3, rfl⟩⟩, ⟨⟨2, rfl⟩, ⟨5, rfl⟩⟩, by decide, by decide,
    by decide⟩

/-- The row of the withheld cell is reached through its cells in the columns `0` and `2`. -/
lemma held_row : Pinned heldCells heldCert (.inl 2) :=
  .row (.anchored ⟨0, rfl⟩ ⟨3, rfl⟩) (.anchored ⟨2, rfl⟩ ⟨5, rfl⟩) (by decide) ⟨6, rfl⟩ ⟨7, rfl⟩

/-- The column of the withheld cell is observed in both rows of the rectangle. -/
lemma held_col : Pinned heldCells heldCert (.inr 1) := .anchored ⟨1, rfl⟩ ⟨4, rfl⟩

example := heldCert_accepts.agree held_fits held_row (.anchored ⟨0, rfl⟩ ⟨3, rfl⟩) ⟨6, rfl⟩
example := anchor_rigid (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
