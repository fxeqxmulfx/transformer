module

public import Transformer.Grokking.FlatSpectrum.ScoreShift
public import Mathlib.Logic.Relation

/-!
# What the seen cells determine of a comparison of two free scores

Source: `review/flat_spectrum.md`, *A comparison of two free scores* (added), items CS3 to CS5.
A link of a chain (`scoreChain`) goes from a column up to a row of a seen cell with
`h j ≤ f i`, and from a row up to a column of a seen cell with `f i < h j`. Fitting scores
do not decrease along a chain (`ScoreFit.le_of_chain`). So a chain from the column of a cell
to its row gives `h' j ≤ f' i` for all fitting scores, and a chain from its row to its column
gives `f' i < h' j` (`ScoreFit.lt_of_chain`). The rows and columns that the chains reach from
a row or a column are closed upwards (`scoreUpClosed_chain`), and with `ScoreShift`:

* some fitting scores have `f' i < h' j` exactly when no chain goes from the column `j` to
  the row `i` (`exists_scoreFit_lt_iff`);
* some fitting scores have `h' j ≤ f' i` exactly when no chain goes from the row `i` to the
  column `j` (`exists_scoreFit_le_iff`).

At the boundary no chain helps. A chain from the column to the row of a cell with equal
scores is the cell itself (`mem_of_chain_of_eq`), and so is a chain from the row to the column
when no score of a column lies strictly between the two (`mem_of_chain_of_gap`). So an unseen
cell with `f i = h j` has fitting scores with `f' i < h' j`, and an unseen cell with integer
scores and `f i + 1 = h j` has fitting scores with `h' j ≤ f' i`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Relation

variable {m n : Type*} {Ω : Set (m × n)} {f f' : m → ℝ} {h h' : n → ℝ}

/-- One link of a chain of seen cells: from a column up to a row where `h j ≤ f i`, and from
a row up to a column where `f i < h j`. -/
def scoreChain (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) : m ⊕ n → m ⊕ n → Prop
  | .inr j, .inl i => (i, j) ∈ Ω ∧ h j ≤ f i
  | .inl i, .inr j => (i, j) ∈ Ω ∧ f i < h j
  | _, _ => False

/-- **CS4**, `review/flat_spectrum.md`: fitting scores do not decrease along a chain. -/
theorem ScoreFit.le_of_chain (hfit : ScoreFit Ω f h f' h') {u v : m ⊕ n}
    (huv : ReflTransGen (scoreChain Ω f h) u v) : Sum.elim f' h' u ≤ Sum.elim f' h' v := by
  induction huv with
  | refl => exact le_rfl
  | @tail b c _ hstep ih =>
    refine ih.trans ?_
    rcases b with i | j <;> rcases c with i' | j'
    · exact hstep.elim
    · exact (not_le.mp fun hle => not_le.mpr hstep.2 ((hfit _ hstep.1).mp hle)).le
    · exact (hfit _ hstep.1).mpr hstep.2
    · exact hstep.elim

/-- **CS4**, `review/flat_spectrum.md`: a chain from the row of a cell to its column puts
the row below the column in all fitting scores. -/
theorem ScoreFit.lt_of_chain (hfit : ScoreFit Ω f h f' h') {i : m} {j : n}
    (hc : ReflTransGen (scoreChain Ω f h) (.inl i) (.inr j)) : f' i < h' j := by
  rcases hc.cases_tail with h0 | ⟨w, hw, hstep⟩
  · cases h0
  · rcases w with i₁ | j₁
    · exact lt_of_le_of_lt (hfit.le_of_chain hw)
        (not_le.mp fun hle => not_le.mpr hstep.2 ((hfit _ hstep.1).mp hle))
    · exact hstep.elim

/-- One seen cell that wraps is a chain from its column to its row. -/
example : (fun _ : Unit => (0 : ℝ)) () ≤ (fun _ : Unit => (1 : ℝ)) () :=
  (scoreFit_self {((), ())} (fun _ => 1) (fun _ => 0)).le_of_chain
    (u := .inr ()) (v := .inl ()) (.single ⟨rfl, by norm_num⟩)

/-- One seen cell that does not wrap is a chain from its row to its column. -/
example : (fun _ : Unit => (0 : ℝ)) () < (fun _ : Unit => (1 : ℝ)) () :=
  (scoreFit_self {((), ())} (fun _ => 0) (fun _ => 1)).lt_of_chain
    (.single ⟨rfl, by norm_num⟩)

/-- The rows and the columns that the chains reach from `u` are closed upwards. -/
theorem scoreUpClosed_chain (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) (u : m ⊕ n) :
    ScoreUpClosed Ω f h {i | ReflTransGen (scoreChain Ω f h) u (.inl i)}
      {j | ReflTransGen (scoreChain Ω f h) u (.inr j)} :=
  fun _ hc => ⟨fun hle hj => hj.tail ⟨hc, hle⟩, fun hlt hi => hi.tail ⟨hc, hlt⟩⟩

/-- **CS3 and CS4**, `review/flat_spectrum.md`: some fitting scores put the row `i` below
the column `j` exactly when no chain of seen cells goes from the column to the row. -/
theorem exists_scoreFit_lt_iff (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) (i : m) (j : n) :
    (∃ f' h', ScoreFit Ω f h f' h' ∧ f' i < h' j) ↔
      ¬ReflTransGen (scoreChain Ω f h) (.inr j) (.inl i) := by
  constructor
  · rintro ⟨f', h', hfit, hlt⟩ hc
    exact not_le.mpr hlt (hfit.le_of_chain hc)
  · exact fun hc => exists_scoreFit_lt (scoreUpClosed_chain Ω f h (.inr j)) hc .refl

/-- **CS3 and CS4**, `review/flat_spectrum.md`: some fitting scores put the column `j` at
most at the row `i` exactly when no chain of seen cells goes from the row to the column. -/
theorem exists_scoreFit_le_iff (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) (i : m) (j : n) :
    (∃ f' h', ScoreFit Ω f h f' h' ∧ h' j ≤ f' i) ↔
      ¬ReflTransGen (scoreChain Ω f h) (.inl i) (.inr j) := by
  constructor
  · rintro ⟨f', h', hfit, hle⟩ hc
    exact not_le.mpr (hfit.lt_of_chain hc) hle
  · exact fun hc => exists_scoreFit_le (scoreUpClosed_chain Ω f h (.inl i)) .refl hc

/-- **CS5**, `review/flat_spectrum.md`: a chain from the column to the row of a cell with
equal scores is the cell itself, which is then seen. -/
theorem mem_of_chain_of_eq {i : m} {j : n} (heq : f i = h j)
    (hc : ReflTransGen (scoreChain Ω f h) (.inr j) (.inl i)) : (i, j) ∈ Ω := by
  rcases hc.cases_head with h0 | ⟨w, hstep, hw⟩
  · cases h0
  · rcases w with i₁ | j₁
    · rcases hw.cases_head with h1 | ⟨w₂, hstep₂, hw₂⟩
      · obtain rfl := Sum.inl.inj h1
        exact hstep.1
      · rcases w₂ with i₂ | j₂
        · exact hstep₂.elim
        · have hle := (scoreFit_self Ω f h).le_of_chain hw₂
          simp only [Sum.elim_inr, Sum.elim_inl] at hle
          linarith [hstep.2, hstep₂.2]
    · exact hstep.elim

/-- **CS5**, `review/flat_spectrum.md`: a chain from the row to the column of a cell is the
cell itself when no score of a column lies strictly between its two scores. -/
theorem mem_of_chain_of_gap {i : m} {j : n} (hgap : ∀ j', ¬(f i < h j' ∧ h j' < h j))
    (hc : ReflTransGen (scoreChain Ω f h) (.inl i) (.inr j)) : (i, j) ∈ Ω := by
  rcases hc.cases_head with h0 | ⟨w, hstep, hw⟩
  · cases h0
  · rcases w with i₁ | j₁
    · exact hstep.elim
    · rcases hw.cases_head with h1 | ⟨w₂, hstep₂, hw₂⟩
      · obtain rfl := Sum.inr.inj h1
        exact hstep.1
      · rcases w₂ with i₂ | j₂
        · exact absurd ⟨hstep.2, lt_of_le_of_lt hstep₂.2
            ((scoreFit_self Ω f h).lt_of_chain hw₂)⟩ (hgap j₁)
        · exact hstep₂.elim

example : ((), ()) ∈ (Set.univ : Set (Unit × Unit)) :=
  mem_of_chain_of_eq (f := fun _ => 0) (h := fun _ => 0) rfl (.single ⟨trivial, le_rfl⟩)

example : ((), ()) ∈ (Set.univ : Set (Unit × Unit)) :=
  mem_of_chain_of_gap (f := fun _ => 0) (h := fun _ => 1) (fun _ hj => lt_irrefl _ hj.2)
    (.single ⟨trivial, by norm_num⟩)

/-- **CS5**, `review/flat_spectrum.md`: an unseen cell with equal scores has fitting scores
with the row below the column. -/
theorem exists_scoreFit_lt_of_eq {i : m} {j : n} (heq : f i = h j) (hij : (i, j) ∉ Ω) :
    ∃ f' h', ScoreFit Ω f h f' h' ∧ f' i < h' j :=
  (exists_scoreFit_lt_iff Ω f h i j).mpr fun hc => hij (mem_of_chain_of_eq heq hc)

/-- **CS5**, `review/flat_spectrum.md`: an unseen cell with integer scores, the row one below
the column, has fitting scores with the column at most at the row. -/
theorem exists_scoreFit_le_of_add_one {F : m → ℤ} {H : n → ℤ} {i : m} {j : n}
    (hF : F i + 1 = H j) (hij : (i, j) ∉ Ω) :
    ∃ f' h', ScoreFit Ω (fun i => (F i : ℝ)) (fun j => (H j : ℝ)) f' h' ∧ h' j ≤ f' i := by
  refine (exists_scoreFit_le_iff Ω _ _ i j).mpr fun hc => hij (mem_of_chain_of_gap ?_ hc)
  rintro j' ⟨h1, h2⟩
  have h1' : F i < H j' := by exact_mod_cast h1
  have h2' : H j' < H j := by exact_mod_cast h2
  omega

/-- A table with no seen cell: both layers at the boundary are open. -/
example : ∃ f' h', ScoreFit (∅ : Set (Unit × Unit)) (fun _ => 3) (fun _ => 3) f' h' ∧
    f' () < h' () :=
  exists_scoreFit_lt_of_eq rfl (Set.notMem_empty _)

example : ∃ f' h', ScoreFit (∅ : Set (Unit × Unit)) (fun _ => ((2 : ℤ) : ℝ))
    (fun _ => ((3 : ℤ) : ℝ)) f' h' ∧ h' () ≤ f' () :=
  exists_scoreFit_le_of_add_one (F := fun _ => 2) (H := fun _ => 3) rfl (Set.notMem_empty _)

end Transformer.Grokking.FlatSpectrum
