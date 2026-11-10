module

public import Transformer.Grokking.FlatSpectrum.StepHalfPlane

/-!
# A comparison of two free scores keeps its seen cells under a shift

Source: `review/flat_spectrum.md`, *A comparison of two free scores* (added), items CS1 to CS3.
A table of comparisons has a real score `f i` for each row and `h j` for each column, and its
cell `(i, j)` says whether `h j ≤ f i`. The wrap of the next step is such a table in the two
steps read (`modStep_neg_iff_score_le`): it wraps exactly when
`p − 3 d₂ − 2p [d₂ < 0] ≤ 2 d₁`.

A learner that knows that the table is a comparison, and not its scores, has from a seen cell
one inequality. Scores `f'`, `h'` fit the seen cells `Ω` when they compare there as `f`, `h`
do (`ScoreFit`). A set `A` of rows with a set `B` of columns is closed upwards
(`ScoreUpClosed`) when a seen cell with `h j ≤ f i` and `j ∈ B` has `i ∈ A`, and a seen cell
with `f i < h j` and `i ∈ A` has `j ∈ B`. Raising the scores of such sets by one constant
`c ≥ 0` fits the seen cells (`scoreFit_indicator`). So an unseen cell that the sets separate
is open: with `i ∉ A` and `j ∈ B` some fitting scores have `f' i < h' j`
(`exists_scoreFit_lt`), and with `i ∈ A` and `j ∉ B` some have `h' j ≤ f' i`
(`exists_scoreFit_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

variable {m n : Type*}

/-- **CS1**, `review/flat_spectrum.md`: the wrap of the next step is a comparison of a score
of each of the two steps read. With `d₁ = b − a` and `d₂ = modStep p a b` it wraps exactly
when `p − 3 d₂ − 2p [d₂ < 0] ≤ 2 d₁`. -/
theorem modStep_neg_iff_score_le {p a b : ℕ} (ha : a < p) (hb : b < p) :
    modStep p b ((a + b) % p) < 0 ↔
      (p : ℤ) - 3 * modStep p a b - 2 * (if modStep p a b < 0 then (p : ℤ) else 0) ≤
        2 * ((b : ℤ) - a) := by
  rw [modStep_neg_iff_steps ha hb]
  constructor <;> intro h <;> linarith

/-- Modulo `97`, from `50` and `60`: `97 − 3 · (−47) − 2 · 97 = 44 > 20`, and `60 + 13 < 97`. -/
example := modStep_neg_iff_score_le (p := 97) (a := 50) (b := 60) (by norm_num) (by norm_num)

/-- The scores `f'`, `h'` compare as `f`, `h` at every cell of `Ω`. -/
def ScoreFit (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) (f' : m → ℝ) (h' : n → ℝ) : Prop :=
  ∀ c ∈ Ω, (h' c.2 ≤ f' c.1 ↔ h c.2 ≤ f c.1)

/-- The rows `A` and the columns `B` are closed upwards along the cells of `Ω`: a cell with
`h j ≤ f i` and `j ∈ B` has `i ∈ A`, and a cell with `f i < h j` and `i ∈ A` has `j ∈ B`. -/
def ScoreUpClosed (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) (A : Set m) (B : Set n) : Prop :=
  ∀ c ∈ Ω, (h c.2 ≤ f c.1 → c.2 ∈ B → c.1 ∈ A) ∧ (f c.1 < h c.2 → c.1 ∈ A → c.2 ∈ B)

/-- The scores themselves fit. -/
theorem scoreFit_self (Ω : Set (m × n)) (f : m → ℝ) (h : n → ℝ) : ScoreFit Ω f h f h :=
  fun _ _ => Iff.rfl

/-- **CS2**, `review/flat_spectrum.md`: raising the scores of rows and columns closed upwards
by one constant `c ≥ 0` keeps every seen comparison. -/
theorem scoreFit_indicator {Ω : Set (m × n)} {f : m → ℝ} {h : n → ℝ} {A : Set m} {B : Set n}
    (hAB : ScoreUpClosed Ω f h A B) {c : ℝ} (hc : 0 ≤ c) :
    ScoreFit Ω f h (f + A.indicator fun _ => c) (h + B.indicator fun _ => c) := by
  intro x hx
  obtain ⟨h1, h2⟩ := hAB x hx
  simp only [Pi.add_apply]
  by_cases hA : x.1 ∈ A <;> by_cases hB : x.2 ∈ B
  · rw [Set.indicator_of_mem hA, Set.indicator_of_mem hB]
    constructor <;> intro h' <;> linarith
  · have hle : h x.2 ≤ f x.1 := not_lt.mp fun hlt => hB (h2 hlt hA)
    rw [Set.indicator_of_mem hA, Set.indicator_of_notMem hB]
    exact ⟨fun _ => hle, fun _ => by linarith⟩
  · have hlt : ¬h x.2 ≤ f x.1 := fun hle => hA (h1 hle hB)
    rw [Set.indicator_of_notMem hA, Set.indicator_of_mem hB]
    exact ⟨fun h' => absurd (by linarith) hlt, fun h' => absurd h' hlt⟩
  · rw [Set.indicator_of_notMem hA, Set.indicator_of_notMem hB]
    simp

/-- One seen cell, with the score of its row below that of its column: the column alone is
closed upwards, and raising it keeps the cell. -/
example : ScoreFit ({((), ())} : Set (Unit × Unit)) (fun _ => 0) (fun _ => 1)
    ((fun _ => 0) + (∅ : Set Unit).indicator fun _ => 5)
    ((fun _ => 1) + (Set.univ : Set Unit).indicator fun _ => 5) :=
  scoreFit_indicator (fun _ _ => ⟨fun hle _ => absurd hle (by norm_num), fun _ hA => hA.elim⟩)
    (by norm_num)

/-- **CS3**, `review/flat_spectrum.md`: a cell whose column is in sets closed upwards and
whose row is not has fitting scores with the row below the column. -/
theorem exists_scoreFit_lt {Ω : Set (m × n)} {f : m → ℝ} {h : n → ℝ} {A : Set m} {B : Set n}
    (hAB : ScoreUpClosed Ω f h A B) {i : m} {j : n} (hi : i ∉ A) (hj : j ∈ B) :
    ∃ f' h', ScoreFit Ω f h f' h' ∧ f' i < h' j := by
  refine ⟨_, _, scoreFit_indicator hAB (c := |f i - h j| + 1) (by positivity), ?_⟩
  simp only [Pi.add_apply, Set.indicator_of_notMem hi, Set.indicator_of_mem hj, add_zero]
  linarith [le_abs_self (f i - h j)]

/-- **CS3**, `review/flat_spectrum.md`: a cell whose row is in sets closed upwards and whose
column is not has fitting scores with the column at most the row. -/
theorem exists_scoreFit_le {Ω : Set (m × n)} {f : m → ℝ} {h : n → ℝ} {A : Set m} {B : Set n}
    (hAB : ScoreUpClosed Ω f h A B) {i : m} {j : n} (hi : i ∈ A) (hj : j ∉ B) :
    ∃ f' h', ScoreFit Ω f h f' h' ∧ h' j ≤ f' i := by
  refine ⟨_, _, scoreFit_indicator hAB (c := |h j - f i|) (abs_nonneg _), ?_⟩
  simp only [Pi.add_apply, Set.indicator_of_mem hi, Set.indicator_of_notMem hj, add_zero]
  linarith [le_abs_self (h j - f i)]

/-- Two rows and one column; the cell of the row `false` is seen and does not wrap. The cell
of the row `true` wraps, `1 ≤ 5`, and is not seen: some fitting scores do not wrap there. -/
example : ∃ f' h', ScoreFit ({(false, ())} : Set (Bool × Unit)) (fun i => if i then 5 else 0)
    (fun _ => 1) f' h' ∧ f' true < h' () :=
  exists_scoreFit_lt (A := ∅) (B := Set.univ)
    (by rintro _ rfl; exact ⟨fun hle _ => absurd hle (by norm_num), fun _ hA => hA.elim⟩)
    (Set.notMem_empty _) (Set.mem_univ _)

/-- The same table with the cell of the row `true` seen and the other not: the row `true`
alone is closed upwards, and some fitting scores wrap at the cell of the row `false`. -/
example : ∃ f' h', ScoreFit ({(true, ())} : Set (Bool × Unit)) (fun i => if i then 5 else 0)
    (fun _ => 1) f' h' ∧ h' () ≤ f' false :=
  exists_scoreFit_le (A := Set.univ) (B := ∅)
    (by rintro _ rfl; exact ⟨fun _ _ => Set.mem_univ _, fun hlt _ => absurd hlt (by norm_num)⟩)
    (Set.mem_univ _) (Set.notMem_empty _)

end Transformer.Grokking.FlatSpectrum
