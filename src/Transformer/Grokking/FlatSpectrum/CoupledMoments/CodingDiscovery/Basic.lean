module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.PhaseCandidate

/-!
# The codings that the training rectangles of a categorical table allow

Source: `review/flat_spectrum.md`, CM17, which continues CM14: "Discovery of a coding remains a
discrete proposal step". Native reference: `experiments/convex_stress_relabel/src/coding.rs` at
414431c7, `discover`: "Stream rectangle constraints z[a]+z[d]-z[b]-z[c]=0 over class phases.
With prime vocabulary V, fix z[0]=0 and require exactly one remaining direction, a bijection,
and consistency on every observed edge"; its header: "This discrete search has no global
optimization or recovery certificate."

A training set is a family of cells `cell : S → n × n` of a square table with their labels
`labels : S → C`. Rows, columns and classes are names: nothing here adds or compares them, and
the only arithmetic is that of the values of a coding `z : C → ZMod p`.

* `rectangleCodes p cell labels` is the space of the codings that satisfy every observed
  rectangle, and `pinnedCodes p cell labels c₀` its part with `z c₀ = 0`.
* A coding that is a sum of a row phase and a column phase at every sample satisfies every
  rectangle (`mem_rectangleCodes_of_fit`), and so does every constant.
* The rectangles through one shared column give the others of their two rows
  (`rectangle_of_anchored`): the native elimination keeps only those.

The witness of the development is a `5 × 5` table with twelve observed cells, whose coding
exchanges two class names (`fiveCells`, `fiveLabels`, `fiveCode`). `Invariance` shows that the
spaces depend only on the set of observations.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

variable {p : ℕ} {C n S : Type*}

/-- Four training samples at the corners of a rectangle of the table: `t₁` and `t₂` in one row,
`t₃` and `t₄` in one row, `t₁` and `t₃` in one column, `t₂` and `t₄` in one column. -/
def Rectangle (cell : S → n × n) (t₁ t₂ t₃ t₄ : S) : Prop :=
  (cell t₁).1 = (cell t₂).1 ∧ (cell t₃).1 = (cell t₄).1 ∧
    (cell t₁).2 = (cell t₃).2 ∧ (cell t₂).2 = (cell t₄).2

instance [DecidableEq n] (cell : S → n × n) (t₁ t₂ t₃ t₄ : S) :
    Decidable (Rectangle cell t₁ t₂ t₃ t₄) :=
  inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

/-- **The codings of the training rectangles** (CM17): the maps `z : C → ZMod p` with
`z a + z d = z b + z c` for the labels `a`, `b` of two cells of one row and the labels `c`, `d`
of the cells of another row in the same two columns. -/
def rectangleCodes (p : ℕ) (cell : S → n × n) (labels : S → C) :
    Submodule (ZMod p) (C → ZMod p) where
  carrier := {z | ∀ t₁ t₂ t₃ t₄, Rectangle cell t₁ t₂ t₃ t₄ →
    z (labels t₁) + z (labels t₄) = z (labels t₂) + z (labels t₃)}
  add_mem' := by
    intro z w hz hw t₁ t₂ t₃ t₄ h
    simp only [Pi.add_apply]
    linear_combination hz t₁ t₂ t₃ t₄ h + hw t₁ t₂ t₃ t₄ h
  zero_mem' := fun _ _ _ _ _ => rfl
  smul_mem' := by
    intro a z hz t₁ t₂ t₃ t₄ h
    simp only [Pi.smul_apply, smul_eq_mul]
    linear_combination a * hz t₁ t₂ t₃ t₄ h

lemma mem_rectangleCodes {cell : S → n × n} {labels : S → C} {z : C → ZMod p} :
    z ∈ rectangleCodes p cell labels ↔ ∀ t₁ t₂ t₃ t₄, Rectangle cell t₁ t₂ t₃ t₄ →
      z (labels t₁) + z (labels t₄) = z (labels t₂) + z (labels t₃) :=
  Iff.rfl

/-- The codings of the training rectangles that vanish at the class `c₀` (CM17): a constant can
be added to a coding, and the native elimination fixes the phase of one class. -/
def pinnedCodes (p : ℕ) (cell : S → n × n) (labels : S → C) (c₀ : C) :
    Submodule (ZMod p) (C → ZMod p) :=
  rectangleCodes p cell labels ⊓ LinearMap.ker (LinearMap.proj c₀ : (C → ZMod p) →ₗ[ZMod p] _)

lemma mem_pinnedCodes {cell : S → n × n} {labels : S → C} {c₀ : C} {z : C → ZMod p} :
    z ∈ pinnedCodes p cell labels c₀ ↔ z ∈ rectangleCodes p cell labels ∧ z c₀ = 0 :=
  Iff.rfl

/-- A coding that is a sum of a row phase and a column phase at every training sample
satisfies every observed rectangle. -/
lemma mem_rectangleCodes_of_fit {cell : S → n × n} {labels : S → C} {z : C → ZMod p}
    {ρ σ : n → ZMod p} (hfit : ∀ t, z (labels t) = ρ (cell t).1 + σ (cell t).2) :
    z ∈ rectangleCodes p cell labels := by
  rintro t₁ t₂ t₃ t₄ ⟨h12, h34, h13, h24⟩
  rw [hfit t₁, hfit t₂, hfit t₃, hfit t₄, h12, h34, h13, h24]
  ring

/-- Every constant satisfies every observed rectangle. -/
lemma const_mem_rectangleCodes (cell : S → n × n) (labels : S → C) (β : ZMod p) :
    (fun _ => β) ∈ rectangleCodes p cell labels :=
  fun _ _ _ _ _ => rfl

/-- A constant can be subtracted from a coding of the training rectangles. -/
lemma sub_const_mem_rectangleCodes {cell : S → n × n} {labels : S → C} {z : C → ZMod p}
    (hz : z ∈ rectangleCodes p cell labels) (β : ZMod p) :
    (fun c => z c - β) ∈ rectangleCodes p cell labels := fun t₁ t₂ t₃ t₄ h => by
  linear_combination hz t₁ t₂ t₃ t₄ h

/-- **The rectangles through one shared column give the others of their two rows**: with the
samples `t₀`, `t₀'` of two rows in one column, the rectangles `t₀ t₁ / t₀' t₃` and
`t₀ t₂ / t₀' t₄` give `t₁ t₂ / t₃ t₄`. The native elimination keeps, for a pair of rows, only
the rectangles through their first shared column. -/
lemma rectangle_of_anchored {labels : S → C} {z : C → ZMod p} {t₀ t₀' t₁ t₂ t₃ t₄ : S}
    (h₁ : z (labels t₀) + z (labels t₃) = z (labels t₁) + z (labels t₀'))
    (h₂ : z (labels t₀) + z (labels t₄) = z (labels t₂) + z (labels t₀')) :
    z (labels t₁) + z (labels t₄) = z (labels t₂) + z (labels t₃) := by
  linear_combination h₂ - h₁

/-! ## The witness

Twelve observed cells of the `5 × 5` table `fiveCode⁻¹ (i + j)`, whose coding exchanges the
class names `0` and `1`, so that it is not the table of `i + j`: the rows `0` and `1` at the
columns `0`–`3`, three more cells of the diagonal and the cell `(4, 0)`. Thirteen cells are held
out. The learner is given `fiveCells` and `fiveLabels` only. -/

/-- The observed cells of the witness. -/
def fiveCells : Fin 12 → Fin 5 × Fin 5 :=
  ![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2), (3, 3), (4, 4),
    (4, 0)]

/-- The observed labels of the witness, as class names. -/
def fiveLabels : Fin 12 → ZMod 5 := ![1, 0, 2, 3, 0, 2, 3, 4, 4, 0, 3, 4]

/-- The coding behind the witness, which the learner is not given: it exchanges the class names
`0` and `1`. -/
def fiveCode : ZMod 5 ≃ ZMod 5 := Equiv.swap 0 1

/-- The phases behind the witness, which the learner is not given: the number of the row or of
the column. -/
def fivePhase (i : Fin 5) : ZMod 5 := (i.val : ZMod 5)

/-- The labels of the witness are those of the table `fiveCode⁻¹ (i + j)`. -/
lemma five_real (t : Fin 12) :
    fiveCode (fiveLabels t) = fivePhase (fiveCells t).1 + fivePhase (fiveCells t).2 := by
  revert t
  decide

example : (fiveCode : ZMod 5 → ZMod 5) ∈ rectangleCodes 5 fiveCells fiveLabels :=
  mem_rectangleCodes_of_fit five_real

example : (fun c => fiveCode c - 2) ∈ rectangleCodes 5 fiveCells fiveLabels :=
  sub_const_mem_rectangleCodes (mem_rectangleCodes_of_fit five_real) 2

/-- The samples `0`, `1` of the row `0` and `4`, `5` of the row `1` are a rectangle. -/
example : Rectangle fiveCells 0 1 4 5 := by decide

/-- The rectangles of the samples `0, 1 / 4, 5` and `0, 2 / 4, 6` give that of `1, 2 / 5, 6`,
here for the coding of the witness. -/
example : fiveCode 0 + fiveCode 3 = fiveCode 2 + fiveCode 2 :=
  rectangle_of_anchored (labels := fiveLabels) (z := fiveCode) (t₀ := 0) (t₀' := 4) (t₁ := 1)
    (t₂ := 2) (t₃ := 5) (t₄ := 6) (by decide) (by decide)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
