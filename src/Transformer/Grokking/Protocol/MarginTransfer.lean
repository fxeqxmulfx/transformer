module

public import Transformer.Grokking.Protocol.Decision
public import Mathlib.Logic.Equiv.Bool

/-!
# A row of logits near a relabelled row keeps its margin

Source: `experiments/grok_division/README.md`, *The decoders that answer every equation are not
quotient-equivariant*, and `experiments/grok_division/src/bin/quotient.rs`: the bound "every
margin of the orbit is then at least the anchor's less `2 ε`"; arXiv:2201.02177v1 (Power et
al.), appendix *Additional experimental details*, for the accuracy that the margins decide.

Let `u` and `v` be two rows of logits over the same classes and `π` a relabelling of the
classes. If `v`, read through `π`, is within `ε` of `u` in every class after one constant `κ`
is taken off, then:

* the margin of the class `π y` in `v` is at least the margin of `y` in `u` less `2 ε`
  (`margin_sub_le_of_abs_sub_le`): the constant cancels in a margin, and each of the two logits
  of a difference moves by at most `ε`;
* so a margin of `u` above `2 ε` forces the prediction of `v` (`predict_eq_of_abs_sub_le`).

The constant is allowed because a row of logits decides the same with a constant added.

The bound is attained: the witness has two classes, `ε = 1/2`, a margin of `2` in `u` and of
`2 − 2 ε = 1` in `v`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {C : Type*} [Fintype C] [LinearOrder C] [Nonempty C]

omit [LinearOrder C] [Nonempty C] in
/-- **Margins move with a relabelling of the classes**
(`experiments/grok_division/src/bin/quotient.rs`, `certified_margin`). If the row `v`, read
through the relabelling `π`, is within `ε` of the row `u` in every class after the constant
`κ` is taken off, the margin of `π y` in `v` is at least the margin of `y` in `u` less
`2 ε`. -/
theorem margin_sub_le_of_abs_sub_le {u v : C → ℝ} {π : C ≃ C} {κ ε : ℝ}
    (h : ∀ c, |v (π c) - u c - κ| ≤ ε) (y : C) : margin u y - 2 * ε ≤ margin v (π y) := by
  have hy := abs_le.1 (h y)
  rcases isEmpty_or_nonempty {c // c ≠ π y} with he | hn
  · have he' : IsEmpty {c // c ≠ y} := ⟨fun c => he.elim ⟨π c, fun e => c.2 (π.injective e)⟩⟩
    rw [margin, margin, Real.iInf_of_isEmpty, Real.iInf_of_isEmpty]
    linarith [abs_nonneg (v (π y) - u y - κ), h y]
  · refine le_ciInf fun c => ?_
    have hc : π.symm c ≠ y := fun e => c.2 ((Equiv.symm_apply_eq π).1 e)
    have h1 := abs_le.1 (h (π.symm c))
    rw [Equiv.apply_symm_apply] at h1
    linarith [margin_le u hc]

/-- **An anchor certifies a relabelled row**
(`experiments/grok_division/README.md`, the same section): under the hypothesis of
`margin_sub_le_of_abs_sub_le`, a margin of `y` in `u` above `2 ε` makes `π y` the prediction
of `v`. -/
theorem predict_eq_of_abs_sub_le {u v : C → ℝ} {π : C ≃ C} {κ ε : ℝ}
    (h : ∀ c, |v (π c) - u c - κ| ≤ ε) {y : C} (hm : 2 * ε < margin u y) : predict v = π y :=
  predict_eq_of_margin_pos (by linarith [margin_sub_le_of_abs_sub_le h y])

/-! The witnesses: two classes, the anchor row `u = (0, 2)` and the rows `(1/2, 3/2)`, read
as it stands, and `(3/2, 1/2)`, read through the exchange of the classes. Each is within
`1/2` of the anchor, with no constant. -/

/-- The margin of the second of two classes. -/
lemma margin_bool_true (z : Bool → ℝ) : margin z true = z true - z false := by
  have : Nonempty {c : Bool // c ≠ true} := ⟨⟨false, by simp⟩⟩
  refine le_antisymm (margin_le z (by simp)) (le_ciInf fun c => ?_)
  obtain ⟨c, hc⟩ := c
  cases c
  · exact le_rfl
  · exact absurd rfl hc

/-- The anchor row `(0, 2)`. -/
def anchorRow : Bool → ℝ := fun c => if c then 2 else 0

/-- The row `(1/2, 3/2)`. -/
noncomputable def nearRow : Bool → ℝ := fun c => if c then 3 / 2 else 1 / 2

lemma abs_nearRow_sub_anchorRow (c : Bool) :
    |nearRow (Equiv.refl Bool c) - anchorRow c - 0| ≤ 1 / 2 := by
  cases c <;> norm_num [nearRow, anchorRow]

lemma abs_nearRow_not_sub_anchorRow (c : Bool) :
    |(fun b => nearRow (!b)) (Equiv.boolNot c) - anchorRow c - 0| ≤ 1 / 2 := by
  cases c <;> norm_num [nearRow, anchorRow]

example : margin anchorRow true - 2 * (1 / 2) ≤ margin nearRow true :=
  margin_sub_le_of_abs_sub_le abs_nearRow_sub_anchorRow true

/-- The bound is attained: the margin `2` of the anchor becomes `2 − 2 · ½ = 1`. -/
example : margin nearRow true = margin anchorRow true - 2 * (1 / 2) := by
  rw [margin_bool_true, margin_bool_true]
  norm_num [nearRow, anchorRow]

example : predict nearRow = true :=
  predict_eq_of_abs_sub_le (u := anchorRow) (ε := 1 / 2) abs_nearRow_sub_anchorRow (y := true)
    (by rw [margin_bool_true]; norm_num [anchorRow])

/-- Read through the exchange of the classes, the row `(3/2, 1/2)` is decided for the first
class. -/
example : predict (fun b => nearRow (!b)) = false :=
  predict_eq_of_abs_sub_le (u := anchorRow) (π := Equiv.boolNot) (ε := 1 / 2)
    abs_nearRow_not_sub_anchorRow (y := true) (by rw [margin_bool_true]; norm_num [anchorRow])

end Transformer.Grokking.Protocol
