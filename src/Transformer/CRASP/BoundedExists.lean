/-
# Bounded existentials over the past, written as counts

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.2 and the proof of `lem:piecewise_testable_depth` in §2.4.

`TL[◁#]` has no quantifiers, but a count can stand in for one: `∃ j ≤ i [ψ]`
is `1 ≤ ◁#[ψ]`.  For the strict `∃ j < i [ψ]` the count needs correcting at
the current position, which `◁#` includes, so the threshold is `2` where `ψ`
holds and `1` where it does not.  Both cost one level of depth, exactly like
the count they are made of; this is what the subsequence formulas of
`Transformer.CRASP.Subsequence` are built from.
-/

import Transformer.CRASP.Basic

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u}

section Counting

variable [DecidableEq σ]

/-- `◁#` takes one more position in at each step: its value at `i + 1` is its
value at `i` plus the indicator of `φ` at `i + 1`. -/
theorem val_countL_succ (w : List σ) (i : ℕ) (φ : Form σ) :
    (Term.countL φ).val w (i + 1) =
      (Term.countL φ).val w i + (if φ.sat w (i + 1) = true then 1 else 0) := by
  rw [Term.val, Term.val, List.range'_concat, List.filter_append, List.length_append]
  have hi : 1 + 1 * i = i + 1 := by omega
  rw [hi]
  cases h : φ.sat w (i + 1) <;> simp [h]

/-- `◁#[φ] > 0` at `i` says that some position `1 ≤ j ≤ i` satisfies `φ`. -/
theorem val_countL_pos_iff (w : List σ) (i : ℕ) (φ : Form σ) :
    0 < (Term.countL φ).val w i ↔ ∃ j, 1 ≤ j ∧ j ≤ i ∧ φ.sat w j = true := by
  rw [Term.val, List.length_pos_iff_exists_mem]
  constructor
  · rintro ⟨j, hj⟩
    rw [List.mem_filter, List.mem_range'_1] at hj
    exact ⟨j, hj.1.1, by omega, hj.2⟩
  · rintro ⟨j, hj1, hj2, hj⟩
    exact ⟨j, by rw [List.mem_filter, List.mem_range'_1]; exact ⟨⟨hj1, by omega⟩, hj⟩⟩

end Counting

namespace Form

/-- `∃ j ≤ i [ψ]`, written as the logic can write it: `1 ≤ ◁#[ψ]`. -/
def exAt (ψ : Form σ) : Form σ := Form.le .one (.countL ψ)

/-- `∃ j < i [ψ]`.  `◁#` counts the current position too, so the threshold is
raised by one exactly when `ψ` holds there. -/
def exBefore (ψ : Form σ) : Form σ :=
  Form.or (.and ψ (Form.le (.add .one .one) (.countL ψ)))
    (.and (.neg ψ) (Form.le .one (.countL ψ)))

@[simp] theorem depth_exAt (ψ : Form σ) : (exAt ψ).depth = ψ.depth + 1 := by
  simp only [exAt, le, Form.depth, Term.depth]
  omega

@[simp] theorem depth_exBefore (ψ : Form σ) : (exBefore ψ).depth = ψ.depth + 1 := by
  simp only [exBefore, or, le, Form.depth, Term.depth]
  omega

@[simp] theorem past_exAt (ψ : Form σ) : (exAt ψ).past = ψ.past := by
  simp [exAt, le, Form.past, Term.past]

@[simp] theorem past_exBefore (ψ : Form σ) : (exBefore ψ).past = ψ.past := by
  simp [exBefore, or, le, Form.past, Term.past]

@[simp] theorem pnpFree_exAt (ψ : Form σ) : (exAt ψ).pnpFree = ψ.pnpFree := by
  simp [exAt, le, Form.pnpFree, Term.pnpFree]

@[simp] theorem pnpFree_exBefore (ψ : Form σ) : (exBefore ψ).pnpFree = ψ.pnpFree := by
  simp [exBefore, or, le, Form.pnpFree, Term.pnpFree]

variable [DecidableEq σ]

/-- `Form.exAt ψ` holds at `i` exactly when `ψ` holds at some `1 ≤ j ≤ i`. -/
theorem sat_exAt (w : List σ) (i : ℕ) (ψ : Form σ) :
    (exAt ψ).sat w i = true ↔ ∃ j, 1 ≤ j ∧ j ≤ i ∧ ψ.sat w j = true := by
  rw [exAt, sat_le, decide_eq_true_eq, ← val_countL_pos_iff]
  exact Iff.rfl

/-- `Form.exBefore ψ` holds at `i` exactly when `ψ` holds at some
`1 ≤ j ≤ i - 1`; at `i = 0` and `i = 1` there is no such position. -/
theorem sat_exBefore (w : List σ) (i : ℕ) (ψ : Form σ) :
    (exBefore ψ).sat w i = true ↔ ∃ j, 1 ≤ j ∧ j ≤ i - 1 ∧ ψ.sat w j = true := by
  rw [← val_countL_pos_iff]
  cases i with
  | zero => simp [exBefore, or, Form.sat, Term.val]
  | succ m =>
      have h1 : (Term.one : Term σ).val w (m + 1) = 1 := rfl
      have h2 : (Term.add .one .one : Term σ).val w (m + 1) = 2 := rfl
      simp only [exBefore, sat_or, sat_le, Form.sat, h1, h2, val_countL_succ, Nat.add_sub_cancel]
      generalize (Term.countL ψ).val w m = c
      cases ψ.sat w (m + 1) <;> simp <;> omega

end Form

end CRASP
end Transformer
