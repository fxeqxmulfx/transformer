/-
# Temporal logic with counting: derived operators, and the Dyck example

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §2.2 and Appendix A.2.

§2.2 says of the grammar of `TL[◁#, ▷#]`:

> Other operators, like `≤, ≥, ∨, −`, multiplication by integer constants,
> and integer constants other than `1`, can be defined in terms of the above.

This file writes those definitions out and proves that each has the value the
paper intends, stays in the past-only fragment when its arguments do, and —
what the depth hierarchy needs — does not raise the depth.  Note that a
constant `0` is *not* among them: the only base term is `1` and terms are
never subtracted, so every term is at least `1` unless it is a count.  What
the paper writes as `t = 0` is `t < 1`, which is `Form.isZero` here.

Appendix A.2 traces the Dyck formula

    φ_Dyck = (◁#[Q_(] = ◁#[Q_)]) ∧ (◁#[◁#[Q_(] < ◁#[Q_)]] = 0)

over `(())()` and `())()(`, position by position.  The two `example`s at the
end check this development against that trace.
-/

import Transformer.CRASP.Defs

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

namespace Term

/-- The constant `n + 1`, as `1 + ⋯ + 1` (§2.2). -/
def ofPos : ℕ → Term σ
  | 0 => .one
  | n + 1 => .add (ofPos n) .one

@[simp] theorem val_ofPos (w : List σ) (i n : ℕ) : (ofPos n : Term σ).val w i = n + 1 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [ofPos, Term.val, ih, Term.val]

omit [DecidableEq σ] in
@[simp] theorem depth_ofPos (n : ℕ) : (ofPos n : Term σ).depth = 0 := by
  induction n with
  | zero => rfl
  | succ n ih => rw [ofPos, Term.depth, ih, Term.depth]; rfl

omit [DecidableEq σ] in
@[simp] theorem past_ofPos (n : ℕ) : (ofPos n : Term σ).past = true := by
  induction n with
  | zero => rfl
  | succ n ih => rw [ofPos, Term.past, ih, Term.past]; rfl

/-- Multiplication by the positive integer constant `n + 1` (§2.2). -/
def nsmul : ℕ → Term σ → Term σ
  | 0, t => t
  | n + 1, t => .add (nsmul n t) t

@[simp] theorem val_nsmul (w : List σ) (i n : ℕ) (t : Term σ) :
    (nsmul n t).val w i = (n + 1) * t.val w i := by
  induction n with
  | zero => simp [nsmul]
  | succ n ih => rw [nsmul, Term.val, ih]; ring

omit [DecidableEq σ] in
@[simp] theorem depth_nsmul (n : ℕ) (t : Term σ) : (nsmul n t).depth = t.depth := by
  induction n with
  | zero => rfl
  | succ n ih => rw [nsmul, Term.depth, ih]; omega

omit [DecidableEq σ] in
@[simp] theorem past_nsmul (n : ℕ) (t : Term σ) : (nsmul n t).past = t.past := by
  induction n with
  | zero => rfl
  | succ n ih => rw [nsmul, Term.past, ih, Bool.and_self]

end Term

namespace Form

/-- `φ₁ ∨ φ₂` (§2.2). -/
def or (φ₁ φ₂ : Form σ) : Form σ := .neg (.and (.neg φ₁) (.neg φ₂))

@[simp] theorem sat_or (w : List σ) (i : ℕ) (φ₁ φ₂ : Form σ) :
    (φ₁.or φ₂).sat w i = (φ₁.sat w i || φ₂.sat w i) := by
  simp [Form.or, Form.sat]

omit [DecidableEq σ] in
@[simp] theorem depth_or (φ₁ φ₂ : Form σ) : (φ₁.or φ₂).depth = max φ₁.depth φ₂.depth := rfl

omit [DecidableEq σ] in
@[simp] theorem past_or (φ₁ φ₂ : Form σ) : (φ₁.or φ₂).past = (φ₁.past && φ₂.past) := rfl

/-- `t₁ ≤ t₂` (§2.2). -/
def le (t₁ t₂ : Term σ) : Form σ := .neg (.lt t₂ t₁)

@[simp] theorem sat_le (w : List σ) (i : ℕ) (t₁ t₂ : Term σ) :
    (le t₁ t₂).sat w i = decide (t₁.val w i ≤ t₂.val w i) := by
  rw [Bool.eq_iff_iff]
  simp [le, Form.sat]

omit [DecidableEq σ] in
@[simp] theorem depth_le (t₁ t₂ : Term σ) : (le t₁ t₂).depth = max t₁.depth t₂.depth := by
  simp only [le, Form.depth]
  omega

/-- `t = 0`.  There is no constant `0` among the terms, so the paper's `t = 0`
is read as `t < 1`. -/
def isZero (t : Term σ) : Form σ := .lt t .one

@[simp] theorem sat_isZero (w : List σ) (i : ℕ) (t : Term σ) :
    (isZero t).sat w i = decide (t.val w i = 0) := by
  rw [Bool.eq_iff_iff]
  simp only [isZero, Form.sat, Term.val, decide_eq_true_eq, Nat.lt_one_iff]

omit [DecidableEq σ] in
@[simp] theorem depth_isZero (t : Term σ) : (isZero t).depth = t.depth := by
  simp only [isZero, Form.depth, Term.depth]
  omega

omit [DecidableEq σ] in
@[simp] theorem past_isZero (t : Term σ) : (isZero t).past = t.past := by
  simp [isZero, Form.past, Term.past]

/-- `▷#[⊤] < 2`, with `⊤` written `¬(1 < 1)`: the position is the last one of
the string.  The suffix `w[i:n]` the term counts has `n - i + 1` positions, so
the formula holds exactly when `n ≤ i`; off the string it holds as well, the
suffix being empty there (§2.2). -/
def atEnd : Form σ := .lt (.countR (.neg (.lt .one .one))) (Term.ofPos 1)

@[simp] theorem sat_atEnd (w : List σ) (i : ℕ) :
    (atEnd : Form σ).sat w i = decide (w.length ≤ i) := by
  simp only [atEnd, Form.sat, Term.val, Term.val_ofPos, lt_self_iff_false, decide_false,
    Bool.not_false, List.filter_true, List.length_range', decide_eq_decide]
  omega

omit [DecidableEq σ] in
@[simp] theorem depth_atEnd : (atEnd : Form σ).depth = 1 := by
  simp [atEnd, Form.depth, Term.depth]

omit [DecidableEq σ] in
@[simp] theorem pnpFree_atEnd : (atEnd : Form σ).pnpFree = true := rfl

end Form

section Dyck

/-- The parenthesis alphabet of Example 2.3: `false` is `(` and `true` is `)`. -/
abbrev Paren : Type := Bool

/-- `◁#[Q_(] = ◁#[Q_)]`: every prefix counted at the current position has as
many left as right parentheses (Example 2.3). -/
def balance : Form Paren :=
  Form.eq (.countL (.sym false)) (.countL (.sym true))

/-- `◁#[◁#[Q_(] < ◁#[Q_)]] = 0`: no prefix has more right than left
parentheses (Example 2.3). -/
def matched : Form Paren :=
  .isZero (.countL (.lt (.countL (.sym false)) (.countL (.sym true))))

/-- The Dyck formula of Example 2.3. -/
def dyck : Form Paren := .and balance matched

/-- It is a formula of the past-only fragment, of depth `2`. -/
theorem dyck_mem : dyck ∈ TLCl Paren 2 :=
  ⟨rfl, rfl, by decide⟩

/-- The trace of Appendix A.2 on `(())()`, which is in the Dyck language: the
formula holds at the last position. -/
example : dyck.models [false, false, true, true, false, true] := by decide

/-- And on `())()(`, which is not: it fails. -/
example : ¬ dyck.models [false, true, true, false, true, false] := by decide

/-- The `matched` conjunct is what rejects it: the string is balanced, but has
two positions at which the right parentheses run ahead. -/
example : balance.models [false, true, true, false, true, false] ∧
    ¬ matched.models [false, true, true, false, true, false] := by
  constructor
  · decide
  · decide

end Dyck

end CRASP
end Transformer
