/-
# The depth hierarchy of `TL[◁#, ▷#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix B: `lem:cropping` and `thm:TLC_depth`.

Adding future counting costs a factor of two in the separating language.  The
reason is stated at the head of the appendix: "if the Parikh vector of a word
is fixed we can rewrite `▷#` in terms of `◁#` and `n⃗`" — a minimal depth-1
subformula is still a half-plane in the counts of `a` and `b`, so cropping
still works, but now it must crop away from *every* side of the interval, and
each reduction step therefore eats a block at the left and a block at the
right.  The separating family is

    D_k = L_{2k-1} = (a⁺b⁺)^{k-1} a⁺,

and `D_{k+1}` separates depth `k` from depth `k+1`.
-/

import Transformer.CRASP.Depth

namespace Transformer
namespace CRASP

/-- An interval sticks to *no* side of another: it sits strictly inside it in
every one of the four directions (Appendix B, `lem:cropping`). -/
def SticksToNoSide (I' I : Interval Bool) : Prop :=
  I'.Subset I ∧ ¬ SticksTo true I' I ∧ ¬ SticksTo false I' I ∧
    ¬ SticksToLo true I' I ∧ ¬ SticksToLo false I' I

/-- **Lemma `lem:cropping` (Cropping Lemma for `TL[◁#, ▷#]`).**  For a formula
of `TL[◁#,▷#]^P` and an accommodating family of intervals inside `[0⃗, n⃗]` on
which its PNPs are constant, there is an accommodating sub-family sticking to
no side of it on which the minimal depth-1 subformulas are constant too. -/
theorem cropping (φ : Form Bool) (I : IntervalFamily Bool) (hI : Accommodating I)
    (hsub : ∀ n, (I n).Subset ⟨fun _ => 0, n⟩) (hpnp : PnpsConstantOn φ I) :
    ∃ I' : IntervalFamily Bool, Accommodating I' ∧ (∀ n, SticksToNoSide (I' n) (I n)) ∧
      MinimalOneConstantOn φ I' ∧ PnpsConstantOn φ I' :=
  sorry

/-- The hypotheses of `lem:cropping` are satisfiable: `n⃗ ↦ [0⃗, n⃗]` is
accommodating, is contained in `[0⃗, n⃗]`, and a PNP-free formula has nothing to
keep constant. -/
example (a : Bool) :
    Accommodating (fun n => ⟨fun _ => 0, n⟩ : IntervalFamily Bool) ∧
      (∀ n, ((fun n => ⟨fun _ => 0, n⟩ : IntervalFamily Bool) n).Subset ⟨fun _ => 0, n⟩) ∧
      PnpsConstantOn (Form.isZero (.countL (.sym a)))
        (fun n => ⟨fun _ => 0, n⟩ : IntervalFamily Bool) := by
  refine ⟨fun s => ⟨s, fun _ => by simp⟩, fun _ _ hv => hv, ?_⟩
  intro ψ hψ
  simp [Form.isZero, Form.pnps, Term.pnps] at hψ

/-- `D_k = L_{2k-1} = (a⁺b⁺)^{k-1} a⁺`, the family separating the depth levels
of `TL[◁#, ▷#]` (Appendix B, `thm:TLC_depth`). -/
def altPlusDouble (k : ℕ) : Set (List Bool) := altPlus false (2 * k - 1)

/-- `D_{k+1} = L_{2k+1}`, the form in which the theorem and its piecewise-
testability corollary use it. -/
theorem altPlusDouble_succ (k : ℕ) : altPlusDouble (k + 1) = altPlus false (2 * k + 1) := by
  rw [altPlusDouble]
  congr 1

/-- The closing remark of the proof: "by `lem:piecewise_testable`,
`D_{k+1} = L_{2k+1}` is a `(2k+1)`-piecewise testable language", which with
`lem:piecewise_testable_depth` is what puts it inside `TL[◁#,▷#]_{k+1}`. -/
theorem kPiecewiseTestable_altPlusDouble (k : ℕ) :
    KPiecewiseTestable (2 * k + 1) (altPlusDouble (k + 1)) := by
  rw [altPlusDouble_succ]
  exact kPiecewiseTestable_altPlus (2 * k + 1) (by omega)

/-- **Theorem `thm:TLC_depth`.**  `D_{k+1}` is definable in `TL[◁#,▷#]_{k+1}`
but not in `TL[◁#,▷#]_k`.

The positive half is the closing remark of the paper's proof, read in this
order: `D_{k+1}` is `(2k+1)`-piecewise testable, and
`lem:piecewise_testable_depth` turns that into a depth-`(k+1)` definition.
The negative half is the one that needs `lem:cropping`, and only it is left
open. -/
theorem definable_altPlusDouble (k : ℕ) (hk : 0 < k) :
    Definable (altPlusDouble (k + 1)) (k + 1) ∧ ¬ Definable (altPlusDouble (k + 1)) k :=
  ⟨definable_of_kPiecewiseTestable k _ (kPiecewiseTestable_altPlusDouble k), sorry⟩

end CRASP
end Transformer
