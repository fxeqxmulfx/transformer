/-
# The depth hierarchy of `TL[◁#, ▷#]`

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix D: `lem:cropping` and `thm:TLC_depth`.

Adding future counting costs a factor of two in the separating language.  The
reason is stated at the head of the appendix: "if the Parikh vector of a word
is fixed we can rewrite `▷#` in terms of `◁#` and `n⃗`" — a minimal depth-1
subformula is still a half-plane in the counts of `a` and `b`, so cropping
still works, but now it must crop away from *every* side of the interval, and
each reduction step therefore eats a block at the left and a block at the
right.  The separating family is

    D_k = L_{2k-1} = (a⁺b⁺)^{k-1} a⁺,

and `D_{k+1}` separates depth `k` from depth `k+1`.

`lem:cropping` is false as stated, for the reason `lem:cropping_oneway` is
(`cropping_unsound`): a count reads the positions before the interval too.  The
lower bound is proved without it, in `CRASP.LowerBoundTwoSided`.
-/

import Transformer.CRASP.Depth
import Transformer.CRASP.LowerBoundTwoSided

namespace Transformer
namespace CRASP

/-- An interval sticks to *no* side of another: it sits strictly inside it in
every one of the four directions (Appendix D, `lem:cropping`). -/
def SticksToNoSide (I' I : Interval Bool) : Prop :=
  I'.Subset I ∧ ¬ SticksTo true I' I ∧ ¬ SticksTo false I' I ∧
    ¬ SticksToLo true I' I ∧ ¬ SticksToLo false I' I

/-- **`lem:cropping` (Cropping Lemma for `TL[◁#, ▷#]`) is false.**  "For any
formula `φ` of `TL[◁#,▷#]^P` and any accommodating family of intervals `I`,
such that `I(n⃗) ⊆ [0⃗, n⃗]` and the PNPs of `φ` are constant on `I`, there
exists an accommodating family of intervals `I'` such that `I'(n⃗) ⊆ I(n⃗)` but
does not stick to any side of `I(n⃗)` for all `n⃗`, and all of the minimal
depth-1 subformulas (and PNPs) of `φ` are constant on `I'`."  The past-only
`firstNotA` on `n⃗ ↦ [(1,1), n⃗]` refutes it as it refutes
`lem:cropping_oneway`: no accommodating family inside that one keeps its
minimal depth-1 subformula constant.

Source: arXiv:2506.16055v3, Appendix D, `lem:cropping` and its proof: "now
each `ψ_ℓ` defines a half-plane over `◁#[Q_a]` and `◁#[Q_b]`". -/
theorem cropping_unsound :
    ¬ ∀ (φ : Form Bool) (I : IntervalFamily Bool), Accommodating I →
      (∀ n, (I n).Subset ⟨fun _ => 0, n⟩) → PnpsConstantOn φ I →
      ∃ I' : IntervalFamily Bool, Accommodating I' ∧ (∀ n, SticksToNoSide (I' n) (I n)) ∧
        MinimalOneConstantOn φ I' ∧ PnpsConstantOn φ I' :=
  fun h => by
    obtain ⟨I', hI', hstick, hmin, -⟩ := h Form.firstNotA _ accommodating_one
      (fun _ _ hv => ⟨fun _ => Nat.zero_le _, hv.2⟩) pnpsConstantOn_firstNotA
    exact not_minimalOneConstantOn_firstNotA hI' (fun n => (hstick n).1) hmin

/-- `D_k = L_{2k-1} = (a⁺b⁺)^{k-1} a⁺`, the family separating the depth levels
of `TL[◁#, ▷#]` (Appendix D, `thm:TLC_depth`). -/
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
The negative half is the one the paper derives from `lem:cropping`, which is
false as stated; it is `not_definable_altPlus_double`, which crops frames of
fixed Parikh vector instead. -/
theorem definable_altPlusDouble (k : ℕ) (hk : 0 < k) :
    Definable (altPlusDouble (k + 1)) (k + 1) ∧ ¬ Definable (altPlusDouble (k + 1)) k :=
  ⟨definable_of_kPiecewiseTestable k _ (kPiecewiseTestable_altPlusDouble k), by
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_lt hk
    rw [altPlusDouble_succ, show 2 * (0 + k + 1) + 1 = 2 * k + 3 by omega, Nat.zero_add]
    exact not_definable_altPlus_double k⟩

/-- The hypothesis of `definable_altPlusDouble` is satisfiable: `D_2 = a⁺b⁺a⁺`
separates depth `1` from depth `2`. -/
example : 0 < 1 := Nat.one_pos

end CRASP
end Transformer
