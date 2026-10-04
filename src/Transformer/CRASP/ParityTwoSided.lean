/-
# PARITY is definable in `TL[◁#, ▷#]`, at depth 2

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), §3: the first half of the remark the authors left commented out
after `thm:transformer_equivalence`, "PARITY is in `TL[◁#, ▷#]` but not
`TL[◁#]`"; the second half is `Transformer.CRASP.Parity`.  The remark gives
no formula; this file gives one.

A position `j` holding a `1` counts the `1`s up to it, `◁#[Q_1]`, and from it
on, `▷#[Q_1]`.  Both count `j` itself, so for `m` ones in all they sum to
`m + 1` (`countL_add_countR`), and they agree exactly at the `(m + 1)/2`-th
`1`, which exists exactly when `m` is odd:

    φ_PARITY = 1 ≤ ◁#[Q_1 ∧ ◁#[Q_1] = ▷#[Q_1]],

a formula of depth `2` (`definable_parity`).  With `not_definableL_parity`,
the future count `▷#` is what PARITY needs.
-/

import Transformer.CRASP.Parity

namespace Transformer
namespace CRASP

universe u

section Counts

variable {σ : Type u} [DecidableEq σ]

/-- `◁#[Q_a]` at a position `j` of the string is the number of `a`s in
`w[1:j]` (Definition `def:TLC_semantics`). -/
theorem val_countL_sym (w : List σ) (a : σ) : ∀ j, j ≤ w.length →
    (Term.countL (.sym a)).val w j = (w.take j).count a
  | 0, _ => by simp [Term.val]
  | j + 1, hj => by
      rw [val_countL_succ, val_countL_sym w a j (by omega), List.take_add_one, List.count_append,
        List.getElem?_eq_getElem (by omega)]
      simp [Form.sat, List.getElem?_eq_getElem (show j < w.length by omega), List.count_singleton]

/-- The hypothesis of `val_countL_sym` is satisfiable: position `1` of `a`. -/
example (a : σ) : 1 ≤ [a].length := le_rfl

/-- **`◁#[φ]` up to `j` and `▷#[φ]` from `j + 1` on make up `◁#[φ]` over
the whole string**, at every `j ≤ |w|`: the two counts split the positions
`1, …, |w|` between them (Definition `def:TLC_semantics`). -/
theorem countL_add_countR (w : List σ) (φ : Form σ) : ∀ j, j ≤ w.length →
    (Term.countL φ).val w j + (Term.countR φ).val w (j + 1) = (Term.countL φ).val w w.length := by
  have hend : (Term.countR φ).val w (w.length + 1) = 0 := by simp [Term.val]
  have hstep : ∀ j, j + 1 ≤ w.length → (Term.countL φ).val w j + (Term.countR φ).val w (j + 1) =
      (Term.countL φ).val w (j + 1) + (Term.countR φ).val w (j + 1 + 1) := fun j hj => by
    rw [val_countL_succ, val_countR_eq_succ w (j + 1) φ]
    by_cases h : φ.sat w (j + 1) = true <;> simp [h, hj]
    omega
  have hall : ∀ d j, j + d = w.length → (Term.countL φ).val w j + (Term.countR φ).val w (j + 1) =
      (Term.countL φ).val w w.length := by
    intro d
    induction d with
    | zero =>
        intro j hj
        rw [Nat.add_zero] at hj
        subst hj
        rw [hend, Nat.add_zero]
    | succ d ih => intro j hj; rw [hstep j (by omega), ih (j + 1) (by omega)]
  exact fun j hj => hall (w.length - j) j (by omega)

/-- The hypothesis of `countL_add_countR` is satisfiable: `j = 0`. -/
example (w : List σ) : 0 ≤ w.length := Nat.zero_le _

end Counts

/-- `Q_1 ∧ ◁#[Q_1] = ▷#[Q_1]`: the position holds the `1` with as many `1`s
up to it as from it on. -/
def middleOne : Form Bool :=
  .and (.sym true) (Form.eq (.countL (.sym true)) (.countR (.sym true)))

/-- `1 ≤ ◁#[Q_1 ∧ ◁#[Q_1] = ▷#[Q_1]]`: some position holds that `1`. -/
def parityForm : Form Bool := Form.exAt middleOne

/-- At a position `j` of the string, `middleOne` holds exactly when `j` holds
a `1` and twice the `1`s up to `j` is one more than the `1`s in all. -/
theorem sat_middleOne (w : List Bool) {j : ℕ} (hj : j ≤ w.length) :
    middleOne.sat w j = true ↔
      w[j - 1]? = some true ∧ 2 * (Term.countL (.sym true)).val w j = w.count true + 1 := by
  have hS := countL_add_countR w (.sym true) j hj
  rw [val_countL_sym w true w.length le_rfl, List.take_length] at hS
  have hR := val_countR_eq_succ w j (.sym true)
  simp only [middleOne, Form.sat, Form.sat_eq, Bool.and_eq_true, decide_eq_true_eq] at hR ⊢
  refine and_congr_right fun hsym => ?_
  rw [ite_eq_left ⟨hj, hsym⟩] at hR
  omega

/-- The hypothesis of `sat_middleOne` is satisfiable: position `1` of `1`. -/
example : 1 ≤ [true].length := le_rfl

/-- **PARITY is definable in `TL[◁#, ▷#]` at depth 2** (the first half of the
remark after `thm:transformer_equivalence`, §3).  A position satisfies
`middleOne` only when twice a count is one more than the number `m` of `1`s,
so `m` is odd; and when `m` is odd, the first position up to which the `1`s
reach `(m + 1)/2` holds a `1` and satisfies it. -/
theorem definable_parity : Definable parity 2 := by
  refine ⟨parityForm, ⟨rfl, by decide⟩, Set.ext fun w => ?_⟩
  change (Form.exAt middleOne).sat w w.length = true ↔ Odd (w.count true)
  rw [Form.sat_exAt]
  constructor
  · rintro ⟨j, -, hj, hsat⟩
    rw [Nat.odd_iff]
    have := ((sat_middleOne w hj).mp hsat).2
    omega
  · rintro ⟨r, hr⟩
    have hm := val_countL_sym w true w.length le_rfl
    rw [List.take_length] at hm
    have hex : ∃ j, r + 1 ≤ (Term.countL (.sym true : Form Bool)).val w j :=
      ⟨w.length, by omega⟩
    have hle : Nat.find hex ≤ w.length := Nat.find_min' hex (by omega)
    obtain ⟨i, hi⟩ : ∃ i, Nat.find hex = i + 1 := by
      refine Nat.exists_eq_succ_of_ne_zero fun h0 => ?_
      have := Nat.find_spec hex
      rw [h0] at this
      simp [Term.val] at this
    have hspec := Nat.find_spec hex
    have hmin := Nat.find_min hex (show i < Nat.find hex by omega)
    rw [hi, val_countL_succ] at hspec
    rw [hi] at hle
    have hone : (Form.sym true : Form Bool).sat w (i + 1) = true := by
      by_contra h
      rw [ite_eq_right_iff.mpr fun h' => absurd h' h] at hspec
      exact hmin (by omega)
    rw [ite_eq_left hone] at hspec
    refine ⟨i + 1, by omega, hle, (sat_middleOne w hle).mpr ⟨?_, ?_⟩⟩
    · simpa [Form.sat] using hone
    · rw [val_countL_succ, ite_eq_left hone]
      omega

/-- The formula on `1101`, which has three `1`s: the second `1` has two `1`s
up to it and two from it on. -/
example : parityForm.models [true, true, false, true] := by decide

/-- And on `1001`, which has two: no `1` has as many up to it as from it on. -/
example : ¬ parityForm.models [true, false, false, true] := by decide

end CRASP
end Transformer
