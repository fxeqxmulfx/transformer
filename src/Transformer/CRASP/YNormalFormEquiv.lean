/-
# The `Y`-normal form of `TL[◁#]^pos`: equivalence

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix F (`app:tlclpos`, "Depth Hierarchy"): `thm:ynf`.

The invariant of `N^c⟦·⟧` is that `N^c⟦φ⟧` holds where `Y^c φ` does, and
that `N^c⟦t⟧` takes at `i` the value `t` takes at `i − c`.  The paper states it
with `i` and `i − c` exchanged, and as its transformation is written the
invariant fails at the first `c` positions.  `Y^c φ` is false there whatever
`φ` is, but `¬` and `<` do not preserve falsity: `N^1⟦¬Q_a⟧ = ¬Y Q_a` holds at
position `1`, where `Y ¬Q_a` does not (`pushY_unguarded_unsound`).  With the
guard of `FormP.pushY` the invariant holds everywhere, and `thm:ynf` follows at
`c = 0`.
-/

import Transformer.CRASP.YNormalForm

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- `Y^c φ` holds at `i` when `φ` holds at `i − c` and, for `c ≥ 1`, the
position `i − c` is on the string, i.e. `c < i` (Appendix F, the semantics of
`Y`). -/
theorem FormP.sat_prevN (w : List σ) (φ : FormP σ) :
    ∀ c i : ℕ, (FormP.prevN c φ).sat w i = (decide (c = 0 ∨ c < i) && φ.sat w (i - c))
  | 0, i => by simp [FormP.prevN]
  | c + 1, i => by
      rw [FormP.prevN, FormP.sat, FormP.sat_prevN w φ c (i - 1), Nat.sub_sub, Nat.add_comm 1 c]
      cases φ.sat w (i - (c + 1)) with
      | false => simp
      | true =>
          rw [Bool.and_true, Bool.and_true, ← Bool.decide_and, decide_eq_decide]
          omega

/-- The guard `Y^c MOD_1^0` holds exactly at the positions `i` where `Y^c`
does not fail for want of predecessors (Appendix F, proof of `thm:ynf`). -/
theorem FormP.sat_guard (w : List σ) (c i : ℕ) :
    (FormP.prevN c (.mod 1 0)).sat w i = decide (c = 0 ∨ c < i) := by
  rw [FormP.sat_prevN, FormP.sat, Nat.mod_one, Nat.zero_mod, decide_eq_true rfl, Bool.and_true]

/-- Counting a predicate delayed by `c` over `[1, i]` is counting the
predicate itself over `[1, i − c]`: the count behind the `◁#` case of
`thm:ynf`. -/
theorem length_filter_range'_delay (p : ℕ → Bool) (c i : ℕ) :
    ((List.range' 1 i).filter fun j => decide (c = 0 ∨ c < j) && p (j - c)).length =
      ((List.range' 1 (i - c)).filter p).length := by
  induction i with
  | zero => simp
  | succ i ih =>
      rw [List.range'_concat, List.filter_append, List.length_append, ih, List.filter_singleton]
      rcases Nat.lt_or_ge i c with h | h
      · rw [show i + 1 - c = i - c by omega,
          decide_eq_false (by omega : ¬ (c = 0 ∨ c < 1 + 1 * i)), Bool.false_and]
        rfl
      · rw [show i + 1 - c = i - c + 1 by omega, List.range'_concat, List.filter_append,
          List.length_append, List.filter_singleton,
          decide_eq_true (by omega : c = 0 ∨ c < 1 + 1 * i), Bool.true_and,
          show 1 + 1 * i - c = 1 + 1 * (i - c) by omega]
        cases p (1 + 1 * (i - c)) <;> rfl

mutual

/-- **The invariant of `N^c⟦·⟧`**, for formulas: `N^c⟦φ⟧` holds where `Y^c φ`
does (Appendix F, proof of `thm:ynf`). -/
theorem FormP.sat_pushY (w : List σ) :
    ∀ (φ : FormP σ) (c i : ℕ), (φ.pushY c).sat w i = (FormP.prevN c φ).sat w i
  | .sym _, _, _ => rfl
  | .mod _ _, _, _ => rfl
  | .prev φ, c, i => by
      rw [FormP.pushY, FormP.sat_pushY w φ (c + 1) i, FormP.prevN_succ']
  | .lt t₁ t₂, c, i => by
      rw [FormP.pushY, FormP.sat, FormP.sat_guard, FormP.sat, TermP.val_pushY w t₁ c i,
        TermP.val_pushY w t₂ c i, FormP.sat_prevN, FormP.sat]
  | .neg φ, c, i => by
      rw [FormP.pushY, FormP.sat, FormP.sat_guard, FormP.sat, FormP.sat_pushY w φ c i,
        FormP.sat_prevN, FormP.sat_prevN, FormP.sat]
      cases decide (c = 0 ∨ c < i) <;> rfl
  | .and φ₁ φ₂, c, i => by
      rw [FormP.pushY, FormP.sat, FormP.sat_pushY w φ₁ c i, FormP.sat_pushY w φ₂ c i,
        FormP.sat_prevN, FormP.sat_prevN, FormP.sat_prevN, FormP.sat]
      cases decide (c = 0 ∨ c < i) <;> rfl

/-- **The invariant of `N^c⟦·⟧`**, for terms: `N^c⟦t⟧` takes at `i` the value
`t` takes at `i − c` (Appendix F, proof of `thm:ynf`). -/
theorem TermP.val_pushY (w : List σ) :
    ∀ (t : TermP σ) (c i : ℕ), (t.pushY c).val w i = t.val w (i - c)
  | .countL φ, c, i => by
      rw [TermP.pushY, TermP.val, TermP.val, ← length_filter_range'_delay]
      exact congrArg List.length (List.filter_congr fun j _ => by
        rw [FormP.sat_pushY w φ c j, FormP.sat_prevN])
  | .add t₁ t₂, c, i => by
      rw [TermP.pushY, TermP.val, TermP.val, TermP.val_pushY w t₁ c i, TermP.val_pushY w t₂ c i]
  | .one, _, _ => rfl

end

/-- **Lemma `thm:ynf`.**  Every `TL[◁#]^pos_k` formula has an equivalent
`TL[◁#]^pos_k` formula in `Y`-normal form: `N^0⟦φ⟧`, with the guards of
`FormP.pushY`.

Source: arXiv:2506.16055v3, Appendix F, `thm:ynf`. -/
theorem exists_yNormal (k : ℕ) (φ : FormP σ) (hφ : φ ∈ TLClPos σ k) :
    ∃ φ' ∈ TLClPos σ k, YNormal φ' ∧ ∀ (w : List σ) (i : ℕ), φ.sat w i = φ'.sat w i :=
  ⟨φ.pushY 0, (φ.depth_pushY 0).trans_le hφ, φ.yNormal_pushY 0,
    fun w i => (FormP.sat_pushY w φ 0 i).symm⟩

/-- The hypothesis is satisfiable: `Q_a` lies in `TL[◁#]^pos_k` at every
depth. -/
example (a : σ) (k : ℕ) : (FormP.sym a : FormP σ) ∈ TLClPos σ k := Nat.zero_le k

/-- **`N^c⟦·⟧` needs the guard.**  Without it the transformation of `thm:ynf`
sends `Y ¬Q_a` to `N^1⟦¬Q_a⟧ = ¬Y Q_a`, and the two disagree on the
one-letter string `a`: its only position has no predecessor, so `Y ¬Q_a` fails
there while `¬Y Q_a` holds.  `Y (1 < 1 + 1)` and `N^1⟦1 < 1 + 1⟧ = 1 < 1 + 1`
disagree in the same way.

Source: arXiv:2506.16055v3, Appendix F, proof of `thm:ynf`. -/
theorem pushY_unguarded_unsound (a : σ) :
    ¬ (FormP.prev (.neg (.sym a))).models [a] ∧ (FormP.neg (.prev (.sym a))).models [a] := by
  simp [FormP.models, FormP.sat]

end CRASP
end Transformer
