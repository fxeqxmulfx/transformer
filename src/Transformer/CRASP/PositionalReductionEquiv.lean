/-
# `TL[◁#]^pos` reduces to `TL[◁#]`: the translation preserves meaning

arXiv:2506.16055v3, "Knee-Deep in C-RASP: A Transformer Depth Hierarchy"
(COLM 2025), Appendix E, the proof of `lem:tlclpos_reduction`, equation
`eq:y_invariant` and the final step `f(w) ⊨ φ ⟺ w ⊨ T_r⟦φ⟧`.

The invariant is proved at every admissible `(i, ρ)`: `i ∈ [|w|]` with
`ρ ∈ [r]`, as in the paper, and also `i = 0` with `ρ = r`, which the final step
needs for `w = ε`.  The block size is `reach ≤ r` instead of the paper's
`r > Y`, and `tr_paperBlockSize_unsound` shows that the paper's `r = M(Y+1)`
is too small there.
-/

import Transformer.CRASP.PositionalReduction
import Transformer.CRASP.PositionalReductionAtomEquiv
import Transformer.CRASP.PositionalReductionCount

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

mutual

/-- **Equation `eq:y_invariant` for formulas** (Appendix E, proof of
`lem:tlclpos_reduction`): at an admissible `(i, ρ)`, `T_ρ⟦φ⟧` holds at `i` of
`w` exactly where `φ` holds at `r·i + ρ` of `f(w)`, for `φ` in `Y`-normal form,
`reach φ ≤ r` and `r` a multiple of the moduli of `φ`. -/
theorem FormP.sat_tr (w : List σ) {r : ℕ} :
    ∀ φ : FormP (Option σ), YNormal φ → φ.reach ≤ r → φ.period ∣ r →
      ∀ {i ρ : ℕ}, ((1 ≤ i ∧ i ≤ w.length ∧ 1 ≤ ρ ∧ ρ ≤ r) ∨ (i = 0 ∧ ρ = r)) →
        (φ.tr r ρ).sat w i = φ.sat (spread r w) (r * i + ρ)
  | .sym a, _, hreach, hper, _, _, hadm => by
      rw [FormP.tr]
      exact FormP.sat_atomTr w hadm (.sym a) 0 (by omega) hper
  | .mod m s, _, hreach, hper, _, _, hadm => by
      rw [FormP.tr]
      exact FormP.sat_atomTr w hadm (.mod m s) 0 (by omega) hper
  | .prev φ, h, hreach, hper, _, _, hadm => by
      have hφ : YAtomic φ := by
        cases h with
        | atom ha => cases ha with
          | prev hφ => exact hφ
      rw [FormP.reach] at hreach
      rw [FormP.period] at hper
      rw [FormP.tr]
      exact FormP.sat_atomTr w hadm hφ 1 (by omega) hper
  | .lt t₁ t₂, h, hreach, hper, _, _, hadm => by
      obtain ⟨h₁, h₂⟩ : YNormalT t₁ ∧ YNormalT t₂ := by
        cases h with
        | lt h₁ h₂ => exact ⟨h₁, h₂⟩
        | atom ha => cases ha
      rw [FormP.reach, max_le_iff] at hreach
      rw [FormP.period] at hper
      have e₁ := t₁.val_tr w h₁ hreach.1 ((Nat.dvd_mul_right _ _).trans hper) hadm
      have e₂ := t₂.val_tr w h₂ hreach.2 ((Nat.dvd_mul_left _ _).trans hper) hadm
      rw [FormP.tr, Form.sat_ltSum, FormP.sat, decide_eq_decide]
      omega
  | .neg φ, h, hreach, hper, _, _, hadm => by
      have hφ : YNormal φ := by
        cases h with
        | neg hφ => exact hφ
        | atom ha => cases ha
      rw [FormP.reach] at hreach
      rw [FormP.period] at hper
      rw [FormP.tr, Form.sat, FormP.sat, φ.sat_tr w hφ hreach hper hadm]
  | .and φ₁ φ₂, h, hreach, hper, _, _, hadm => by
      obtain ⟨h₁, h₂⟩ : YNormal φ₁ ∧ YNormal φ₂ := by
        cases h with
        | and h₁ h₂ => exact ⟨h₁, h₂⟩
        | atom ha => cases ha
      rw [FormP.reach, max_le_iff] at hreach
      rw [FormP.period] at hper
      rw [FormP.tr, Form.sat, FormP.sat,
        φ₁.sat_tr w h₁ hreach.1 ((Nat.dvd_mul_right _ _).trans hper) hadm,
        φ₂.sat_tr w h₂ hreach.2 ((Nat.dvd_mul_left _ _).trans hper) hadm]

/-- **Equation `eq:y_invariant` for terms** (Appendix E, proof of
`lem:tlclpos_reduction`), in the form of `Summands`: the value of `t` at
`r·i + ρ` of `f(w)`, plus the indicators `T_ρ⟦t⟧` subtracts, is its constant
plus its counts at `i` of `w`. -/
theorem TermP.val_tr (w : List σ) {r : ℕ} :
    ∀ t : TermP (Option σ), YNormalT t → t.reach ≤ r → t.period ∣ r →
      ∀ {i ρ : ℕ}, ((1 ≤ i ∧ i ≤ w.length ∧ 1 ≤ ρ ∧ ρ ≤ r) ∨ (i = 0 ∧ ρ = r)) →
        t.val (spread r w) (r * i + ρ) + (t.tr r ρ).negs.countP (·.sat w i) =
          (t.tr r ρ).const + ((t.tr r ρ).terms.map (·.val w i)).sum
  | .countL φ, h, hreach, hper, i, ρ, hadm => by
      have hφ : YNormal φ := by
        cases h with
        | countL hφ => exact hφ
      rw [TermP.reach] at hreach
      rw [TermP.period] at hper
      have h₁ := TermP.val_countL_blockEnd w φ (fun ρ' => φ.tr r ρ') (r := r) (i := i) (ρ := ρ)
        (by omega) fun ρ' _ _ => φ.sat_tr w hφ hreach hper (by omega)
      have h₂ := TermP.val_countL_spread w φ (fun ρ' => φ.tr r ρ') r i
        fun _ ρ' _ _ _ _ => φ.sat_tr w hφ hreach hper (by omega)
      rw [TermP.tr]
      exact h₁.trans h₂
  | .add t₁ t₂, h, hreach, hper, _, _, hadm => by
      obtain ⟨h₁, h₂⟩ : YNormalT t₁ ∧ YNormalT t₂ := by
        cases h with
        | add h₁ h₂ => exact ⟨h₁, h₂⟩
      rw [TermP.reach, max_le_iff] at hreach
      rw [TermP.period] at hper
      have e₁ := t₁.val_tr w h₁ hreach.1 ((Nat.dvd_mul_right _ _).trans hper) hadm
      have e₂ := t₂.val_tr w h₂ hreach.2 ((Nat.dvd_mul_left _ _).trans hper) hadm
      rw [TermP.tr, TermP.val]
      dsimp only
      rw [List.countP_append, List.map_append, List.sum_append]
      omega
  | .one, _, _, _, _, _, _ => rfl

end

/-- **The final step of `lem:tlclpos_reduction`** (Appendix E):
`f(w) ⊨ φ ⟺ w ⊨ T_r⟦φ⟧`, the invariant at the last position `r·|w| + r` of
`f(w)`.  For `w = ε` that position is the end of the first block, outside the
positions `i ∈ [|w|]` the paper proves the invariant for. -/
theorem FormP.models_spread_iff {r : ℕ} (hr : 1 ≤ r) (w : List σ) {φ : FormP (Option σ)}
    (hφ : YNormal φ) (hreach : φ.reach ≤ r) (hper : φ.period ∣ r) :
    FormP.models (spread r w) φ ↔ (φ.tr r r).models w := by
  rw [FormP.models, Form.models, length_spread hr, Nat.mul_add_one,
    ← FormP.sat_tr w φ hφ hreach hper (i := w.length) (ρ := r) (by omega)]

/-- The hypotheses of `FormP.sat_tr`, `TermP.val_tr` and
`FormP.models_spread_iff` are satisfiable: `Q_e` and `1` at the end of the
first block, for `r = 2`. -/
example :
    1 ≤ 2 ∧ YNormal (FormP.sym none : FormP (Option σ)) ∧
      (FormP.sym none : FormP (Option σ)).reach ≤ 2 ∧
      (FormP.sym none : FormP (Option σ)).period ∣ 2 ∧
      YNormalT (TermP.one : TermP (Option σ)) ∧ (TermP.one : TermP (Option σ)).reach ≤ 2 ∧
      (TermP.one : TermP (Option σ)).period ∣ 2 ∧
      ((1 ≤ 0 ∧ 0 ≤ ([] : List σ).length ∧ 1 ≤ 2 ∧ 2 ≤ 2) ∨ (0 = 0 ∧ 2 = 2)) :=
  ⟨by decide, .atom (.sym none), le_rfl, one_dvd 2, .one, Nat.zero_le 2, one_dvd 2,
    Or.inr ⟨rfl, rfl⟩⟩

/-- **The paper's block size is too small.**  For `φ = Y Q_e` there are no
moduli and one `Y`, so the proof of `lem:tlclpos_reduction` takes
`r = M(Y+1) = 2`.  Then `T_2⟦Y Q_e⟧ = ⊥`, since `ρ − c = 2 − 1 = 1` makes it
read the letter `w_i`, yet `f(ε) = ee` satisfies `Y Q_e`: at `i = 0` the first
block has no letter to read.  `FormP.reach` asks for `r ≥ 3` here.

Source: arXiv:2506.16055v3, Appendix E, proof of `lem:tlclpos_reduction`. -/
theorem tr_paperBlockSize_unsound :
    FormP.models (spread 2 ([] : List σ)) (.prev (.sym none)) ∧
      ¬ ((FormP.prev (.sym none)).tr 2 2 : Form σ).models [] := by
  simp [FormP.models, FormP.sat, spread, FormP.tr, FormP.atomTr, Form.models]

end CRASP
end Transformer
