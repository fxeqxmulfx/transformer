/-
# Logical cells for the corrected ALiBi quotient

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
Once a profile specifies the integer corrections, the generic rounded
quotient cells describe the actual ALiBi attention coordinate.
-/

import Transformer.CRASP.AlibiWindowCorrections
import Transformer.CRASP.WindowCountCells

namespace Transformer.CRASP.AlibiTables

universe u
variable {σ : Type u} {p s d k j : ℕ}

/-- An ALiBi attention cell with known finite-window corrections (F). -/
noncomputable def cell (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (q : Fin d → Fx p s) (c : Fin d)
    (dz nz : ℤ) (ψ : (Fin d → Fx p s) → FormP σ) (y : Fx p s) : FormP σ :=
  CountCells.windowFormula (T.actAt [] ℓ 0) (fun r => B (q, none, r))
    (fun r => B (q, some c, r)) (fun r => T.WV ℓ r c) dz nz ψ y

/-- Each corrected attention cell adds one counting layer (Appendix F). -/
theorem cell_mem (T : PTfr (Option σ) p s d k) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (q : Fin d → Fx p s) (c : Fin d)
    (dz nz : ℤ) (ψ : (Fin d → Fx p s) → FormP σ)
    (hψ : ∀ r, ψ r ∈ TLClY σ j) (y : Fx p s) :
    cell T ℓ B q c dz nz ψ y ∈ TLClY σ (j + 1) :=
  CountCells.windowFormula_mem _ _ _ _ _ _ _ hψ _

variable [DecidableEq σ]

/-- Exact corrected sums make the cell equivalent to the attention value.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem sat_cell (T : PTfr (Option σ) p s d k) (a : ℝ) (ℓ : ℕ)
    (B : Entry p s d → Fx p s) (q : Fin d → Fx p s) (c : Fin d)
    (dz nz : ℤ) (ψ : (Fin d → Fx p s) → FormP σ) (w : List σ)
    {i : ℕ} (hi : 0 < i)
    (hψ : ∀ t ∈ Finset.Icc 1 i, ∀ r, (ψ r).sat w t = decide (T.actAt w ℓ t = r))
    (hD : integerSum T a ℓ q none w i =
      CountCells.sum (T.actAt [] ℓ 0) (fun r => B (q, none, r)) (T.actAt w ℓ) i + dz)
    (hN : integerSum T a ℓ q (some c) w i =
      CountCells.sum (T.actAt [] ℓ 0) (fun r => B (q, some c, r)) (T.actAt w ℓ) i + nz)
    (y : Fx p s) :
    (cell T ℓ B q c dz nz ψ y).sat w i = true ↔ attentionValue T a ℓ w i q c = y := by
  have hd : 0 ≤ CountCells.sum (T.actAt [] ℓ 0) (fun r => B (q, none, r))
      (T.actAt w ℓ) i + dz := by
    rw [← hD]
    exact integerSum_nonneg T a ℓ q w i
  rw [cell, CountCells.sat_windowFormula _ _ _ _ _ _ _ _ _ hi hψ hd,
    ← hD, ← hN]
  rfl

/-- Fragment, source agreement and both corrected-sum hypotheses hold in a
one-dimensional zero model, with an actual output coordinate (Appendix F). -/
example : ∃ (T : PTfr (Option Bool) 2 0 1 0)
    (ψ : (Fin 1 → Fx 2 0) → FormP Bool),
    (∀ r, ψ r ∈ TLClY Bool 0) ∧
    (∀ t ∈ Finset.Icc 1 1, ∀ r, (ψ r).sat [true] t = decide (T.actAt [true] 0 t = r)) ∧
    ∃ dz nz : ℤ,
      integerSum T 0 0 (fun _ => 0) none [true] 1 =
        CountCells.sum (T.actAt [] 0 0) (fun _ => (0 : Fx 2 0)) (T.actAt [true] 0) 1 + dz ∧
      integerSum T 0 0 (fun _ => 0) (some 0) [true] 1 =
        CountCells.sum (T.actAt [] 0 0) (fun _ => (0 : Fx 2 0)) (T.actAt [true] 0) 1 + nz := by
  classical
  let T : PTfr (Option Bool) 2 0 1 0 := {
    E := fun _ _ => 0
    WQ := fun _ _ => 0
    WK := fun _ _ => 0
    WV := fun _ _ => 0
    ff := fun _ _ => 0
    Wout := fun _ => 0
    pe := .alibi 0 }
  let ψ := fun r : Fin 1 → Fx 2 0 => (FormP.truth (decide ((fun _ => 0) = r)) : FormP Bool)
  refine ⟨T, ψ, fun _ => FormP.truth_mem_Y _ _, ?_, ?_⟩
  · intro t ht r
    simp [ψ, T, PTfr.actAt, PTfr.act, PosEnc.emb, Fx.add_zero]
  · refine ⟨integerSum T 0 0 (fun _ => 0) none [true] 1,
      integerSum T 0 0 (fun _ => 0) (some 0) [true] 1, ?_, ?_⟩ <;>
      simp [CountCells.sum]

end Transformer.CRASP.AlibiTables
