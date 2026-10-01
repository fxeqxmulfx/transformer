/-
# Quotient cells with a finite-window correction

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
The stable tail is counted over all source states. A fixed recent-state
profile contributes an integer constant correcting the numerator and
denominator; their quotient still costs only one counting layer.
-/

import Transformer.CRASP.PreviousPredicates
import Transformer.CRASP.PositionalStateCounts

namespace Transformer.CRASP.CountCells

universe u v
variable {α : Type u} {σ : Type v} [Fintype α] {p s k : ℕ}

/-- A stable-state count with its finite-window correction added (Appendix F). -/
noncomputable def corrected (b : α) (f : α → Fx p s) (z : ℤ) : LinearCount α :=
  (states b f).add ⟨z, []⟩

/-- A constant correction adds no counted predicates (Appendix F). -/
theorem corrected_good (b : α) (f : α → Fx p s) (z : ℤ) : (corrected b f z).Good 0 :=
  LinearCount.good_add (states_good b f) (by simp [LinearCount.Good])

/-- A quotient cell with corrected denominator and numerator (Appendix F). -/
noncomputable def windowFormula (b : α) (D N V : α → Fx p s) (dz nz : ℤ)
    (ψ : α → FormP σ) (y : Fx p s) : FormP σ :=
  (formula (corrected b D dz) (corrected b N nz) (states b V) y).substPos ψ

/-- Window quotient cells lie at one more level in the MOD-free fragment (F). -/
theorem windowFormula_mem (b : α) (D N V : α → Fx p s) (dz nz : ℤ)
    (ψ : α → FormP σ) (hψ : ∀ a, ψ a ∈ TLClY σ k) (y : Fx p s) :
    windowFormula b D N V dz nz ψ y ∈ TLClY σ (k + 1) := by
  have h := formula_mem (corrected_good b D dz) (corrected_good b N nz) (states_good b V) y
  refine ⟨Form.modFree_substPos ψ (fun a => (hψ a).1) _, ?_⟩
  have hd := Form.depth_substPos_le ψ k (fun a => (hψ a).2)
    (formula (corrected b D dz) (corrected b N nz) (states b V) y)
  change (Form.substPos ψ _).depth ≤ _
  have hb := h.2.2
  omega

variable [DecidableEq α] [DecidableEq σ]

/-- Corrected counts evaluate to the table sum plus its correction (F). -/
theorem val_corrected (b : α) (f : α → Fx p s) (z : ℤ) (q : ℕ → α)
    {i : ℕ} (hi : 0 < i) :
    (corrected b f z).val (stateWord q i) i = sum b f q i + z := by
  rw [corrected, LinearCount.val_add, val_states b f q hi]
  simp [LinearCount.val]

/-- A window cell defines the corrected quotient on a valid source profile.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`. -/
theorem sat_windowFormula (b : α) (D N V : α → Fx p s) (dz nz : ℤ)
    (ψ : α → FormP σ) (q : ℕ → α) (w : List σ) {i : ℕ} (hi : 0 < i)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a, (ψ a).sat w j = decide (q j = a))
    (hD : 0 ≤ sum b D q i + dz) (y : Fx p s) :
    (windowFormula b D N V dz nz ψ y).sat w i = true ↔
      (if sum b D q i + dz = 0 then
        Fx.round p s (((sum b V q i : ℝ) / (i + 1 : ℕ)) / 2 ^ s)
       else Fx.round p s (((sum b N q i + nz : ℤ) : ℝ) /
        ((sum b D q i + dz : ℤ) : ℝ))) = y := by
  let φ := formula (corrected b D dz) (corrected b N nz) (states b V) y
  have hφ := formula_mem (corrected_good b D dz) (corrected_good b N nz) (states_good b V) y
  have hletters : ∀ j, RTfr.Readable (stateWord q i) j → ∀ a,
      (ψ a).sat w j = (Form.sym a).sat (stateWord q i) j := by
    intro j hj a
    rw [sat_sym_stateWord q hi hj a]
    have hjle : j ≤ i := by simpa [stateWord] using hj.1
    have hjpos : 0 < j := hj.2.resolve_right (by
      intro h; have := congrArg List.length h; simp [stateWord] at this; omega)
    exact hψ j (Finset.mem_Icc.mpr ⟨hjpos, hjle⟩) a
  rw [windowFormula, Form.sat_substPos ψ (stateWord q i) w hletters φ hφ.1 hφ.2.1 i
    ⟨by simp [stateWord], Or.inl hi⟩]
  change (formula (corrected b D dz) (corrected b N nz) (states b V) y).sat
    (stateWord q i) i = true ↔ _
  have hn : 0 ≤ (corrected b D dz).val (stateWord q i) i := by
    rw [val_corrected b D dz q hi]
    exact hD
  rw [sat_formula _ _ _ y _ i hn, val_corrected b D dz q hi,
    val_corrected b N nz q hi, val_states b V q hi]

/-- Source agreement, positivity and fragment hypotheses have constant witnesses (F). -/
example : (∀ _ : Unit, (FormP.truth true : FormP Bool) ∈ TLClY Bool 0) ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Unit,
      (FormP.truth true : FormP Bool).sat [true] j = decide (() = a)) ∧
    0 ≤ sum () (fun _ => (0 : Fx 2 0)) (fun _ => ()) 1 + 1 := by
  simp [FormP.truth_mem_Y, sum]

end Transformer.CRASP.CountCells
