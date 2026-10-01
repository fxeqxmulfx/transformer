/-
# Layer induction for a finite attention-table representation

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod`.
A periodic position encoding can be included in a finite activation state.
Its attention then has finite query/source tables, to which the same
one-counting-layer induction as Appendix B.2 applies.
-/

import Transformer.CRASP.PositionalNextState

namespace Transformer.CRASP.PositionalNext

universe u v
variable {α : Type u} {σ : Type v} [Fintype α] [DecidableEq α] [DecidableEq σ]
variable {p s d : ℕ}

/-- State predicates define each ordinary input position (Appendix F). -/
def Defines (H : ℕ → List σ → ℕ → α) (ℓ : ℕ) (ψ : α → FormP σ) : Prop :=
  ∀ w i, 0 < i → i ≤ w.length → ∀ a, (ψ a).sat w i = decide (H ℓ w i = a)

/-- Finite attention tables admit state predicates of the same counting depth.
The two hypotheses specify the embedding and the actual layer recurrence;
no unproved simulation is used.

Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLClmod`, and
Appendix B.2, Equation `eq:phi_h`. -/
theorem exists_formulas (H : ℕ → List σ → ℕ → α) (b : ℕ → α)
    (D : ℕ → α → α → Fx p s) (N : ℕ → α → Fin d → α → Fx p s)
    (V : ℕ → Fin d → α → Fx p s)
    (update : ℕ → α → (Fin d → Fx p s) → α)
    (hD : ∀ ℓ q a, 0 ≤ (D ℓ q a).m)
    (hinit : ∃ ψ : α → FormP σ, (∀ a, ψ a ∈ TLClMod σ 0) ∧ Defines H 0 ψ)
    (hstep : ∀ ℓ w i, 0 < i → i ≤ w.length →
      H (ℓ + 1) w i = update ℓ (H ℓ w i) (fun c =>
        CountCells.value (b ℓ) (D ℓ (H ℓ w i)) (N ℓ (H ℓ w i) c)
          (V ℓ c) (H ℓ w) i)) (ℓ : ℕ) :
    ∃ ψ : α → FormP σ, (∀ a, ψ a ∈ TLClMod σ ℓ) ∧ Defines H ℓ ψ := by
  induction ℓ with
  | zero => exact hinit
  | succ ℓ ih =>
      obtain ⟨ψ, hb, hs⟩ := ih
      refine ⟨formula (b ℓ) (D ℓ) (N ℓ) (V ℓ) (update ℓ) ψ,
        formula_mem _ _ _ _ _ ψ hb, ?_⟩
      intro w i hi hilen a
      rw [sat_formula _ _ _ _ _ _ (hD ℓ) (H ℓ w) w hi, hstep ℓ w i hi hilen]
      intro j hj a
      have hjb := Finset.mem_Icc.mp hj
      exact hs w j (by omega) (by omega) a

/-- The table recurrence and initial-state hypotheses have a constant witness.
Source: arXiv:2506.16055v3, Appendix F, finite-state enumeration. -/
example :
    (∀ (_ℓ : ℕ) (_q _a : Unit), 0 ≤ (0 : Fx 2 0).m) ∧
    (∃ ψ : Unit → FormP Bool, (∀ a, ψ a ∈ TLClMod Bool 0) ∧
      Defines (fun _ _ _ => ()) 0 ψ) ∧
    (∀ (_ : ℕ) (w : List Bool) (i : ℕ), 0 < i → i ≤ w.length →
      (fun (_a : Unit) (_v : Fin 1 → Fx 2 0) => ()) ()
        (fun _ => CountCells.value () (fun _ => (0 : Fx 2 0))
          (fun _ => 0) (fun _ => 0) (fun _ => ()) i) = ()) := by
  refine ⟨by simp, ⟨fun _ => FormP.truth true, fun _ => FormP.truth_mem _ _, ?_⟩,
    fun _ _ _ _ _ => rfl⟩
  intro w i hi hilen a
  cases a
  simp

end Transformer.CRASP.PositionalNext
