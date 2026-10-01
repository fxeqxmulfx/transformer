/-
# Free variables and satisfaction in `MAJ²`

arXiv:2506.16055v3, Appendix E, `def:MAJtwo` and `thm:majtwo_to_tlc`:
satisfaction depends only on the assignment of free variables.
-/

import Transformer.CRASP.MajTwo

namespace Transformer.CRASP.Maj2

universe u
variable {σ : Type u} [DecidableEq σ]

/-- Agreeing on free variables is enough for equal satisfaction (Appendix E). -/
theorem sat_eq_of_agree_free (φ : Maj2 σ) (w : List σ) (ξ η : Var → ℕ)
    (h : ∀ v, φ.freeIn v = true → ξ v = η v) : φ.sat w ξ = φ.sat w η := by
  induction φ generalizing ξ η with
  | sym a v => simp only [sat, h v (by simp [freeIn])]
  | lt v u => simp only [sat, h v (by simp [freeIn]), h u (by simp [freeIn])]
  | neg φ ih =>
      rw [sat, sat, ih ξ η fun v hv => h v (by simpa [freeIn] using hv)]
  | and φ ψ ihφ ihψ =>
      rw [sat, sat, ihφ ξ η (fun v hv => h v (by simp [freeIn, hv])),
        ihψ ξ η (fun v hv => h v (by simp [freeIn, hv]))]
  | maj v m φ ih =>
      simp only [sat]
      apply congrArg (fun z : ℕ => decide (w.length * (m + 1) < 2 * z))
      apply Finset.sum_congr rfl
      intro j hj
      apply Finset.sum_congr rfl
      intro t ht
      rw [ih t (Function.update ξ v j) (Function.update η v j) ?_]
      intro u hu
      by_cases huv : u = v
      · subst u; simp
      · simp only [Function.update_of_ne huv]
        apply h u
        simp only [freeIn, ite_eq_right (Ne.symm huv), decide_eq_true_eq]
        exact ⟨t, hu⟩

/-- Updating an unused variable leaves satisfaction unchanged (Appendix E). -/
theorem sat_update_of_not_free (φ : Maj2 σ) (w : List σ) (ξ : Var → ℕ) (v : Var)
    (hv : φ.freeIn v = false) (j : ℕ) :
    φ.sat w (Function.update ξ v j) = φ.sat w ξ := by
  apply sat_eq_of_agree_free
  intro u hu
  have hne : u ≠ v := by intro he; subst u; rw [hv] at hu; cases hu
  simp [hne]

/-- A closed formula has the same value under every assignment (Appendix E). -/
theorem sat_eq_of_closed (φ : Maj2 σ) (hc : φ.Closed) (w : List σ) (ξ η : Var → ℕ) :
    φ.sat w ξ = φ.sat w η := by
  apply sat_eq_of_agree_free
  intro v hv
  rw [hc v] at hv
  cases hv

/-- The free-variable hypotheses are satisfiable (Appendix E). -/
example : (closedTop : Maj2 Bool).Closed ∧ (sym true .x : Maj2 Bool).freeIn .y = false ∧
    (∀ v, (sym true .x : Maj2 Bool).freeIn v = true → (fun _ => 1) v = (fun _ => 1) v) :=
  ⟨closed_closedTop, rfl, fun _ _ => rfl⟩

end Transformer.CRASP.Maj2
