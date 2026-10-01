/-
# Correct Boolean evaluation from memory and comparison signs

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
The feed-forward map takes Boolean combinations of already correct
symbol features and newly computed comparison features.
-/

import Transformer.CRASP.TemporalProgramState

namespace Transformer.CRASP.TemporalProgram

universe u
variable {σ : Type u} [DecidableEq σ]

/-- Correct atom inputs give correct Boolean combinations (Appendix B.2). -/
theorem evaluate_eq_sat (φ : Form σ) (H : State φ) (w : List σ) (i j : ℕ)
    (hsym : ∀ a, Form.sym a ∈ φ.subformulas → read φ H (.sym a) = (Form.sym a).sat w i)
    (hcmp : ∀ t u, Form.lt t u ∈ φ.subformulas → (Form.lt t u).depth ≤ j →
      readComparison φ H (.lt t u) = (Form.lt t u).sat w i) :
    ∀ ψ : Form σ, ψ ∈ φ.subformulas → ψ.pnpFree = true → ψ.depth ≤ j →
      evaluate φ H ψ = ψ.sat w i
  | .sym a, hψ, hf, hd => hsym a hψ
  | .lt t u, hψ, hf, hd => hcmp t u hψ hd
  | .neg ψ, hψ, hf, hd => by
      have hm : ψ ∈ φ.subformulas := φ.subformulas_trans _ _ hψ
        (List.mem_cons_of_mem _ ψ.mem_subformulas)
      rw [evaluate, Form.sat, evaluate_eq_sat φ H w i j hsym hcmp ψ hm hf hd]
  | .and ψ χ, hψ, hf, hd => by
      have hmψ : ψ ∈ φ.subformulas := φ.subformulas_trans _ _ hψ
        (List.mem_cons_of_mem _ (List.mem_append_left _ ψ.mem_subformulas))
      have hmχ : χ ∈ φ.subformulas := φ.subformulas_trans _ _ hψ
        (List.mem_cons_of_mem _ (List.mem_append_right _ χ.mem_subformulas))
      simp only [Form.pnpFree, Bool.and_eq_true] at hf
      rw [evaluate, Form.sat, evaluate_eq_sat φ H w i j hsym hcmp ψ hmψ hf.1
        ((le_max_left _ _).trans hd), evaluate_eq_sat φ H w i j hsym hcmp χ hmχ hf.2
        ((le_max_right _ _).trans hd)]
  | .pnp π, hψ, hf, hd => by cases hf

/-- The atom-input, membership, fragment and bound hypotheses have witnesses (B.2). -/
example : (Form.sym true) ∈ (Form.sym true).subformulas ∧
    (Form.sym true).pnpFree = true ∧ (Form.sym true).depth ≤ 0 ∧
    (∀ a : Bool, Form.sym a ∈ (Form.sym true).subformulas →
      read (Form.sym true) (fun _ => 0) (.sym a) = (Form.sym a).sat [] 0) ∧
    (∀ t u : Term Bool, Form.lt t u ∈ (Form.sym true).subformulas →
      (Form.lt t u).depth ≤ 0 → readComparison (Form.sym true) (fun _ => 0) (.lt t u) =
        (Form.lt t u).sat [] 0) := by
  refine ⟨Form.mem_subformulas _, rfl, le_rfl, ?_, ?_⟩
  · intro a ha
    unfold read
    split_ifs
    simp [Form.sat]
  · intro t u hmem hd
    simp [Form.subformulas] at hmem

end Transformer.CRASP.TemporalProgram
