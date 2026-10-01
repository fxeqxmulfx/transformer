/-
# The finite collection of subformulas

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
The simulated formula and all its Boolean/counting descendants are
stored in parallel Boolean coordinates.
-/

import Transformer.CRASP.DepthZero

namespace Transformer.CRASP

universe u
variable {σ : Type u}

mutual

/-- All formula nodes, including those beneath counting terms (Appendix B.2). -/
def Form.subformulas : Form σ → List (Form σ)
  | φ@(.sym _) | φ@(.pnp _) => [φ]
  | φ@(.lt t u) => φ :: (t.subformulas ++ u.subformulas)
  | φ@(.neg ψ) => φ :: ψ.subformulas
  | φ@(.and ψ χ) => φ :: (ψ.subformulas ++ χ.subformulas)

/-- Formula nodes reached through a term (Appendix B.2). -/
def Term.subformulas : Term σ → List (Form σ)
  | .countL φ | .countR φ => φ.subformulas
  | .add t u => t.subformulas ++ u.subformulas
  | .one => []

end

/-- Every formula is among its own subformulas (Appendix B.2). -/
theorem Form.mem_subformulas (φ : Form σ) : φ ∈ φ.subformulas := by
  cases φ <;> exact List.mem_cons_self

mutual

/-- A subformula's descendants remain subformulas of the original (B.2). -/
theorem Form.subformulas_trans : ∀ (φ ψ χ : Form σ),
    ψ ∈ φ.subformulas → χ ∈ ψ.subformulas → χ ∈ φ.subformulas
  | .sym a, ψ, χ, hψ, hχ => by
      have he : ψ = .sym a := by simpa [Form.subformulas] using hψ
      subst ψ
      exact hχ
  | .pnp a, ψ, χ, hψ, hχ => by
      have he : ψ = .pnp a := by simpa [Form.subformulas] using hψ
      subst ψ
      exact hχ
  | .lt t u, ψ, χ, hψ, hχ => by
      rcases List.mem_cons.mp hψ with hψ | hψ
      · subst ψ; exact hχ
      · apply List.mem_cons_of_mem
        rcases List.mem_append.mp hψ with hψ | hψ
        · exact List.mem_append_left _ (t.subformulas_trans ψ χ hψ hχ)
        · exact List.mem_append_right _ (u.subformulas_trans ψ χ hψ hχ)
  | .neg φ, ψ, χ, hψ, hχ => by
      rcases List.mem_cons.mp hψ with hψ | hψ
      · subst ψ; exact hχ
      · exact List.mem_cons_of_mem _ (φ.subformulas_trans ψ χ hψ hχ)
  | .and φ θ, ψ, χ, hψ, hχ => by
      rcases List.mem_cons.mp hψ with hψ | hψ
      · subst ψ; exact hχ
      · apply List.mem_cons_of_mem
        rcases List.mem_append.mp hψ with hψ | hψ
        · exact List.mem_append_left _ (φ.subformulas_trans ψ χ hψ hχ)
        · exact List.mem_append_right _ (θ.subformulas_trans ψ χ hψ hχ)

/-- Descendants of a term's formula node remain nodes of that term (B.2). -/
theorem Term.subformulas_trans : ∀ (t : Term σ) (ψ χ : Form σ),
    ψ ∈ t.subformulas → χ ∈ ψ.subformulas → χ ∈ t.subformulas
  | .countL φ, ψ, χ, hψ, hχ | .countR φ, ψ, χ, hψ, hχ =>
      φ.subformulas_trans ψ χ hψ hχ
  | .add t u, ψ, χ, hψ, hχ => by
      rcases List.mem_append.mp hψ with hψ | hψ
      · exact List.mem_append_left _ (t.subformulas_trans ψ χ hψ hχ)
      · exact List.mem_append_right _ (u.subformulas_trans ψ χ hψ hχ)
  | .one, ψ, χ, hψ, hχ => by cases hψ

end

mutual

/-- Descendants inherit the fragment and depth bounds (Appendix B.2). -/
theorem Form.subformulas_properties : ∀ (φ : Form σ), φ.past = true → φ.pnpFree = true →
    ∀ ψ ∈ φ.subformulas, ψ.past = true ∧ ψ.pnpFree = true ∧ ψ.depth ≤ φ.depth
  | .sym a, hp, hf, ψ, hψ => by
      have he : ψ = .sym a := by simpa [Form.subformulas] using hψ
      subst ψ
      exact ⟨hp, hf, le_rfl⟩
  | .pnp a, hp, hf, ψ, hψ => by
      have he : ψ = .pnp a := by simpa [Form.subformulas] using hψ
      subst ψ
      exact ⟨hp, hf, le_rfl⟩
  | .lt t u, hp, hf, ψ, hψ => by
      rcases List.mem_cons.mp hψ with hψ | hψ
      · subst ψ; exact ⟨hp, hf, le_rfl⟩
      · simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hf
        rcases List.mem_append.mp hψ with hψ | hψ
        · obtain ⟨h₁, h₂, hd⟩ := t.subformulas_properties hp.1 hf.1 ψ hψ
          exact ⟨h₁, h₂, hd.trans (le_max_left _ _)⟩
        · obtain ⟨h₁, h₂, hd⟩ := u.subformulas_properties hp.2 hf.2 ψ hψ
          exact ⟨h₁, h₂, hd.trans (le_max_right _ _)⟩
  | .neg φ, hp, hf, ψ, hψ => by
      rcases List.mem_cons.mp hψ with hψ | hψ
      · subst ψ; exact ⟨hp, hf, le_rfl⟩
      · exact φ.subformulas_properties hp hf ψ hψ
  | .and φ θ, hp, hf, ψ, hψ => by
      rcases List.mem_cons.mp hψ with hψ | hψ
      · subst ψ; exact ⟨hp, hf, le_rfl⟩
      · simp only [Form.past, Form.pnpFree, Bool.and_eq_true] at hp hf
        rcases List.mem_append.mp hψ with hψ | hψ
        · obtain ⟨h₁, h₂, hd⟩ := φ.subformulas_properties hp.1 hf.1 ψ hψ
          exact ⟨h₁, h₂, hd.trans (le_max_left _ _)⟩
        · obtain ⟨h₁, h₂, hd⟩ := θ.subformulas_properties hp.2 hf.2 ψ hψ
          exact ⟨h₁, h₂, hd.trans (le_max_right _ _)⟩

/-- A term's formula descendants inherit its fragment and depth bound (B.2). -/
theorem Term.subformulas_properties : ∀ (t : Term σ), t.past = true → t.pnpFree = true →
    ∀ ψ ∈ t.subformulas, ψ.past = true ∧ ψ.pnpFree = true ∧ ψ.depth ≤ t.depth
  | .countL φ, hp, hf, ψ, hψ => by
      obtain ⟨h₁, h₂, hd⟩ := φ.subformulas_properties hp hf ψ hψ
      exact ⟨h₁, h₂, hd.trans (Nat.le_succ _)⟩
  | .countR φ, hp, hf, ψ, hψ => by cases hp
  | .add t u, hp, hf, ψ, hψ => by
      simp only [Term.past, Term.pnpFree, Bool.and_eq_true] at hp hf
      rcases List.mem_append.mp hψ with hψ | hψ
      · obtain ⟨h₁, h₂, hd⟩ := t.subformulas_properties hp.1 hf.1 ψ hψ
        exact ⟨h₁, h₂, hd.trans (le_max_left _ _)⟩
      · obtain ⟨h₁, h₂, hd⟩ := u.subformulas_properties hp.2 hf.2 ψ hψ
        exact ⟨h₁, h₂, hd.trans (le_max_right _ _)⟩
  | .one, hp, hf, ψ, hψ => by cases hψ

end

/-- Subformula membership and the fragment hypotheses have witnesses (B.2). -/
example : (Form.sym true).past = true ∧ (Form.sym true).pnpFree = true ∧
    (Form.sym true) ∈ (Form.neg (.sym true)).subformulas ∧
    (Form.sym true) ∈ (Term.countL (.sym true)).subformulas := by
  simp [Form.past, Form.pnpFree, Form.subformulas, Term.subformulas]

end Transformer.CRASP
