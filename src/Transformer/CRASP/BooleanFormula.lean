/-
# Boolean functions of finitely many formulas

arXiv:2506.16055v3, Appendix E, the Boolean normalization in
`thm:majtwo_to_tlc`. Shannon expansion expresses every finite Boolean
function without increasing counting depth.
-/

import Transformer.CRASP.Conjunctions

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- Shannon expansion of a Boolean function of a list of formulas (Appendix E). -/
def Form.booleanComb : List (Form σ) → (List Bool → Bool) → Form σ
  | [], f => if f [] then Form.topAt 0 else .lt .one .one
  | φ :: L, f => Form.or
      (.and φ (booleanComb L fun b => f (true :: b)))
      (.and (.neg φ) (booleanComb L fun b => f (false :: b)))

/-- Boolean normalization adds no counting depth (Appendix E). -/
theorem Form.depth_booleanComb_le (L : List (Form σ)) (f : List Bool → Bool)
    (d : ℕ) (hL : ∀ φ ∈ L, φ.depth ≤ d) : (Form.booleanComb L f).depth ≤ d := by
  induction L generalizing f with
  | nil => cases hf : f [] <;>
      simp [Form.booleanComb, hf, Form.topAt, Form.depth, Term.depth]
  | cons φ L ih =>
      have hφ := hL φ (List.mem_cons_self ..)
      have ht : ∀ ψ ∈ L, ψ.depth ≤ d := fun ψ hψ => hL ψ (List.mem_cons_of_mem _ hψ)
      simp only [Form.booleanComb, Form.depth_or, Form.depth, max_le_iff]
      exact ⟨⟨hφ, ih _ ht⟩, hφ, ih _ ht⟩

/-- Boolean normalization preserves the plain-logic fragment (Appendix E). -/
theorem Form.pnpFree_booleanComb (L : List (Form σ)) (f : List Bool → Bool)
    (hL : ∀ φ ∈ L, φ.pnpFree = true) : (Form.booleanComb L f).pnpFree = true := by
  induction L generalizing f with
  | nil => cases hf : f [] <;> simp [Form.booleanComb, hf, Form.topAt,
      Form.pnpFree, Term.pnpFree]
  | cons φ L ih =>
      have hφ := hL φ (List.mem_cons_self ..)
      have ht : ∀ ψ ∈ L, ψ.pnpFree = true := fun ψ hψ => hL ψ (List.mem_cons_of_mem _ hψ)
      simp only [Form.booleanComb, Form.or, Form.pnpFree, Bool.and_eq_true]
      exact ⟨⟨hφ, ih _ ht⟩, hφ, ih _ ht⟩

/-- The normalization hypotheses are satisfiable (Appendix E). -/
example : (∀ φ ∈ [Form.sym true], (φ : Form Bool).depth ≤ 0) ∧
    (∀ φ ∈ [Form.sym true], (φ : Form Bool).pnpFree = true) := by
  simp [Form.depth, Form.pnpFree]

variable [DecidableEq σ]

/-- The constructed formula computes the given Boolean function (Appendix E). -/
theorem Form.sat_booleanComb (L : List (Form σ)) (f : List Bool → Bool)
    (w : List σ) (i : ℕ) :
    (Form.booleanComb L f).sat w i = f (L.map fun φ => φ.sat w i) := by
  induction L generalizing f with
  | nil => cases hf : f [] <;> simp [Form.booleanComb, hf, Form.sat, Term.val]
  | cons φ L ih =>
      simp only [Form.booleanComb, Form.sat_or, Form.sat, ih, List.map_cons]
      cases φ.sat w i <;> simp

end Transformer.CRASP
