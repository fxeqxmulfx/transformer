/-
# Substituting positional state predicates into counting formulas

arXiv:2506.16055v3, Appendix F, adaptations of `thm:rtfr_to_TLCl`.
The finite-state count comparisons from Appendix B.2 are reused by replacing
their letter predicates with positional formulas.
-/

import Transformer.CRASP.PositionalEmbedding
import Transformer.CRASP.FormulaBounds

namespace Transformer.CRASP

universe u v
variable {α : Type u} {σ : Type v}

mutual

/-- Substitute state predicates into the past-counting syntax (Appendix F). -/
def Form.substPos (ψ : α → FormP σ) : Form α → FormP σ
  | .sym a => ψ a
  | .lt t u => .lt (t.substPos ψ) (u.substPos ψ)
  | .neg φ => .neg (φ.substPos ψ)
  | .and φ χ => .and (φ.substPos ψ) (χ.substPos ψ)
  | .pnp _ => .lt .one .one

/-- Substitute state predicates into a counting term (Appendix F). -/
def Term.substPos (ψ : α → FormP σ) : Term α → TermP σ
  | .countL φ => .countL (φ.substPos ψ)
  | .countR _ => .one
  | .add t u => .add (t.substPos ψ) (u.substPos ψ)
  | .one => .one

end

mutual

/-- Substitution adds only the depth of the inserted predicates (Appendix F). -/
theorem Form.depth_substPos_le (ψ : α → FormP σ) (m : ℕ)
    (hψ : ∀ a, (ψ a).depth ≤ m) :
    ∀ φ : Form α, (φ.substPos ψ).depth ≤ φ.depth + m
  | .sym a => by simpa only [Form.substPos, Form.depth, Nat.zero_add] using hψ a
  | .lt t u => by
      simp only [Form.substPos, FormP.depth, Form.depth, max_le_iff]
      exact ⟨(t.depth_substPos_le ψ m hψ).trans (by omega),
        (u.depth_substPos_le ψ m hψ).trans (by omega)⟩
  | .neg φ => φ.depth_substPos_le ψ m hψ
  | .and φ χ => by
      simp only [Form.substPos, FormP.depth, Form.depth, max_le_iff]
      exact ⟨(φ.depth_substPos_le ψ m hψ).trans (by omega),
        (χ.depth_substPos_le ψ m hψ).trans (by omega)⟩
  | .pnp _ => Nat.zero_le _

/-- The same depth estimate holds beneath counts (Appendix F). -/
theorem Term.depth_substPos_le (ψ : α → FormP σ) (m : ℕ)
    (hψ : ∀ a, (ψ a).depth ≤ m) :
    ∀ t : Term α, (t.substPos ψ).depth ≤ t.depth + m
  | .countL φ => by
      have h := φ.depth_substPos_le ψ m hψ
      change (φ.substPos ψ).depth + 1 ≤ φ.depth + 1 + m
      omega
  | .countR _ => Nat.zero_le _
  | .add t u => by
      simp only [Term.substPos, TermP.depth, Term.depth, max_le_iff]
      exact ⟨(t.depth_substPos_le ψ m hψ).trans (by omega),
        (u.depth_substPos_le ψ m hψ).trans (by omega)⟩
  | .one => Nat.zero_le _

end

mutual

/-- Substitution introduces no previous-position operators (Appendix F). -/
theorem Form.prevFree_substPos (ψ : α → FormP σ)
    (hψ : ∀ a, (ψ a).prevFree = true) :
    ∀ φ : Form α, (φ.substPos ψ).prevFree = true
  | .sym a => hψ a
  | .lt t u => by simp [Form.substPos, FormP.prevFree,
      t.prevFree_substPos ψ hψ, u.prevFree_substPos ψ hψ]
  | .neg φ => φ.prevFree_substPos ψ hψ
  | .and φ χ => by simp [Form.substPos, FormP.prevFree,
      φ.prevFree_substPos ψ hψ, χ.prevFree_substPos ψ hψ]
  | .pnp _ => rfl

/-- No previous-position operators appear in substituted terms (Appendix F). -/
theorem Term.prevFree_substPos (ψ : α → FormP σ)
    (hψ : ∀ a, (ψ a).prevFree = true) :
    ∀ t : Term α, (t.substPos ψ).prevFree = true
  | .countL φ => φ.prevFree_substPos ψ hψ
  | .countR _ => rfl
  | .add t u => by simp [Term.substPos, TermP.prevFree,
      t.prevFree_substPos ψ hψ, u.prevFree_substPos ψ hψ]
  | .one => rfl

end

/-- The depth and fragment hypotheses admit constant state predicates (F). -/
example : (∀ _ : Bool, (FormP.lt .one .one : FormP Bool).depth ≤ 0) ∧
    (∀ _ : Bool, (FormP.lt .one .one : FormP Bool).prevFree = true) :=
  ⟨fun _ => le_rfl, fun _ => rfl⟩

end Transformer.CRASP
