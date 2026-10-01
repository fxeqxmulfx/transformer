/-
# PNP-free elimination of counting sugar

arXiv:2506.16055v3, Appendix A.3: `thm:strict` and conditional-term
elimination. The guarded-piece translation stays in the plain logic when
its input has no Parikh numerical predicates.
-/

import Transformer.CRASP.ExtensionsElim
import Transformer.CRASP.Conjunctions

namespace Transformer.CRASP

universe u
variable {σ : Type u}

mutual

/-- Absence of Parikh numerical predicates in the extended syntax (A.3). -/
def FormX.pnpFree : FormX σ → Bool
  | .sym _ => true
  | .lt t u => t.pnpFree && u.pnpFree
  | .neg φ => φ.pnpFree
  | .and φ ψ => φ.pnpFree && ψ.pnpFree
  | .pnp _ => false

/-- The corresponding test for extended terms (Appendix A.3). -/
def TermX.pnpFree : TermX σ → Bool
  | .countL φ | .countR φ | .countAll φ | .countLStrict φ | .countRStrict φ => φ.pnpFree
  | .cond φ t u => φ.pnpFree && t.pnpFree && u.pnpFree
  | .add t u => t.pnpFree && u.pnpFree
  | .one => true

end

mutual

/-- The plain-to-extended embedding preserves the fragment (Appendix A.3). -/
@[simp] theorem Form.pnpFree_toX : ∀ φ : Form σ, φ.toX.pnpFree = φ.pnpFree
  | .sym _ => rfl
  | .lt t u => by simp [Form.toX, FormX.pnpFree, Form.pnpFree, t.pnpFree_toX, u.pnpFree_toX]
  | .neg φ => by simp [Form.toX, FormX.pnpFree, Form.pnpFree, φ.pnpFree_toX]
  | .and φ ψ => by
      simp [Form.toX, FormX.pnpFree, Form.pnpFree, φ.pnpFree_toX, ψ.pnpFree_toX]
  | .pnp _ => rfl

/-- The embedding preserves the fragment on terms (Appendix A.3). -/
@[simp] theorem Term.pnpFree_toX : ∀ t : Term σ, t.toX.pnpFree = t.pnpFree
  | .countL φ => by simp [Term.toX, TermX.pnpFree, Term.pnpFree, φ.pnpFree_toX]
  | .countR φ => by simp [Term.toX, TermX.pnpFree, Term.pnpFree, φ.pnpFree_toX]
  | .add t u => by simp [Term.toX, TermX.pnpFree, Term.pnpFree, t.pnpFree_toX, u.pnpFree_toX]
  | .one => rfl

end

/-- Adding a natural constant introduces no PNP (Appendix A.3). -/
@[simp] theorem Term.pnpFree_addNat (t : Term σ) (n : ℕ) :
    (t.addNat n).pnpFree = t.pnpFree := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Term.addNat, Term.pnpFree, ih]

/-- Both the guards and the terms of every piece are PNP-free (A.3). -/
def PnpFreePieces (P : List (Form σ × Term σ × ℕ)) : Prop :=
  ∀ x ∈ P, x.1.pnpFree = true ∧ x.2.1.pnpFree = true

/-- Comparing PNP-free pieces introduces no PNP (Appendix A.3). -/
private theorem pnpFree_ltPieces {P Q : List (Form σ × Term σ × ℕ)}
    (hP : PnpFreePieces P) (hQ : PnpFreePieces Q) : (ltPieces P Q).pnpFree = true := by
  apply Form.pnpFree_any
  simp only [List.mem_flatMap, List.mem_map]
  rintro _ ⟨x, hx, y, hy, rfl⟩
  simp [Form.pnpFree, (hP x hx).1, (hP x hx).2, (hQ y hy).1, (hQ y hy).2]

/-- A conditional with PNP-free guards and branches stays PNP-free (A.3). -/
private theorem pnpFree_condPieces {c : Form σ} {P Q : List (Form σ × Term σ × ℕ)}
    (hc : c.pnpFree = true) (hP : PnpFreePieces P) (hQ : PnpFreePieces Q) :
    PnpFreePieces (condPieces c P Q) := by
  simp only [PnpFreePieces, condPieces, List.mem_append, List.mem_map]
  rintro _ (⟨x, hx, rfl⟩ | ⟨x, hx, rfl⟩)
  · simp [Form.pnpFree, hc, (hP x hx).1, (hP x hx).2]
  · simp [Form.pnpFree, hc, (hQ x hx).1, (hQ x hx).2]

/-- Adding PNP-free pieces introduces no PNP (Appendix A.3). -/
private theorem pnpFree_addPieces {P Q : List (Form σ × Term σ × ℕ)}
    (hP : PnpFreePieces P) (hQ : PnpFreePieces Q) : PnpFreePieces (addPieces P Q) := by
  simp only [PnpFreePieces, addPieces, List.mem_flatMap, List.mem_map]
  rintro _ ⟨x, hx, y, hy, rfl⟩
  simp [Form.pnpFree, Term.pnpFree, (hP x hx).1, (hP x hx).2, (hQ y hy).1, (hQ y hy).2]

mutual

/-- Eliminating sugar preserves the plain-logic fragment (Appendix A.3). -/
theorem FormX.pnpFree_elim : ∀ φ : FormX σ, φ.pnpFree = true → φ.elim.pnpFree = true
  | .sym _, _ => rfl
  | .lt t u, h => by
      simp only [FormX.pnpFree, Bool.and_eq_true] at h
      exact pnpFree_ltPieces (t.pnpFree_pieces h.1) (u.pnpFree_pieces h.2)
  | .neg φ, h => φ.pnpFree_elim h
  | .and φ ψ, h => by
      simp only [FormX.pnpFree, Bool.and_eq_true] at h
      simp only [FormX.elim, Form.pnpFree, Bool.and_eq_true]
      exact ⟨φ.pnpFree_elim h.1, ψ.pnpFree_elim h.2⟩
  | .pnp _, h => by cases h

/-- All the pieces used in elimination are PNP-free (Appendix A.3). -/
theorem TermX.pnpFree_pieces : ∀ t : TermX σ, t.pnpFree = true → PnpFreePieces t.pieces
  | .countL φ, h | .countR φ, h | .countAll φ, h |
      .countLStrict φ, h | .countRStrict φ, h => by
      have hp := φ.pnpFree_elim h
      simp only [PnpFreePieces, TermX.pieces, List.mem_cons, List.not_mem_nil, or_false]
      rintro x (rfl | rfl | rfl) <;>
        simp [Form.pnpFree, Term.pnpFree, Form.pos, Form.topAt, hp]
  | .cond φ t u, h => by
      simp only [TermX.pnpFree, Bool.and_eq_true] at h
      exact pnpFree_condPieces (φ.pnpFree_elim h.1.1)
        (t.pnpFree_pieces h.1.2) (u.pnpFree_pieces h.2)
  | .add t u, h => by
      simp only [TermX.pnpFree, Bool.and_eq_true] at h
      exact pnpFree_addPieces (t.pnpFree_pieces h.1) (u.pnpFree_pieces h.2)
  | .one, _ => by simp [PnpFreePieces, TermX.pieces, Form.topAt, Form.pnpFree, Term.pnpFree]

end

/-- The fragment hypotheses above have witnesses (Appendix A.3). -/
example : (FormX.sym true).pnpFree = true ∧ (TermX.countAll (FormX.sym true)).pnpFree = true ∧
    PnpFreePieces [((Form.sym true), (Term.one : Term Bool), 0)] := by
  simp [FormX.pnpFree, TermX.pnpFree, PnpFreePieces, Form.pnpFree, Term.pnpFree]

end Transformer.CRASP
