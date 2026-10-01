/-
# A past term as a constant and a list of counts

arXiv:2506.16055v3, Appendix B.2, `thm:TLCl_to_rtfr`.
A comparison is an integer constant plus signed counts of formulas
from the previous layer; its uniform average has the same sign.
-/

import Transformer.CRASP.Subformulas
import Transformer.CRASP.StateCounts

namespace Transformer.CRASP

universe u
variable {σ : Type u}

/-- The constant summands of a term (Appendix B.2). -/
def Term.constant : Term σ → ℕ
  | .countL _ | .countR _ => 0
  | .add t u => t.constant + u.constant
  | .one => 1

/-- The bodies of the term's outermost counts, with multiplicities (B.2). -/
def Term.countBodies : Term σ → List (Form σ)
  | .countL φ | .countR φ => [φ]
  | .add t u => t.countBodies ++ u.countBodies
  | .one => []

/-- Each count body is a formula node beneath the term (Appendix B.2). -/
theorem Term.countBody_mem_subformulas : ∀ (t : Term σ) (ψ : Form σ),
    ψ ∈ t.countBodies → ψ ∈ t.subformulas
  | .countL φ, ψ, h | .countR φ, ψ, h => by
      have he : ψ = φ := by simpa [Term.countBodies] using h
      subst ψ
      exact φ.mem_subformulas
  | .add t u, ψ, h => by
      rcases List.mem_append.mp h with h | h
      · exact List.mem_append_left _ (t.countBody_mem_subformulas ψ h)
      · exact List.mem_append_right _ (u.countBody_mem_subformulas ψ h)
  | .one, ψ, h => by cases h

/-- Every body lies one level below its surrounding term (Appendix B.2). -/
theorem Term.countBody_depth : ∀ (t : Term σ) (ψ : Form σ),
    ψ ∈ t.countBodies → ψ.depth + 1 ≤ t.depth
  | .countL φ, ψ, h | .countR φ, ψ, h => by
      have he : ψ = φ := by simpa [Term.countBodies] using h
      subst ψ
      exact le_rfl
  | .add t u, ψ, h => by
      rcases List.mem_append.mp h with h | h
      · exact (t.countBody_depth ψ h).trans (le_max_left _ _)
      · exact (u.countBody_depth ψ h).trans (le_max_right _ _)
  | .one, ψ, h => by cases h

/-- The size needed to bound one integer value-projection coordinate (B.2). -/
def Term.weightSize (t : Term σ) : ℕ := t.constant + t.countBodies.length

/-- The total bound for a comparison coordinate (Appendix B.2). -/
def Form.weightSize : Form σ → ℕ
  | .lt t u => t.weightSize + u.weightSize
  | _ => 0

/-- A common bound large enough for every comparison's source values (B.2). -/
def Form.capacity (φ : Form σ) : ℕ := (φ.subformulas.map Form.weightSize).sum

/-- Every comparison fits the common bound (Appendix B.2). -/
theorem Form.weightSize_le_capacity {φ ψ : Form σ} (hψ : ψ ∈ φ.subformulas) :
    ψ.weightSize ≤ φ.capacity :=
  List.single_le_sum (fun _ _ => Nat.zero_le _) _ (List.mem_map_of_mem hψ)

variable [DecidableEq σ]

/-- Flattening a past term preserves its value (Appendix B.2). -/
theorem Term.val_countBodies : ∀ (t : Term σ), t.past = true → ∀ (w : List σ) (i : ℕ),
    t.val w i = t.constant + (t.countBodies.map fun ψ => Term.countL ψ |>.val w i).sum
  | .countL φ, hp, w, i => by simp [Term.constant, Term.countBodies]
  | .countR φ, hp, w, i => by cases hp
  | .add t u, hp, w, i => by
      simp only [Term.past, Bool.and_eq_true] at hp
      rw [Term.val, t.val_countBodies hp.1, u.val_countBodies hp.2]
      simp only [Term.constant, Term.countBodies, List.map_append, List.sum_append]
      omega
  | .one, hp, w, i => rfl

/-- Counting true source-state bits and then summing sources gives the past counts.
Source: arXiv:2506.16055v3, Appendix B.2, uniform-attention construction. -/
theorem sum_countP_eq_counts (L : List (Form σ)) (w : List σ) (i : ℕ)
    (b : ℕ → Form σ → Bool)
    (hb : ∀ ψ ∈ L, ∀ j ∈ Finset.Icc 1 i, b j ψ = ψ.sat w j) :
    (∑ j ∈ Finset.Icc 1 i, (L.countP (b j) : ℤ)) =
      ((L.map fun ψ => Term.countL ψ |>.val w i).sum : ℤ) := by
  induction L with
  | nil => simp
  | cons ψ L ih =>
      have hψ : ∀ j ∈ Finset.Icc 1 i, b j ψ = ψ.sat w j := hb ψ (List.mem_cons_self ..)
      have ht : ∀ χ ∈ L, ∀ j ∈ Finset.Icc 1 i, b j χ = χ.sat w j :=
        fun χ hχ => hb χ (List.mem_cons_of_mem _ hχ)
      simp only [List.countP_cons, Nat.cast_add, Nat.cast_ite, Nat.cast_one, Nat.cast_zero,
        Finset.sum_add_distrib, List.map_cons, List.sum_cons]
      rw [ih ht]
      have hsingle : (∑ j ∈ Finset.Icc 1 i, if b j ψ then (1 : ℤ) else 0) =
          (Term.countL ψ |>.val w i : ℤ) := by
        rw [LinearCount.val_countL_sum]
        apply Finset.sum_congr rfl
        intro j hj
        rw [hψ j hj]
      rw [hsingle]
      push_cast
      ring

/-- The subformula, body and fragment hypotheses have witnesses (B.2). -/
example : (Form.sym true) ∈ (Term.countL (.sym true)).countBodies ∧
    (Form.sym true) ∈ (Form.neg (.sym true)).subformulas ∧
    (Term.countL (.sym true) : Term Bool).past = true ∧
    (∀ ψ ∈ [Form.sym true], ∀ j ∈ Finset.Icc 1 1, ψ.sat [true] j = ψ.sat [true] j) := by
  simp [Term.countBodies, Form.subformulas, Term.past, Form.past]

end Transformer.CRASP
