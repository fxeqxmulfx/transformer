/-
# Boolean formulas in the past-counting fragment

arXiv:2506.16055v3, Appendix B.2, `lem:finite_function` and the
finite-state normal forms in `thm:rtfr_to_TLCl`.
-/

import Transformer.CRASP.Conjunctions

namespace Transformer.CRASP.Form

universe u
variable {σ : Type u} {k j : ℕ}

/-- A Boolean constant written in the original grammar (Appendix B.2). -/
def truth (b : Bool) : Form σ := if b then .neg (.lt .one .one) else .lt .one .one

/-- Boolean constants have depth zero (Appendix B.2). -/
theorem truth_mem (b : Bool) (k : ℕ) : (truth b : Form σ) ∈ TLCl σ k := by
  cases b <;> exact ⟨rfl, rfl, Nat.zero_le k⟩

/-- Increasing a depth bound preserves membership (Appendix B.2). -/
theorem mem_mono {φ : Form σ} (hφ : φ ∈ TLCl σ k) (hk : k ≤ j) : φ ∈ TLCl σ j :=
  ⟨hφ.1, hφ.2.1, hφ.2.2.trans hk⟩

/-- Negation preserves the fragment and bound (Appendix B.2). -/
theorem neg_mem {φ : Form σ} (hφ : φ ∈ TLCl σ k) : φ.neg ∈ TLCl σ k := hφ

/-- Conjunction preserves the fragment and bound (Appendix B.2). -/
theorem and_mem {φ ψ : Form σ} (hφ : φ ∈ TLCl σ k) (hψ : ψ ∈ TLCl σ k) :
    (φ.and ψ) ∈ TLCl σ k := by
  exact ⟨by simp [Form.past, hφ.1, hψ.1],
    by simp [Form.pnpFree, hφ.2.1, hψ.2.1], max_le hφ.2.2 hψ.2.2⟩

/-- Disjunction preserves the fragment and bound (Appendix B.2). -/
theorem or_mem {φ ψ : Form σ} (hφ : φ ∈ TLCl σ k) (hψ : ψ ∈ TLCl σ k) :
    (φ.or ψ) ∈ TLCl σ k := neg_mem (and_mem (neg_mem hφ) (neg_mem hψ))

/-- Finite disjunction preserves the fragment and bound (Appendix B.2). -/
theorem any_mem (L : List (Form σ)) (hL : ∀ φ ∈ L, φ ∈ TLCl σ k) :
    Form.any L ∈ TLCl σ k :=
  ⟨Form.past_any L fun φ hφ => (hL φ hφ).1,
    Form.pnpFree_any L fun φ hφ => (hL φ hφ).2.1,
    Form.depth_any_le L fun φ hφ => (hL φ hφ).2.2⟩

/-- Finite conjunction preserves the fragment and bound (Appendix B.2). -/
theorem all_mem (L : List (Form σ)) (hL : ∀ φ ∈ L, φ ∈ TLCl σ k) :
    Form.all L ∈ TLCl σ k :=
  ⟨Form.past_all L fun φ hφ => (hL φ hφ).1,
    Form.pnpFree_all L fun φ hφ => (hL φ hφ).2.1,
    Form.depth_all_le L fun φ hφ => (hL φ hφ).2.2⟩

/-- The Boolean-operation hypotheses have witnesses (Appendix B.2). -/
example : (Form.sym true : Form Bool) ∈ TLCl Bool 0 ∧ 0 ≤ 1 ∧
    (∀ φ ∈ [Form.sym true], φ ∈ TLCl Bool 0) := by
  simp [TLCl, Form.past, Form.pnpFree, Form.depth]

variable [DecidableEq σ]

/-- The Boolean constant has its stated truth value (Appendix B.2). -/
@[simp] theorem sat_truth (b : Bool) (w : List σ) (i : ℕ) :
    (truth b : Form σ).sat w i = b := by
  cases b <;> simp [truth, Form.sat, Term.val]

end Transformer.CRASP.Form
