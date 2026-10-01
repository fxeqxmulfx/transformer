/-
# Bounded previous-position tests for ALiBi windows

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
Repeated `Y` reads the finitely many recent source states without adding
counting depth and detects whether the ordinary prefix is long enough.
-/

import Transformer.CRASP.PositionalBoolean
import Transformer.CRASP.PositionalSubstitution

namespace Transformer.CRASP

universe u v
variable {α : Type u} {σ : Type v}

mutual

/-- Substituting MOD-free source predicates introduces no MOD atoms (F). -/
theorem Form.modFree_substPos (ψ : α → FormP σ)
    (hψ : ∀ a, (ψ a).modFree = true) :
    ∀ φ : Form α, (φ.substPos ψ).modFree = true
  | .sym a => hψ a
  | .lt t u => by simp [Form.substPos, FormP.modFree,
      t.modFree_substPos ψ hψ, u.modFree_substPos ψ hψ]
  | .neg φ => φ.modFree_substPos ψ hψ
  | .and φ χ => by simp [Form.substPos, FormP.modFree,
      φ.modFree_substPos ψ hψ, χ.modFree_substPos ψ hψ]
  | .pnp _ => rfl

/-- The same MOD-free substitution property holds for counting terms (F). -/
theorem Term.modFree_substPos (ψ : α → FormP σ)
    (hψ : ∀ a, (ψ a).modFree = true) :
    ∀ t : Term α, (t.substPos ψ).modFree = true
  | .countL φ => φ.modFree_substPos ψ hψ
  | .countR _ => rfl
  | .add t u => by simp [Term.substPos, TermP.modFree,
      t.modFree_substPos ψ hψ, u.modFree_substPos ψ hψ]
  | .one => rfl

end

namespace FormP

variable {k j : ℕ}

/-- A fixed number of previous-position operators (Appendix F). -/
def shift : ℕ → FormP σ → FormP σ
  | 0, φ => φ
  | n + 1, φ => .prev (shift n φ)

/-- Previous-position tests add no counting depth (Appendix F). -/
@[simp] theorem depth_shift (n : ℕ) (φ : FormP σ) : (shift n φ).depth = φ.depth := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih

/-- Previous-position tests introduce no MOD predicates (Appendix F). -/
@[simp] theorem modFree_shift (n : ℕ) (φ : FormP σ) :
    (shift n φ).modFree = φ.modFree := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih

/-- Constants lie in every depth of the previous-position fragment (F). -/
theorem truth_mem_Y (b : Bool) (k : ℕ) : (truth b : FormP σ) ∈ TLClY σ k := by
  cases b <;> exact ⟨rfl, Nat.zero_le _⟩

/-- Negation preserves the previous-position fragment (Appendix F). -/
theorem neg_mem_Y {φ : FormP σ} (hφ : φ ∈ TLClY σ k) : φ.neg ∈ TLClY σ k := hφ

/-- Conjunction preserves the previous-position fragment (Appendix F). -/
theorem and_mem_Y {φ ψ : FormP σ} (hφ : φ ∈ TLClY σ k) (hψ : ψ ∈ TLClY σ k) :
    (φ.and ψ) ∈ TLClY σ k :=
  ⟨by simp [FormP.modFree, hφ.1, hψ.1], max_le hφ.2 hψ.2⟩

/-- Disjunction preserves the previous-position fragment (Appendix F). -/
theorem or_mem_Y {φ ψ : FormP σ} (hφ : φ ∈ TLClY σ k) (hψ : ψ ∈ TLClY σ k) :
    (φ.or ψ) ∈ TLClY σ k := neg_mem_Y (and_mem_Y (neg_mem_Y hφ) (neg_mem_Y hψ))

/-- Increasing the bound preserves previous-position formulas (Appendix F). -/
theorem mem_mono_Y {φ : FormP σ} (hφ : φ ∈ TLClY σ k) (hk : k ≤ j) :
    φ ∈ TLClY σ j := ⟨hφ.1, hφ.2.trans hk⟩

/-- A bounded previous-position test keeps the same fragment and bound (F). -/
theorem shift_mem_Y {φ : FormP σ} (hφ : φ ∈ TLClY σ k) (n : ℕ) :
    shift n φ ∈ TLClY σ k := by
  change (shift n φ).modFree = true ∧ (shift n φ).depth ≤ k
  rw [modFree_shift, depth_shift]
  exact hφ

/-- Finite state disjunctions preserve the previous-position fragment (F). -/
theorem any_mem_Y (L : List (FormP σ)) (hL : ∀ φ ∈ L, φ ∈ TLClY σ k) :
    any L ∈ TLClY σ k := by
  induction L with
  | nil => exact truth_mem_Y _ _
  | cons φ L ih =>
      exact or_mem_Y (hL φ (List.mem_cons_self ..))
        (ih fun ψ hψ => hL ψ (List.mem_cons_of_mem _ hψ))

/-- Finite window conjunctions preserve the previous-position fragment (F). -/
theorem all_mem_Y (L : List (FormP σ)) (hL : ∀ φ ∈ L, φ ∈ TLClY σ k) :
    all L ∈ TLClY σ k := by
  induction L with
  | nil => exact truth_mem_Y _ _
  | cons φ L ih =>
      exact and_mem_Y (hL φ (List.mem_cons_self ..))
        (ih fun ψ hψ => hL ψ (List.mem_cons_of_mem _ hψ))

/-- The fragment and depth hypotheses admit ordinary letter predicates (F). -/
example : (FormP.sym true) ∈ TLClY Bool 0 ∧ 0 ≤ 1 ∧
    (∀ φ ∈ [FormP.sym true], φ ∈ TLClY Bool 0) := by
  simp [TLClY, FormP.modFree, FormP.depth]

variable [DecidableEq σ]

/-- At a positive position, `Y^n` reads position `i-n` exactly when it exists.
Source: arXiv:2506.16055v3, Appendix F, semantic rule for `Y`. -/
theorem sat_shift (n : ℕ) (φ : FormP σ) (w : List σ) (i : ℕ) (hi : 0 < i) :
    (shift n φ).sat w i = (decide (n < i) && φ.sat w (i - n)) := by
  induction n generalizing i with
  | zero => simp [shift, hi]
  | succ n ih =>
      rw [shift, FormP.sat]
      by_cases h : 1 < i
      · rw [show decide (1 < i) = true by simp [h], Bool.true_and, ih _ (by omega)]
        rw [show i - 1 - n = i - (n + 1) by omega]
        congr 1
        apply decide_eq_decide.mpr
        omega
      · have hn : ¬ n + 1 < i := by omega
        simp only [h, hn, decide_false, Bool.false_and]

end FormP

end Transformer.CRASP
