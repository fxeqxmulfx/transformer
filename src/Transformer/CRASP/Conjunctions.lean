/-
# The conjunction of a list, and the test that a position carries a letter

Support for arXiv:2506.16055v3, Appendix B.2.  `Form.any` (in
`Transformer.CRASP.ExtensionsPieces`) is the disjunction of a list of
formulas; `Form.all` here is its dual, and both keep the fragment: a Boolean
combination of past-only formulas is past-only, and of `PNP`-free formulas is
`PNP`-free.

`Form.onStr` is `⋁_{a ∈ Σ} Q_a`, true exactly at the positions that carry a
letter.  The logic judges strings at their last position but its semantics is
defined at every `i`, and a formula that is to describe a *string-indexed*
quantity has to say where the string stops.  The alphabet is finite — it is a
vocabulary in the paper — which is what lets the test be written at depth `0`
without a Parikh predicate; over an infinite alphabet no depth-`0` `PNP`-free
formula can express it, since any one of them mentions finitely many letters.
-/

import Transformer.CRASP.ExtensionsPieces

namespace Transformer
namespace CRASP

universe u

variable {σ : Type u} [DecidableEq σ]

/-- The conjunction of a list of formulas; the empty conjunction is `⊤`. -/
def Form.all (L : List (Form σ)) : Form σ := L.foldr Form.and (.neg (.lt .one .one))

theorem Form.sat_all (w : List σ) (i : ℕ) (L : List (Form σ)) :
    (Form.all L).sat w i = true ↔ ∀ φ ∈ L, φ.sat w i = true := by
  induction L with
  | nil => simp [Form.all, Form.sat, Term.val]
  | cons φ L ih => simp [Form.all, Form.sat, ← ih]

omit [DecidableEq σ] in
theorem Form.depth_all_le {d : ℕ} (L : List (Form σ)) (h : ∀ φ ∈ L, φ.depth ≤ d) :
    (Form.all L).depth ≤ d := by
  induction L with
  | nil => simp [Form.all, Form.depth, Term.depth]
  | cons φ L ih =>
      simp only [Form.all, List.foldr_cons, Form.depth, max_le_iff]
      exact ⟨h φ (List.mem_cons_self ..), ih fun ψ hψ => h ψ (List.mem_cons_of_mem _ hψ)⟩

omit [DecidableEq σ] in
theorem Form.past_all (L : List (Form σ)) (h : ∀ φ ∈ L, φ.past = true) :
    (Form.all L).past = true := by
  induction L with
  | nil => rfl
  | cons φ L ih =>
      simp only [Form.all, List.foldr_cons, Form.past, Bool.and_eq_true]
      exact ⟨h φ (List.mem_cons_self ..), ih fun ψ hψ => h ψ (List.mem_cons_of_mem _ hψ)⟩

omit [DecidableEq σ] in
theorem Form.pnpFree_all (L : List (Form σ)) (h : ∀ φ ∈ L, φ.pnpFree = true) :
    (Form.all L).pnpFree = true := by
  induction L with
  | nil => rfl
  | cons φ L ih =>
      simp only [Form.all, List.foldr_cons, Form.pnpFree, Bool.and_eq_true]
      exact ⟨h φ (List.mem_cons_self ..), ih fun ψ hψ => h ψ (List.mem_cons_of_mem _ hψ)⟩

omit [DecidableEq σ] in
theorem Form.past_any (L : List (Form σ)) (h : ∀ φ ∈ L, φ.past = true) :
    (Form.any L).past = true := by
  induction L with
  | nil => rfl
  | cons φ L ih =>
      simp only [Form.any, List.foldr_cons, Form.past_or, Bool.and_eq_true]
      exact ⟨h φ (List.mem_cons_self ..), ih fun ψ hψ => h ψ (List.mem_cons_of_mem _ hψ)⟩

omit [DecidableEq σ] in
theorem Form.pnpFree_any (L : List (Form σ)) (h : ∀ φ ∈ L, φ.pnpFree = true) :
    (Form.any L).pnpFree = true := by
  induction L with
  | nil => rfl
  | cons φ L ih =>
      simp only [Form.any, List.foldr_cons, Form.or, Form.pnpFree, Bool.and_eq_true]
      exact ⟨h φ (List.mem_cons_self ..), ih fun ψ hψ => h ψ (List.mem_cons_of_mem _ hψ)⟩

variable [Fintype σ]

/-- `⋁_{a ∈ Σ} Q_a`: the current position carries a letter of the string. -/
noncomputable def Form.onStr : Form σ := Form.any ((Finset.univ : Finset σ).toList.map Form.sym)

/-- It holds exactly at the positions of the string: `i - 1` is an index of
`w`, positions being numbered from `1`. -/
theorem Form.sat_onStr (w : List σ) (i : ℕ) :
    (Form.onStr : Form σ).sat w i = decide (i - 1 < w.length) := by
  rw [Bool.eq_iff_iff, decide_eq_true_iff]
  unfold Form.onStr
  rw [Form.sat_any]
  constructor
  · rintro ⟨φ, hφ, hsat⟩
    obtain ⟨a, -, rfl⟩ := List.mem_map.mp hφ
    have ha : w[i - 1]? = some a := by simpa [Form.sat] using hsat
    obtain ⟨h, -⟩ := List.getElem?_eq_some_iff.mp ha
    exact h
  · intro h
    refine ⟨Form.sym w[i - 1], List.mem_map_of_mem ?_, ?_⟩
    · exact Finset.mem_toList.mpr (Finset.mem_univ _)
    · simp [Form.sat]

omit [DecidableEq σ] in
@[simp] theorem Form.depth_onStr : (Form.onStr : Form σ).depth = 0 :=
  Nat.le_zero.mp (Form.depth_any_le _ (by
    rintro φ hφ
    obtain ⟨a, -, rfl⟩ := List.mem_map.mp hφ
    exact le_rfl))

omit [DecidableEq σ] in
@[simp] theorem Form.past_onStr : (Form.onStr : Form σ).past = true :=
  Form.past_any _ (by
    rintro φ hφ
    obtain ⟨a, -, rfl⟩ := List.mem_map.mp hφ
    rfl)

omit [DecidableEq σ] in
@[simp] theorem Form.pnpFree_onStr : (Form.onStr : Form σ).pnpFree = true :=
  Form.pnpFree_any _ (by
    rintro φ hφ
    obtain ⟨a, -, rfl⟩ := List.mem_map.mp hφ
    rfl)

/-- `Form.onStr` is not trivially true: off the end of the string it fails. -/
example : (Form.onStr : Form Bool).sat [true] 2 = false := by
  rw [Form.sat_onStr]
  norm_num

end CRASP
end Transformer
