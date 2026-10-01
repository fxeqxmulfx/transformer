/-
# Finite Boolean tests in positional logic

arXiv:2506.16055v3, Appendix F, finite-function constructions for position
encodings. Boolean combinations do not add counting depth or operators.
-/

import Transformer.CRASP.Positional

namespace Transformer.CRASP.FormP

universe u
variable {σ : Type u} {k j : ℕ}

/-- Boolean constants in the positional grammar (Appendix F). -/
def truth (b : Bool) : FormP σ := if b then .neg (.lt .one .one) else .lt .one .one

/-- Disjunction in the positional grammar (Appendix F). -/
def or (φ ψ : FormP σ) : FormP σ := .neg (.and (.neg φ) (.neg ψ))

/-- A disjunction of finitely many state predicates (Appendix F). -/
def any : List (FormP σ) → FormP σ
  | [] => truth false
  | φ :: L => φ.or (any L)

/-- A conjunction of finitely many coordinate predicates (Appendix F). -/
def all : List (FormP σ) → FormP σ
  | [] => truth true
  | φ :: L => φ.and (all L)

/-- Constants belong to every depth of the periodic fragment (Appendix F). -/
theorem truth_mem (b : Bool) (k : ℕ) : (truth b : FormP σ) ∈ TLClMod σ k := by
  cases b <;> exact ⟨rfl, Nat.zero_le _⟩

/-- Negation preserves the periodic fragment and depth bound (Appendix F). -/
theorem neg_mem {φ : FormP σ} (hφ : φ ∈ TLClMod σ k) : φ.neg ∈ TLClMod σ k := hφ

/-- Conjunction preserves the periodic fragment and depth bound (Appendix F). -/
theorem and_mem {φ ψ : FormP σ} (hφ : φ ∈ TLClMod σ k) (hψ : ψ ∈ TLClMod σ k) :
    (φ.and ψ) ∈ TLClMod σ k :=
  ⟨by simp [FormP.prevFree, hφ.1, hψ.1], max_le hφ.2 hψ.2⟩

/-- Disjunction preserves the periodic fragment and depth bound (Appendix F). -/
theorem or_mem {φ ψ : FormP σ} (hφ : φ ∈ TLClMod σ k) (hψ : ψ ∈ TLClMod σ k) :
    (φ.or ψ) ∈ TLClMod σ k := neg_mem (and_mem (neg_mem hφ) (neg_mem hψ))

/-- A larger depth bound includes the same formulas (Appendix F). -/
theorem mem_mono {φ : FormP σ} (hφ : φ ∈ TLClMod σ k) (hk : k ≤ j) :
    φ ∈ TLClMod σ j := ⟨hφ.1, hφ.2.trans hk⟩

/-- Finite state disjunctions preserve the fragment and bound (Appendix F). -/
theorem any_mem (L : List (FormP σ)) (hL : ∀ φ ∈ L, φ ∈ TLClMod σ k) :
    any L ∈ TLClMod σ k := by
  induction L with
  | nil => exact truth_mem _ _
  | cons φ L ih =>
      exact or_mem (hL φ (List.mem_cons_self ..))
        (ih fun ψ hψ => hL ψ (List.mem_cons_of_mem _ hψ))

/-- Finite coordinate conjunctions preserve the fragment and bound (Appendix F). -/
theorem all_mem (L : List (FormP σ)) (hL : ∀ φ ∈ L, φ ∈ TLClMod σ k) :
    all L ∈ TLClMod σ k := by
  induction L with
  | nil => exact truth_mem _ _
  | cons φ L ih =>
      exact and_mem (hL φ (List.mem_cons_self ..))
        (ih fun ψ hψ => hL ψ (List.mem_cons_of_mem _ hψ))

/-- Boolean-operation hypotheses have a single-letter witness (Appendix F). -/
example : (FormP.sym true) ∈ TLClMod Bool 0 ∧ 0 ≤ 1 ∧
    (∀ φ ∈ [FormP.sym true], φ ∈ TLClMod Bool 0) := by
  simp [TLClMod, FormP.prevFree, FormP.depth]

variable [DecidableEq σ]

/-- Constants have their specified truth values (Appendix F). -/
@[simp] theorem sat_truth (b : Bool) (w : List σ) (i : ℕ) :
    (truth b : FormP σ).sat w i = b := by
  cases b <;> simp [truth, FormP.sat, TermP.val]

/-- The derived disjunction has Boolean-or semantics (Appendix F). -/
@[simp] theorem sat_or (φ ψ : FormP σ) (w : List σ) (i : ℕ) :
    (φ.or ψ).sat w i = (φ.sat w i || ψ.sat w i) := by
  simp [or, FormP.sat]

/-- A finite disjunction holds exactly when one member holds (Appendix F). -/
theorem sat_any (L : List (FormP σ)) (w : List σ) (i : ℕ) :
    (any L).sat w i = true ↔ ∃ φ ∈ L, φ.sat w i = true := by
  induction L with
  | nil => simp [any]
  | cons φ L ih => simp [any, ih]

/-- A finite conjunction holds exactly when every member holds (Appendix F). -/
theorem sat_all (L : List (FormP σ)) (w : List σ) (i : ℕ) :
    (all L).sat w i = true ↔ ∀ φ ∈ L, φ.sat w i = true := by
  induction L with
  | nil => simp [all]
  | cons φ L ih => simp [all, FormP.sat, ih]

end Transformer.CRASP.FormP
