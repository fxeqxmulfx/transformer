/-
# Exact recent-state profiles in previous-position logic

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
`Y` checks an exact finite profile and separates short prefixes from those
long enough that their recent window excludes BOS.
-/

import Transformer.CRASP.PreviousPredicates

namespace Transformer.CRASP.WindowPredicates

universe u v
variable {α : Type u} {σ : Type v} {k m : ℕ}

/-- The conjunction reading a finite profile of recent source states (F). -/
noncomputable def profile (ψ : α → FormP σ) (P : Fin m → α) : FormP σ :=
  FormP.all (Finset.univ.toList.map fun δ : Fin m => FormP.shift δ.val (ψ (P δ)))

/-- A recent-state profile has the depth of its state predicates (Appendix F). -/
theorem profile_mem (ψ : α → FormP σ) (hψ : ∀ a, ψ a ∈ TLClY σ k) (P : Fin m → α) :
    profile ψ P ∈ TLClY σ k := by
  apply FormP.all_mem_Y
  rintro φ hφ
  obtain ⟨δ, hδ, rfl⟩ := List.mem_map.mp hφ
  exact FormP.shift_mem_Y (hψ (P δ)) δ.val

/-- A positive ordinary position is exactly `n` (Appendix F, `Y` semantics). -/
def atPosition (n : ℕ) : FormP σ :=
  .and (FormP.shift (n - 1) (FormP.truth true)) (.neg (FormP.shift n (FormP.truth true)))

/-- The position test adds no counting depth (Appendix F). -/
theorem atPosition_mem (n k : ℕ) : (atPosition n : FormP σ) ∈ TLClY σ k :=
  FormP.and_mem_Y (FormP.shift_mem_Y (FormP.truth_mem_Y _ _) _)
    (FormP.neg_mem_Y (FormP.shift_mem_Y (FormP.truth_mem_Y _ _) _))

variable [DecidableEq α] [DecidableEq σ]

/-- A profile guard agrees exactly with the preceding finite activation states.
Source: arXiv:2506.16055v3, Appendix F, `Y` window construction. -/
theorem sat_profile (ψ : α → FormP σ) (q : ℕ → α) (w : List σ) (i : ℕ) (hi : 0 < i)
    (hψ : ∀ j ∈ Finset.Icc 1 i, ∀ a, (ψ a).sat w j = decide (q j = a))
    (P : Fin m → α) :
    (profile ψ P).sat w i = true ↔ ∀ δ : Fin m, δ.val < i ∧ q (i - δ.val) = P δ := by
  rw [profile, FormP.sat_all]
  constructor
  · intro h δ
    have hs := h _ (List.mem_map_of_mem (Finset.mem_toList.mpr (Finset.mem_univ δ)))
    rw [FormP.sat_shift _ _ _ _ hi] at hs
    have hh := Bool.and_eq_true_iff.mp hs
    have hd : δ.val < i := of_decide_eq_true hh.1
    refine ⟨hd, ?_⟩
    have hj : i - δ.val ∈ Finset.Icc 1 i := Finset.mem_Icc.mpr ⟨by omega, Nat.sub_le _ _⟩
    rw [hψ _ hj] at hh
    exact of_decide_eq_true hh.2
  · intro h φ hφ
    obtain ⟨δ, hδ, rfl⟩ := List.mem_map.mp hφ
    have hh := h δ
    rw [FormP.sat_shift _ _ _ _ hi,
      hψ _ (Finset.mem_Icc.mpr ⟨by omega, Nat.sub_le _ _⟩), hh.2]
    simp [hh.1]

omit [DecidableEq α] in
/-- The short-prefix guard detects its exact positive index (Appendix F). -/
theorem sat_atPosition (n : ℕ) (hn : 0 < n) (w : List σ) (i : ℕ) (hi : 0 < i) :
    (atPosition n : FormP σ).sat w i = decide (i = n) := by
  rw [atPosition, FormP.sat, FormP.sat, FormP.sat_shift _ _ _ _ hi,
    FormP.sat_shift _ _ _ _ hi]
  simp only [FormP.sat_truth, Bool.and_true]
  rw [Bool.eq_iff_iff]
  simp only [Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true', decide_eq_false_iff_not]
  omega

/-- Profile agreement and positive-index hypotheses have a one-state witness (F). -/
example : 0 < 1 ∧
    (∀ j ∈ Finset.Icc 1 1, ∀ a : Unit,
      (FormP.truth true : FormP Bool).sat [true] j = decide (() = a)) ∧
    (∀ δ : Fin 1, δ.val < 1 ∧ (() : Unit) = ()) := by simp

end Transformer.CRASP.WindowPredicates
