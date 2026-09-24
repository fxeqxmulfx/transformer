/-
# The combinatorial recurrence for strict sign patterns

Adding one nonzero vector contributes one sign pattern for every old feasible
pattern, plus one extra for each old sign cone that meets the new vector's
orthogonal hyperplane. This is the counting step behind Wendel's recurrence.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelConeSlice

namespace Transformer.Perspective

/-- The number of strict sign patterns realizable by a linear functional.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def strictSignCount (d n : ℕ) (v : Idx n → EucSpace d) : ℕ := by
  classical
  exact ∑ mask : Idx n → Bool, if StrictSignPattern d n v mask then 1 else 0

/-- The number of old sign cones meeting the orthogonal hyperplane of `x`.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
noncomputable def sliceSignCount (d n : ℕ) (v : Idx n → EucSpace d)
    (x : EucSpace d) : ℕ := by
  classical
  exact ∑ mask : Idx n → Bool,
    if ∃ w : EucSpace d,
      (∀ i : Idx n, 0 < inner (𝕜 := ℝ) (if mask i then -v i else v i) w) ∧
        inner (𝕜 := ℝ) x w = 0 then 1 else 0

/-- The sign-pattern count satisfies the one-vector recurrence, with the
hyperplane-slice count as the correction term.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem strictSignCount_snoc (d n : ℕ) (v : Idx n → EucSpace d)
    (x : EucSpace d) (hx : x ≠ 0) :
    strictSignCount d (n + 1) (Fin.snoc v x) =
      strictSignCount d n v + sliceSignCount d n v x := by
  classical
  let e : (Bool × (Idx n → Bool)) ≃ (Idx (n + 1) → Bool) :=
    Fin.snocEquiv (fun _ => Bool)
  have hreindex : strictSignCount d (n + 1) (Fin.snoc v x) =
      ∑ mask : Idx n → Bool,
        ((if StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask true)
            then 1 else 0) +
          (if StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask false)
            then 1 else 0)) := by
    unfold strictSignCount
    calc
      (∑ mask : Idx (n + 1) → Bool,
          if StrictSignPattern d (n + 1) (Fin.snoc v x) mask then 1 else 0) =
        ∑ pair : Bool × (Idx n → Bool),
          if StrictSignPattern d (n + 1) (Fin.snoc v x) (e pair) then 1 else 0 := by
            exact (Equiv.sum_comp e _).symm
      _ = _ := by
        rw [Fintype.sum_prod_type_right]
        apply Finset.sum_congr rfl
        intro mask _
        rw [Fintype.sum_bool]
        rfl
  rw [hreindex]
  change (∑ mask : Idx n → Bool,
      ((if StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask true)
          then 1 else 0) +
        (if StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask false)
          then 1 else 0))) =
    (∑ mask : Idx n → Bool, if StrictSignPattern d n v mask then 1 else 0) +
      ∑ mask : Idx n → Bool,
        if ∃ w : EucSpace d,
          (∀ i : Idx n, 0 < inner (𝕜 := ℝ) (if mask i then -v i else v i) w) ∧
            inner (𝕜 := ℝ) x w = 0 then 1 else 0
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro mask _
  have hsome := strictSignPattern_snoc_exists_iff d n v mask x hx
  have hboth := strictSignPattern_snoc_both_iff_hyperplane d n v mask x hx
  rw [← hsome, ← hboth]
  by_cases hp : StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask true) <;>
    by_cases hq : StrictSignPattern d (n + 1) (Fin.snoc v x) (Fin.snoc mask false) <;>
      simp [hp, hq]

/-- The nonzero-vector hypothesis of the recurrence holds for a point of the
one-dimensional unit sphere. -/
example : (eOne : EucSpace 1) ≠ 0 := by
  intro h
  have hh := inner_eOne_eOne
  simp [h] at hh

end Transformer.Perspective
