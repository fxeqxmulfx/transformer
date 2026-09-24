/-
# Base cases for Wendel's strict-sign count

An independent family realizes every sign pattern. In zero-dimensional space,
a nonempty family realizes none. These are the boundary cases of the
dimension-reduction recurrence.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelProjectionGeneralPosition

namespace Transformer.Perspective

/-- An independent nonempty family realizes every strict sign pattern.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignCount_eq_pow_of_linearIndependent (d n : ℕ) (hn : 1 ≤ n)
    (v : Idx n → EucSpace d) (hv : LinearIndependent ℝ v) :
    strictSignCount d n v = 2 ^ n := by
  classical
  have hgood (mask : Idx n → Bool) : StrictSignPattern d n v mask := by
    let s : Idx n → ℝˣ := fun i => if mask i then -1 else 1
    have heq : (fun i : Idx n => if mask i then -v i else v i) = s • v := by
      funext i
      cases h : mask i with
      | false => simp [s, h]
      | true => simp [s, h]
    have hflip : LinearIndependent ℝ
        (fun i : Idx n => if mask i then -v i else v i) := by
      rw [heq]
      exact (LinearIndependent.units_smul_iff _ s).2 hv
    obtain ⟨w, hw⟩ := exists_common_hemisphere_of_linearIndependent d _ hflip hn
    exact ⟨(w : EucSpace d), hw⟩
  simp [strictSignCount, hgood, Fintype.card_bool]

/-- A nonempty family in `ℝ⁰` has no strict sign pattern.
Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignCount_zero (n : ℕ) (hn : 1 ≤ n)
    (v : Idx n → EucSpace 0) : strictSignCount 0 n v = 0 := by
  classical
  have hv (i : Idx n) : v i = 0 := Subsingleton.elim _ _
  have hbad (mask : Idx n → Bool) : ¬StrictSignPattern 0 n v mask := by
    rintro ⟨w, hw⟩
    have hi := hw ⟨0, hn⟩
    simp [hv] at hi
  simp [strictSignCount, hbad]

/-- The nonempty-family hypothesis is realizable, and the one-vector
standard basis family is independent. -/
example : LinearIndependent ℝ (fun _ : Idx 1 => (eOne : EucSpace 1)) ∧
    1 ≤ 1 := by
  constructor
  · rw [linearIndependent_unique_iff]
    intro h
    have hh := inner_eOne_eOne
    simp [h] at hh
  · exact le_rfl

end Transformer.Perspective
