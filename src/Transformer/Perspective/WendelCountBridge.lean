/-
# From strict linear sign patterns to spherical hemispheres

For a nonempty spherical sample, a strict sign pattern realized by a nonzero
linear functional is exactly a sign-flipped sample in an open hemisphere.
This connects the deterministic recurrence with the probability reduction in
`WendelSignAverage`.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelCountRecurrence
import Transformer.Perspective.WendelCircuitSigns

namespace Transformer.Perspective

/-- Strict feasibility of a sign pattern is equivalent to the common-open-
hemisphere event after applying its coordinatewise sign flips.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignPattern_iff_openHemisphere (d n : ℕ) (hn : 1 ≤ n)
    (X : SphereTuple d n) (mask : Idx n → Bool) :
    StrictSignPattern d n (fun i => (X i : EucSpace d)) mask ↔
      ∃ w : SSphere d, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((flipSigns d n mask X i : EucSpace d))
          ((w : EucSpace d)) := by
  constructor
  · rintro ⟨w₀, hw₀⟩
    have hwne : w₀ ≠ 0 := by
      intro hz
      have hi := hw₀ ⟨0, hn⟩
      simp [hz] at hi
    have hnorm : 0 < ‖w₀‖ := norm_pos_iff.mpr hwne
    refine ⟨⟨‖w₀‖⁻¹ • w₀, ?_⟩, fun i => ?_⟩
    · rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, Real.norm_eq_abs,
        abs_of_pos hnorm, inv_mul_cancel₀ (ne_of_gt hnorm)]
    · rw [flipSigns_coe]
      show 0 < inner (𝕜 := ℝ)
        (if mask i then -(X i : EucSpace d) else (X i : EucSpace d))
        (‖w₀‖⁻¹ • w₀)
      rw [real_inner_smul_right]
      exact mul_pos (inv_pos.mpr hnorm) (hw₀ i)
  · rintro ⟨w, hw⟩
    refine ⟨(w : EucSpace d), fun i => ?_⟩
    simpa only [flipSigns_coe] using hw i

/-- The finite sum used by the recurrence is the same count of successful
hemisphere signs used by the probabilistic identity.

Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem strictSignCount_eq_hemisphereSignCount (d n : ℕ) (hn : 1 ≤ n)
    (X : SphereTuple d n) :
    strictSignCount d n (fun i => (X i : EucSpace d)) = hemisphereSignCount d n X := by
  classical
  simp only [strictSignCount, hemisphereSignCount]
  simp_rw [strictSignPattern_iff_openHemisphere d n hn X]
  simp

/-- The hemisphere sign count for a nonempty spherical sample satisfies the
one-point recurrence. The unresolved term counts the old sign cones cut by
the new point's orthogonal hyperplane.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem hemisphereSignCount_snoc (d n : ℕ) (hn : 1 ≤ n)
    (X : SphereTuple d (n + 1)) :
    hemisphereSignCount d (n + 1) X =
      hemisphereSignCount d n (Fin.init X) +
        sliceSignCount d n
          (fun i : Idx n => (Fin.init X i : EucSpace d))
          (X (Fin.last n) : EucSpace d) := by
  let Y : SphereTuple d n := Fin.init X
  let x : EucSpace d := X (Fin.last n)
  have hx : x ≠ 0 := by
    have hxnorm : ‖x‖ = 1 := mem_sphere_zero_iff_norm.mp (X (Fin.last n)).2
    intro h
    simp [h] at hxnorm
  have hXeq : Fin.snoc (fun i : Idx n => (Y i : EucSpace d)) x =
      fun i : Idx (n + 1) => (X i : EucSpace d) := by
    simpa [Y, x, Function.comp_def, Fin.snoc_init_self] using
      (Fin.comp_snoc (fun z : SSphere d => (z : EucSpace d))
        (Fin.init X) (X (Fin.last n))).symm
  have hrec := strictSignCount_snoc d n (fun i : Idx n => (Y i : EucSpace d)) x hx
  rw [hXeq, strictSignCount_eq_hemisphereSignCount d (n + 1) (by omega) X,
    strictSignCount_eq_hemisphereSignCount d n hn Y] at hrec
  exact hrec

/-- The nonempty-sample hypothesis is satisfiable at one point. -/
example : 1 ≤ 1 := le_rfl

end Transformer.Perspective
