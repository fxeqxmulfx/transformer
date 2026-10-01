/-
# Uniform estimates near the image of a compact fiber

Local image-gradient bounds on a compact fiber give a common power bound
for all nearby images of its compact ambient set. This supplies the compact
step of a parametrization route to Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.GradientPullback
import Mathlib.Data.Finset.Lattice.Fold

open Filter Set

namespace Transformer.Normalization

/-- Local gradient bounds at all points over `z` uniformize for nearby
images of a compact source set. Points away from the fiber are excluded
using compactness of the remaining image. Auxiliary compactness argument
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem uniform_gradient_inequality_at_compact_fiber {M N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : ContinuousAt E z)
    (phi : EucSpace M → EucSpace N) (K : Set (EucSpace M))
    (hK : IsCompact K) (hphi : ContinuousOn phi K)
    (hlocal : ∀ w ∈ K, phi w = z → ∃ alpha k : ℝ, ∃ V : Set (EucSpace M),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ w ∈ V ∧
        ∀ y ∈ V, |E (phi y) - E z| ^ alpha ≤ k * ‖gradient E (phi y)‖) :
    ∃ alpha k : ℝ, ∃ W : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen W ∧ z ∈ W ∧
        ∀ y ∈ K, phi y ∈ W → |E (phi y) - E z| ^ alpha ≤ k * ‖gradient E (phi y)‖ := by
  classical
  let T : Set (EucSpace M) := {w | w ∈ K ∧ phi w = z}
  have hT : IsCompact T := hK.of_isClosed_subset
    (hK.isClosed.isClosed_eq hphi continuousOn_const) (fun _ hw => hw.1)
  by_cases hT0 : T.Nonempty
  · have hcover (w : T) := hlocal w w.property.1 w.property.2
    choose alpha k V ha ha1 hk hV hwV hineq using hcover
    obtain ⟨s, hs⟩ := hT.elim_finite_subcover V hV (fun w hw =>
      Set.mem_iUnion.mpr ⟨⟨w, hw⟩, hwV ⟨w, hw⟩⟩)
    have hs0 : s.Nonempty := by
      obtain ⟨w, hw⟩ := hT0
      obtain ⟨v, hv, _⟩ := Set.mem_iUnion₂.mp (hs hw)
      exact ⟨v, hv⟩
    let A := s.sup' hs0 alpha
    let B := s.sup' hs0 k
    let U : Set (EucSpace M) := ⋃ w ∈ s, V w
    have hU : IsOpen U := isOpen_iUnion (fun w => isOpen_iUnion (fun _ => hV w))
    have hA : 0 < A := (ha hs0.choose).trans_le (Finset.le_sup' alpha hs0.choose_spec)
    have hA1 : A < 1 := (Finset.sup'_lt_iff hs0).mpr (fun w _ => ha1 w)
    have hB : 0 < B := (hk hs0.choose).trans_le (Finset.le_sup' k hs0.choose_spec)
    have hKU : IsCompact (K \ U) := by
      rw [Set.sdiff_eq]
      exact hK.inter_right hU.isClosed_compl
    have himage : IsCompact (phi '' (K \ U)) :=
      hKU.image_of_continuousOn (hphi.mono Set.sdiff_subset)
    have hz : z ∈ (phi '' (K \ U))ᶜ := by
      rintro ⟨w, hw, hwz⟩
      exact hw.2 (hs ⟨hw.1, hwz⟩)
    obtain ⟨r, hr, hsmall⟩ := Metric.continuousAt_iff.mp hE 1 zero_lt_one
    refine ⟨A, B, (phi '' (K \ U))ᶜ ∩ Metric.ball z r, hA, hA1, hB,
      himage.isClosed.isOpen_compl.inter Metric.isOpen_ball,
      ⟨hz, Metric.mem_ball_self hr⟩, ?_⟩
    intro y hyK hyW
    have hyU : y ∈ U := by
      by_contra hyU
      exact hyW.1 (Set.mem_image_of_mem phi ⟨hyK, hyU⟩)
    obtain ⟨w, hw, hyV⟩ := Set.mem_iUnion₂.mp hyU
    have hgap : |E (phi y) - E z| ≤ 1 := by
      simpa only [Real.dist_eq] using (hsmall hyW.2).le
    by_cases hzero : E (phi y) - E z = 0
    · rw [hzero, abs_zero, Real.zero_rpow hA.ne']
      exact mul_nonneg hB.le (norm_nonneg _)
    · calc
        |E (phi y) - E z| ^ A ≤ |E (phi y) - E z| ^ alpha w :=
          Real.rpow_le_rpow_of_exponent_ge (abs_pos.mpr hzero) hgap
            (Finset.le_sup' alpha hw)
        _ ≤ k w * ‖gradient E (phi y)‖ := hineq w y hyV
        _ ≤ B * ‖gradient E (phi y)‖ :=
          mul_le_mul_of_nonneg_right (Finset.le_sup' k hw) (norm_nonneg _)
  · have himage := hK.image_of_continuousOn hphi
    have hz : z ∈ (phi '' K)ᶜ := by
      rintro ⟨w, hw, hwz⟩
      exact hT0 ⟨w, hw, hwz⟩
    refine ⟨1 / 2, 1, (phi '' K)ᶜ, by norm_num, by norm_num,
      zero_lt_one, himage.isClosed.isOpen_compl, hz, ?_⟩
    intro y hy hyW
    exact False.elim (hyW (Set.mem_image_of_mem phi hy))

/-- A singleton source, identity map, and constant zero energy satisfy all
compact-fiber hypotheses; Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ alpha k : ℝ, ∃ W : Set (EucSpace 1),
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen W ∧ (0 : EucSpace 1) ∈ W ∧
      ∀ y ∈ ({0} : Set (EucSpace 1)), y ∈ W → |(0 : ℝ)| ^ alpha ≤ k *
        ‖gradient (fun _ : EucSpace 1 => (0 : ℝ)) y‖ := by
  simpa only [id_eq, sub_self] using uniform_gradient_inequality_at_compact_fiber
    (fun _ : EucSpace 1 => (0 : ℝ)) 0 continuousAt_const id {0}
      isCompact_singleton continuousOn_id (fun w _ _ =>
        ⟨1 / 2, 1, Set.univ, by norm_num, by norm_num, zero_lt_one,
          isOpen_univ, Set.mem_univ _, fun y _ => by simp⟩)

end Transformer.Normalization
