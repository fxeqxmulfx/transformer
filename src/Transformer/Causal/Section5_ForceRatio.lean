module

public import Transformer.Causal.SectionB_FarLogSlope

/-!
# Preserving the original force inequality at half the peak

ArXiv:2411.04990v2, §5, `thm:fixed_centers`.
Write the physical neighborhood radius as `u` and the fixed
part of the original far radius as `d`. The original force
comparison is `N h(d-2u)<h(u)`.
On the interval from half the actual interaction peak to the
original radius, its ratio `h(u)/h(d-2u)` strictly decreases
provided the far radius stays beyond twice the peak.
The derivative is the original cross product of `h` and `g`,
whose strict sign has been proved from the logarithmic slopes.

Consequently the original force inequality is preserved when
such a radius is reduced to half the peak. The mass `N` is
unchanged. These statements use the actual original force;
the endpoint comparison introduces no replacement potential.
Both denominator arguments stay strictly inside the first
semicircle, where the original force is strictly positive.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

/-- The original force ratio as a function of the physical neighborhood radius.
Source: arXiv:2411.04990v2, §5, the original `h` condition in `thm:fixed_centers`, with `d=(c-1) beta^(-1/2)`. -/
noncomputable def frozenForceWindowRatio (β d u : ℝ) : ℝ := h_pot β u / h_pot β (d - 2 * u)

/-- The original force ratio has the derivative obtained from its actual force and curvature.
Source: arXiv:2411.04990v2, §5, differentiating the original force comparison in `thm:fixed_centers`; the moving far argument gives the factor two. -/
theorem hasDerivAt_frozenForceWindowRatio (β d u : ℝ) (hden : h_pot β (d - 2 * u) ≠ 0) :
    HasDerivAt (frozenForceWindowRatio β d)
      ((g_pot β u * h_pot β (d - 2 * u) + 2 * h_pot β u * g_pot β (d - 2 * u)) /
        h_pot β (d - 2 * u) ^ 2) u := by
  have hi : HasDerivAt (fun x : ℝ => d - 2 * x) (-2) u := by
    convert (hasDerivAt_const u d).fun_sub ((hasDerivAt_id u).const_mul 2) using 1 <;> simp
  have hd := (hasDerivAt_h_pot β (d - 2 * u)).comp u hi
  apply ((hasDerivAt_h_pot β u).fun_div hd hden).congr_deriv
  dsimp only [Function.comp_def]
  ring

/-- The far argument remains in the proved far-slope interval throughout the original half-peak window.
Source: arXiv:2411.04990v2, §5, the actual far radius in the force condition of `thm:fixed_centers`. -/
theorem frozen_force_window_far_mem (β d e u : ℝ) (hβ : 0 < β)
    (hfar : 2 * interactionPeak β ≤ d - 2 * e) (hd : d < Real.pi)
    (hu : u ∈ Icc (interactionPeak β / 2) e) : d - 2 * u ∈ Ico (2 * interactionPeak β) Real.pi := by
  have ht := (interactionPeak_properties β hβ).1
  exact ⟨by linarith [hu.2], by linarith [hu.1]⟩

/-- The actual force ratio is strictly positive throughout an admissible original half-peak window.
Source: arXiv:2411.04990v2, Appendix B and §5, positivity of the original force in the original parameter comparison. -/
theorem frozenForceWindowRatio_pos (β d e u : ℝ) (hβ : 0 < β)
    (hfar : 2 * interactionPeak β ≤ d - 2 * e) (hd : d < Real.pi)
    (hu : u ∈ Icc (interactionPeak β / 2) e) : 0 < frozenForceWindowRatio β d u := by
  have ht := interactionPeak_properties β hβ
  have hv := frozen_force_window_far_mem β d e u hβ hfar hd hu
  have hue : u < Real.pi := by linarith only [hu.2, hfar, hd, ht.1, Real.pi_pos]
  have hpu : 0 < h_pot β u := mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi
    (by linarith [hu.1, ht.1]) hue)
  have hpv : 0 < h_pot β (d - 2 * u) := mul_pos (Real.exp_pos _)
    (Real.sin_pos_of_pos_of_lt_pi (by linarith [hv.1, ht.1]) hv.2)
  exact div_pos hpu hpv

/-- The actual original force ratio strictly decreases from half the peak to any admissible original radius below the peak.
Source: arXiv:2411.04990v2, §5, the force-window comparison for `thm:fixed_centers`; the far-radius premise will be derived from its original hypotheses. -/
theorem frozenForceWindowRatio_strictAntiOn (β d e : ℝ) (hβ : 0 < β)
    (he : e ≤ interactionPeak β) (hfar : 2 * interactionPeak β ≤ d - 2 * e) (hd : d < Real.pi) :
    StrictAntiOn (frozenForceWindowRatio β d) (Icc (interactionPeak β / 2) e) := by
  have ht := interactionPeak_properties β hβ
  have hv : ∀ u ∈ Icc (interactionPeak β / 2) e, d - 2 * u ∈ Ico (2 * interactionPeak β) Real.pi :=
    fun u hu => frozen_force_window_far_mem β d e u hβ hfar hd hu
  have hp : ∀ u ∈ Icc (interactionPeak β / 2) e, 0 < h_pot β (d - 2 * u) := by
    intro u hu
    have hx := hv u hu
    exact mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi (by linarith [hx.1, ht.1]) hx.2)
  apply strictAntiOn_of_deriv_neg (convex_Icc _ _)
    (fun u hu => (hasDerivAt_frozenForceWindowRatio β d u (hp u hu).ne').continuousAt.continuousWithinAt)
  intro u hu
  rw [interior_Icc] at hu
  rw [(hasDerivAt_frozenForceWindowRatio β d u (hp u ⟨hu.1.le, hu.2.le⟩).ne').deriv]
  exact div_neg_of_neg_of_pos
    (g_pot_force_window_cross_neg β hβ u (d - 2 * u) ⟨hu.1.le, hu.2.le.trans he⟩
      (hv u ⟨hu.1.le, hu.2.le⟩)) (sq_pos_of_pos (hp u ⟨hu.1.le, hu.2.le⟩))

/-- Decreasing the original radius within the admissible half-peak window preserves the original force inequality with its unchanged mass.
Source: arXiv:2411.04990v2, §5, the original `h` parameter condition in `thm:fixed_centers`, compared at any smaller physical radius in this window. -/
theorem frozen_force_window_radius_mono (N β d e u : ℝ) (hβ : 0 < β)
    (he : e ≤ interactionPeak β) (hu : u ∈ Icc (interactionPeak β / 2) e)
    (hfar : 2 * interactionPeak β ≤ d - 2 * e) (hd : d < Real.pi)
    (hh : N * h_pot β (d - 2 * e) < h_pot β e) :
    N * h_pot β (d - 2 * u) < h_pot β u := by
  have ht := interactionPeak_properties β hβ
  have hv := frozen_force_window_far_mem β d e e hβ hfar hd ⟨hu.1.trans hu.2, le_rfl⟩
  have hve := frozen_force_window_far_mem β d e u hβ hfar hd hu
  have hp : 0 < h_pot β (d - 2 * e) := mul_pos (Real.exp_pos _)
    (Real.sin_pos_of_pos_of_lt_pi (by linarith [hv.1, ht.1]) hv.2)
  have hp0 : 0 < h_pot β (d - 2 * u) := mul_pos (Real.exp_pos _)
    (Real.sin_pos_of_pos_of_lt_pi (by linarith [hve.1, ht.1]) hve.2)
  have hR := (frozenForceWindowRatio_strictAntiOn β d e hβ he hfar hd).antitoneOn hu
    (show e ∈ Icc (interactionPeak β / 2) e from ⟨hu.1.trans hu.2, le_rfl⟩) hu.2
  have hstart : N < frozenForceWindowRatio β d e := (lt_div_iff₀ hp).mpr hh
  exact (lt_div_iff₀ hp0).mp (hstart.trans_le hR)

/-- Reducing an admissible original radius to half the actual peak preserves the original force inequality with its unchanged mass.
Source: arXiv:2411.04990v2, §5, the original `h` parameter condition in `thm:fixed_centers`, compared at the smaller physical radius. -/
theorem frozen_force_window_half_peak (N β d e : ℝ) (hβ : 0 < β)
    (he : e ∈ Icc (interactionPeak β / 2) (interactionPeak β))
    (hfar : 2 * interactionPeak β ≤ d - 2 * e) (hd : d < Real.pi)
    (hh : N * h_pot β (d - 2 * e) < h_pot β e) :
    N * h_pot β (d - interactionPeak β) < h_pot β (interactionPeak β / 2) := by
  have hbound := frozen_force_window_radius_mono N β d e (interactionPeak β / 2)
    hβ he.2 ⟨le_rfl, he.1⟩ hfar hd hh
  simpa only [show 2 * (interactionPeak β / 2) = interactionPeak β by ring] using hbound

/-- A genuine original temperature has a nonempty full half-peak window whose endpoint force inequality holds at positive mass one.
Source: arXiv:2411.04990v2, Appendix B and §5, the original force ratio in `thm:fixed_centers`. -/
example : (0 : ℝ) < 2 ∧
    interactionPeak 2 ∈ Icc (interactionPeak 2 / 2) (interactionPeak 2) ∧
    2 * interactionPeak 2 ≤ 4 * interactionPeak 2 - 2 * interactionPeak 2 ∧
    4 * interactionPeak 2 < Real.pi ∧
    1 * h_pot 2 (4 * interactionPeak 2 - 2 * interactionPeak 2) < h_pot 2 (interactionPeak 2) := by
  have hβ : (0 : ℝ) < 2 := by norm_num
  have ht := interactionPeak_properties 2 hβ
  have hfront := double_interactionPeak_lt_pi_div_two 2 le_rfl
  have hanti := (h_pot_monotone_peak 2 hβ).2
    (show interactionPeak 2 ∈ Icc (interactionPeak 2) Real.pi from ⟨le_rfl, by linarith [ht.2.1, Real.pi_pos]⟩)
    (show 2 * interactionPeak 2 ∈ Icc (interactionPeak 2) Real.pi from ⟨by linarith [ht.1], by linarith [ht.2.1]⟩)
    (show interactionPeak 2 < 2 * interactionPeak 2 by linarith [ht.1])
  refine ⟨hβ, ⟨by linarith [ht.1], le_rfl⟩, by linarith, by linarith, ?_⟩
  simpa only [show 4 * interactionPeak 2 - 2 * interactionPeak 2 = 2 * interactionPeak 2 by ring, one_mul] using hanti

/-- The interior radius at three quarters of the actual peak jointly realizes the moving-radius premises and the nonzero denominator of the derivative.
Source: arXiv:2411.04990v2, Appendix B and §5, a genuine original force window with unchanged temperature and far-radius rule. -/
example : (0 : ℝ) < 2 ∧ interactionPeak 2 ≤ interactionPeak 2 ∧
    3 * interactionPeak 2 / 4 ∈ Icc (interactionPeak 2 / 2) (interactionPeak 2) ∧
    2 * interactionPeak 2 ≤ 4 * interactionPeak 2 - 2 * interactionPeak 2 ∧
    4 * interactionPeak 2 < Real.pi ∧
    h_pot 2 (4 * interactionPeak 2 - 2 * (3 * interactionPeak 2 / 4)) ≠ 0 := by
  have hβ : (0 : ℝ) < 2 := by norm_num
  have ht := interactionPeak_properties 2 hβ
  have hf := double_interactionPeak_lt_pi_div_two 2 le_rfl
  have hv : 4 * interactionPeak 2 - 2 * (3 * interactionPeak 2 / 4) ∈ Ioo (0 : ℝ) Real.pi :=
    ⟨by linarith [ht.1], by linarith [ht.1]⟩
  have hp : 0 < h_pot 2 (4 * interactionPeak 2 - 2 * (3 * interactionPeak 2 / 4)) :=
    mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi hv.1 hv.2)
  exact ⟨hβ, le_rfl, ⟨by linarith [ht.1], by linarith [ht.1]⟩, by linarith, by linarith, hp.ne'⟩

end Transformer.Causal
