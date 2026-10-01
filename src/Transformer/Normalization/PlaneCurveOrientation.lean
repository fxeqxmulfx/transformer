/-
# Orienting accumulation in a prepared plane zero set

The isolated central-axis zero forces nonzero projected parameters. One of
the two parameter signs accumulates, so reflection gives positive-side
accumulation suitable for constrained analytic root lifting.
-/

import Transformer.Normalization.ZeroAxisProjection

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- Nonzero points accumulating at a prepared plane zero set's origin
accumulate from at least one of the two oriented base-parameter sides.
Every finite analytic sign requirement is preserved by the orientation.
Auxiliary for real plane curve selection in Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
theorem prepared_plane_positive_accumulation {d l : ℕ} (F : Ambient 1 → ℝ)
    (hF : AnalyticAt ℝ F 0) (horder : ExactOrderInLastVariable F d)
    (C : Fin l → Ambient 1 → ℝ) (requirement : Fin l → AnalyticSignRequirement)
    (hacc : (0 : Ambient 1) ∈ closure {x | F x = 0 ∧ x ≠ 0 ∧
      ∀ i, (requirement i).Holds (C i x)}) :
    ∃ sign : ℝ, (sign = 1 ∨ sign = -1) ∧
      ∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧
        F ((fun _ => sign * x.1), x.2) = 0 ∧
          ∀ i, (requirement i).Holds (C i ((fun _ => sign * x.1), x.2)) := by
  let E := oneParameterCoordinates
  have hE : Tendsto E (nhds 0) (nhds 0) := by
    simpa only [map_zero] using E.continuous.tendsto (0 : Ambient 1)
  have hinv : Tendsto E.symm (nhds 0) (nhds 0) := by
    simpa only [map_zero] using E.symm.continuous.tendsto (0 : ℝ × ℝ)
  have hsource : ∃ᶠ x in nhds (0 : ℝ × ℝ), F (E.symm x) = 0 ∧ E.symm x ≠ 0 ∧
      ∀ i, (requirement i).Holds (C i (E.symm x)) := by
    apply hE.frequently
    apply (mem_closure_iff_frequently.mp hacc).mono
    intro x hx
    change F x = 0 ∧ x ≠ 0 ∧ ∀ i, (requirement i).Holds (C i x) at hx
    simpa only [ContinuousLinearEquiv.symm_apply_apply] using hx
  have haxis := hinv.eventually (analytic_zero_axis_isolated F hF horder)
  have hnonzero : ∃ᶠ x in nhds (0 : ℝ × ℝ), x.1 ≠ 0 ∧ F (E.symm x) = 0 ∧
      ∀ i, (requirement i).Holds (C i (E.symm x)) := by
    apply (hsource.and_eventually haxis).mono
    intro x hx
    refine ⟨?_, hx.1.1, hx.1.2.2⟩
    intro hx0
    have hbase : (E.symm x).1 = 0 := by
      rw [oneParameterCoordinates_symm_apply]
      ext i
      exact hx0
    exact hx.1.2.1 (hx.2 hbase hx.1.1)
  have hsplit : (∃ᶠ x in nhds (0 : ℝ × ℝ), 0 < x.1 ∧ F (E.symm x) = 0 ∧
      ∀ i, (requirement i).Holds (C i (E.symm x))) ∨
      (∃ᶠ x in nhds (0 : ℝ × ℝ), x.1 < 0 ∧ F (E.symm x) = 0 ∧
        ∀ i, (requirement i).Holds (C i (E.symm x))) := by
    apply frequently_or_distrib.mp
    apply hnonzero.mono
    intro x hx
    rcases lt_or_gt_of_ne hx.1 with hn | hp
    · exact Or.inr ⟨hn, hx.2⟩
    · exact Or.inl ⟨hp, hx.2⟩
  rcases hsplit with hp | hn
  · refine ⟨1, Or.inl rfl, ?_⟩
    simpa only [one_mul, E, oneParameterCoordinates_symm_apply] using hp
  · refine ⟨-1, Or.inr rfl, ?_⟩
    let flip : ℝ × ℝ → ℝ × ℝ := fun x => (-x.1, x.2)
    have hflip : Tendsto flip (nhds 0) (nhds 0) := by
      have hc : ContinuousAt flip 0 := by fun_prop
      simpa [flip, Prod.mk_zero_zero] using hc.tendsto
    apply hflip.frequently
    apply hn.mono
    intro x hx
    simpa only [flip, neg_one_mul, neg_neg, E, oneParameterCoordinates_symm_apply] using
      (show 0 < -x.1 ∧ F (E.symm x) = 0 ∧
        ∀ i, (requirement i).Holds (C i (E.symm x)) from ⟨neg_pos.mpr hx.1, hx.2⟩)

end Transformer.Normalization
