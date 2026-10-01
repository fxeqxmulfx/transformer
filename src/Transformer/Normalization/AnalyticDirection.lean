/-
# A finite-order direction for a nonconstant real analytic germ

Every nonconstant analytic energy germ has a nonconstant restriction to
some line through its base point. Its finite order gives a regularizing
direction for analytic preparation; it does not by itself supply the
uniform gradient inequality in Appendix D.1 of arXiv:2510.22026v2.

The Taylor-diagonal argument is adapted to real scalars and an arbitrary
base point from Bochao Kong's LocalComplexGeometry/Analytic/Regularization.lean,
revision 029697242294aea7989bf5d391199696a43490df:
https://github.com/BochaoKong/nullstellensatz
The original code is licensed under Apache-2.0; the license is preserved
in third_party/LocalComplexGeometry/LICENSE. No package dependency is added.
-/

import Transformer.Normalization.AnalyticCurveGradient
import Mathlib.Analysis.Analytic.Uniqueness

open Filter Set

namespace Transformer.Normalization

/-- Every nonconstant analytic energy germ has a nonzero direction on
which the shifted energy has a positive finite order and a nonvanishing
analytic unit. This establishes the directional regularity needed before
analytic preparation in the general case of Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. It asserts no uniform bound over directions. -/
theorem exists_finite_order_analytic_direction {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (hE : AnalyticAt ℝ E z)
    (hconstant : ¬ ∀ᶠ y in nhds z, E y = E z) :
    ∃ (v : EucSpace N) (m : ℕ) (u : ℝ → ℝ), v ≠ 0 ∧ 0 < m ∧
      AnalyticAt ℝ u 0 ∧ u 0 ≠ 0 ∧ ∀ᶠ t in nhds (0 : ℝ),
        E (z + t • v) - E z = t ^ m * u t := by
  let F : EucSpace N → ℝ := fun y => E (z + y) - E z
  have hshift : AnalyticAt ℝ (fun y : EucSpace N => z + y) 0 :=
    analyticAt_const.add analyticAt_id
  have hF : AnalyticAt ℝ F 0 := by
    have he : AnalyticAt ℝ E (z + 0) := by simpa only [add_zero] using hE
    exact (he.comp (f := fun y : EucSpace N => z + y) hshift).sub analyticAt_const
  have hF0 : F 0 = 0 := by simp [F]
  have hFne : ¬ ∀ᶠ y in nhds (0 : EucSpace N), F y = 0 := by
    intro hzero
    apply hconstant
    have ht : Tendsto (fun y : EucSpace N => y - z) (nhds z) (nhds 0) := by
      have hc : ContinuousAt (fun y : EucSpace N => y - z) z :=
        continuousAt_id.sub continuousAt_const
      simpa only [sub_self] using hc.tendsto
    filter_upwards [ht.eventually hzero] with y hy
    have hshift : z + (y - z) = y := by abel
    simpa only [F, hshift, sub_eq_zero] using hy
  obtain ⟨p, hp⟩ := hF
  have hdiagonal : ∃ (d : ℕ) (v : EucSpace N), p d (fun _ => v) ≠ 0 := by
    by_contra hn
    push Not at hn
    apply hFne
    filter_upwards [hp.eventually_hasSum_sub] with y hy
    have hz : HasSum (fun d : ℕ => p d (fun _ => y - 0)) 0 := by
      simpa only [hn] using (hasSum_zero : HasSum (fun _ : ℕ => (0 : ℝ)) 0)
    exact hy.unique hz
  obtain ⟨d, v, hdv⟩ := hdiagonal
  have hd : 0 < d := Nat.pos_of_ne_zero (by
    intro hzero
    subst d
    exact hdv ((hp.coeff_zero (fun _ : Fin 0 => v)).trans hF0))
  have hv : v ≠ 0 := by
    intro hzero
    subst v
    exact hdv ((p d).map_coord_zero (⟨0, hd⟩ : Fin d) rfl)
  let line : ℝ →L[ℝ] EucSpace N := ContinuousLinearMap.toSpanSingleton ℝ v
  have hp0 : HasFPowerSeriesAt F p (line 0) := by simpa only [map_zero] using hp
  have hline := hp0.compContinuousLinearMap (u := line) (x := 0)
  have hc : (p.compContinuousLinearMap line) d (fun _ => (1 : ℝ)) ≠ 0 := by
    change p d (line ∘ (fun _ : Fin d => (1 : ℝ))) ≠ 0
    simpa [line, Function.comp_def] using hdv
  have hlineNe : ¬ ∀ᶠ t in nhds (0 : ℝ), (F ∘ line) t = 0 := by
    intro hzero
    have hpzero := hline.eq_zero_of_eventually hzero
    apply hc
    rw [hpzero]
    simp
  obtain ⟨m, u, hu, hu0, hfactor⟩ :=
    hline.analyticAt.exists_eventuallyEq_pow_smul_nonzero_iff.mpr hlineNe
  have hm : 0 < m := Nat.pos_of_ne_zero (by
    intro hzero
    have heq := hfactor.self_of_nhds
    simp only [hzero, Function.comp_def, map_zero, hF0, pow_zero, one_smul] at heq
    exact hu0 heq.symm)
  refine ⟨v, m, u, hv, hm, hu, hu0, ?_⟩
  simpa only [Function.comp_def, line, ContinuousLinearMap.toSpanSingleton_apply,
    F, sub_zero, smul_eq_mul] using hfactor

/-- The two-variable energy `x²+y⁴` is analytic and nonconstant at its
degenerate critical origin, so all finite-order direction hypotheses are
jointly satisfiable. Auxiliary example for Appendix D.1 of
arXiv:2510.22026v2. -/
example : ∃ (v : EucSpace 2) (m : ℕ) (u : ℝ → ℝ), v ≠ 0 ∧ 0 < m ∧
    AnalyticAt ℝ u 0 ∧ u 0 ≠ 0 ∧ ∀ᶠ t in nhds (0 : ℝ),
      (t * v 0) ^ 2 + (t * v 1) ^ 4 = t ^ m * u t := by
  let E : EucSpace 2 → ℝ := fun y => (y 0) ^ 2 + (y 1) ^ 4
  have hE : AnalyticAt ℝ E 0 :=
    (((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2).add
      (((EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 4)
  have hnonconstant : ¬ ∀ᶠ y in nhds (0 : EucSpace 2), E y = E 0 := by
    intro hzero
    let gamma : ℝ → EucSpace 2 := fun t =>
      t • (PiLp.single 2 (0 : Fin 2) 1 : EucSpace 2)
    have ht : Tendsto gamma (nhds 0) (nhds 0) := by
      have hc : ContinuousAt gamma 0 := continuousAt_id.smul continuousAt_const
      simpa [gamma] using hc.tendsto
    have htzero : ∀ᶠ t in nhds (0 : ℝ), t ^ 2 = 0 := by
      simpa [E, gamma] using ht.eventually hzero
    obtain ⟨delta, hd, hball⟩ := Metric.eventually_nhds_iff.mp htzero
    have hz : (delta / 2) ^ 2 = 0 := hball (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hd)] using half_lt_self hd)
    exact (sq_pos_of_pos (half_pos hd)).ne' hz
  simpa [E, PiLp.smul_apply, PiLp.add_apply] using
    exists_finite_order_analytic_direction E 0 hE hnonconstant

end Transformer.Normalization
