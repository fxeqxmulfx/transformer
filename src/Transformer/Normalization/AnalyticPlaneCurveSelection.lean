/-
# Curve selection in arbitrary nonconstant analytic plane level sets

A finite-order direction supplies invertible preparation coordinates.
Prepared plane curve selection then transports back to the original point,
retaining all finite analytic sign requirements and excluding that point
at every positive curve parameter.
-/

import Transformer.Normalization.PlaneCurveExamples
import Transformer.Normalization.WeierstrassCoordinates

open Filter Set

namespace Transformer.Normalization

open AnalyticPreparation

/-- A nonconstant real analytic level germ in the plane admits analytic
curve selection subject to any finite family of analytic sign conditions.
No preparation coordinates, root simplicity, Hessian condition, or base
curve is assumed: they are constructed from analyticity and accumulation.
Auxiliary geometric theorem for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. This concerns finite sign conditions on a level set;
the general gradient-minimum set also has a quantified fiber condition. -/
theorem analytic_plane_level_curve_selection {l : ℕ} (F : EucSpace 2 → ℝ)
    (z : EucSpace 2) (hF : AnalyticAt ℝ F z)
    (hnonconstant : ¬ ∀ᶠ y in nhds z, F y = F z)
    (C : Fin l → EucSpace 2 → ℝ) (hC : ∀ i, AnalyticAt ℝ (C i) z)
    (requirement : Fin l → AnalyticSignRequirement)
    (hacc : z ∈ closure {y | F y = F z ∧ y ≠ z ∧
      ∀ i, (requirement i).Holds (C i y)}) :
    ∃ (curve : ℝ → EucSpace 2) (epsilon : ℝ), 0 < epsilon ∧
      AnalyticAt ℝ curve 0 ∧ curve 0 = z ∧
        ∀ t ∈ Ioo 0 epsilon, F (curve t) = F z ∧ curve t ≠ z ∧
          ∀ i, (requirement i).Holds (C i (curve t)) := by
  obtain ⟨L, d, _, hG, horder⟩ := exists_regular_energy_coordinates F z hF hnonconstant
  let G : Ambient 1 → ℝ := fun x => F (z + L x) - F z
  let constraints : Fin l → Ambient 1 → ℝ := fun i x => C i (z + L x)
  have hshift : AnalyticAt ℝ (fun x : Ambient 1 => z + L x) 0 :=
    analyticAt_const.fun_add (L.toContinuousLinearMap.analyticAt 0)
  have hc (i : Fin l) : AnalyticAt ℝ (constraints i) 0 :=
    (by simpa only [map_zero, add_zero] using hC i : AnalyticAt ℝ (C i) (z + L 0)).comp
      (f := fun x : Ambient 1 => z + L x) (x := 0) hshift
  let inverse : EucSpace 2 → Ambient 1 := fun y => L.symm (y - z)
  have hinverse : Tendsto inverse (nhds z) (nhds 0) := by
    have hcInv : ContinuousAt inverse z := by fun_prop
    simpa only [inverse, sub_self, map_zero] using hcInv.tendsto
  have hleft (y : EucSpace 2) : z + L (inverse y) = y := by
    dsimp only [inverse]
    rw [ContinuousLinearEquiv.apply_symm_apply]
    exact add_sub_cancel _ _
  have haccG : (0 : Ambient 1) ∈ closure (preparedPlaneSignSet G constraints requirement) := by
    apply mem_closure_iff_frequently.mpr
    apply hinverse.frequently
    apply (mem_closure_iff_frequently.mp hacc).mono
    intro y hy
    change F y = F z ∧ y ≠ z ∧ ∀ i, (requirement i).Holds (C i y) at hy
    refine ⟨?_, ?_, ?_⟩
    · change F (z + L (inverse y)) - F z = 0
      rw [hleft, hy.1, sub_self]
    · intro hzero
      have hz : y = z := by
        have h := hleft y
        simpa only [hzero, map_zero, add_zero] using h.symm
      exact hy.2.1 hz
    · simpa only [constraints, hleft] using hy.2.2
  obtain ⟨gamma, epsilon, he, hg, hg0, hpoints⟩ :=
    prepared_plane_curve_selection G hG horder constraints hc requirement haccG
  let curve : ℝ → EucSpace 2 := fun t => z + L (gamma t)
  have hcurve : AnalyticAt ℝ curve 0 :=
    analyticAt_const.fun_add ((L.toContinuousLinearMap.analyticAt (gamma 0)).comp
      (f := gamma) (x := 0) hg)
  refine ⟨curve, epsilon, he, hcurve, by simp [curve, hg0], ?_⟩
  intro t ht
  obtain ⟨hGt, hne, hCt⟩ := hpoints t ht
  refine ⟨sub_eq_zero.mp hGt, ?_, hCt⟩
  intro hz
  have hL : L (gamma t) = 0 := by
    apply add_left_cancel (a := z)
    simpa only [add_zero] using hz
  exact hne (L.injective (hL.trans L.map_zero.symm))

/-- The degenerate quadratic level set `x₀² = 0`, with positive `x₁`,
accumulates at the origin along a line. The energy germ is nonconstant,
so this satisfies every hypothesis of arbitrary analytic plane level
curve selection without assuming preparation coordinates. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ (curve : ℝ → EucSpace 2) (epsilon : ℝ), 0 < epsilon ∧
    AnalyticAt ℝ curve 0 ∧ curve 0 = 0 ∧ ∀ t ∈ Ioo 0 epsilon,
      (curve t 0) ^ 2 = 0 ∧ curve t ≠ 0 ∧ 0 < curve t 1 := by
  let F : EucSpace 2 → ℝ := fun x => (x 0) ^ 2
  have hF : AnalyticAt ℝ F 0 :=
    ((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2
  have hnonconstant : ¬ ∀ᶠ y in nhds (0 : EucSpace 2), F y = F 0 := by
    intro hzero
    let gamma : ℝ → EucSpace 2 := fun t => t • PiLp.single 2 (0 : Fin 2) 1
    have ht : Tendsto gamma (nhds 0) (nhds 0) := by
      have hc : ContinuousAt gamma 0 := continuousAt_id.smul continuousAt_const
      simpa [gamma] using hc.tendsto
    have htzero : ∀ᶠ t in nhds (0 : ℝ), t ^ 2 = 0 := by
      simpa [F, gamma] using ht.eventually hzero
    obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp htzero
    have hz : (r / 2) ^ 2 = 0 := hball (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hr)] using half_lt_self hr)
    exact (sq_pos_of_pos (half_pos hr)).ne' hz
  have hacc : (0 : EucSpace 2) ∈ closure {y | F y = F 0 ∧ y ≠ 0 ∧
      ∀ i : Fin 1, AnalyticSignRequirement.positive.Holds (y 1)} := by
    have : NeBot (nhdsWithin (0 : ℝ) (Ioi 0)) := nhdsWithin_Ioi_neBot le_rfl
    let gamma : ℝ → EucSpace 2 := fun t => t • PiLp.single 2 (1 : Fin 2) 1
    have ht : Tendsto gamma (nhdsWithin 0 (Ioi 0)) (nhds 0) := by
      have hc : ContinuousAt gamma 0 := continuousAt_id.smul continuousAt_const
      simpa [gamma] using hc.tendsto.mono_left nhdsWithin_le_nhds
    have hpositive : ∀ᶠ t in nhdsWithin (0 : ℝ) (Ioi 0), 0 < t := self_mem_nhdsWithin
    apply mem_closure_iff_frequently.mpr
    apply ht.frequently
    apply hpositive.frequently.mono
    intro t htp
    refine ⟨by simp [F, gamma], ?_, fun i => ?_⟩
    · intro hzero
      have hz : t = 0 := by simpa [gamma] using congrArg (fun y : EucSpace 2 => y 1) hzero
      exact htp.ne' hz
    · simpa [AnalyticSignRequirement.Holds, gamma] using htp
  simpa only [F, PiLp.zero_apply, zero_pow (by omega : (2 : ℕ) ≠ 0),
    AnalyticSignRequirement.Holds, forall_const] using
      analytic_plane_level_curve_selection F 0 hF hnonconstant (l := 1)
        (fun _ y => y 1)
        (fun _ => (EuclideanSpace.proj 1 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0)
        (fun _ => .positive) hacc

end Transformer.Normalization
