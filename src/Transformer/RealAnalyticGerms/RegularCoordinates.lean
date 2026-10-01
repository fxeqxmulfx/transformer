/-
# Regularizing coordinates for real analytic germs

The existing real directional preparation supplies regular coordinates for
nonconstant germs. A nonzero central value gives the degree-zero unit case.
This joins both cases without an isolated-zero or Hessian hypothesis.
-/

import Transformer.RealAnalyticGerms.Coordinates
import Transformer.Normalization.WeierstrassCoordinates

open Filter
open scoped Topology

noncomputable section
namespace Transformer.RealAnalyticGerms

open AnalyticPreparation Normalization

/-- Every analytic real function with a nonzero germ is regular in the
last coordinate after an invertible linear change. A nonzero central
value is allowed, with exact order zero. Auxiliary for Rückert induction
in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_regularizing_coordinateEquiv {n : ℕ}
    {F : Base (n + 1) → ℝ} (hF : AnalyticAt ℝ F 0)
    (hFne : ¬ F =ᶠ[𝓝 0] (fun _ => 0)) :
    ∃ (L : Base (n + 1) ≃L[ℝ] Base (n + 1)) (d : ℕ),
      ExactOrderInLastVariable
        (fun x : Ambient n => F (L ((wptAmbientEquiv n).symm x))) d := by
  by_cases hF0 : F 0 = 0
  · let T : EucSpace (n + 1) ≃L[ℝ] Base (n + 1) :=
      PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin (n + 1) => ℝ)
    let E : EucSpace (n + 1) → ℝ := fun x => F (T x)
    have hE : AnalyticAt ℝ E 0 := by
      have hF' : AnalyticAt ℝ F (T 0) := by simpa only [map_zero] using hF
      exact hF'.compContinuousLinearMap (u := T.toContinuousLinearMap) (x := 0)
    have hnonconstant : ¬ ∀ᶠ x in 𝓝 (0 : EucSpace (n + 1)), E x = E 0 := by
      intro hconstant
      apply hFne
      have ht : Tendsto T.symm (𝓝 0) (𝓝 0) := by
        simpa only [map_zero] using T.symm.continuous.continuousAt.tendsto (x := 0)
      filter_upwards [ht.eventually hconstant] with x hx
      change F (T (T.symm x)) = F (T 0) at hx
      simpa only [T.apply_symm_apply, map_zero, hF0] using hx
    obtain ⟨J, d, hd, hEJ, horder⟩ := exists_regular_energy_coordinates E 0 hE hnonconstant
    let L := ((wptAmbientEquiv n).trans J).trans T
    refine ⟨L, d, ?_⟩
    have hfunctions : (fun x : Ambient n => F (L ((wptAmbientEquiv n).symm x))) =
        (fun x : Ambient n => E (0 + J x) - E 0) := by
      funext x
      simp [L, E, hF0]
    rw [hfunctions]
    exact horder
  · refine ⟨ContinuousLinearEquiv.refl ℝ (Base (n + 1)), 0, ?_⟩
    constructor
    · intro k hk
      omega
    · simpa only [iteratedDeriv_zero, lastSlice, map_zero,
        ContinuousLinearEquiv.refl_apply, ← ambient_zero_eq] using hF0

/-- A singular nonzero quadratic germ supplies all regular-coordinate
hypotheses simultaneously. Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
example : ∃ (L : Base 2 ≃L[ℝ] Base 2) (d : ℕ),
    ExactOrderInLastVariable
      (fun x : Ambient 1 => (L ((wptAmbientEquiv 1).symm x) 0) ^ 2) d := by
  let F : Base 2 → ℝ := fun x => (x 0) ^ 2
  have hF : AnalyticAt ℝ F 0 :=
    ((ContinuousLinearMap.proj (0 : Fin 2) : Base 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2
  have hFne : ¬ F =ᶠ[𝓝 0] (fun _ => 0) := by
    intro hzero
    let line : ℝ →L[ℝ] Base 2 := ContinuousLinearMap.toSpanSingleton ℝ (fun _ => 1)
    have hline : Tendsto line (𝓝 0) (𝓝 0) := by
      simpa only [map_zero] using line.continuous.continuousAt.tendsto (x := 0)
    have hs : (fun t : ℝ => t ^ 2) =ᶠ[𝓝 0] (fun _ => 0) := by
      simpa [Function.comp_def, F, line] using hzero.comp_tendsto hline
    obtain ⟨r, hr, hb⟩ := Metric.eventually_nhds_iff.mp hs
    have hz : (r / 2) ^ 2 = 0 := hb (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hr)] using half_lt_self hr)
    exact (sq_pos_of_pos (half_pos hr)).ne' hz
  exact exists_regularizing_coordinateEquiv hF hFne

end Transformer.RealAnalyticGerms
