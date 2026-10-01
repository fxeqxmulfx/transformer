/-
# Preparation of an arbitrary nonconstant real analytic energy germ

A finite-order direction is extended to invertible coordinates. Real
Weierstrass preparation then gives a monic polynomial in the distinguished
variable with analytic parameter coefficients. No Hessian or isolated
critical-point hypothesis is needed.
-/

import Transformer.Normalization.AnalyticDirection
import Transformer.Normalization.DirectionalCoordinates
import Transformer.AnalyticPreparation

open Filter

namespace Transformer.Normalization

open AnalyticPreparation

/-- Every nonconstant real analytic energy germ has invertible coordinates
in which its distinguished-variable order is positive and finite. Auxiliary
for the general preparation step in Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem exists_regular_energy_coordinates {n : ℕ}
    (E : EucSpace (n + 1) → ℝ) (z : EucSpace (n + 1))
    (hE : AnalyticAt ℝ E z) (hnonconstant : ¬ ∀ᶠ y in nhds z, E y = E z) :
    ∃ (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (d : ℕ), 0 < d ∧
      AnalyticAt ℝ (fun x : Ambient n => E (z + L x) - E z) 0 ∧
      ExactOrderInLastVariable (fun x => E (z + L x) - E z) d := by
  obtain ⟨v, d, unit, hv, hd, hunit, hunit0, hfactor⟩ :=
    exists_finite_order_analytic_direction E z hE hnonconstant
  obtain ⟨j, hvj⟩ : ∃ j : Fin (n + 1), v j ≠ 0 := by
    by_contra h
    push Not at h
    exact hv (PiLp.ext (fun j => by simpa using h j))
  let L : Ambient n ≃L[ℝ] EucSpace (n + 1) := directionCoordinates v j hvj
  let F : Ambient n → ℝ := fun x => E (z + L x) - E z
  have hshift : AnalyticAt ℝ (fun x : Ambient n => z + L x) 0 :=
    analyticAt_const.add (L.toContinuousLinearMap.analyticAt 0)
  have hEshift : AnalyticAt ℝ E (z + L 0) := by simpa using hE
  have hF : AnalyticAt ℝ F 0 :=
    (hEshift.comp (f := fun x : Ambient n => z + L x) hshift).sub analyticAt_const
  have haxis : AnalyticAt ℝ (fun t : ℝ => ((0 : Base n), t)) 0 :=
    analyticAt_const.prod analyticAt_id
  have hslice : AnalyticAt ℝ (lastSlice F) 0 := by
    change AnalyticAt ℝ (fun t : ℝ => F ((0 : Base n), t)) 0
    simpa only [Function.comp_def] using
      hF.comp (f := fun t : ℝ => ((0 : Base n), t)) (x := 0) haxis
  have hsliceFactor : ∀ᶠ t in nhds (0 : ℝ),
      lastSlice F t = (t - 0) ^ d • unit t := by
    filter_upwards [hfactor] with t ht
    simpa only [lastSlice, F, L, directionCoordinates_axis, sub_zero, smul_eq_mul] using ht
  have horder : ExactOrderInLastVariable F d :=
    (analyticOrderAt_eq_nat_iff_iteratedDeriv_eq_zero hslice).mp
      (hslice.analyticOrderAt_eq_natCast.mpr ⟨unit, hunit, hunit0, hsliceFactor⟩)
  exact ⟨L, d, hd, hF, horder⟩

/-- Every nonconstant real analytic energy germ in a positive dimension
admits an invertible linear coordinate change and a Weierstrass factorization
of positive degree. Both the analytic unit and all polynomial coefficients
are constructed, and the degree is the exact directional order. This is
the preparation step for general singular curve selection underlying
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2; it asserts no gradient
inequality before the singular branches are selected. -/
theorem exists_weierstrass_energy_coordinates {n : ℕ}
    (E : EucSpace (n + 1) → ℝ) (z : EucSpace (n + 1))
    (hE : AnalyticAt ℝ E z) (hnonconstant : ¬ ∀ᶠ y in nhds z, E y = E z) :
    ∃ (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (d : ℕ)
      (a : Fin d → Base n → ℝ) (u : Ambient n → ℝ), 0 < d ∧
      ExactOrderInLastVariable (fun x => E (z + L x) - E z) d ∧
      IsWeierstrassPreparation (fun x => E (z + L x) - E z) d a u := by
  obtain ⟨L, d, hd, hF, horder⟩ := exists_regular_energy_coordinates E z hE hnonconstant
  obtain ⟨a, u, hprep⟩ := exists_isWeierstrassPreparation hF horder
  exact ⟨L, d, a, u, hd, horder, hprep⟩

/-- A degenerate critical quadratic energy in two dimensions is analytic
and nonconstant and hence exercises the unrestricted preparation theorem.
Auxiliary example for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : ∃ (L : Ambient 1 ≃L[ℝ] EucSpace 2) (d : ℕ)
    (a : Fin d → Base 1 → ℝ) (u : Ambient 1 → ℝ), 0 < d ∧
    ExactOrderInLastVariable (fun x => ((L x) 0) ^ 2) d ∧
    IsWeierstrassPreparation (fun x => ((L x) 0) ^ 2) d a u := by
  let E : EucSpace 2 → ℝ := fun x => (x 0) ^ 2
  have hE : AnalyticAt ℝ E 0 :=
    ((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2
  have hnonconstant : ¬ ∀ᶠ y in nhds (0 : EucSpace 2), E y = E 0 := by
    intro hzero
    let gamma : ℝ → EucSpace 2 := fun t =>
      t • (PiLp.single 2 (0 : Fin 2) 1 : EucSpace 2)
    have ht : Tendsto gamma (nhds 0) (nhds 0) := by
      have hc : ContinuousAt gamma 0 := continuousAt_id.smul continuousAt_const
      simpa [gamma] using hc.tendsto
    have htzero : ∀ᶠ t in nhds (0 : ℝ), t ^ 2 = 0 := by
      simpa [E, gamma] using ht.eventually hzero
    obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp htzero
    have hz : (r / 2) ^ 2 = 0 := hball (by
      simpa only [Real.dist_eq, sub_zero, abs_of_pos (half_pos hr)] using half_lt_self hr)
    exact (sq_pos_of_pos (half_pos hr)).ne' hz
  simpa [E] using exists_weierstrass_energy_coordinates E 0 hE hnonconstant

end Transformer.Normalization
