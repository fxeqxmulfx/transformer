/-
# Polar equations for compact energy-level gradient minima

Interior minima of squared gradient norm have dependent energy and
objective differentials. On a boundary sphere the radial differential is
included. These analytic equations locate the minima whose curve selection
is needed for Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.GradientEnergyImage
import Transformer.Normalization.QuadraticEnergy
import Mathlib.Analysis.Calculus.LagrangeMultipliers

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- A gradient-norm minimum on an energy fiber in a neighborhood satisfies
the two-differential polar equation. No regularity of the energy level is
assumed: the multipliers need not have a nonzero objective component.
Auxiliary necessary condition for the analytic-curve route to Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem gradient_minimum_polar_interior {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (w : EucSpace N) (hE : AnalyticAt ℝ E w)
    (hK : K ∈ nhds w)
    (hmin : ∀ y ∈ K, E y = E w → ‖gradient E w‖ ≤ ‖gradient E y‖) :
    ∃ a b : ℝ, (a, b) ≠ 0 ∧
      a • fderiv ℝ E w + b • fderiv ℝ (squaredGradientNorm E) w = 0 := by
  have hlocal : IsLocalMinOn (squaredGradientNorm E) {y | E y = E w} w := by
    have hnear : ∀ᶠ y in nhds w, y ∈ K := hK
    filter_upwards [hnear.filter_mono nhdsWithin_le_nhds, self_mem_nhdsWithin] with y hy hyE
    exact pow_le_pow_left₀ (norm_nonneg _) (hmin y hy hyE) 2
  exact IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt_1d (Or.inl hlocal)
    hE.hasStrictFDerivAt (squared_gradient_norm_analyticAt E w hE).hasStrictFDerivAt

/-- A gradient-norm minimum on an energy fiber of a closed ball satisfies
the three-differential polar equation including squared radius. Restricting
to the sphere through the minimizer gives equality constraints, so the
Lagrange multiplier theorem applies also at the boundary and at singular
energy levels. Auxiliary necessary condition for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
theorem gradient_minimum_polar_radial {N : ℕ} (E : EucSpace N → ℝ)
    (z w : EucSpace N) (r : ℝ) (hE : AnalyticAt ℝ E w)
    (hw : w ∈ Metric.closedBall z r)
    (hmin : ∀ y ∈ Metric.closedBall z r,
      E y = E w → ‖gradient E w‖ ≤ ‖gradient E y‖) :
    ∃ a b c : ℝ, (a, b, c) ≠ 0 ∧
      a • fderiv ℝ E w + b • fderiv ℝ (fun y : EucSpace N => ‖y - z‖ ^ 2) w +
        c • fderiv ℝ (squaredGradientNorm E) w = 0 := by
  classical
  let R : EucSpace N → ℝ := fun y => ‖y - z‖ ^ 2
  let f : Fin 2 → EucSpace N → ℝ := fun i => if i = 0 then E else R
  have hsub : AnalyticAt ℝ (fun y : EucSpace N => y - z) w :=
    analyticAt_id.sub analyticAt_const
  have hR : AnalyticAt ℝ R w :=
    (quadratic_energy_analytic (w - z) (Set.mem_univ _)).comp
      (f := fun y : EucSpace N => y - z) (x := w) hsub
  have hf (i : Fin 2) : AnalyticAt ℝ (f i) w := by
    fin_cases i
    · exact hE
    · exact hR
  have hlocal : IsLocalMinOn (squaredGradientNorm E)
      {y | ∀ i, f i y = f i w} w := by
    apply IsMinOn.isLocalMinOn
    intro y hy
    have hyE : E y = E w := by simpa [f] using hy 0
    have hyR : ‖y - z‖ ^ 2 = ‖w - z‖ ^ 2 := by simpa [f, R] using hy 1
    have hyK : y ∈ Metric.closedBall z r := by
      have hwnorm : ‖w - z‖ ≤ r := by
        simpa only [Metric.mem_closedBall, dist_eq_norm] using hw
      have heqnorm : ‖y - z‖ = ‖w - z‖ := by
        nlinarith [norm_nonneg (y - z), norm_nonneg (w - z)]
      simpa only [Metric.mem_closedBall, dist_eq_norm, heqnorm] using hwnorm
    exact pow_le_pow_left₀ (norm_nonneg _) (hmin y hyK hyE) 2
  obtain ⟨L, c, hLc, heq⟩ :=
    IsLocalExtrOn.exists_multipliers_of_hasStrictFDerivAt (Or.inl hlocal)
      (fun i => (hf i).hasStrictFDerivAt)
      (squared_gradient_norm_analyticAt E w hE).hasStrictFDerivAt
  refine ⟨L 0, L 1, c, ?_, ?_⟩
  · intro hzero
    have hzero' : L 0 = 0 ∧ L 1 = 0 ∧ c = 0 := by
      simpa only [Prod.mk_eq_zero] using hzero
    apply hLc
    have hL : L = 0 := by
      ext i
      fin_cases i
      · exact hzero'.1
      · exact hzero'.2.1
    exact Prod.ext hL hzero'.2.2
  · simpa [Fin.sum_univ_two, f, R] using heq

/-- Every attained energy level on a compact analytic ball has a
gradient-norm minimizer satisfying the radial polar equation and, when
interior, the two-differential polar equation. This establishes the
necessary analytic equations for the general selection step of
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem compact_gradient_minimum_with_polar_equations {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (r c : ℝ)
    (hE : AnalyticOnNhd ℝ E (Metric.closedBall z r))
    (hc : c ∈ E '' Metric.closedBall z r) :
    ∃ w ∈ Metric.closedBall z r, E w = c ∧
      (∀ y ∈ Metric.closedBall z r, E y = c → ‖gradient E w‖ ≤ ‖gradient E y‖) ∧
      (∃ a b d : ℝ, (a, b, d) ≠ 0 ∧
        a • fderiv ℝ E w + b • fderiv ℝ (fun y : EucSpace N => ‖y - z‖ ^ 2) w +
          d • fderiv ℝ (squaredGradientNorm E) w = 0) ∧
      (w ∈ Metric.ball z r → ∃ a b : ℝ, (a, b) ≠ 0 ∧
        a • fderiv ℝ E w + b • fderiv ℝ (squaredGradientNorm E) w = 0) := by
  obtain ⟨w, hw, hwc, hmin⟩ := compact_gradient_minimum_on_energy_fiber E
    (Metric.closedBall z r) (isCompact_closedBall _ _) hE c hc
  have hminw : ∀ y ∈ Metric.closedBall z r,
      E y = E w → ‖gradient E w‖ ≤ ‖gradient E y‖ := by
    intro y hy hyE
    exact hmin y hy (hyE.trans hwc)
  refine ⟨w, hw, hwc, hmin,
    gradient_minimum_polar_radial E z w r (hE w hw) hw hminw, ?_⟩
  intro hwi
  exact gradient_minimum_polar_interior E (Metric.closedBall z r) w (hE w hw)
    (Filter.mem_of_superset (Metric.isOpen_ball.mem_nhds hwi) Metric.ball_subset_closedBall) hminw

/-- A squared coordinate on the closed unit ball has a nonempty zero
energy level. Thus all compact minimization and polar-equation hypotheses
are jointly satisfiable. Auxiliary example for Appendix D.1 of
arXiv:2510.22026v2. -/
example : ∃ w ∈ Metric.closedBall (0 : EucSpace 1) 1,
    (w 0) ^ 2 = 0 ∧
      (∃ a b c : ℝ, (a, b, c) ≠ 0 ∧
        a • fderiv ℝ (fun y : EucSpace 1 => (y 0) ^ 2) w +
          b • fderiv ℝ (fun y : EucSpace 1 => ‖y - 0‖ ^ 2) w +
          c • fderiv ℝ (squaredGradientNorm (fun y : EucSpace 1 => (y 0) ^ 2)) w = 0) := by
  have hE : AnalyticOnNhd ℝ (fun y : EucSpace 1 => (y 0) ^ 2)
      (Metric.closedBall 0 1) := fun y hy =>
    ((EuclideanSpace.proj 0 : EucSpace 1 →L[ℝ] ℝ).analyticAt y).fun_pow 2
  have hc : (0 : ℝ) ∈ (fun y : EucSpace 1 => (y 0) ^ 2) '' Metric.closedBall 0 1 :=
    ⟨0, by simp, by simp⟩
  obtain ⟨w, hw, hw0, _, hrad, _⟩ :=
    compact_gradient_minimum_with_polar_equations _ 0 1 0 hE hc
  exact ⟨w, hw, hw0, hrad⟩

end Transformer.Normalization
