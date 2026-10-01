/-
# From energy-level comparison curves to the gradient inequality

Two analytic curves representing the smallest gradient norms on small
positive and negative energy levels suffice for the local inequality.
Curve existence is an explicit geometric input. This is the last analytic
step in an alternative route to Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticCurveGradient
import Transformer.Normalization.GradientMinimumCluster

open Filter Set

namespace Transformer.Normalization

/-- Two analytic curves with the same initial energy represent nearby
energy levels with no larger gradient norm. Parameters of the representing
points can be made arbitrarily small by shrinking the target neighborhood.
The curves need not start at `z`. This genuine predicate records the
geometric input of a curve-selection route for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. It does not assert curve existence for arbitrary
analytic energies. -/
def HasAnalyticGradientComparisonArcs {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) : Prop :=
  ∃ gamma : Fin 2 → ℝ → EucSpace N,
    (∀ j, AnalyticAt ℝ (gamma j) 0) ∧
    (∀ j, AnalyticAt ℝ E (gamma j 0)) ∧
    (∀ j, E (gamma j 0) = E z) ∧
    ∀ delta : ℝ, 0 < delta → ∀ᶠ y in nhds z,
      ∃ (j : Fin 2) (t : ℝ), |t| < delta ∧ E (gamma j t) = E y ∧
        ‖gradient E (gamma j t)‖ ≤ ‖gradient E y‖

/-- Analytic energy-level comparison curves imply the local gradient
inequality, with an exponent strictly below one supplied by the scalar
finite-order theorem along each curve. This proves the analytic implication
in the curve-selection route to Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2; selection of the curves remains an explicit hypothesis. -/
theorem local_gradient_inequality_of_analytic_comparison_arcs {N : ℕ}
    (E : EucSpace N → ℝ) (z : EucSpace N) (hE : ContinuousAt E z)
    (harcs : HasAnalyticGradientComparisonArcs E z) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  classical
  obtain ⟨gamma, hgamma, hEgamma, henergy, hlift⟩ := harcs
  have hcurve (j : Fin 2) :=
    analytic_curve_gradient_inequality E (gamma j) (hEgamma j) (hgamma j)
  choose alpha k delta ha ha1 hk hd hineq using hcurve
  let A := max (alpha 0) (alpha 1)
  let B := max (k 0) (k 1)
  let d := min (delta 0) (delta 1)
  have hA : 0 < A := (ha 0).trans_le (le_max_left _ _)
  have hA1 : A < 1 := max_lt (ha1 0) (ha1 1)
  have hB : 0 < B := (hk 0).trans_le (le_max_left _ _)
  have hd0 : 0 < d := lt_min (hd 0) (hd 1)
  have hAj (j : Fin 2) : alpha j ≤ A := by
    fin_cases j
    · exact le_max_left _ _
    · exact le_max_right _ _
  have hBj (j : Fin 2) : k j ≤ B := by
    fin_cases j
    · exact le_max_left _ _
    · exact le_max_right _ _
  have hdj (j : Fin 2) : d ≤ delta j := by
    fin_cases j
    · exact min_le_left _ _
    · exact min_le_right _ _
  have hsmall : ∀ᶠ y in nhds z, |E y - E z| < 1 := by
    simpa only [Real.dist_eq] using
      (Metric.tendsto_nhds.mp hE.tendsto) 1 zero_lt_one
  have hevent : ∀ᶠ y in nhds z, |E y - E z| ^ A ≤ B * ‖gradient E y‖ := by
    filter_upwards [hlift d hd0, hsmall] with y hy hygap
    obtain ⟨j, t, ht, hEt, hGt⟩ := hy
    have hbound := hineq j t (ht.trans_le (hdj j))
    rw [henergy j, hEt] at hbound
    by_cases hzero : E y - E z = 0
    · rw [hzero, abs_zero, Real.zero_rpow hA.ne']
      exact mul_nonneg hB.le (norm_nonneg _)
    · calc
        |E y - E z| ^ A ≤ |E y - E z| ^ alpha j :=
          Real.rpow_le_rpow_of_exponent_ge (abs_pos.mpr hzero) hygap.le (hAj j)
        _ ≤ k j * ‖gradient E (gamma j t)‖ := hbound
        _ ≤ k j * ‖gradient E y‖ := mul_le_mul_of_nonneg_left hGt (hk j).le
        _ ≤ B * ‖gradient E y‖ := mul_le_mul_of_nonneg_right (hBj j) (norm_nonneg _)
  obtain ⟨V, hV, hVo, hzV⟩ := eventually_nhds_iff.mp hevent
  exact ⟨A, B, V, hA, hA1, hB, hVo, hzV, hV⟩

/-- The quartic energy in one dimension satisfies all comparison-curve
hypotheses, using the coordinate line for both signs. This is a nonconstant
degenerate instance of the analytic implication for Appendix D.1 of
arXiv:2510.22026v2. -/
example : ∃ alpha k : ℝ, ∃ V : Set (EucSpace 1),
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 1) ∈ V ∧
      ∀ y ∈ V, |(y 0) ^ 4| ^ alpha ≤
        k * ‖gradient (fun t : EucSpace 1 => (t 0) ^ 4) y‖ := by
  let v : EucSpace 1 := PiLp.single 2 (0 : Fin 1) 1
  let gamma : ℝ → EucSpace 1 := fun t => t • v
  let E : EucSpace 1 → ℝ := fun y => (y 0) ^ 4
  have hE (y : EucSpace 1) : AnalyticAt ℝ E y :=
    ((EuclideanSpace.proj 0 : EucSpace 1 →L[ℝ] ℝ).analyticAt y).fun_pow 4
  have hgamma : AnalyticAt ℝ gamma 0 := analyticAt_id.fun_smul analyticAt_const
  have harcs : HasAnalyticGradientComparisonArcs E 0 := by
    refine ⟨(fun _ => gamma), (fun _ => hgamma), (fun _ => hE _),
      (fun _ => by simp [E, gamma]), ?_⟩
    intro delta hd
    filter_upwards [Metric.ball_mem_nhds (0 : EucSpace 1) hd] with y hy
    have hcoord : |y 0| < delta := (PiLp.norm_apply_le y 0).trans_lt
      (by simpa only [Metric.mem_ball, dist_zero_right] using hy)
    have hpoint : gamma (y 0) = y := by
      ext i
      fin_cases i
      simp [gamma, v]
    exact ⟨0, y 0, hcoord, by rw [hpoint], by rw [hpoint]⟩
  simpa only [E, PiLp.zero_apply, zero_pow (by norm_num : (4 : ℕ) ≠ 0), sub_zero] using
    local_gradient_inequality_of_analytic_comparison_arcs E 0 (hE 0).continuousAt harcs

end Transformer.Normalization
