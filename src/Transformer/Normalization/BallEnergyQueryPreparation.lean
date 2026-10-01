/-
# Constructing prepared gradient-query patches from analyticity

The degree, real polynomial coefficients, and coverage neighborhood are
constructed for every nonconstant analytic energy germ.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.BallEnergyQueryValues
import Transformer.Normalization.AnalyticFiberPolynomialValues

noncomputable section

namespace Transformer.Normalization

open AnalyticPreparation

/-- Every nonconstant analytic energy germ gives actual scalar-query
data for a closed-ball energy patch. The patch contains all sufficiently
nearby points of the original ball, and both the ball constraint and the
squared gradient norm have exact polynomial values at every retained
root. Coordinates and all coefficients are constructed; no analytic base
curve, root branch, minimum, or prepared data is assumed. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem exists_ball_energy_query_preparation {n : ℕ}
    (E : EucSpace (n + 1) → ℝ) (w center : EucSpace (n + 1)) (R : ℝ) (hR : 0 ≤ R)
    (hE : AnalyticAt ℝ E w) (hnonconstant : ¬ ∀ᶠ x in nhds w, E x = E w) :
    ∃ (L : Ambient n ≃L[ℝ] EucSpace (n + 1)) (d : ℕ)
      (a b v : Fin d → Base (n + 1) → ℝ) (r delta : ℝ),
      0 < d ∧ 0 < r ∧ 0 < delta ∧
      (∀ i, AnalyticAt ℝ (a i) 0) ∧ (∀ i, a i 0 = 0) ∧
      (∀ i, AnalyticAt ℝ (b i) 0) ∧ (∀ i, AnalyticAt ℝ (v i) 0) ∧
      BallEnergyQueryValues E w center L R r a b v ∧
      ∀ x ∈ Metric.closedBall center R,
        x ∈ Metric.ball w delta → x ∈ ballEnergyPatchSource E w center L R r := by
  obtain ⟨L, d, hd, _, horder⟩ := exists_regular_energy_coordinates E w hE hnonconstant
  have hphi := (energyChartMap_analytic w L).1
  have hphi0 := (energyChartMap_analytic w L).2
  have hEphi : AnalyticAt ℝ E (energyChartMap w L 0) := by simpa only [hphi0] using hE
  have hF : AnalyticAt ℝ
      (fun x => E (energyChartMap w L x) - energyChartLevel E w x.1) 0 :=
    (hEphi.comp (f := energyChartMap w L) (x := 0) hphi).fun_sub
      ((energyChartLevel_analytic E w).comp
        (f := fun x : Ambient (n + 1) => x.1) (x := 0) analyticAt_fst)
  have hobjective : AnalyticAt ℝ (fun x => squaredGradientNorm E (energyChartMap w L x)) 0 :=
    (squared_gradient_norm_analyticAt E (energyChartMap w L 0) hEphi).comp
      (f := energyChartMap w L) (x := 0) hphi
  obtain ⟨a, b, v, r, hr, ha, ha0, hb, hv, hvalues⟩ :=
    analytic_fiber_polynomial_values _ hF (energyChart_exact_order E w L horder)
      (fun _ : Fin 1 => energyChartBallConstraint w center L R)
      (fun _ => energyChartBallConstraint_analytic w center L R) _ hobjective
  obtain ⟨delta, hdelta, hcover⟩ :=
    ballEnergyPatchSource_contains_neighborhood E w center L hE hR hr
  refine ⟨L, d, a, b 0, v, r, delta, hd, hr, hdelta, ha, ha0, hb 0, hv, ?_, hcover⟩
  intro z hz y hy
  have hzy := hvalues z hz y hy
  exact ⟨hzy.1, fun hP => hzy.2.1 hP 0, hzy.2.2⟩

/-- The nonconstant quadratic energy and the closed unit ball witness
all inputs jointly. The constructed value relations also inhabit the
hypotheses of the patch lower-bound query theorem. Auxiliary for Appendix
D.1, `lem: loj`, of arXiv:2510.22026v2. -/
example : let E : EucSpace 1 → ℝ := fun x => ‖x‖ ^ 2
    ∃ (L : Ambient 0 ≃L[ℝ] EucSpace 1) (d : ℕ)
      (a b v : Fin d → Base 1 → ℝ) (r delta : ℝ),
      0 < d ∧ 0 < r ∧ 0 < delta ∧
      BallEnergyQueryValues E 0 0 L 1 r a b v ∧
      ∀ x ∈ Metric.closedBall (0 : EucSpace 1) 1,
        x ∈ Metric.ball 0 delta → x ∈ ballEnergyPatchSource E 0 0 L 1 r := by
  intro E
  obtain ⟨L, d, a, b, v, r, delta, hd, hr, hdelta, _, _, _, _, hvalues, hcover⟩ :=
    exists_ball_energy_query_preparation E 0 0 1 (by norm_num)
      (quadratic_energy_analytic 0 (Set.mem_univ _))
      (nonflatEnergyLevel_nonconstant E (Metric.closedBall 0 1) 0 zero_mem_nonflat_quadratic_level)
  exact ⟨L, d, a, b, v, r, delta, hd, hr, hdelta, hvalues, hcover⟩

end Transformer.Normalization
