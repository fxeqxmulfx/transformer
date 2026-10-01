/-
# The local gradient inequality for normal-crossing energy germs

This proves the monomial-times-unit case of the analytic input in
Appendix D.1 of arXiv:2510.22026v2, including multivariable degenerate
critical points and positive-dimensional critical loci.
-/

import Transformer.Normalization.MonomialGradientBound
import Transformer.Normalization.ModulatedCritical

open Filter Set
open scoped BigOperators

namespace Transformer.Normalization

/-- The local analytic gradient inequality for a monomial times a
nonvanishing analytic unit, with exponent `1 - 1 / max (sum p) 2`. This
proves a normal-crossing case of the analytic input in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2;
it does not assert that arbitrary analytic germs have this form. -/
theorem analytic_gradient_inequality_of_monomial_unit {N : ℕ}
    (p : Fin N → ℕ) (hp : 0 < ∑ j, p j) (E u : EucSpace N → ℝ) (z : EucSpace N)
    (hE : AnalyticAt ℝ E z) (hu : AnalyticAt ℝ u z) (hu0 : u z ≠ 0)
    (heq : ∀ᶠ y in nhds z, E y - E z = u y * coordinateMonomial p (y - z)) :
    ∃ alpha k : ℝ, ∃ V : Set (EucSpace N),
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |E y - E z| ^ alpha ≤ k * ‖gradient E y‖ := by
  let m : ℕ := ∑ j, p j
  let beta : ℝ := 1 / (max m 2 : ℕ)
  let b : ℝ := |u z| / 2
  let B : ℝ := ‖gradient u z‖ + 1
  let delta : ℝ := min 1 (b / (2 * B))
  have hb : 0 < b := half_pos (abs_pos.mpr hu0)
  have hB : 0 < B := by dsimp [B]; positivity
  have hd : 0 < delta := lt_min zero_lt_one (div_pos hb (by positivity))
  have hM : (2 : ℝ) ≤ max m 2 := by exact_mod_cast Nat.le_max_right m 2
  have hMpos : (0 : ℝ) < max m 2 := by linarith
  have hbeta : 0 < beta := div_pos zero_lt_one hMpos
  have hbeta1 : beta < 1 := (div_lt_one hMpos).mpr (by linarith)
  have huBound : ∀ᶠ y in nhds z, b ≤ |u y| := by
    exact ((hu.continuousAt.abs).eventually
      (eventually_gt_nhds (half_lt_self (abs_pos.mpr hu0)))).mono (fun y hy => hy.le)
  have hgBound : ∀ᶠ y in nhds z, ‖gradient u y‖ ≤ B := by
    have hBz : ‖gradient u z‖ < B := by dsimp [B]; linarith
    exact ((analytic_gradient_continuousAt u z hu).norm.eventually
      (eventually_lt_nhds hBz)).mono (fun y hy => hy.le)
  have hradius : ∀ᶠ y in nhds z, ‖y - z‖ < delta := by
    filter_upwards [Metric.ball_mem_nhds z hd] with y hy
    simpa only [Metric.mem_ball, dist_eq_norm] using hy
  have hineq : ∀ᶠ y in nhds z,
      |E y - E z| ^ (1 - beta) ≤ (2 / b ^ beta) * ‖gradient E y‖ := by
    filter_upwards [hE.eventually_analyticAt, hu.eventually_analyticAt,
      heq.eventually_nhds, huBound, hgBound, hradius] with y hyE hyu hyeq hyb hyB hyr
    obtain ⟨i, hpi, hmin⟩ := coordinateMonomial_min_power p hp (y - z)
    have hri : |(y - z) i| ≤ ‖y - z‖ := by
      simpa only [Real.norm_eq_abs] using PiLp.norm_apply_le (y - z) i
    have hri1 : |(y - z) i| ≤ 1 :=
      hri.trans (hyr.le.trans (min_le_left _ _))
    have hsmall : ‖gradient u y‖ * |(y - z) i| ≤ |u y| / 2 := by
      calc
        ‖gradient u y‖ * |(y - z) i| ≤ B * (b / (2 * B)) :=
          mul_le_mul hyB (hri.trans (hyr.le.trans (min_le_right _ _)))
            (abs_nonneg _) hB.le
        _ = b / 2 := by field_simp
        _ ≤ |u y| / 2 := div_le_div_of_nonneg_right hyb (by norm_num)
    have hscale : b * |coordinateMonomial p (y - z)| ≤ |E y - E z| := by
      rw [hyeq.self_of_nhds, abs_mul]
      exact mul_le_mul_of_nonneg_right hyb (abs_nonneg _)
    exact monomial_gradient_power_bound m b |(y - z) i|
      |coordinateMonomial p (y - z)| |E y - E z| ‖gradient E y‖ hb
      (abs_nonneg _) hri1 (abs_nonneg _) (norm_nonneg _) hmin hscale
      (coordinateMonomial_unit_gradient_bound p E u z y (E z)
        hyE.differentiableAt hyu.differentiableAt hyeq i hpi hsmall)
  obtain ⟨V, hV, hVo, hzV⟩ := eventually_nhds_iff.mp hineq
  exact ⟨1 - beta, 2 / b ^ beta, V, by linarith, by linarith,
    div_pos (by norm_num) (Real.rpow_pos_of_pos hb beta), hVo, hzV, hV⟩

/-- A nonconstant analytic unit times `x²y²` satisfies all local hypotheses.
This multivariable degenerate model is covered by the local inequality
from Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ alpha k : ℝ, ∃ V : Set (EucSpace 2),
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : EucSpace 2) ∈ V ∧
      ∀ y ∈ V, |(1 + (y 0) ^ 2) * coordinateMonomial (fun _ => 2) y| ^ alpha ≤
        k * ‖gradient (fun w : EucSpace 2 =>
          (1 + (w 0) ^ 2) * coordinateMonomial (fun _ => 2) w) y‖ := by
  let u : EucSpace 2 → ℝ := fun y => 1 + (y 0) ^ 2
  let E : EucSpace 2 → ℝ := fun y => u y * coordinateMonomial (fun _ => 2) y
  have hu : AnalyticAt ℝ u 0 :=
    analyticAt_const.add
      (((EuclideanSpace.proj 0 : EucSpace 2 →L[ℝ] ℝ).analyticAt 0).fun_pow 2)
  have hE : AnalyticAt ℝ E 0 := hu.mul (coordinateMonomial_analytic (fun _ => 2) 0)
  have hz : E 0 = 0 := by simp [E, coordinateMonomial, Fin.prod_univ_two]
  simpa only [hz, sub_zero] using
    analytic_gradient_inequality_of_monomial_unit (fun _ : Fin 2 => 2)
      (by decide) E u 0 hE hu (by norm_num [u])
      (Filter.Eventually.of_forall (fun y => by simp [E, hz]))


end Transformer.Normalization
