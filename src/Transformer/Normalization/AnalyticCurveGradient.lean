/-
# Gradient estimates along analytic curves

The scalar finite-order inequality and the chain rule give an exponent
below one along every analytic curve. This is the analytic part of an
energy-level curve-selection route to Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.ScalarGradientInequality
import Transformer.Normalization.ModulatedCritical

open Filter Set

namespace Transformer.Normalization

/-- The derivative of the energy along a differentiable curve is bounded
by the gradient norm times the curve speed. Auxiliary chain-rule estimate
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem energy_curve_deriv_norm_le {N : ℕ} (E : EucSpace N → ℝ)
    (gamma : ℝ → EucSpace N) (t : ℝ)
    (hE : DifferentiableAt ℝ E (gamma t)) (hgamma : DifferentiableAt ℝ gamma t) :
    |deriv (fun s => E (gamma s)) t| ≤ ‖gradient E (gamma t)‖ * ‖deriv gamma t‖ := by
  have hchain := hE.hasFDerivAt.comp_hasDerivAt t hgamma.hasDerivAt
  have hderiv : deriv (fun s => E (gamma s)) t =
      inner (𝕜 := ℝ) (gradient E (gamma t)) (deriv gamma t) := by
    simpa only [Function.comp_def, hE.hasGradientAt.fderiv_apply] using hchain.deriv
  rw [hderiv]
  exact norm_inner_le_norm (𝕜 := ℝ) _ _

/-- Every analytic energy restricted to an analytic curve satisfies a
local gradient power bound using the original ambient gradient. The curve
may start at any point, including a degenerate critical point, and no
immersion or nonzero velocity is assumed. Auxiliary analytic-curve input
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem analytic_curve_gradient_inequality {N : ℕ} (E : EucSpace N → ℝ)
    (gamma : ℝ → EucSpace N) (hE : AnalyticAt ℝ E (gamma 0))
    (hgamma : AnalyticAt ℝ gamma 0) :
    ∃ alpha k delta : ℝ, 0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ 0 < delta ∧
      ∀ t : ℝ, |t| < delta →
        |E (gamma t) - E (gamma 0)| ^ alpha ≤ k * ‖gradient E (gamma t)‖ := by
  have hF : AnalyticAt ℝ (fun t => E (gamma t)) 0 := hE.comp hgamma
  obtain ⟨alpha, k, V, ha, ha1, hk, hV, h0V, hineq⟩ :=
    scalar_analytic_gradient_inequality_at_any_point _ 0 hF
  let B : ℝ := ‖deriv gamma 0‖ + 1
  have hB : 0 < B := by dsimp [B]; positivity
  have hspeed : ∀ᶠ t in nhds (0 : ℝ), ‖deriv gamma t‖ ≤ B :=
    (hgamma.deriv.continuousAt.norm.eventually
      (eventually_lt_nhds (by dsimp [B]; linarith : ‖deriv gamma 0‖ < B))).mono
        (fun _ ht => ht.le)
  have hevent : ∀ᶠ t in nhds (0 : ℝ),
      |E (gamma t) - E (gamma 0)| ^ alpha ≤ (k * B) * ‖gradient E (gamma t)‖ := by
    filter_upwards [hgamma.eventually_analyticAt,
      hgamma.continuousAt.tendsto.eventually hE.eventually_analyticAt,
      hspeed, hV.mem_nhds h0V] with t htG htE htB htV
    calc
      |E (gamma t) - E (gamma 0)| ^ alpha ≤ k *
          |deriv (fun s => E (gamma s)) t| := hineq t htV
      _ ≤ k * (‖gradient E (gamma t)‖ * ‖deriv gamma t‖) :=
        mul_le_mul_of_nonneg_left
          (energy_curve_deriv_norm_le E gamma t htE.differentiableAt htG.differentiableAt) hk.le
      _ ≤ k * (‖gradient E (gamma t)‖ * B) := mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left htB (norm_nonneg _)) hk.le
      _ = (k * B) * ‖gradient E (gamma t)‖ := by ring
  obtain ⟨delta, hd, hball⟩ := Metric.eventually_nhds_iff.mp hevent
  exact ⟨alpha, k * B, delta, ha, ha1, mul_pos hk hB, hd,
    fun t ht => hball (by simpa only [Real.dist_eq, sub_zero] using ht)⟩

/-- The quartic energy and the curve with coordinate `t²` satisfy all
analytic and derivative hypotheses. The curve has zero velocity at zero,
so this also exercises the non-immersed case. Auxiliary example for
Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ alpha k delta : ℝ,
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ 0 < delta ∧
      ∀ t : ℝ, |t| < delta → |t ^ 8| ^ alpha ≤ k *
        ‖gradient (fun y : EucSpace 1 => (y 0) ^ 4)
          ((t ^ 2) • (PiLp.single 2 (0 : Fin 1) 1 : EucSpace 1))‖ := by
  let v : EucSpace 1 := PiLp.single 2 (0 : Fin 1) 1
  let gamma : ℝ → EucSpace 1 := fun t => t ^ 2 • v
  have hgamma : AnalyticAt ℝ gamma 0 :=
    (analyticAt_id.fun_pow 2).fun_smul analyticAt_const
  have hE : AnalyticAt ℝ (fun y : EucSpace 1 => (y 0) ^ 4) (gamma 0) :=
    ((EuclideanSpace.proj 0 : EucSpace 1 →L[ℝ] ℝ).analyticAt _).fun_pow 4
  simpa [gamma, v, ← pow_mul] using
    analytic_curve_gradient_inequality (fun y : EucSpace 1 => (y 0) ^ 4) gamma hE hgamma

end Transformer.Normalization
