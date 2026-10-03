/-
# Convergence of original AMSGradW in a decay-dominated regime

User-requested extension of arXiv:1904.03590v4, Algorithm 1 and §4–6,
with decoupled weight decay as discussed in arXiv:2606.25971v2, §2.
This is the actual unguarded update, with constant learning rate and
coefficients and no bias correction. The sufficient condition
`L < decay*epsilon` is deliberately explicit; small experimental decay
does not automatically satisfy it. All limits are derived from the
objective and parameter conditions, including the limiting metric.
-/

import Transformer.AMSGradW.MetricLimit
import Transformer.AMSGradW.Error
import Transformer.AMSGradW.Perturbation

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGradW

variable {d : ℕ}

/-- Original AMSGradW's actual weights, first moments, denominators and
loss converge. The limit obeys the history-dependent weighted-decay
equation, which generally differs from `gradient f = 0`. The objective
is differentiable with a maximum-norm Lipschitz gradient; sufficient
conditions are positive eta/epsilon/decay, `0<=beta<1`, `0<=beta2<=1`,
`eta*decay<=1` and `L<decay*epsilon`. No boundedness, alignment, metric
limit, stationary point or convergence hypothesis is supplied.
Source: arXiv:1904.03590v4, Algorithm 1 and §4–6, AMSGradW extension. -/
theorem trainingRun_convergence (η ε wd β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hf : Differentiable ℝ f) (hgrad : LipschitzGradient f L)
    (hη : 0 < η) (hε : 0 < ε) (hwd : 0 < wd) (hL : 0 ≤ L)
    (hβ : 0 ≤ β) (hβ' : β < 1) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1)
    (hstep : η * wd ≤ 1) (hdom : L < wd * ε) :
    ∃ (D : Fin d → ℝ) (star : TrainingSpace d), (∀ i, ε ≤ D i) ∧
      Tendsto (fun t => (trainingRun η ε wd β β₂ f initial t).position) atTop (𝓝 star) ∧
      Tendsto (fun t => (trainingRun η ε wd β β₂ f initial t).momentum)
        atTop (𝓝 (WithLp.ofLp (gradient f star))) ∧
      Tendsto (fun t => AMSGrad.trainingDenominator ε
        (trainingRun η ε wd β β₂ f initial t)) atTop (𝓝 D) ∧
      Tendsto (fun t => f (trainingRun η ε wd β β₂ f initial t).position)
        atTop (𝓝 (f star)) ∧
      (∀ i, gradient f star i + wd * D i * star i = 0) := by
  obtain ⟨D, hD, hmetric⟩ := trainingRun_metric_limit η ε wd β β₂ L f initial hgrad
    hη.le hε hwd.le hL hβ hβ'.le hβ₂ hβ₂' hstep hdom
  obtain ⟨x, hx, _⟩ := exists_unique_equilibrium ε wd L D f hε hwd hL hdom hD hgrad
  have hnext : Tendsto (fun t => AMSGrad.trainingDenominator ε
      (trainingRun η ε wd β β₂ f initial (t + 1))) atTop (𝓝 D) :=
    (Filter.tendsto_add_atTop_iff_nat 1).mpr hmetric
  have hforcing : Tendsto (fun t => metricForcing η f x D
      (AMSGrad.trainingDenominator ε (trainingRun η ε wd β β₂ f initial (t + 1))))
      atTop (𝓝 0) := by
    have hv : Tendsto (fun t i => coordinateGradient f x i *
        (1 / AMSGrad.trainingDenominator ε (trainingRun η ε wd β β₂ f initial (t + 1)) i -
          1 / D i)) atTop (𝓝 (0 : Fin d → ℝ)) := by
      apply tendsto_pi_nhds.mpr
      intro i
      have hi := ((tendsto_const_nhds (x := (1 : ℝ))).div (tendsto_pi_nhds.mp hnext i)
        (hε.trans_le (hD i)).ne').sub_const (1 / D i)
      simpa only [sub_self, mul_zero, Pi.zero_apply, Pi.div_apply] using
        hi.const_mul (coordinateGradient f x i)
    simpa only [metricForcing, norm_zero, mul_zero] using hv.norm.const_mul η
  let E := fun t => stateError ε wd f x (trainingRun η ε wd β β₂ f initial t)
  obtain ⟨hq0, hq⟩ := contractionFactor_bounds η ε wd β L hη hε hwd hL hβ hβ' hdom
  have he : Tendsto E atTop (𝓝 0) := contraction_forcing_tendsto_zero
    (contractionFactor η ε wd β L) E _ hq0 hq
    (fun t => (norm_nonneg _).trans (le_max_left _ _)) hforcing
    (fun t => stateError_step η ε wd β β₂ L f
      (trainingRun η ε wd β β₂ f initial t) x D hη.le hε hwd hL hβ hβ'.le
      hstep hgrad (fun i => hε.trans_le (hD i)) hx)
  have hxnorm : Tendsto (fun t =>
      ‖WithLp.ofLp (trainingRun η ε wd β β₂ f initial t).position - x‖) atTop (𝓝 0) :=
    squeeze_zero (fun t => norm_nonneg _) (fun t => le_max_left _ _) he
  have hxcoord : Tendsto (fun t => WithLp.ofLp
      (trainingRun η ε wd β β₂ f initial t).position) atTop (𝓝 x) :=
    tendsto_iff_norm_sub_tendsto_zero.mpr hxnorm
  have hxpos : Tendsto (fun t => (trainingRun η ε wd β β₂ f initial t).position)
      atTop (𝓝 (WithLp.toLp 2 x)) := by
    simpa only [Function.comp_def, WithLp.toLp_ofLp] using
      (PiLp.continuous_toLp 2 (fun _ : Fin d => ℝ)).tendsto x |>.comp hxcoord
  have hmnorm : Tendsto (fun t =>
      ‖(trainingRun η ε wd β β₂ f initial t).momentum - coordinateGradient f x‖)
      atTop (𝓝 0) := by
    have hup : Tendsto (fun t => wd * ε * E t) atTop (𝓝 0) := by
      simpa only [mul_zero] using he.const_mul (wd * ε)
    apply squeeze_zero (fun t => norm_nonneg _) (fun t => ?_) hup
    simpa only [mul_comm] using (div_le_iff₀ (mul_pos hwd hε)).mp
      (show ‖(trainingRun η ε wd β β₂ f initial t).momentum - coordinateGradient f x‖ /
        (wd * ε) ≤ E t from le_max_right _ _)
  exact ⟨D, WithLp.toLp 2 x, hD, hxpos,
    tendsto_iff_norm_sub_tendsto_zero.mpr hmnorm, hmetric,
    hf.continuous.tendsto _ |>.comp hxpos, hx⟩

/-- All convergence hypotheses hold on a nonconstant loss with nonzero
initial gradient, positive momentum and positive decay. Source:
arXiv:1904.03590v4, §4, AMSGradW extension. -/
example :
    let f := (Optimization.energy : TrainingSpace 1 → ℝ)
    let initial : TrainingSpace 1 := WithLp.toLp 2 (fun _ => 1)
    Differentiable ℝ f ∧ LipschitzGradient f 1 ∧
      (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
      (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (1 / 4 : ℝ) * 2 ≤ 1 ∧
      (1 : ℝ) < 2 * 1 ∧ gradient f initial ≠ 0 := by
  refine ⟨Optimization.energy_models.1.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_⟩
  rw [Optimization.energy_gradient]
  intro h
  have hi := congrArg (fun y : TrainingSpace 1 => y 0) h
  norm_num at hi

end Transformer.AMSGradW
