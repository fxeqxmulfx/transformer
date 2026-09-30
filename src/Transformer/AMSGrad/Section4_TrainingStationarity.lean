/-
# AMSGrad — actual full-gradient convergence to stationarity

Deterministic fixed-objective extension of arXiv:1904.03590v4, Algorithm 1
and §4, using the positive denominator regularizer of §6. This proves
convergence of the actual objective gradients along training; it is
distinct from regret bounds and the limit of a matrix iteration.
-/

import Transformer.AMSGrad.Section4_TrainingGradientEnergy
import Transformer.AMSGrad.Section4_TrainingVelocityLimit

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- Every true gradient norm tends to zero for epsilon-stabilized AMSGrad
with constant momentum, constant learning rate and the sufficient step
condition `L * eta <= epsilon`. No bounded-gradient, bounded-weight,
gradient-alignment or convergence hypothesis is supplied. A fixed smooth
lower-bounded objective suffices; global optimality is not claimed for
a nonconvex objective. Source: arXiv:1904.03590v4, Algorithm 1 and §4,
explicit full-gradient, constant-step, unprojected extension with §6 epsilon. -/
theorem trainingRun_gradient_tendsto_zero (η ε β β₂ L lower : ℝ)
    (f : TrainingSpace d → ℝ) (initial : TrainingSpace d) (hf : SmoothObjective f L)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1) (hstep : L * η ≤ ε) (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => ‖gradient f (trainingRun η ε β β₂ f initial t).position‖)
      atTop (𝓝 0) := by
  let velocity := fun t => ‖trainingVelocity η ε (trainingRun η ε β β₂ f initial t)‖
  let C := 2 * (ε + Real.sqrt (2 * L * (f initial - lower))) ^ 2 /
    (η ^ 2 * (1 - β) ^ 2)
  have hv : Tendsto velocity atTop (𝓝 0) :=
    trainingRun_velocity_tendsto_zero η ε β β₂ L lower f initial hf
      hη hε hL hβ hβ' hstep hlower
  have hvnext : Tendsto (fun t : ℕ => velocity (t + 1)) atTop (𝓝 0) :=
    (Filter.tendsto_add_atTop_iff_nat 1).mpr hv
  have hupper : Tendsto (fun t : ℕ => C * (velocity t ^ 2 + velocity (t + 1) ^ 2))
      atTop (𝓝 0) := by
    simpa only [zero_pow (by decide : 2 ≠ 0), zero_add, mul_zero] using
      (hv.pow 2 |>.add (hvnext.pow 2)).const_mul C
  have hsq : Tendsto (fun t : ℕ =>
      ‖gradient f (trainingRun η ε β β₂ f initial t).position‖ ^ 2) atTop (𝓝 0) :=
    squeeze_zero (fun t => sq_nonneg _)
      (fun t => trainingRun_gradient_energy_bound η ε β β₂ L lower f initial hf
        hη hε hL hβ hβ' hβ₂ hβ₂' hstep hlower t) hupper
  have h := (Real.continuous_sqrt.tendsto 0).comp hsq
  simpa only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] using h

/-- Training convergence hypotheses are jointly satisfiable with a
nonconstant loss, nonzero initial gradient and nonzero momentum.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example :
    let initial : TrainingSpace 1 := WithLp.toLp 2 (fun _ => 1)
    SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
      (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
      (∀ x : TrainingSpace 1, 0 ≤ energy x) ∧ gradient energy initial ≠ 0 := by
  refine ⟨energy_models.1, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num), ?_⟩
  rw [energy_gradient]
  intro h
  have hi := congrArg (fun x : TrainingSpace 1 => x 0) h
  norm_num at hi

end Transformer.AMSGrad
