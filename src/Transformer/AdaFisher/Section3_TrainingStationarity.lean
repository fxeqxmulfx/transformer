/-
# AdaFisher — convergence of true gradients along full training

arXiv:2405.16397v3, §3.4, deterministic constant-step extension.
The computed KF metric and bias-corrected nonzero momentum are retained.
This is a learning convergence theorem, not merely a bounded-metric
premise or a conditional stochastic energy extraction.
-/

import Transformer.AdaFisher.Section3_TrainingErrorLimit
import Transformer.AdaFisher.Section3_TrainingGradientEnergy

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AdaFisher

open Optimization

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- Every actual full gradient norm tends to zero along AdaFisher training
of a fixed smooth lower-bounded objective with Lipschitz gradient. The
sufficient constant-step condition is `2*L*eta <= damping*(1-beta)`.
No bounded-gradient, bounded-weight, alignment, monotone-Fisher or
convergence hypothesis is supplied. Bias correction and both KF EMAs
are computed by Algorithm 1's actual corrected state transition.
Source: arXiv:2405.16397v3, §3.3–3.4, explicit deterministic constant-step
extension, distinct from the printed stochastic Proposition 3.4. -/
theorem trainingRun_gradient_tendsto_zero (η β γ δ L lower : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1)
    (hδ : 0 < δ) (hL : 0 < L) (hstep : 2 * L * η ≤ δ * (1 - β))
    (hlower : ∀ x, lower ≤ f x) :
    Tendsto (fun t : ℕ => ‖gradient f
      (trainingPosition (trainingRun η β γ δ f factors initial t))‖) atTop (𝓝 0) := by
  obtain ⟨hv, he⟩ := trainingRun_velocity_error_tendsto_zero η β γ δ L lower f factors initial
    hf hgrad hη hβ hβ' hδ hL hstep hlower
  have hvnext := (Filter.tendsto_add_atTop_iff_nat 1).mpr hv
  have hupper : Tendsto (fun t : ℕ =>
      2 * ((1 + δ) ^ 2 / η ^ 2 *
          ‖trainingVelocity η β δ (trainingRun η β γ δ f factors initial (t + 1))‖ ^ 2 +
        β ^ 2 * ‖trainingError β f (trainingRun η β γ δ f factors initial t)‖ ^ 2))
      atTop (𝓝 0) := by
    simpa only [zero_pow (by decide : 2 ≠ 0), mul_zero, zero_add] using
      (((hvnext.pow 2).const_mul ((1 + δ) ^ 2 / η ^ 2)).add
        ((he.pow 2).const_mul (β ^ 2))).const_mul 2
  have hsq : Tendsto (fun t : ℕ => ‖gradient f
      (trainingPosition (trainingRun η β γ δ f factors initial t))‖ ^ 2) atTop (𝓝 0) :=
    squeeze_zero (fun t => sq_nonneg _)
      (fun t => trainingRun_gradient_energy_bound η β γ δ f factors initial hη hβ hβ' hδ t) hupper
  have h := (Real.continuous_sqrt.tendsto 0).comp hsq
  simpa only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] using h

/-- Actual gradient convergence applies to a nonconstant loss with
nonzero initial gradient and beta=0.9. Source: arXiv:2405.16397v3,
§3.4, deterministic training extension. -/
example :
    let initial : TrainingSpace 2 2 := WithLp.toLp 2 (fun i => (i : ℝ) + 1)
    SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
      LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
      (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
      (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) ∧
      (∀ x : TrainingSpace 2 2, 0 ≤ energy x) ∧ gradient energy initial ≠ 0 := by
  refine ⟨energy_models.1, energy_lipschitz, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num,
    fun x => div_nonneg (sq_nonneg _) (by norm_num), ?_⟩
  rw [energy_gradient]
  intro h
  have hi := congrArg (fun x : TrainingSpace 2 2 => x 0) h
  norm_num at hi

end Transformer.AdaFisher
