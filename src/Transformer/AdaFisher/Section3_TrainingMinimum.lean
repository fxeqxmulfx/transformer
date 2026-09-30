/-
# AdaFisher — convergence of actual last weights to a global minimum

arXiv:2405.16397v3, §3.4 and Appendix A.2, strong-convexity extension
of the complete momentum/bias-corrected Algorithm 1. The loss Hessian
is not identified with the computed Fisher matrix. Strong convexity
is an explicit additional objective assumption, not a claim about DNNs.
-/

import Transformer.AdaFisher.Section3_TrainingStationarity

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AdaFisher

open Optimization

variable {a b : ℕ} [NeZero a] [NeZero b]

/-- Actual AdaFisher weights and losses converge to a global minimizer
when the fixed smooth loss is strongly convex and has a stationary point.
Nonzero momentum, positive-time bias correction, both KF EMAs and the
varying min-max/damped Fisher are retained. Source: arXiv:2405.16397v3,
Proposition 3.3 and Appendix A.2, explicit extension from its unmomented
iteration to full deterministic Algorithm 1 with the sufficient step
`2*L*eta <= damping*(1-beta)`. The minimum and lower loss bound are derived. -/
theorem trainingRun_strong_convergence (η β γ δ L μ : ℝ)
    (f : TrainingSpace a b → ℝ) (factors : TrainingFactors a b)
    (initial star : TrainingSpace a b) (hf : SmoothObjective f L)
    (hgrad : LipschitzGradient f L) (hstrong : Optimization.StrongLowerModel f μ)
    (hη : 0 < η) (hβ : 0 ≤ β) (hβ' : β < 1) (hδ : 0 < δ) (hL : 0 < L) (hμ : 0 < μ)
    (hstep : 2 * L * η ≤ δ * (1 - β)) (hstar : gradient f star = 0) :
    (∀ x, f star ≤ f x) ∧
      Tendsto (fun t : ℕ => trainingPosition (trainingRun η β γ δ f factors initial t))
        atTop (𝓝 star) ∧
      Tendsto (fun t : ℕ => f (trainingPosition (trainingRun η β γ δ f factors initial t)))
        atTop (𝓝 (f star)) := by
  have hmin : ∀ x, f star ≤ f x := by
    intro x
    have h := (strong_gap_bounds f μ star x hstrong hμ hstar).1
    have hn : 0 ≤ μ / 2 * ‖x - star‖ ^ 2 := by positivity
    linarith
  let position := fun t => trainingPosition (trainingRun η β γ δ f factors initial t)
  have hg := trainingRun_gradient_tendsto_zero η β γ δ L (f star) f factors initial hf
    hgrad hη hβ hβ' hδ hL hstep hmin
  have hgap : Tendsto (fun t : ℕ => f (position t) - f star) atTop (𝓝 0) := by
    apply squeeze_zero (fun t => sub_nonneg.mpr (hmin _))
      (g := fun t => ‖gradient f (position t)‖ ^ 2 / (2 * μ))
    · intro t
      apply (le_div_iff₀ (show 0 < 2 * μ by positivity)).mpr
      simpa only [mul_comm] using
        (strong_gap_bounds f μ star (position t) hstrong hμ hstar).2
    · simpa only [zero_pow (by decide : 2 ≠ 0), zero_div] using (hg.pow 2).div_const (2 * μ)
  have hdist : Tendsto (fun t : ℕ => ‖position t - star‖ ^ 2) atTop (𝓝 0) := by
    apply squeeze_zero (fun t => sq_nonneg _)
      (g := fun t => 2 * (f (position t) - f star) / μ)
    · intro t
      apply (le_div_iff₀ hμ).mpr
      have h := (strong_gap_bounds f μ star (position t) hstrong hμ hstar).1
      nlinarith
    · simpa only [mul_zero, zero_div] using (hgap.const_mul 2).div_const μ
  have hn : Tendsto (fun t : ℕ => ‖position t - star‖) atTop (𝓝 0) := by
    have h := (Real.continuous_sqrt.tendsto 0).comp hdist
    simpa only [Function.comp_def, Real.sqrt_sq (norm_nonneg _), Real.sqrt_zero] using h
  refine ⟨hmin, tendsto_iff_norm_sub_tendsto_zero.mpr hn, ?_⟩
  simpa only [sub_add_cancel, zero_add] using hgap.add_const (f star)

/-- The full strong-convergence assumptions are satisfiable with
nonzero momentum and a nonoptimal initial block. Source:
arXiv:2405.16397v3, §3.4, deterministic training extension. -/
example :
    let initial : TrainingSpace 2 2 := WithLp.toLp 2 (fun i => (i : ℝ) + 1)
    SmoothObjective (energy : TrainingSpace 2 2 → ℝ) 1 ∧
      LipschitzGradient (energy : TrainingSpace 2 2 → ℝ) 1 ∧
      Optimization.StrongLowerModel (energy : TrainingSpace 2 2 → ℝ) 1 ∧
      (0 : ℝ) < 1 / 40 ∧ (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
      (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      2 * (1 : ℝ) * (1 / 40) ≤ 1 * (1 - 9 / 10) ∧
      gradient (energy : TrainingSpace 2 2 → ℝ) 0 = 0 ∧ initial ≠ 0 := by
  refine ⟨energy_models.1, energy_lipschitz, energy_models.2, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by rw [energy_gradient]; rfl, ?_⟩
  intro h
  have hi := congrArg (fun x : TrainingSpace 2 2 => x 0) h
  norm_num at hi

end Transformer.AdaFisher
