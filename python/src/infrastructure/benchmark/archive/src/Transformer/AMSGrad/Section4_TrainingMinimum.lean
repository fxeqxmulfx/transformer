/-
# AMSGrad — actual weights converge under strong convexity

Additional objective assumption for the full-gradient training extension
of arXiv:1904.03590v4, Algorithm 1 and §4. Strong convexity is not claimed
for neural-network losses. In this regime the actual last iterates, not
only their averages or loss values, converge to a global minimizer.
-/

import Transformer.AMSGrad.Section4_TrainingStationarity

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGrad

open Optimization

variable {d : ℕ}

/-- Constant-momentum epsilon-stabilized AMSGrad converges in its actual
weights and losses to a global minimizer when the fixed smooth objective
is strongly convex and has a stationary point. The minimum and the
required lower loss bound are derived from these objective assumptions.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, explicit strong-convexity,
full-gradient, constant-step, unprojected extension with §6 epsilon. -/
theorem trainingRun_strong_convergence (η ε β β₂ L μ : ℝ)
    (f : TrainingSpace d → ℝ) (initial star : TrainingSpace d)
    (hf : SmoothObjective f L) (hstrong : StrongLowerModel f μ)
    (hη : 0 < η) (hε : 0 < ε) (hL : 0 < L) (hμ : 0 < μ)
    (hβ : 0 ≤ β) (hβ' : β < 1) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1)
    (hstep : L * η ≤ ε) (hstar : gradient f star = 0) :
    (∀ x, f star ≤ f x) ∧
      Tendsto (fun t : ℕ => (trainingRun η ε β β₂ f initial t).position) atTop (𝓝 star) ∧
      Tendsto (fun t : ℕ => f (trainingRun η ε β β₂ f initial t).position)
        atTop (𝓝 (f star)) := by
  have hmin : ∀ x, f star ≤ f x := by
    intro x
    have h := (strong_gap_bounds f μ star x hstrong hμ hstar).1
    have hn : 0 ≤ μ / 2 * ‖x - star‖ ^ 2 := by positivity
    linarith
  let position := fun t => (trainingRun η ε β β₂ f initial t).position
  have hg := trainingRun_gradient_tendsto_zero η ε β β₂ L (f star) f initial hf
    hη hε hL hβ hβ' hβ₂ hβ₂' hstep hmin
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

/-- All strong-convergence hypotheses are simultaneously satisfied by a
nonconstant loss, a nonoptimal initial weight and nonzero momentum.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, training extension. -/
example :
    let initial : TrainingSpace 1 := WithLp.toLp 2 (fun _ => 1)
    SmoothObjective (energy : TrainingSpace 1 → ℝ) 1 ∧
      StrongLowerModel (energy : TrainingSpace 1 → ℝ) 1 ∧
      (0 : ℝ) < 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧
      (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (1 : ℝ) * (1 / 2) ≤ 1 ∧
      gradient (energy : TrainingSpace 1 → ℝ) 0 = 0 ∧ initial ≠ 0 := by
  refine ⟨energy_models.1, energy_models.2, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by rw [energy_gradient]; rfl, ?_⟩
  intro h
  have hi := congrArg (fun x : TrainingSpace 1 => x 0) h
  norm_num at hi

end Transformer.AMSGrad
