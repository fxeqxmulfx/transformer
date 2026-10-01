/-
# AMSGradW need not approach unregularized stationarity

arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. On the
smooth strongly convex loss `(w-1)^2/2`, actual training with positive
decay and momentum converges, but its original gradient does not vanish.
This refutes inheritance of the ordinary AMSGrad stationarity conclusion.
-/

import Transformer.AMSGradW.Minimum

open scoped Topology
open Filter

noncomputable section

namespace Transformer.AMSGradW

/-- A genuine shifted quadratic with minimizer at one.
Source: arXiv:1904.03590v4, §4, AMSGradW counterexample extension. -/
def shiftedEnergy (x : TrainingSpace 1) : ℝ :=
  Optimization.energy (x - WithLp.toLp 2 (fun _ => 1))

/-- The derivative of the actual nonconstant counterexample loss.
Source: arXiv:1904.03590v4, §4, AMSGradW counterexample extension. -/
theorem shiftedEnergy_hasGradientAt (x : TrainingSpace 1) :
    HasGradientAt shiftedEnergy (x - WithLp.toLp 2 (fun _ => 1)) x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  have h := (Optimization.energy_hasGradientAt
    (x - WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ)))).hasFDerivAt.comp x
      ((hasFDerivAt_id x).sub_const (WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))))
  convert h using 1 <;> ext y <;> rfl

/-- The true gradient is `w-1`, not the weighted-decay residual.
Source: arXiv:1904.03590v4, §4, AMSGradW counterexample extension. -/
theorem shiftedEnergy_gradient : gradient shiftedEnergy =
    fun x => x - WithLp.toLp 2 (fun _ => 1) :=
  gradient_eq shiftedEnergy_hasGradientAt

/-- The loss is differentiable with maximum-norm Lipschitz constant one
and a strong-convexity constant one, and has a finite lower bound.
Source: arXiv:1904.03590v4, §4, AMSGradW counterexample extension. -/
theorem shiftedEnergy_models : Differentiable ℝ shiftedEnergy ∧
    LipschitzGradient shiftedEnergy 1 ∧ Optimization.StrongLowerModel shiftedEnergy 1 ∧
      (∀ x, 0 ≤ shiftedEnergy x) := by
  refine ⟨fun x => (shiftedEnergy_hasGradientAt x).differentiableAt, ?_, ?_,
    fun x => div_nonneg (sq_nonneg _) (by norm_num)⟩
  · intro x y
    have heq : coordinateGradient shiftedEnergy x - coordinateGradient shiftedEnergy y = x - y := by
      ext i
      simp only [coordinateGradient, shiftedEnergy_gradient, PiLp.sub_apply, Pi.sub_apply]
      ring
    rw [heq, one_mul]
  · intro x y
    have h := Optimization.energy_models.2
      (x - WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ)))
      (y - WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ)))
    have heq : (y - WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) -
        (x - WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ))) = y - x := by abel
    rw [Optimization.energy_gradient, id_eq, heq] at h
    simpa only [shiftedEnergy, shiftedEnergy_gradient] using h

/-- Actual AMSGradW with nonzero momentum and both moment histories.
Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW counterexample extension. -/
def shiftedRun : ℕ → TrainingState 1 :=
  trainingRun (1 / 4) 1 2 (9 / 10) (1 / 2) shiftedEnergy 0

/-- The true original gradient converges to a nonzero value, although
actual AMSGradW weights and losses converge on a smooth strongly convex
loss with admissible positive learning rate, epsilon, decay and momentum.
The weighted equilibrium is `w*=1/(1+2D)`, with `D>=1`.
Source: arXiv:1904.03590v4, §4, AMSGradW counterexample extension. -/
theorem original_stationarity_counterexample :
    ∃ (D : ℝ) (star : TrainingSpace 1), 1 ≤ D ∧
      star 0 = 1 / (1 + 2 * D) ∧
      Tendsto (fun t => (shiftedRun t).position) atTop (𝓝 star) ∧
      Tendsto (fun t => shiftedEnergy (shiftedRun t).position) atTop (𝓝 (shiftedEnergy star)) ∧
      Tendsto (fun t => gradient shiftedEnergy (shiftedRun t).position 0)
        atTop (𝓝 (star 0 - 1)) ∧
      ¬ Tendsto (fun t => gradient shiftedEnergy (shiftedRun t).position 0) atTop (𝓝 0) := by
  obtain ⟨D, star, hD, hx, hm, hmetric, hloss, heq⟩ := trainingRun_convergence
    (1 / 4) 1 2 (9 / 10) (1 / 2) 1 shiftedEnergy 0 shiftedEnergy_models.1
    shiftedEnergy_models.2.1 (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have hh := heq 0
  rw [shiftedEnergy_gradient] at hh
  change star 0 - 1 + 2 * D 0 * star 0 = 0 at hh
  have hs : star 0 = 1 / (1 + 2 * D 0) := by
    apply (eq_div_iff (show 1 + 2 * D 0 ≠ 0 by linarith [hD 0])).mpr
    nlinarith
  have hg : Tendsto (fun t => gradient shiftedEnergy (shiftedRun t).position 0)
      atTop (𝓝 (star 0 - 1)) := by
    rw [shiftedEnergy_gradient]
    simpa only [PiLp.sub_apply, Function.comp_def, shiftedRun] using
      ((PiLp.continuous_apply 2 (fun _ : Fin 1 => ℝ) 0).tendsto star |>.comp hx).sub_const 1
  refine ⟨D 0, star, hD 0, hs, hx, hloss, hg, ?_⟩
  intro hzero
  have hn : star 0 - 1 = 0 := tendsto_nhds_unique hg hzero
  nlinarith [hD 0]

end Transformer.AMSGradW
