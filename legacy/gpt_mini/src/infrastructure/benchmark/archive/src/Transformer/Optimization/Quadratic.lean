/-
# A genuine matrix-training objective witnessing the assumptions

The nonconstant objective `‖x‖²/2` works in every real Hilbert space,
including rectangular matrices equipped with their Frobenius geometry.
Source: convergence extension of arXiv:2502.16982, §2.1–2.2,
and arXiv:2602.02016v2, §2–4.
-/

import Transformer.Optimization.StrongConvexity
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped InnerProductSpace

noncomputable section

namespace Transformer.Optimization

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- A nonconstant quadratic loss on the optimizer's parameter space.
Source: training extension of arXiv:2502.16982, §2.1, and
arXiv:2602.02016v2, §2, Algorithm 1. -/
def energy (x : E) : ℝ := ‖x‖ ^ 2 / 2

/-- The loss has the current parameter as its genuine gradient.
Source: training extension of arXiv:2502.16982, §2.1, and
arXiv:2602.02016v2, §2, Algorithm 1. -/
theorem energy_hasGradientAt (x : E) : HasGradientAt energy x x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  convert ((hasStrictFDerivAt_norm_sq x).hasFDerivAt.const_smul (1 / 2 : ℝ)) using 1
  · ext y
    simp only [energy, Pi.smul_apply, smul_eq_mul]
    ring
  · ext y
    simp [InnerProductSpace.toDual_apply_apply]

/-- Actual gradient of the quadratic witness,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, training extension. -/
theorem energy_gradient : gradient (energy : E → ℝ) = id :=
  gradient_eq energy_hasGradientAt

/-- The quadratic has smoothness constant one and strong convexity
constant one. Source: training extension of arXiv:2502.16982, §2.1,
and arXiv:2602.02016v2, §2–4. -/
theorem energy_models : SmoothObjective (energy : E → ℝ) 1 ∧
    StrongLowerModel (energy : E → ℝ) 1 := by
  have hmodel : ∀ x y : E,
      energy y = energy x + ⟪x, y - x⟫_ℝ + 1 / 2 * ‖y - x‖ ^ 2 := by
    intro x y
    simp only [energy, inner_sub_right, real_inner_self_eq_norm_sq, norm_sub_sq_real]
    rw [real_inner_comm y x]
    ring
  refine ⟨⟨fun x => (energy_hasGradientAt x).differentiableAt, ?_⟩, ?_⟩
  · intro x y
    rw [energy_gradient]
    exact (hmodel x y).le
  · intro x y
    rw [energy_gradient]
    exact (hmodel x y).ge

end Transformer.Optimization
