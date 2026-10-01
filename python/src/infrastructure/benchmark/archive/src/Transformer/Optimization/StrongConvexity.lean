/-
# Strong-convexity consequences for the training correction

Additional objective assumptions for arXiv:2502.16982, §2.1–2.2,
and arXiv:2602.02016v2, §2–4. Strong convexity is not asserted for
neural-network training losses. It identifies the regime in which the
corrected algorithms converge to a unique global minimizer.
-/

import Transformer.Optimization.Basic

open scoped InnerProductSpace

noncomputable section

namespace Transformer.Optimization

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- Strong-convexity lower model, expressed using the objective's actual
gradient. Additional assumption for the training extension of
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, Algorithm 1. -/
def StrongLowerModel (f : E → ℝ) (μ : ℝ) : Prop := ∀ x y,
  f x + ⟪gradient f x, y - x⟫_ℝ + μ / 2 * ‖y - x‖ ^ 2 ≤ f y

/-- A stationary point is a global minimizer, and the objective gap
controls distance while its gradient energy controls the objective gap.
These are derived from strong convexity, not supplied as convergence
assumptions about the optimizer. Source: training extension of
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2. -/
theorem strong_gap_bounds (f : E → ℝ) (μ : ℝ) (star x : E)
    (hf : StrongLowerModel f μ) (hμ : 0 < μ) (hstar : gradient f star = 0) :
    μ / 2 * ‖x - star‖ ^ 2 ≤ f x - f star ∧
      2 * μ * (f x - f star) ≤ ‖gradient f x‖ ^ 2 := by
  constructor
  · have h := hf star x
    rw [hstar, inner_zero_left, add_zero] at h
    linarith
  · have h := hf x star
    have hinner : ⟪gradient f x, star - x⟫_ℝ = -⟪gradient f x, x - star⟫_ℝ := by
      rw [← neg_sub x star, inner_neg_right]
    rw [hinner, norm_sub_rev star x] at h
    have hc := real_inner_le_norm (gradient f x) (x - star)
    have hm := mul_le_mul_of_nonneg_left h hμ.le
    have hcm := mul_le_mul_of_nonneg_left hc hμ.le
    nlinarith [sq_nonneg (‖gradient f x‖ - μ * ‖x - star‖)]

/-- The nonconstant quadratic is strongly convex,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, training extension. -/
theorem quadratic_strong : StrongLowerModel quadratic 1 := by
  intro x y
  rw [quadratic_gradient]
  simp only [quadratic, id_eq, RCLike.inner_apply, conj_trivial, Real.norm_eq_abs, sq_abs]
  nlinarith

/-- The strong-convexity and stationary-point assumptions are satisfiable,
arXiv:2502.16982, §2.1, and arXiv:2602.02016v2, §2, training extension. -/
example : StrongLowerModel quadratic 1 ∧ (0 : ℝ) < 1 ∧ gradient quadratic 0 = 0 := by
  exact ⟨quadratic_strong, by norm_num, by rw [quadratic_gradient]; rfl⟩

end Transformer.Optimization
