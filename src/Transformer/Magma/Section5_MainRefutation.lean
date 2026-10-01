/-
# Refutation of the manuscript's uncorrected finite-horizon theorem

Formalization of arXiv:2602.15322v1, Section 5, theorem:main_result,
eq:magma:conv, and Appendix A.4. The source's assumptions are satisfied
by a lower-bounded two-block polynomial with exact gradients and zero noise.
Its published one-step horizon estimate nevertheless fails.
-/

import Transformer.Magma.Section5_SmoothnessRefutation
import Transformer.Magma.Section5_AlignmentRefutation

noncomputable section

namespace Transformer.Magma

/-- At the first dense step from zero momentum, g=2 and beta=0.9
give current momentum=1/5. With previous scale=1/2, the actual alignment
EMA is 21/40, even though both gradient and momentum are positively
aligned. Source: arXiv:2602.15322v1, Sections 2--3 and Algorithm 1. -/
theorem positive_momentum_damping :
    damping (1 / Real.log 3) (1 / 2) (1 / 5 : ℝ) 2 = 21 / 40 := by
  norm_num [damping, cosine, div_div, Real.sigmoid, Real.exp_neg,
    Real.exp_log (by norm_num : (0 : ℝ) < 3)]

/-- The advertised stationarity bound is false on the source's stated
domain. Choose T=1, eta=1, p=1/2, both L_b=1, zero gradient noise, and
theta_0=(1,1). Dense momentum starts at zero, beta=0.9, and previous
EMA scales are 1/2, giving both actual s_b=21/40. With gamma=1 the true-gradient events have
probability one, so both proof-defined alpha_b=3/4 and c_b=0.
Thus alpha_eff=9/16, effective L_max=(21/40)^2/(1/2)=441/800, and the printed
step bound admits eta=1. The left side is 8, while the right side is 64/9.
This refutes the original statement, not a corrected globally smooth
version. Source: arXiv:2602.15322v1, Section 5 and Appendix A.4. -/
theorem paper_main_result_counterexample :
    CoordinateSmooth coupledQuadratic 1 1 ∧
      (∀ x y, 0 ≤ coupledQuadratic x y) ∧
      coupledQuadratic 0 0 = 0 ∧
      let gx := deriv (fun z => coupledQuadratic z 1) 1
      let gy := deriv (fun z => coupledQuadratic 1 z) 1
      let α := Real.sigmoid (1 / (1 / Real.log 3))
      let s := damping (1 / Real.log 3) (1 / 2) (1 / 5 : ℝ) 2
      let A := ((α * gx) ^ 2 + (α * gy) ^ 2) / (gx ^ 2 + gy ^ 2)
      (0 : ℝ) < 1 ∧ 1 ≤ A / (s ^ 2 / (1 / 2)) ∧
        gx ^ 2 + gy ^ 2 > 2 * (coupledQuadratic 1 1 - 0) / (1 * A * 1) := by
  refine ⟨coupledQuadratic_coordinateSmooth.1, coupledQuadratic_coordinateSmooth.2,
    by norm_num [coupledQuadratic], ?_⟩
  dsimp only
  rw [coupledQuadratic_deriv_x, coupledQuadratic_deriv_y, positive_momentum_damping]
  norm_num [coupledQuadratic, div_div, Real.sigmoid, Real.exp_neg,
    Real.exp_log (by norm_num : (0 : ℝ) < 3)]

/-- The counterexample uses a genuine first dense moment step from zero
momentum and variance, with an admissible initial scale 1/2. Source:
arXiv:2602.15322v1, Section 2, first-moment recurrence, and Algorithm 1. -/
theorem first_dense_momentum_half_scale (direction : ℝ) :
    let after := scalarMagmaStep (9 / 10) (999 / 1000) (1 / Real.log 3)
      2 direction false ⟨1, 0, 0, 1 / 2⟩
    after.momentum = 1 / 5 ∧ after.scale = 21 / 40 := by
  dsimp [scalarMagmaStep]
  norm_num only
  simp only [true_and]
  exact positive_momentum_damping

/-- The claimed SGD specialization also fails when the source's
coordinate constants are treated as a global smoothness constant.
At theta_0=(1,1), exact gradients, L_b=1, eta=1 and T=1, the stationarity
left side is 8 while its advertised zero-noise right side is 4.
Source: arXiv:2602.15322v1, Section 5, paragraph after theorem:main_result. -/
theorem sgd_coordinate_constants_counterexample :
    CoordinateSmooth coupledQuadratic 1 1 ∧
      let gx := deriv (fun z => coupledQuadratic z 1) 1
      let gy := deriv (fun z => coupledQuadratic 1 z) 1
      gx ^ 2 + gy ^ 2 > 2 * (coupledQuadratic 1 1 - 0) / (1 * 1) := by
  refine ⟨coupledQuadratic_coordinateSmooth.1, ?_⟩
  dsimp only
  rw [coupledQuadratic_deriv_x, coupledQuadratic_deriv_y]
  norm_num [coupledQuadratic]

/-- The vanilla-SGD comparison also contains an unsquared-gradient typo:
it subtracts eta*norm(grad) instead of eta*norm(grad)^2. Even the actual
globally smooth quadratic refutes that displayed bound at x=eta=1/2.
Source: arXiv:2602.15322v1, Section 5, paragraph after
eq:one_step_descent_magma. -/
theorem vanilla_sgd_unsquared_gradient_counterexample :
    let loss := Transformer.Optimization.quadratic
    let g := gradient loss (1 / 2 : ℝ)
    loss ((1 / 2 : ℝ) - (1 / 2) * g) >
      loss (1 / 2) - (1 / 2) * ‖g‖ + (1 / 2) ^ 2 / 2 * ‖g‖ ^ 2 := by
  norm_num [Transformer.Optimization.quadratic_gradient,
    Transformer.Optimization.quadratic, Real.norm_eq_abs]

/-- Loewner order S<=I cannot order the mixed bilinear expression when
the stochastic gradient opposes the true gradient. Refuted auxiliary
claim: g^T S grad<=g^T grad almost surely solely because S<=I.
Source: arXiv:2602.15322v1, Section 5, paragraph after
eq:one_step_descent_magma. -/
theorem loewner_bilinear_order_counterexample :
    (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
      (-1 : ℝ) * (1 / 2) * 1 > (-1 : ℝ) * 1 := by norm_num

end Transformer.Magma
