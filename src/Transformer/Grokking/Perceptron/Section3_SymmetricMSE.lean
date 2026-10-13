import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Convex.Function
import Mathlib.Tactic

/-!
# An optimized convex perceptron, not a prescribed accuracy curve

Source: Žunkovič and Ilievski, arXiv:2210.15435v1, section 3,
the affine perceptron and mean-squared-error gradient flow.
Specialization: one dimension, the two training examples `(1, 1)` and
`(-1, -1)`, and zero regularization. This is not the paper's exponential
population model, its random-sample theorem, or AdamW on a transformer.

The loss is defined from the actual two predictions. Its quadratic form,
convexity and partial derivatives are proved. The exponential trajectories
then solve the negative-gradient equations of that same loss. Delayed
held-out classification is derived in the next module, not put in the flow.
-/

namespace Transformer.Grokking.Perceptron

/-- The affine preactivation in arXiv:2210.15435v1, section 3, equation
`eq:model`, specialized to one input coordinate. -/
def score (w b x : ℝ) : ℝ := w * x + b

/-- Mean of the half-squared residuals on `(1, 1)` and `(-1, -1)`.
Source: arXiv:2210.15435v1, section 3, the risk equation with `N = 1`
and both regularization coefficients zero. -/
noncomputable def trainMSE (w b : ℝ) : ℝ :=
  ((score w b 1 - 1) ^ 2 + (score w b (-1) + 1) ^ 2) / 4

/-- Exact closed form of the actual sample loss, derived rather than
assumed. Source specialization: arXiv:2210.15435v1, section 3. -/
theorem trainMSE_eq_quadratic (w b : ℝ) :
    trainMSE w b = ((w - 1) ^ 2 + b ^ 2) / 2 := by
  unfold trainMSE score
  ring

/-- The quadratic loss has the exact strong Jensen remainder in its two
coordinates. Source: the section 3 MSE in arXiv:2210.15435v1, for the
explicit symmetric two-sample specialization used in this module. -/
theorem trainMSE_strongJensen_identity (a w b v c : ℝ) :
    trainMSE (a * w + (1 - a) * v) (a * b + (1 - a) * c) =
      a * trainMSE w b + (1 - a) * trainMSE v c -
        a * (1 - a) / 2 * ((w - v) ^ 2 + (b - c) ^ 2) := by
  simp only [trainMSE_eq_quadratic]
  ring

/-- Convexity is a property of the loss on jointly trained weight and
bias, not a frozen parameter premise. Source specialization:
arXiv:2210.15435v1, section 3; stronger information is in the preceding
exact Jensen identity. -/
theorem trainMSE_convex :
    ConvexOn ℝ Set.univ (fun p : ℝ × ℝ => trainMSE p.1 p.2) := by
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  change trainMSE (a * x.1 + b * y.1) (a * x.2 + b * y.2) ≤
    a * trainMSE x.1 x.2 + b * trainMSE y.1 y.2
  have hb' : b = 1 - a := by linarith
  rw [hb', trainMSE_strongJensen_identity]
  have hprod : 0 ≤ a * (1 - a) := mul_nonneg ha (by linarith)
  have hsquares : 0 ≤ (x.1 - y.1) ^ 2 + (x.2 - y.2) ^ 2 := by positivity
  nlinarith [mul_nonneg hprod hsquares]

/-- Actual weight derivative of the training objective. Source:
arXiv:2210.15435v1, section 3, negative-gradient dynamics, specialized
to the symmetric training samples. -/
theorem trainMSE_weight_deriv (w b : ℝ) :
    HasDerivAt (fun u => trainMSE u b) (w - 1) w := by
  have h := ((((hasDerivAt_id w).sub_const 1).pow 2).add_const (b ^ 2)).div_const 2
  convert h using 1
  · funext u
    exact trainMSE_eq_quadratic u b
  · norm_num

/-- Actual bias derivative of the same training objective. Source:
arXiv:2210.15435v1, section 3, the symmetric two-sample specialization. -/
theorem trainMSE_bias_deriv (w b : ℝ) :
    HasDerivAt (fun u => trainMSE w u) b b := by
  have h := (((hasDerivAt_id b).pow 2).const_add ((w - 1) ^ 2)).div_const 2
  convert h using 1
  · funext u
    exact trainMSE_eq_quadratic w u
  · norm_num

/-- Candidate weight trajectory from an arbitrary starting weight.
Source: solving the section 3 gradient flow in arXiv:2210.15435v1 for
the explicit symmetric sample loss; verified below. -/
noncomputable def flowWeight (w₀ t : ℝ) : ℝ :=
  1 + (w₀ - 1) * Real.exp (-t)

/-- Candidate bias trajectory from an arbitrary starting bias.
Source: the same section 3 gradient-flow specialization. -/
noncomputable def flowBias (b₀ t : ℝ) : ℝ := b₀ * Real.exp (-t)

/-- The weight trajectory solves the negative weight-gradient equation
of `trainMSE`; its derivative is not supplied as a hypothesis. Source:
arXiv:2210.15435v1, section 3, specialized as stated above. -/
theorem flowWeight_deriv (w₀ t : ℝ) :
    HasDerivAt (flowWeight w₀) (-(flowWeight w₀ t - 1)) t := by
  have he := (Real.hasDerivAt_exp (-t)).comp t (hasDerivAt_id t).neg
  have h := (he.const_mul (w₀ - 1)).const_add 1
  convert h using 1
  · rfl
  · unfold flowWeight
    ring

/-- The bias trajectory solves the negative bias-gradient equation of
the actual sample loss. Source: arXiv:2210.15435v1, section 3,
specialized as stated above. -/
theorem flowBias_deriv (b₀ t : ℝ) :
    HasDerivAt (flowBias b₀) (-flowBias b₀ t) t := by
  have he := (Real.hasDerivAt_exp (-t)).comp t (hasDerivAt_id t).neg
  convert he.const_mul b₀ using 1
  · rfl
  · unfold flowBias
    ring

/-- These trajectories have the stated initial data. Source: the section
3 gradient-flow initial-value problem in arXiv:2210.15435v1, specialized
to `trainMSE`, without an assumed convergence property. -/
theorem flow_initial (w₀ b₀ : ℝ) :
    flowWeight w₀ 0 = w₀ ∧ flowBias b₀ 0 = b₀ := by
  unfold flowWeight flowBias
  norm_num

/-- A fitted initial weight leaves a genuinely decreasing quadratic bias
loss. Source: arXiv:2210.15435v1, section 3, the specialization above.
The next module uses this trajectory to derive its classification delay. -/
theorem trainMSE_along_flow (b₀ t : ℝ) :
    trainMSE (flowWeight 1 t) (flowBias b₀ t) =
      b₀ ^ 2 * Real.exp (-t) ^ 2 / 2 := by
  rw [trainMSE_eq_quadratic]
  unfold flowWeight flowBias
  ring

/-- Training loss strictly decreases throughout the delayed-decision
trajectory, including its initial perfect-training-classification plateau.
Source specialization: arXiv:2210.15435v1, section 3. This is a statement
about the sample loss, not a surrogate test-accuracy curve. -/
theorem trainMSE_flow_strictAnti :
    StrictAnti (fun t => trainMSE (flowWeight 1 t) (flowBias (1 / 2) t)) := by
  intro s t hst
  change trainMSE (flowWeight 1 t) (flowBias (1 / 2) t) <
    trainMSE (flowWeight 1 s) (flowBias (1 / 2) s)
  rw [trainMSE_along_flow, trainMSE_along_flow]
  have he : Real.exp (-t) < Real.exp (-s) := Real.exp_lt_exp.mpr (by linarith)
  have hs := (sq_lt_sq₀ (Real.exp_pos (-t)).le (Real.exp_pos (-s)).le).mpr he
  nlinarith

end Transformer.Grokking.Perceptron
