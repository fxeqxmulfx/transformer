import Transformer.Grokking.AdamW.LossDirection

/-!
# First-step amplification of an error in a zero gradient coordinate

Source: native PyTorch 2.14.1 AdamW first-step equations at lab commit
43d4d66. Comparison: the exact zero first coordinate in the centered
effective quotient of Liu et al., arXiv:2205.10343v2, eq:l_eff, and the
CPU float64 sanity check recorded in grokking_lean_plan.md on 2026-10-08.

An approximate gradient is supplied as a real input perturbation. Derive
the difference between the actual two native coordinate updates; no
floating-point autograd program or error-generation mechanism is assumed
verified. Decay and the current parameter cancel from this difference.
The amplification depends on epsilon, saturates at the learning rate,
and can reach half a step with arbitrarily small absolute error when
epsilon is allowed to decrease with it. This is a conditional transfer
bound, not a claim that every tiny numerical gradient causes such motion.
-/

namespace Transformer.Grokking.AdamW

/-- Actual discrepancy of two finite native coordinate updates, one
using the correct zero gradient and one using a supplied error. Source:
PyTorch 2.14.1 AdamW equations at 43d4d66; all update inputs are retained. -/
noncomputable def zeroGradientUpdateError
    (b1 b2 eps decay eta p delta : ℝ) : ℝ :=
  |firstUpdate b1 b2 eps decay eta p delta - firstUpdate b1 b2 eps decay eta p 0|

/-- Exact coordinate discrepancy, including bias correction and decay
cancellation. Source: native first-step AdamW at 43d4d66. The supplied
gradient error is not identified with a verified floating-point trace. -/
theorem zero_gradient_update_error_exact (b1 b2 eps decay eta p delta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) :
    zeroGradientUpdateError b1 b2 eps decay eta p delta =
      |eta| * |delta| / (|delta| + eps) := by
  unfold zeroGradientUpdateError firstUpdate
  rw [firstDirection_eq b1 b2 eps delta h1 h2, firstDirection_eq b1 b2 eps 0 h1 h2]
  have hd : 0 < |delta| + eps := by positivity
  have hid : (1 - eta * decay) * p - eta * (delta / (|delta| + eps)) -
      ((1 - eta * decay) * p - eta * (0 / (|0| + eps))) =
      -eta * delta / (|delta| + eps) := by ring
  rw [hid, abs_div, abs_mul, abs_neg, abs_of_pos hd]

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 < (1 / 100000000 : ℝ) := by norm_num

/-- The error contribution is bounded by one learning-rate magnitude.
Source: native bias-corrected first step at 43d4d66, with positive
epsilon. This saturation bound does not force the contribution to be small. -/
theorem zero_gradient_error_le_rate (b1 b2 eps decay eta p delta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) :
    zeroGradientUpdateError b1 b2 eps decay eta p delta ≤ |eta| := by
  rw [zero_gradient_update_error_exact _ _ _ _ _ _ _ h1 h2 he]
  apply (div_le_iff₀ (by positivity : 0 < |delta| + eps)).mpr
  have hp : 0 ≤ |eta| * eps := by positivity
  nlinarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 < (1 : ℝ) := by norm_num

/-- A conditional gradient-error transfer bound with explicit epsilon
dependence. Source: native first-step equations at 43d4d66; it concerns
a truly zero coordinate and does not assert full-gradient kernel accuracy. -/
theorem zero_gradient_error_le_epsilon_bound (b1 b2 eps decay eta p delta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) :
    zeroGradientUpdateError b1 b2 eps decay eta p delta ≤ |eta| * |delta| / eps := by
  rw [zero_gradient_update_error_exact _ _ _ _ _ _ _ h1 h2 he]
  apply (div_le_div_iff₀ (by positivity : 0 < |delta| + eps) he).mpr
  have hp : 0 ≤ |eta| * |delta| ^ 2 := by positivity
  nlinarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ 0 < (1 / 100000000 : ℝ) := by norm_num

/-- Errors comparable with epsilon give a step fraction determined by
their ratio. Source: native first-step normalization at 43d4d66;
small absolute magnitude alone is not the relevant condition. -/
theorem zero_gradient_error_at_epsilon_scale (b1 b2 eps decay eta p rho : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hr : 0 ≤ rho) :
    zeroGradientUpdateError b1 b2 eps decay eta p (rho * eps) =
      |eta| * rho / (rho + 1) := by
  rw [zero_gradient_update_error_exact _ _ _ _ _ _ _ h1 h2 he,
    abs_of_nonneg (by positivity : 0 ≤ rho * eps)]
  have hen : eps ≠ 0 := by positivity
  have hden : rho + 1 ≠ 0 := by positivity
  field_simp

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 ≤ (1 : ℝ) := by norm_num

/-- A verified gradient-error budget relative to epsilon bounds the
fraction of a step that the zero coordinate can move. Source: native
first-step equations at 43d4d66; a kernel error bound would still have
to be proved separately to apply this conditional certificate. -/
theorem zero_gradient_error_relative_budget (b1 b2 eps decay eta p delta rho : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hr : 0 ≤ rho)
    (hd : |delta| ≤ rho * eps) :
    zeroGradientUpdateError b1 b2 eps decay eta p delta ≤ |eta| * rho / (rho + 1) := by
  rw [zero_gradient_update_error_exact _ _ _ _ _ _ _ h1 h2 he]
  apply (div_le_div_iff₀ (by positivity : 0 < |delta| + eps)
    (by positivity : 0 < rho + 1)).mpr
  have hp : 0 ≤ |eta| * (rho * eps - |delta|) := by positivity
  nlinarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 : ℝ) ∧ 0 ≤ (1 / 100 : ℝ) ∧ |(1 / 100 : ℝ)| ≤ (1 / 100 : ℝ) * 1 := by norm_num

/-- An error at least as large as epsilon moves at least half of the
learning-rate magnitude. Source: native first-step equations at 43d4d66;
the comparison with epsilon is an explicit hypothesis. -/
theorem zero_gradient_error_ge_half_rate (b1 b2 eps decay eta p delta : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hd : eps ≤ |delta|) :
    |eta| / 2 ≤ zeroGradientUpdateError b1 b2 eps decay eta p delta := by
  rw [zero_gradient_update_error_exact _ _ _ _ _ _ _ h1 h2 he]
  apply (le_div_iff₀ (by positivity : 0 < |delta| + eps)).mpr
  have hp : 0 ≤ |eta| * (|delta| - eps) := by positivity
  nlinarith

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ (1 / 100000000 : ℝ) ≤ |(1 / 100000000 : ℝ)| := by norm_num

/-- Arbitrarily small supplied errors can move half a fixed positive
step when epsilon varies with them. Source: the numerical-invariance
question for native AdamW at 43d4d66, raised by the exact centered
gradient in arXiv:2205.10343v2, eq:l_eff. Epsilon is not held fixed. -/
theorem arbitrarily_small_error_moves_half_step (b1 b2 decay eta p tolerance : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (heta : 0 < eta) (htol : 0 < tolerance) :
    ∃ eps delta : ℝ, 0 < eps ∧ 0 < delta ∧ delta < tolerance ∧
      eta / 2 ≤ zeroGradientUpdateError b1 b2 eps decay eta p delta := by
  have he : 0 < tolerance / 2 := by positivity
  refine ⟨tolerance / 2, tolerance / 2, he, he, by linarith, ?_⟩
  have h := zero_gradient_error_ge_half_rate b1 b2 (tolerance / 2) decay eta p
    (tolerance / 2) h1 h2 he (by rw [abs_of_pos he])
  simpa [abs_of_pos heta] using h

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 1000 : ℝ) ∧ 0 < (1 / 1000000000000 : ℝ) := by norm_num

/-- Quantified coordinate effect of the supplied float64-check error
`-2^-31` at the experiment's hyperparameters. Source: native AdamW
at 43d4d66 and the 2026-10-08 sanity check in grokking_lean_plan.md.
This verifies the effect of that rational input, not its autograd origin. -/
theorem native_nearzero_coordinate_error :
    (1 / 25000 : ℝ) < zeroGradientUpdateError (9 / 10) (49 / 50)
      (1 / 100000000) (1 / 10) (1 / 1000) (1 / 1000000) (-1 / 2147483648) ∧
    zeroGradientUpdateError (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (1 / 1000) (1 / 1000000) (-1 / 2147483648) < 1 / 20000 := by
  rw [zero_gradient_update_error_exact _ _ _ _ _ _ _
    (by norm_num) (by norm_num) (by norm_num)]
  norm_num

end Transformer.Grokking.AdamW
