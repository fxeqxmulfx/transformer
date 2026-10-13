import Transformer.Grokking.EffectiveTheory.Section3_Conservation
import Mathlib.Analysis.Real.Sqrt

/-!
# Euclidean norm conservation does not transfer to native AdamW

Source: the native AdamW builder, `lab.infrastructure.optim` at 43d4d66,
and the installed PyTorch 2.14.1 algorithm in `torch/optim/adamw.py`,
equations for zero initial moments, bias correction and decoupled decay.
Comparison: Liu et al., arXiv:2205.10343v2, appendix conservation laws,
equation `eq:Z0_conservation` for Euclidean gradient flow.

This is exact real arithmetic for the documented first-step algorithm;
kernel rounding is not modeled. At the actual normalized-loss point
`(1, 2, 0)`, the Euclidean gradient is tangent to the norm sphere. AdamW's
first adaptive direction is not tangent, even with zero weight decay.
The squared norm has strictly negative learning-rate derivative at zero,
and the finite update at the experimental recipe's rate also lowers it.
This refutes an unrestricted conservation transfer, not grokking itself.
-/

namespace Transformer.Grokking.AdamW

open EffectiveTheory

/-- First bias-corrected direction from zero moment buffers. Source:
PyTorch 2.14.1 `AdamW` algorithm, first update; beta equals one is excluded
in the algebraic theorem below, as it is by the native parameter checks. -/
noncomputable def firstDirection (b1 b2 eps g : ℝ) : ℝ :=
  ((1 - b1) * g / (1 - b1)) /
    (Real.sqrt ((1 - b2) * g ^ 2 / (1 - b2)) + eps)

/-- One coordinate after decoupled decay and the adaptive first update.
Source: native AdamW builder at 43d4d66, PyTorch 2.14.1 algorithm;
the gradient is evaluated before applying decay. -/
noncomputable def firstUpdate (b1 b2 eps decay eta p g : ℝ) : ℝ :=
  (1 - eta * decay) * p - eta * firstDirection b1 b2 eps g

/-- Actual squared norm after applying that update to the normalized
parallelogram gradient. Source: Liu et al., arXiv:2205.10343v2, appendix
effective loss, combined with native AdamW at 43d4d66. -/
noncomputable def squaredNormAfterFirstStep (b1 b2 eps decay eta x y z : ℝ) : ℝ :=
  squaredNorm (firstUpdate b1 b2 eps decay eta x (quotientGradient x y z).1)
    (firstUpdate b1 b2 eps decay eta y (quotientGradient x y z).2.1)
    (firstUpdate b1 b2 eps decay eta z (quotientGradient x y z).2.2)

/-- Bias correction removes both beta factors on the first step, leaving
coordinate-wise normalization. Source: PyTorch 2.14.1 AdamW algorithm;
this exact algebra uses only the nonzero bias-correction denominators. -/
theorem firstDirection_eq (b1 b2 eps g : ℝ) (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) :
    firstDirection b1 b2 eps g = g / (|g| + eps) := by
  have hc1 : 1 - b1 ≠ 0 := by intro h; apply h1; linarith
  have hc2 : 1 - b2 ≠ 0 := by intro h; apply h2; linarith
  have hm : (1 - b1) * g / (1 - b1) = g := by field_simp
  have hv : (1 - b2) * g ^ 2 / (1 - b2) = g ^ 2 := by field_simp
  unfold firstDirection
  rw [hm, hv, Real.sqrt_sq_eq_abs]

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 := by norm_num

/-- A concrete nonzero normalized-loss point has a tangent Euclidean
gradient. Source: arXiv:2205.10343v2, appendix norm law; the gradient
here comes from actual quotient differentiation, not a frozen denominator. -/
theorem quotientGradient_test_point :
    quotientGradient 1 2 0 = (-48 / 25, 24 / 25, -6 / 5) ∧
      (quotientGradient 1 2 0).1 + 2 * (quotientGradient 1 2 0).2.1 = 0 := by
  have hZ : squaredNorm 1 2 0 ≠ 0 := by unfold squaredNorm; norm_num
  rw [quotientGradient_eq 1 2 0 hZ]
  unfold numerator EffectiveTheory.residual squaredNorm
  norm_num

/-- The adaptive radial component is positive at the same point. Source
comparison: arXiv:2205.10343v2, appendix norm law, versus the first-step
native AdamW algorithm at 43d4d66. Both experimental betas are retained. -/
theorem firstDirection_radial_formula (eps : ℝ) (he : 0 < eps) :
    firstDirection (9 / 10) (49 / 50) eps (quotientGradient 1 2 0).1 +
      2 * firstDirection (9 / 10) (49 / 50) eps (quotientGradient 1 2 0).2.1 =
        2 * (24 / 25 : ℝ) ^ 2 / ((24 / 25 + eps) * (48 / 25 + eps)) := by
  rw [quotientGradient_test_point.1]
  have hd (g : ℝ) := firstDirection_eq (9 / 10) (49 / 50) eps g (by norm_num) (by norm_num)
  simp only [hd]
  norm_num
  have hsmall : (24 / 25 : ℝ) + eps ≠ 0 := by positivity
  have hlarge : (48 / 25 : ℝ) + eps ≠ 0 := by positivity
  field_simp
  ring

example : 0 < (1 / 100000000 : ℝ) := by norm_num

/-- The norm derivative of the actual finite update with respect to its
learning rate. Source: native AdamW algorithm at 43d4d66; this is an
update derivative, not an assertion that AdamW follows a Euclidean ODE. -/
theorem first_step_norm_deriv (b1 b2 eps decay x y z : ℝ) :
    HasDerivAt (fun eta => squaredNormAfterFirstStep b1 b2 eps decay eta x y z)
      (-2 * (x * firstDirection b1 b2 eps (quotientGradient x y z).1 +
        y * firstDirection b1 b2 eps (quotientGradient x y z).2.1 +
        z * firstDirection b1 b2 eps (quotientGradient x y z).2.2 +
        decay * squaredNorm x y z)) 0 := by
  have hu (p g : ℝ) : HasDerivAt (fun eta => firstUpdate b1 b2 eps decay eta p g)
      (-decay * p - firstDirection b1 b2 eps g) 0 := by
    convert ((hasDerivAt_id (0 : ℝ)).mul_const
      (decay * p + firstDirection b1 b2 eps g)).const_sub p using 1
    · funext eta
      dsimp only [id]
      unfold firstUpdate
      ring
    · ring
  have h := (((hu x (quotientGradient x y z).1).pow 2).add
    ((hu y (quotientGradient x y z).2.1).pow 2)).add
    ((hu z (quotientGradient x y z).2.2).pow 2)
  convert h using 1
  · rfl
  · simp [firstUpdate, squaredNorm]
    ring

/-- Strictly negative norm derivative even when decay is zero. Source
counterexample to transferring arXiv:2205.10343v2's Euclidean norm law
to the unchanged native AdamW algorithm at 43d4d66. -/
theorem adamw_norm_conservation_counterexample (eps decay : ℝ)
    (he : 0 < eps) (hd : 0 ≤ decay) :
    HasDerivAt (fun eta => squaredNormAfterFirstStep (9 / 10) (49 / 50) eps decay eta 1 2 0)
      (-2 * (2 * (24 / 25 : ℝ) ^ 2 / ((24 / 25 + eps) * (48 / 25 + eps)) + 5 * decay)) 0 ∧
    -2 * (2 * (24 / 25 : ℝ) ^ 2 / ((24 / 25 + eps) * (48 / 25 + eps)) + 5 * decay) < 0 := by
  constructor
  · have h := first_step_norm_deriv (9 / 10) (49 / 50) eps decay 1 2 0
    have hc : decay * squaredNorm 1 2 0 = 5 * decay := by
      unfold squaredNorm
      norm_num
      ring
    simpa only [one_mul, zero_mul, add_zero, firstDirection_radial_formula eps he,
      hc] using h
  · have hp : 0 < 2 * (24 / 25 : ℝ) ^ 2 / ((24 / 25 + eps) * (48 / 25 + eps)) := by positivity
    nlinarith

example : 0 < (1 / 100000000 : ℝ) ∧ 0 ≤ (0 : ℝ) := by norm_num
example : 0 < (1 / 100000000 : ℝ) ∧ 0 ≤ (1 / 10 : ℝ) := by norm_num

/-- A finite first step at the experimental learning rate and epsilon
breaks the norm law with and without decay. Source: native AdamW at
43d4d66; exact reals, beta `(0.9, 0.98)`, rate `0.001`, epsilon `1e-8`.
This point is an effective-loss counterexample, not GPTMini's actual state. -/
theorem finite_native_step_counterexample :
    squaredNormAfterFirstStep (9 / 10) (49 / 50) (1 / 100000000) 0 (1 / 1000) 1 2 0 < 5 ∧
    squaredNormAfterFirstStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000) 1 2 0 < 5 := by
  unfold squaredNormAfterFirstStep firstUpdate
  rw [quotientGradient_test_point.1]
  have hd (g : ℝ) := firstDirection_eq (9 / 10) (49 / 50) (1 / 100000000) g
    (by norm_num) (by norm_num)
  simp only [hd]
  norm_num [squaredNorm]

end Transformer.Grokking.AdamW
