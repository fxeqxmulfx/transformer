import Transformer.Grokking.AdamW.FirstStep

/-!
# Actual effective-loss derivative under the native first update

Source: Liu et al., arXiv:2205.10343v2, section 3.2, equation eq:l_eff,
and appendix conservation laws; the exact first-step AdamW equations in
PyTorch 2.14.1, torch/optim/adamw.py, used by lab.infrastructure.optim
at 43d4d66. Initial moment buffers are zero and decay is decoupled.

Differentiate the actual normalized loss along the finite update as its
learning rate varies. This is not an ODE for AdamW. The loss gradient is
the already verified quotient gradient, including denominator derivatives.
Its radial identity cancels the decay contribution to this derivative.
Bias correction then gives a negative weighted gradient-square sum.

The one-constraint numerator's averaging factors remain suppressed, as
in the existing effective model. Under AdamW, restoring a positive factor
also changes effective epsilon; it cannot generally be identified with
a uniform time rescaling. No GPTMini loss, floating-point kernel or
later nonzero-moment state is identified with this scalar calculation.
-/

namespace Transformer.Grokking.AdamW

open Transformer.Grokking.EffectiveTheory

/-- The existing effective loss evaluated at the actual finite first
update. Source: arXiv:2205.10343v2, eq:l_eff, with the documented native
AdamW first update at 43d4d66; the gradient is taken before decay. -/
noncomputable def lossAfterFirstStep (b1 b2 eps decay eta x y z : ℝ) : ℝ :=
  normalizedLoss (firstUpdate b1 b2 eps decay eta x (quotientGradient x y z).1)
    (firstUpdate b1 b2 eps decay eta y (quotientGradient x y z).2.1)
    (firstUpdate b1 b2 eps decay eta z (quotientGradient x y z).2.2)

/-- Weighted gradient energy obtained from the bias-corrected first
direction. Source: PyTorch 2.14.1 AdamW first-step equations. Positivity
requires positive epsilon and a nonzero gradient; neither is defined in. -/
noncomputable def firstGradientEnergy (eps : ℝ) (g : ℝ × ℝ × ℝ) : ℝ :=
  g.1 ^ 2 / (|g.1| + eps) + g.2.1 ^ 2 / (|g.2.1| + eps) +
    g.2.2 ^ 2 / (|g.2.2| + eps)

/-- Actual directional derivative of the normalized quotient along a
linear embedding path. Source: arXiv:2205.10343v2, section 3.2,
eq:l_eff and appendix quotient rule; all coordinate velocities are free. -/
theorem normalizedLoss_linear_path_deriv (x y z u v w : ℝ)
    (hZ : squaredNorm x y z ≠ 0) :
    HasDerivAt (fun eta => normalizedLoss (x + eta * u) (y + eta * v) (z + eta * w))
      ((quotientGradient x y z).1 * u + (quotientGradient x y z).2.1 * v +
        (quotientGradient x y z).2.2 * w) 0 := by
  have hl (a b : ℝ) : HasDerivAt (fun eta => a + eta * b) b 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).mul_const b).const_add a
  have hr := ((hl x u).add (hl z w)).sub ((hl y v).const_mul 2)
  have hn := hr.pow 2
  have hz := (((hl x u).pow 2).add ((hl y v).pow 2)).add ((hl z w).pow 2)
  have h := hn.div hz (by simpa [squaredNorm] using hZ)
  rw [quotientGradient_eq x y z hZ]
  convert h using 1
  · rfl
  · dsimp
    unfold numerator EffectiveTheory.residual squaredNorm
    field_simp
    ring

example : squaredNorm 1 (-3) 2 ≠ 0 := by norm_num [squaredNorm]

/-- The finite first update is an affine learning-rate path. Source:
the decoupled native AdamW update at 43d4d66, PyTorch 2.14.1 equations. -/
theorem firstUpdate_as_linear_path (b1 b2 eps decay eta p g : ℝ) :
    firstUpdate b1 b2 eps decay eta p g =
      p + eta * (-decay * p - firstDirection b1 b2 eps g) := by
  unfold firstUpdate
  ring

/-- A zero learning rate evaluates the unchanged effective loss. Source:
the finite native update at 43d4d66 and arXiv:2205.10343v2, eq:l_eff. -/
theorem loss_after_zero_rate (b1 b2 eps decay x y z : ℝ) :
    lossAfterFirstStep b1 b2 eps decay 0 x y z = normalizedLoss x y z := by
  unfold lossAfterFirstStep firstUpdate
  norm_num

/-- Decoupled decay cancels from the actual first loss derivative by
the verified quotient-gradient radial identity. Source: native AdamW
at 43d4d66 and arXiv:2205.10343v2, appendix conservation laws.
This concerns the learning-rate derivative at zero, not finite-step decay. -/
theorem first_step_loss_deriv (b1 b2 eps decay x y z : ℝ)
    (hZ : squaredNorm x y z ≠ 0) :
    HasDerivAt (fun eta => lossAfterFirstStep b1 b2 eps decay eta x y z)
      (-((quotientGradient x y z).1 * firstDirection b1 b2 eps (quotientGradient x y z).1 +
        (quotientGradient x y z).2.1 * firstDirection b1 b2 eps (quotientGradient x y z).2.1 +
        (quotientGradient x y z).2.2 * firstDirection b1 b2 eps (quotientGradient x y z).2.2)) 0 := by
  have h := normalizedLoss_linear_path_deriv x y z
    (-decay * x - firstDirection b1 b2 eps (quotientGradient x y z).1)
    (-decay * y - firstDirection b1 b2 eps (quotientGradient x y z).2.1)
    (-decay * z - firstDirection b1 b2 eps (quotientGradient x y z).2.2) hZ
  convert h using 1
  · funext eta
    unfold lossAfterFirstStep
    rw [firstUpdate_as_linear_path, firstUpdate_as_linear_path, firstUpdate_as_linear_path]
  · have hr := quotientGradient_radial_zero x y z hZ
    linear_combination decay * hr

example : squaredNorm 1 2 0 ≠ 0 := by norm_num [squaredNorm]

/-- Bias correction makes the first loss derivative minus the explicit
weighted gradient energy. Source: PyTorch 2.14.1 first-step equations,
combined with arXiv:2205.10343v2, eq:l_eff. This real-division identity
does not supply positivity when epsilon is nonpositive. -/
theorem first_step_loss_deriv_eq_energy (b1 b2 eps decay x y z : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (hZ : squaredNorm x y z ≠ 0) :
    HasDerivAt (fun eta => lossAfterFirstStep b1 b2 eps decay eta x y z)
      (-firstGradientEnergy eps (quotientGradient x y z)) 0 := by
  have h := first_step_loss_deriv b1 b2 eps decay x y z hZ
  have hd (g : ℝ) := firstDirection_eq b1 b2 eps g h1 h2
  simp only [hd] at h
  convert h using 1
  unfold firstGradientEnergy
  ring

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧ squaredNorm 1 2 0 ≠ 0 := by
  norm_num [squaredNorm]

/-- Positive rescaling of the gradient changes the effective epsilon.
Source: suppressed averaging in arXiv:2205.10343v2, eq:l_eff, and native
AdamW first-step normalization at 43d4d66. No time-only invariance follows. -/
theorem first_direction_positive_gradient_scaling (b1 b2 eps c g : ℝ)
    (h1 : b1 ≠ 1) (h2 : b2 ≠ 1) (he : 0 < eps) (hc : 0 < c) :
    firstDirection b1 b2 eps (c * g) = firstDirection b1 b2 (eps / c) g := by
  rw [firstDirection_eq b1 b2 eps (c * g) h1 h2,
    firstDirection_eq b1 b2 (eps / c) g h1 h2, abs_mul, abs_of_pos hc]
  have hcn : c ≠ 0 := by positivity
  have hd1 : c * |g| + eps ≠ 0 := by positivity
  have hd2 : |g| + eps / c ≠ 0 := by positivity
  field_simp

example : (9 / 10 : ℝ) ≠ 1 ∧ (49 / 50 : ℝ) ≠ 1 ∧
    0 < (1 / 100000000 : ℝ) ∧ 0 < (2 : ℝ) := by norm_num

/-- A gradient factor of two cannot be absorbed by one time factor for
both coordinates of the native first direction. Source: the proposed
extension of arXiv:2205.10343v2's averaging/time argument to AdamW at
43d4d66. The counterexample uses valid betas and epsilon 1, not the
experiment's epsilon 1e-8; it refutes an unrestricted transfer. -/
theorem gradient_scaling_not_uniform_time_rescaling :
    ¬∃ t : ℝ,
      firstDirection (9 / 10) (49 / 50) 1 2 = t * firstDirection (9 / 10) (49 / 50) 1 1 ∧
      firstDirection (9 / 10) (49 / 50) 1 4 = t * firstDirection (9 / 10) (49 / 50) 1 2 := by
  rintro ⟨t, hfirst, hsecond⟩
  have hd (g : ℝ) := firstDirection_eq (9 / 10) (49 / 50) 1 g (by norm_num) (by norm_num)
  simp only [hd] at hfirst hsecond
  norm_num at hfirst hsecond
  linarith

end Transformer.Grokking.AdamW
