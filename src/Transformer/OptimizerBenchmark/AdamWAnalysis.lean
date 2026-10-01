/-
# Decoupled decay can help or hurt: a limit on explaining AdamW's win

First-step moments and epsilon match the benchmark's coordinate.py:
beta1=0.9, beta2=0.999, epsilon=1e-8, with zero initial moments.
Adam retains raw moments; AdamW corrects both moments and adds decay.
These are real-arithmetic specifications, not a claim about exact
floating-point PyTorch execution or the entire 1000-update training run.
The objectives below use their actual derivatives and satisfy both smooth
upper and strongly convex lower models. This refutes a proposed universal
explanation of our experiment; the paper makes no such universal claim.
The improvement example fixes all controls except decay. The deterioration
example already works at the learning rates selected by the benchmark.

The decoupled update is from arXiv:2502.16982, Section 2.2, equation
equation:weightdecay.
-/

import Transformer.Optimization.StrongConvexity
import Mathlib.Analysis.Calculus.Deriv.Add

open scoped InnerProductSpace

noncomputable section

namespace Transformer.OptimizerBenchmark

open Transformer.Optimization

/-- First Adam update from zero moments, without debiasing, exactly as the
benchmark's Adam reference. Source: coordinate.py, propose; Adam context in
arXiv:2502.16982, Section 2.1. -/
def firstAdamStep (rate : ℝ) (loss : ℝ → ℝ) (x : ℝ) : ℝ :=
  let g := gradient loss x
  x - rate * ((1 - 9 / 10) * g) / (Real.sqrt ((1 - 999 / 1000) * g ^ 2) + 1 / 10 ^ 8)

/-- First AdamW update from zero moments, with both moment corrections and
decay. Source: coordinate.py, propose; decoupled decay in arXiv:2502.16982,
Section 2.2, equation equation:weightdecay. -/
def firstAdamWStep (rate decay : ℝ) (loss : ℝ → ℝ) (x : ℝ) : ℝ :=
  let g := gradient loss x
  let m := (1 - 9 / 10) * g
  let v := (1 - 999 / 1000) * g ^ 2
  x - rate * (m / (1 - 9 / 10) / (Real.sqrt (v / (1 - 999 / 1000)) +
    1 / 10 ^ 8) + decay * x)

/-- Actual moment debiasing recovers g and g^2 at the first step. This
identity removes initialization bias, not optimization or generalization
error. Source: coordinate.py, propose; AdamW context in arXiv:2502.16982,
Section 2.2, Weight Decay. -/
theorem firstAdamWStep_formula (rate decay : ℝ) (loss : ℝ → ℝ) (x : ℝ) :
    firstAdamWStep rate decay loss x =
      x - rate * (gradient loss x / (Real.sqrt ((gradient loss x) ^ 2) +
        1 / 10 ^ 8) + decay * x) := by
  unfold firstAdamWStep
  norm_num

/-- The isolated effect of decay is a displacement of rate*decay*x. This
identity leaves the adaptive direction intact. Source: arXiv:2502.16982,
Section 2.2, equation equation:weightdecay, as used in coordinate.py. -/
theorem firstAdamWStep_decay (rate decay : ℝ) (loss : ℝ → ℝ) (x : ℝ) :
    firstAdamWStep rate decay loss x = firstAdamWStep rate 0 loss x - rate * decay * x := by
  rw [firstAdamWStep_formula, firstAdamWStep_formula]
  ring

/-- A nonconstant objective with a nonzero unique minimizer, used to test
the claimed decay explanation. Source: user-requested extension of
arXiv:2502.16982, Section 2.2, equation equation:weightdecay. -/
def shiftedQuadratic (x : ℝ) : ℝ := (x - 1) ^ 2 / 2

/-- The counterexample's gradient is its actual derivative. Source:
user-requested decay analysis, arXiv:2502.16982, Section 2.2,
equation equation:weightdecay. -/
theorem shiftedQuadratic_hasGradientAt (x : ℝ) :
    HasGradientAt shiftedQuadratic (x - 1) x := by
  apply HasDerivAt.hasGradientAt'
  convert (((hasDerivAt_id x).sub_const 1).pow 2).div_const 2 using 1 <;>
    first | rfl | simp

/-- Actual gradient of the shifted quadratic. Source: decay analysis for
arXiv:2502.16982, Section 2.2, equation equation:weightdecay. -/
theorem shiftedQuadratic_gradient : gradient shiftedQuadratic = fun x => x - 1 :=
  gradient_eq shiftedQuadratic_hasGradientAt

/-- The witness is 1-smooth and 1-strongly convex, so failure of a universal
AdamW ranking is not caused by nonsmoothness or nonconvexity. Source:
user-requested decay analysis of arXiv:2502.16982, Section 2.2. -/
theorem shiftedQuadratic_models :
    SmoothObjective shiftedQuadratic 1 ∧ StrongLowerModel shiftedQuadratic 1 := by
  constructor
  · refine ⟨fun x => (shiftedQuadratic_hasGradientAt x).differentiableAt, ?_⟩
    intro x y
    rw [shiftedQuadratic_gradient]
    simp only [shiftedQuadratic, RCLike.inner_apply, conj_trivial, Real.norm_eq_abs, sq_abs]
    nlinarith
  · intro x y
    rw [shiftedQuadratic_gradient]
    simp only [shiftedQuadratic, RCLike.inner_apply, conj_trivial, Real.norm_eq_abs, sq_abs]
    nlinarith

/-- At the nonzero minimizer, Adam stays optimal while AdamW's decay moves
away and strictly increases the unregularized loss for every positive rate
and decay. Refuted claim: AdamW always beats Adam on smooth strongly convex
objectives; this is not a claim of the paper. Source: user-requested analysis
of arXiv:2502.16982, Section 2.2, equation equation:weightdecay. -/
theorem decay_hurts_at_minimizer (rate decay : ℝ) (hrate : 0 < rate) (hdecay : 0 < decay) :
    shiftedQuadratic (firstAdamStep rate shiftedQuadratic 1) <
      shiftedQuadratic (firstAdamWStep rate decay shiftedQuadratic 1) := by
  rw [firstAdamWStep_formula]
  norm_num [firstAdamStep, shiftedQuadratic_gradient, shiftedQuadratic]
  nlinarith [sq_pos_of_pos (mul_pos hrate hdecay)]

/-- Positive parameters include the winning benchmark's rate and decay.
Source: REPORT.md, Implementation and proof scope; decoupled update in
arXiv:2502.16982, Section 2.2, equation equation:weightdecay. -/
example : (0 : ℝ) < 1 / 1000 ∧ (0 : ℝ) < 1 / 100 := by norm_num

/-- Even with the separately selected benchmark rates, AdamW cannot
uniformly dominate Adam over smooth strongly convex objectives. The
universal claim is refuted at the shifted quadratic's minimizer after one
update. Source: REPORT.md, selected rates; user-requested analysis of
arXiv:2502.16982, Section 2.2, equation equation:weightdecay. -/
theorem adamw_not_uniformly_better :
    ¬ ∀ loss : ℝ → ℝ, SmoothObjective loss 1 → StrongLowerModel loss 1 →
      ∀ x, loss (firstAdamWStep (1 / 1000) (1 / 100) loss x) ≤
        loss (firstAdamStep (3 / 10000) loss x) := by
  intro h
  have hbad := h shiftedQuadratic shiftedQuadratic_models.1 shiftedQuadratic_models.2 1
  rw [firstAdamWStep_formula] at hbad
  norm_num [firstAdamStep, shiftedQuadratic_gradient, shiftedQuadratic] at hbad

/-- On a quadratic centered at zero, decay can instead improve the first
step. Both sides use the identical bias-corrected rule and rate, isolating
decay itself; this is not the benchmark's Adam-versus-AdamW comparison.
Source: user-requested analysis of arXiv:2502.16982, Section 2.2,
equation equation:weightdecay. -/
theorem decay_can_help :
    quadratic (firstAdamWStep (1 / 1000) (1 / 100) quadratic 1) <
      quadratic (firstAdamWStep (1 / 1000) 0 quadratic 1) := by
  rw [firstAdamWStep_formula, firstAdamWStep_formula, quadratic_gradient]
  norm_num [quadratic]

end Transformer.OptimizerBenchmark
