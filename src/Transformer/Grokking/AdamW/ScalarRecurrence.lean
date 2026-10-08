import Transformer.Grokking.AdamW.PartialReset

/-!
# Scalar native AdamW state and its retained update

Source: PyTorch 2.14.1 AdamW at lab commit 88aa892, documented
parameter decay, moment recurrences and completed-update bias clock.
Keep the parameter, both actual buffers and clock together. The input
gradient is supplied here and may already be clipped. Its relation
to the current parameters and objective belongs to the application.

These are exact-real non-AMSGrad, non-maximizing updates. Floating-
point kernels and clipping are not encoded. Zero-state identities
use the real division convention; native beta and epsilon conditions
are explicit in the later sign and trajectory results. No buffer is
reset between steps and no convergence is presumed by the state type.
-/

namespace Transformer.Grokking.AdamW

/-- Numerical state of one always-updated coordinate. Source: native
AdamW at 88aa892; fields are values, not optimizer-success claims. -/
structure ScalarState where
  parameter : ℝ
  moment : ℝ
  variance : ℝ
  clock : ℕ

/-- Standard native initialization at a supplied parameter. Source:
PyTorch 2.14.1 AdamW at 88aa892, zero moments and zero update clock. -/
def seededScalarState (parameter : ℝ) : ScalarState := ⟨parameter, 0, 0, 0⟩

/-- A zero coordinate and buffers at a supplied completed clock.
Source: numerical state representation of the native zero-gradient
invariant; the clock is retained rather than restarted. -/
def zeroScalarStateAt (clock : ℕ) : ScalarState := ⟨0, 0, 0, clock⟩

/-- One simultaneous native insertion and parameter update. Source:
PyTorch 2.14.1 AdamW at 88aa892; use the gradient at the old parameter,
and keep the moments that will supply the following iteration. -/
noncomputable def scalarNativeStep (b1 b2 eps decay rate : ℝ)
    (state : ScalarState) (gradient : ℝ) : ScalarState :=
  { parameter := (1 - rate * decay) * state.parameter -
      rate * nextBufferDirection b1 b2 eps state.moment state.variance gradient state.clock
    moment := b1 * state.moment + (1 - b1) * gradient
    variance := b2 * state.variance + (1 - b2) * gradient ^ 2
    clock := state.clock + 1 }

/-- The initialized scalar update agrees with the already checked
native first-step parameter formula. Source: PyTorch 2.14.1 AdamW
at 88aa892; this checks buffers and bias-clock indexing. -/
theorem scalar_native_first_parameter (b1 b2 eps decay rate parameter gradient : ℝ) :
    (scalarNativeStep b1 b2 eps decay rate (seededScalarState parameter) gradient).parameter =
      firstUpdate b1 b2 eps decay rate parameter gradient := by
  change (1 - rate * decay) * parameter - rate * nextBufferDirection b1 b2 eps 0 0 gradient 0 = _
  rw [fresh_buffer_direction]
  unfold firstUpdate
  ring

/-- Each supplied gradient increments its coordinate clock exactly
once. Source: native AdamW at 88aa892; skipped-gradient coordinates
are outside this always-updated recurrence. -/
theorem scalar_native_clock (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState) :
    (scalarNativeStep b1 b2 eps decay rate state gradient).clock = state.clock + 1 := by
  unfold scalarNativeStep
  dsimp

/-- A zero coordinate with zero actual moments remains zero after
a zero input, but its clock still advances. Source: native AdamW
at 88aa892, exact-real formula; no moment reset is performed. -/
theorem scalar_native_zero_step (b1 b2 eps decay rate : ℝ) (clock : ℕ) :
    scalarNativeStep b1 b2 eps decay rate (zeroScalarStateAt clock) 0 =
      zeroScalarStateAt (clock + 1) := by
  simp [scalarNativeStep, nextBufferDirection, zeroScalarStateAt]

/-- A zero current gradient does not erase the retained numerator.
Source: native AdamW at 88aa892; unlike its initialized first step,
the parameter can move from a nonzero historical first moment. -/
theorem scalar_zero_gradient_retained_update (b1 b2 eps decay rate : ℝ) (state : ScalarState) :
    (scalarNativeStep b1 b2 eps decay rate state 0).parameter =
      (1 - rate * decay) * state.parameter - rate *
        ((b1 * state.moment / (1 - b1 ^ (state.clock + 1))) /
          (Real.sqrt (b2 * state.variance / (1 - b2 ^ (state.clock + 1))) + eps)) := by
  unfold scalarNativeStep nextBufferDirection
  simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0), mul_zero, add_zero]

/-- Nonpositive current and historical gradients keep the native
first moment nonpositive. Source: PyTorch 2.14.1 AdamW at 88aa892,
nonnegative exponential averaging factors. -/
theorem scalar_native_moment_nonpos (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState)
    (hb : 0 ≤ b1) (h1 : b1 ≤ 1) (hm : state.moment ≤ 0) (hg : gradient ≤ 0) :
    (scalarNativeStep b1 b2 eps decay rate state gradient).moment ≤ 0 := by
  change b1 * state.moment + (1 - b1) * gradient ≤ 0
  have hl := mul_nonpos_of_nonneg_of_nonpos hb hm
  have hr := mul_nonpos_of_nonneg_of_nonpos (show 0 ≤ 1 - b1 by linarith) hg
  linarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧
    (seededScalarState 1).moment ≤ 0 ∧ (-1 : ℝ) ≤ 0 := by norm_num [seededScalarState]

/-- Every nonnegative old variance remains nonnegative after the
actual squared-gradient insertion. Source: native non-AMSGrad AdamW
at 88aa892; the derivative sign imposes no restriction on variance. -/
theorem scalar_native_variance_nonneg (b1 b2 eps decay rate gradient : ℝ) (state : ScalarState)
    (hb : 0 ≤ b2) (h1 : b2 ≤ 1) (hv : 0 ≤ state.variance) :
    0 ≤ (scalarNativeStep b1 b2 eps decay rate state gradient).variance := by
  change 0 ≤ b2 * state.variance + (1 - b2) * gradient ^ 2
  exact add_nonneg (mul_nonneg hb hv) (mul_nonneg (by linarith) (sq_nonneg _))

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) ≤ 1 ∧
    0 ≤ (seededScalarState 1).variance := by norm_num [seededScalarState]

/-- When the entire retained numerator history and current gradient
are nonpositive, the adaptive direction is nonpositive. Source:
native AdamW at 88aa892; no alignment claim for arbitrary histories
is made. The denominator is positive at positive epsilon. -/
theorem retained_direction_nonpos (b1 b2 eps gradient : ℝ) (state : ScalarState)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps)
    (hm : state.moment ≤ 0) (hg : gradient ≤ 0) :
    nextBufferDirection b1 b2 eps state.moment state.variance gradient state.clock ≤ 0 := by
  have hc := bias_correction_positive b1 (state.clock + 1) hb h1 (by omega)
  have hn : b1 * state.moment + (1 - b1) * gradient ≤ 0 := by
    have hl := mul_nonpos_of_nonneg_of_nonpos hb hm
    have hr := mul_nonpos_of_nonneg_of_nonpos (show 0 ≤ 1 - b1 by linarith) hg
    linarith
  have hd := Real.sqrt_nonneg
    ((b2 * state.variance + (1 - b2) * gradient ^ 2) / (1 - b2 ^ (state.clock + 1)))
  unfold nextBufferDirection
  exact div_nonpos_of_nonpos_of_nonneg
    (div_nonpos_of_nonpos_of_nonneg hn (le_of_lt hc)) (by linarith)

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (seededScalarState 1).moment ≤ 0 ∧ (-1 : ℝ) ≤ 0 := by norm_num [seededScalarState]

/-- A strictly negative current gradient makes that retained
adaptive direction strictly negative. Source: native AdamW at
88aa892; the old first moment is allowed to be exactly zero. -/
theorem retained_direction_negative (b1 b2 eps gradient : ℝ) (state : ScalarState)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps)
    (hm : state.moment ≤ 0) (hg : gradient < 0) :
    nextBufferDirection b1 b2 eps state.moment state.variance gradient state.clock < 0 := by
  have hc := bias_correction_positive b1 (state.clock + 1) hb h1 (by omega)
  have hl := mul_nonpos_of_nonneg_of_nonpos hb hm
  have hn : b1 * state.moment + (1 - b1) * gradient < 0 := by nlinarith
  have hs := Real.sqrt_nonneg
    ((b2 * state.variance + (1 - b2) * gradient ^ 2) / (1 - b2 ^ (state.clock + 1)))
  unfold nextBufferDirection
  exact div_neg_of_neg_of_pos (div_neg_of_neg_of_pos hn hc) (by linarith)

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    (seededScalarState 1).moment ≤ 0 ∧ (-1 : ℝ) < 0 := by norm_num [seededScalarState]

end Transformer.Grokking.AdamW
