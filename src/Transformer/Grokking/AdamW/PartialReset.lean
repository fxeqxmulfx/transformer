import Transformer.Grokking.AdamW.MomentBounds

/-!
# Same-gradient partial resets of native AdamW

Source: PyTorch 2.14.1 AdamW at lab commit 0033b1b, documented
retained-buffer update, and moment_resets.py's explicit CPU interventions.
The supplied first/second moments are values immediately before an
update. Partial resets retain its completed-update clock; a fresh
optimizer resets both buffers and the clock. These transformations
are counterfactual observations, not the archived training algorithm.

Prove how the same current gradient changes the next direction: erasing
the first moment restores scalar adaptive-gradient alignment; erasing
only the second moment can enlarge the adaptive magnitude. Neither
fact implies finite-step loss descent, behavior of decoupled decay,
or a multi-step GPTMini generalization improvement.
-/

namespace Transformer.Grokking.AdamW

/-- One native adaptive direction from retained pre-update buffers.
Source: PyTorch 2.14.1 AdamW at 0033b1b; the old completed clock is
incremented once, and decoupled parameter decay is separate. -/
noncomputable def nextBufferDirection (b1 b2 eps oldM oldV current : ℝ) (clock : ℕ) : ℝ :=
  ((b1 * oldM + (1 - b1) * current) / (1 - b1 ^ (clock + 1))) /
    (Real.sqrt ((b2 * oldV + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1))) + eps)

/-- Reset only the old first moment and retain variance and clock.
Source: moment_resets.py, same-gradient reset_first_moment branch. -/
noncomputable def firstResetDirection (b1 b2 eps oldV current : ℝ) (clock : ℕ) : ℝ :=
  nextBufferDirection b1 b2 eps 0 oldV current clock

/-- Reset only the old second moment and retain numerator and clock.
Source: moment_resets.py, same-gradient reset_second_moment branch. -/
noncomputable def secondResetDirection (b1 b2 eps oldM current : ℝ) (clock : ℕ) : ℝ :=
  nextBufferDirection b1 b2 eps oldM 0 current clock

/-- The actual noninitial history consumes its next gradient through
the retained-buffer formula. Source: PyTorch 2.14.1 AdamW at 0033b1b;
both buffers come from the previously verified zero-initialized history. -/
theorem history_direction_succ (b1 b2 eps : ℝ) (gradient : ℕ → ℝ) (clock : ℕ) :
    historyDirection b1 b2 eps gradient (clock + 1) =
      nextBufferDirection b1 b2 eps (firstMomentAt b1 gradient clock)
        (secondMomentAt b2 gradient clock) (gradient clock) clock := by
  unfold historyDirection nextBufferDirection firstMomentAt secondMomentAt
  simp only [momentAt]

/-- Resetting both buffers and the old clock gives the checked first
direction, rather than a partial reset at the original clock.
Source: moment_resets.py, fresh_adamw branch, native equations. -/
theorem fresh_buffer_direction (b1 b2 eps current : ℝ) :
    nextBufferDirection b1 b2 eps 0 0 current 0 = firstDirection b1 b2 eps current := by
  unfold nextBufferDirection firstDirection
  simp only [mul_zero, zero_add, pow_one]

/-- With physical nonnegative old variance, erasing it cannot reduce
the absolute adaptive direction when the same current gradient,
first moment and clock are retained. Source: native AdamW at 0033b1b,
same-gradient reset_second_moment intervention; beta2 and epsilon
conditions make both actual variance denominators admissible. -/
theorem second_reset_abs_ge_retained (b1 b2 eps oldM oldV current : ℝ) (clock : ℕ)
    (hb : 0 ≤ b2) (h1 : b2 < 1) (he : 0 < eps) (hv : 0 ≤ oldV) :
    |nextBufferDirection b1 b2 eps oldM oldV current clock| ≤
      |secondResetDirection b1 b2 eps oldM current clock| := by
  have hc := bias_correction_positive b2 (clock + 1) hb h1 (by omega)
  have hm : (b2 * 0 + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1)) ≤
      (b2 * oldV + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1)) :=
    div_le_div_of_nonneg_right (by nlinarith) (le_of_lt hc)
  have hsr := Real.sqrt_nonneg
    ((b2 * oldV + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1)))
  have hsz := Real.sqrt_nonneg
    ((b2 * 0 + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1)))
  have horder := Real.sqrt_le_sqrt hm
  unfold secondResetDirection nextBufferDirection
  conv_lhs => rw [abs_div]
  conv_rhs => rw [abs_div]
  rw [abs_of_nonneg (show 0 ≤ Real.sqrt
    ((b2 * oldV + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1))) + eps by linarith),
    abs_of_nonneg (show 0 ≤ Real.sqrt
    ((b2 * 0 + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1))) + eps by linarith)]
  exact div_le_div_of_nonneg_left (abs_nonneg _) (by linarith) (by linarith)

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Erasing the old first moment aligns the scalar adaptive direction
with every nonzero current gradient. Source: native AdamW at 0033b1b,
same-gradient reset_first_moment intervention. Beta2 and oldV remain
arbitrary here because the real square root is always nonnegative;
physical-buffer validity is established by the actual history module. -/
theorem first_reset_current_alignment (b1 b2 eps oldV current : ℝ) (clock : ℕ)
    (hb : 0 ≤ b1) (h1 : b1 < 1) (he : 0 < eps) (hg : current ≠ 0) :
    0 < current * firstResetDirection b1 b2 eps oldV current clock := by
  have hc := bias_correction_positive b1 (clock + 1) hb h1 (by omega)
  have hd : 0 < Real.sqrt
      ((b2 * oldV + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1))) + eps := by
    have hs := Real.sqrt_nonneg
      ((b2 * oldV + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1)))
    linarith
  unfold firstResetDirection nextBufferDirection
  simp only [mul_zero, zero_add, div_div]
  have hp := div_pos (mul_pos (show 0 < 1 - b1 by linarith) (sq_pos_of_ne_zero hg))
    (mul_pos hc hd)
  convert hp using 1
  ring

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 100000000 ∧ (-1 : ℝ) ≠ 0 := by norm_num

/-- A zero current gradient leaves a stale retained first moment in
the variance-only reset direction, divided by epsilon. Source: native
AdamW at 0033b1b, reset_second_moment branch; the expression explains
possible amplification without asserting it is always harmful. -/
theorem second_reset_zero_gradient (b1 b2 eps oldM : ℝ) (clock : ℕ) :
    secondResetDirection b1 b2 eps oldM 0 clock =
      (b1 * oldM / (1 - b1 ^ (clock + 1))) / eps := by
  unfold secondResetDirection nextBufferDirection
  simp only [zero_pow (by norm_num : (2 : ℕ) ≠ 0), mul_zero, zero_add,
    add_zero, zero_div, Real.sqrt_zero]

/-- Erasing the first moment instead gives zero adaptive direction at
a zero current gradient, while decoupled decay can still move weights.
Source: native AdamW at 0033b1b, reset_first_moment branch. -/
theorem first_reset_zero_gradient (b1 b2 eps oldV : ℝ) (clock : ℕ) :
    firstResetDirection b1 b2 eps oldV 0 clock = 0 := by
  unfold firstResetDirection nextBufferDirection
  simp only [mul_zero, zero_add, zero_div]

/-- A nonnegative old variance remains a physical corrected variance
after inserting any current gradient. Source: PyTorch 2.14.1 AdamW
at 0033b1b; valid beta2 makes the correction denominator positive. -/
theorem next_corrected_variance_nonneg (b2 oldV current : ℝ) (clock : ℕ)
    (hb : 0 ≤ b2) (h1 : b2 < 1) (hv : 0 ≤ oldV) :
    0 ≤ (b2 * oldV + (1 - b2) * current ^ 2) / (1 - b2 ^ (clock + 1)) := by
  have hc := bias_correction_positive b2 (clock + 1) hb h1 (by omega)
  exact div_nonneg (add_nonneg (mul_nonneg hb hv)
    (mul_nonneg (by linarith) (sq_nonneg _))) (le_of_lt hc)

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- Even with an exactly constant, already aligned gradient history,
resetting the first moment changes the direction by the stated clock-
dependent factor. Source: native AdamW at 0033b1b, reset_first_moment
branch. Thus its observed benefit cannot be attributed to removing
misalignment without accounting for this change of effective rate. -/
theorem first_reset_constant_history (b1 b2 eps value : ℝ) (clock : ℕ)
    (hb : 0 ≤ b2) (h1 : b2 < 1) :
    firstResetDirection b1 b2 eps (secondMomentAt b2 (fun _ => value) clock) value clock =
      ((1 - b1) / (1 - b1 ^ (clock + 1))) * (value / (|value| + eps)) := by
  have hm := (constant_gradient_corrected 0 b2 value (clock + 1)
    (by norm_num) (by norm_num) hb h1 (by omega)).2
  have hv : (b2 * secondMomentAt b2 (fun _ => value) clock + (1 - b2) * value ^ 2) /
      (1 - b2 ^ (clock + 1)) = value ^ 2 := hm
  unfold firstResetDirection nextBufferDirection
  rw [mul_zero, zero_add, hv, Real.sqrt_sq_eq_abs]
  ring

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 := by norm_num

end Transformer.Grokking.AdamW
