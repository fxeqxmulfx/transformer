import Transformer.Grokking.AdamW.SecondStep

/-!
# Native moment histories and direct forgetting

Source: PyTorch 2.14.1 AdamW at lab commit 0033b1b, documented
moment recurrences, bias corrections and zero-buffer initialization.
The gradient indexed by k is the input at completed-update clock k;
its insertion increments the clock to k+1. Derive first/second-step
compatibility and causal prefix bounds for the actual linear recurrence.

Changing an old moment while supplying the same subsequent gradients
changes its later value by exactly beta^n times the old difference.
This is direct buffer memory, not the effect of different parameters
on future gradients. Those gradients may be actual clipped minibatch
gradients, but a transformer/rounding bridge is not assumed here.
No sign of alignment or delayed generalization is built into the history.
-/

namespace Transformer.Grokking.AdamW

/-- Exponential moment recurrence with an arbitrary initial buffer.
Source: native AdamW at 0033b1b, PyTorch 2.14.1; input squares give
the second moment and unsquared inputs give the first moment. -/
noncomputable def momentAt (beta : ℝ) (input : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => beta * momentAt beta input initial n + (1 - beta) * input n

/-- Actual first moment from zero buffers. Source: native AdamW at
0033b1b, the unsquared-gradient exponential average. -/
noncomputable def firstMomentAt (beta : ℝ) (gradient : ℕ → ℝ) (n : ℕ) : ℝ :=
  momentAt beta gradient 0 n

/-- Actual non-AMSGrad second moment from zero buffers. Source:
native AdamW at 0033b1b, exponential average of squared gradients. -/
noncomputable def secondMomentAt (beta : ℝ) (gradient : ℕ → ℝ) (n : ℕ) : ℝ :=
  momentAt beta (fun k => gradient k ^ 2) 0 n

/-- Bias-corrected native direction after n supplied gradients.
Source: native AdamW at 0033b1b; positive-clock and valid-beta
conditions on its denominators are proved explicitly below. -/
noncomputable def historyDirection (b1 b2 eps : ℝ) (gradient : ℕ → ℝ) (n : ℕ) : ℝ :=
  (firstMomentAt b1 gradient n / (1 - b1 ^ n)) /
    (Real.sqrt (secondMomentAt b2 gradient n / (1 - b2 ^ n)) + eps)

/-- Direct memory of an initial moment contracts by beta at every
insertion of a shared subsequent input. Source: PyTorch 2.14.1 native
AdamW at 0033b1b; no parameter trajectory equality is inferred. -/
theorem moment_initial_difference (beta : ℝ) (input : ℕ → ℝ) (left right : ℝ) (n : ℕ) :
    momentAt beta input left n - momentAt beta input right n = beta ^ n * (left - right) := by
  induction n with
  | zero => simp [momentAt]
  | succ n ih =>
    simp only [momentAt, pow_succ]
    calc
      beta * momentAt beta input left n + (1 - beta) * input n -
          (beta * momentAt beta input right n + (1 - beta) * input n) =
          beta * (momentAt beta input left n - momentAt beta input right n) := by ring
      _ = beta ^ n * beta * (left - right) := by rw [ih]; ring

/-- A constant input has the stated exact exponential weighting.
Source: native AdamW at 0033b1b, solved linear buffer recurrence;
the initial buffer remains distinct from the subsequent input. -/
theorem moment_constant (beta value initial : ℝ) (n : ℕ) :
    momentAt beta (fun _ => value) initial n =
      beta ^ n * initial + (1 - beta ^ n) * value := by
  induction n with
  | zero => simp [momentAt]
  | succ n ih =>
    simp only [momentAt, ih, pow_succ]
    ring

/-- Continuing an actual buffer at clock n agrees with consuming the
shifted future inputs from that buffer. Source: PyTorch 2.14.1 AdamW
at 0033b1b; this identity does not restart bias correction at clock zero. -/
theorem moment_continue (beta initial : ℝ) (input : ℕ → ℝ) (n k : ℕ) :
    momentAt beta input initial (n + k) =
      momentAt beta (fun j => input (n + j)) (momentAt beta input initial n) k := by
  induction k with
  | zero => simp [momentAt]
  | succ k ih =>
    change beta * momentAt beta input initial (n + k) + (1 - beta) * input (n + k) =
      beta * momentAt beta (fun j => input (n + j)) (momentAt beta input initial n) k +
        (1 - beta) * input (n + k)
    rw [ih]

/-- A causal bounded prefix and bounded initial buffer keep the moment
bounded. Source: native AdamW at 0033b1b, convex averaging for beta in
[0,1]. No future gradients enter this bound. A floating-point clipping
certificate is not supplied by the exact-real input hypothesis. -/
theorem moment_prefix_abs_bound (beta bound initial : ℝ) (input : ℕ → ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1) (hi : |initial| ≤ bound)
    (hg : ∀ k : ℕ, k < n → |input k| ≤ bound) :
    |momentAt beta input initial n| ≤ bound := by
  revert hg
  induction n with
  | zero => intro hg; exact hi
  | succ n ih =>
    intro hg
    have hm := ih (fun k hk => hg k (by omega))
    have hc := hg n (by omega)
    have ht : 0 ≤ 1 - beta := by linarith
    calc
      |momentAt beta input initial (n + 1)| ≤
          |beta * momentAt beta input initial n| + |(1 - beta) * input n| := abs_add_le _ _
      _ = beta * |momentAt beta input initial n| + (1 - beta) * |input n| := by
        rw [abs_mul, abs_mul, abs_of_nonneg hb, abs_of_nonneg ht]
      _ ≤ beta * bound + (1 - beta) * bound :=
        add_le_add (mul_le_mul_of_nonneg_left hm hb) (mul_le_mul_of_nonneg_left hc ht)
      _ = bound := by ring

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧ |(0 : ℝ)| ≤ 1 ∧
    (∀ k : ℕ, k < 3 → |(if k = 0 then (1 : ℝ) else -1)| ≤ 1) := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro k hk
  split_ifs <;> norm_num

/-- All actual second moments are nonnegative at valid beta values.
Source: PyTorch 2.14.1 native non-AMSGrad recurrence at 0033b1b. -/
theorem second_moment_nonneg (beta : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1) : 0 ≤ secondMomentAt beta gradient n := by
  unfold secondMomentAt
  induction n with
  | zero => norm_num [momentAt]
  | succ n ih =>
    simp only [momentAt]
    exact add_nonneg (mul_nonneg hb ih) (mul_nonneg (by linarith) (sq_nonneg _))

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) ≤ 1 := by norm_num

/-- Native bias-correction denominators are positive after any positive
number of updates. Source: PyTorch 2.14.1 AdamW at 0033b1b, valid beta
range [0,1); the clock is a completed-update count, not a sample count. -/
theorem bias_correction_positive (beta : ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hn : 0 < n) : 0 < 1 - beta ^ n := by
  have hp := pow_lt_one₀ hb h1 (Nat.ne_of_gt hn)
  linarith

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ 0 < (3 : ℕ) := by norm_num

/-- The general history specializes to the already checked native
first direction. Source: PyTorch 2.14.1 AdamW at 0033b1b, zero moments;
this checks indexing and correction factors against the earlier module. -/
theorem history_direction_first (b1 b2 eps : ℝ) (gradient : ℕ → ℝ) :
    historyDirection b1 b2 eps gradient 1 = firstDirection b1 b2 eps (gradient 0) := by
  unfold historyDirection firstMomentAt secondMomentAt firstDirection
  simp only [momentAt, mul_zero, zero_add, pow_one]

/-- Two insertions specialize to the retained first-gradient history,
without resetting its clock. Source: PyTorch 2.14.1 AdamW at 0033b1b,
compared with the checked SecondStep module. -/
theorem history_direction_second (b1 b2 eps : ℝ) (gradient : ℕ → ℝ) :
    historyDirection b1 b2 eps gradient 2 = secondDirection b1 b2 eps (gradient 0) (gradient 1) := by
  unfold historyDirection firstMomentAt secondMomentAt secondDirection
  simp only [momentAt, mul_zero, zero_add]

end Transformer.Grokking.AdamW
