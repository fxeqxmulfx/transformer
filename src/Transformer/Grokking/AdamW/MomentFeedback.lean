import Transformer.Grokking.AdamW.MomentMemory

/-!
# Persistent input differences in native moment histories

Source: PyTorch 2.14.1 AdamW at lab commit 0033b1b, exponential
buffer recurrences; extensions of the exact shared-input memory law.
Two different parameter paths generally supply different gradients.
Derive an estimate that keeps the old-buffer and later-input effects
separate, including square-gradient feedback into the second moment.

The parameter-feedback theorem uses actual scalar loss derivatives and
a stated local bound along the two visited paths. It does not prove
that bound for GPTMini or bound the parameter paths themselves. A
constant input difference also survives bias correction forever, even
with identical zero initial buffers. Thus shared-input forgetting is
not unrestricted contraction of a training process or a grokking time.
-/

namespace Transformer.Grokking.AdamW

/-- Initial-buffer contribution separates from the zero-buffer
history. Source: native AdamW at 0033b1b, exact linear recurrence. -/
theorem moment_initial_superposition (beta initial : ℝ) (input : ℕ → ℝ) (n : ℕ) :
    momentAt beta input initial n = firstMomentAt beta input n + beta ^ n * initial := by
  have hm := moment_initial_difference beta input initial 0 n
  unfold firstMomentAt
  linarith

/-- With different later inputs, the buffer difference is the
recurrence driven by their differences. Source: PyTorch 2.14.1
AdamW at 0033b1b; no equality of parameter paths is assumed. -/
theorem moment_distinct_input_difference (beta left right : ℝ) (a b : ℕ → ℝ) (n : ℕ) :
    momentAt beta a left n - momentAt beta b right n =
      momentAt beta (fun k => a k - b k) (left - right) n := by
  induction n with
  | zero => simp [momentAt]
  | succ n ih =>
    simp only [momentAt]
    rw [← ih]
    ring

/-- Causal input disagreements add a nonvanishing term to the old-
buffer decay bound. Source: native AdamW at 0033b1b; the estimate
depends only on the first n differences, not future gradients. -/
theorem moment_different_prefix_bound (beta left right bound : ℝ) (a b : ℕ → ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1)
    (hg : ∀ k : ℕ, k < n → |a k - b k| ≤ bound) :
    |momentAt beta a left n - momentAt beta b right n| ≤
      beta ^ n * |left - right| + (1 - beta ^ n) * bound := by
  have hm := first_moment_prefix_abs_bound beta bound (fun k => a k - b k) n hb h1 hg
  rw [moment_distinct_input_difference, moment_initial_superposition]
  calc
    _ ≤ |firstMomentAt beta (fun k => a k - b k) n| + |beta ^ n * (left - right)| :=
      abs_add_le _ _
    _ ≤ (1 - beta ^ n) * bound + |beta ^ n * (left - right)| := add_le_add hm le_rfl
    _ = beta ^ n * |left - right| + (1 - beta ^ n) * bound := by
      rw [abs_mul, abs_of_nonneg (pow_nonneg hb n)]
      ring

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧
    (∀ k : ℕ, k < 3 → |(if k = 0 then (1 : ℝ) else 0) - (-1)| ≤ 2) := by
  refine ⟨by norm_num, by norm_num, ?_⟩
  intro k hk
  split_ifs <;> norm_num

/-- Constant input disagreements attain the zero-initial-buffer
feedback estimate exactly. Source: native AdamW at 0033b1b; the
later-input term cannot be deleted from that bound in general. -/
theorem constant_input_feedback_sharp (beta left right : ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1) :
    |firstMomentAt beta (fun _ => left) n - firstMomentAt beta (fun _ => right) n| =
      (1 - beta ^ n) * |left - right| := by
  have hm : firstMomentAt beta (fun _ => left) n - firstMomentAt beta (fun _ => right) n =
      (1 - beta ^ n) * (left - right) := by
    unfold firstMomentAt
    rw [moment_constant, moment_constant]
    ring
  have hc : 0 ≤ 1 - beta ^ n := by linarith [pow_le_one₀ hb h1 (n := n)]
  rw [hm, abs_mul, abs_of_nonneg hc]

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 := by norm_num

/-- Bounded coordinates convert gradient disagreement into a bound
on variance disagreement. Source: native non-AMSGrad AdamW at
0033b1b, squared gradient inputs. This term need not decay with age
when later gradient differences remain nonzero. -/
theorem second_moment_feedback_bound (beta coordinate difference : ℝ)
    (a b : ℕ → ℝ) (n : ℕ) (hb : 0 ≤ beta) (h1 : beta ≤ 1) (hc : 0 ≤ coordinate)
    (ha : ∀ k : ℕ, k < n → |a k| ≤ coordinate)
    (hbg : ∀ k : ℕ, k < n → |b k| ≤ coordinate)
    (hd : ∀ k : ℕ, k < n → |a k - b k| ≤ difference) :
    |secondMomentAt beta a n - secondMomentAt beta b n| ≤
      (1 - beta ^ n) * (2 * coordinate * difference) := by
  have hs : ∀ k : ℕ, k < n → |a k ^ 2 - b k ^ 2| ≤ 2 * coordinate * difference := by
    intro k hk
    have hsum : |a k + b k| ≤ 2 * coordinate :=
      le_trans (abs_add_le _ _) (by linarith [ha k hk, hbg k hk])
    have he : a k ^ 2 - b k ^ 2 = (a k - b k) * (a k + b k) := by ring
    rw [he, abs_mul]
    calc
      _ ≤ |a k - b k| * (2 * coordinate) := mul_le_mul_of_nonneg_left hsum (abs_nonneg _)
      _ ≤ difference * (2 * coordinate) := mul_le_mul_of_nonneg_right (hd k hk) (by positivity)
      _ = 2 * coordinate * difference := by ring
  have hm := moment_different_prefix_bound beta 0 0 (2 * coordinate * difference)
    (fun k => a k ^ 2) (fun k => b k ^ 2) n hb h1 hs
  simpa [secondMomentAt] using hm

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ k : ℕ, k < 3 → |(1 : ℝ)| ≤ 1) ∧ (∀ k : ℕ, k < 3 → |(-1 : ℝ)| ≤ 1) ∧
    (∀ k : ℕ, k < 3 → |(1 : ℝ) - (-1)| ≤ 2) := by norm_num

/-- A local derivative bound along supplied parameter paths yields
the stated first-moment feedback estimate. Source: native AdamW at
0033b1b and the previous exact recurrence law. Neither the local
smoothness condition nor the parameter-separation bound is derived
for GPTMini; an unconditional training contraction is not asserted. -/
theorem parameter_feedback_bound (beta lipschitz radius : ℝ) (loss : ℝ → ℝ)
    (left right : ℕ → ℝ) (n : ℕ) (hb : 0 ≤ beta) (h1 : beta ≤ 1) (hl : 0 ≤ lipschitz)
    (hr : ∀ k : ℕ, k < n → |left k - right k| ≤ radius)
    (hg : ∀ k : ℕ, k < n → |deriv loss (left k) - deriv loss (right k)| ≤
      lipschitz * |left k - right k|) :
    |firstMomentAt beta (fun k => deriv loss (left k)) n -
        firstMomentAt beta (fun k => deriv loss (right k)) n| ≤
      (1 - beta ^ n) * (lipschitz * radius) := by
  have hm := moment_different_prefix_bound beta 0 0 (lipschitz * radius)
    (fun k => deriv loss (left k)) (fun k => deriv loss (right k)) n hb h1 (fun k hk =>
      le_trans (hg k hk) (mul_le_mul_of_nonneg_left (hr k hk) hl))
  simpa [firstMomentAt] using hm

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 2 ∧
    (∀ k : ℕ, k < 3 → |(0 : ℝ) - 1| ≤ 1) ∧
    (∀ k : ℕ, k < 3 → |deriv (fun x : ℝ => x ^ 2) 0 - deriv (fun x : ℝ => x ^ 2) 1| ≤
      2 * |(0 : ℝ) - 1|) := by
  norm_num [(hasDerivAt_pow 2 (0 : ℝ)).deriv, (hasDerivAt_pow 2 (1 : ℝ)).deriv]

/-- A persistent constant input difference survives correction at
every positive clock, despite identical initial zero buffers. Source:
native AdamW at 0033b1b; countercase to extrapolating the shared-input
forgetting theorem to arbitrary different future gradient streams. -/
theorem corrected_constant_input_difference (beta left right : ℝ) (n : ℕ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (hn : 0 < n) :
    firstMomentAt beta (fun _ => left) n / (1 - beta ^ n) -
      firstMomentAt beta (fun _ => right) n / (1 - beta ^ n) = left - right := by
  have hl := (constant_gradient_corrected beta 0 left n hb h1 (by norm_num) (by norm_num) hn).1
  have hr := (constant_gradient_corrected beta 0 right n hb h1 (by norm_num) (by norm_num) hn).1
  rw [hl, hr]

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧ 0 < (1000 : ℕ) := by norm_num

/-- An explicit difference of two units remains in the corrected
first moments for arbitrarily long positive native histories.
Source: counterexample to input-independent optimizer forgetting;
the prescribed streams are not declared visited GPTMini trajectories. -/
theorem opposing_corrected_histories (n : ℕ) (hn : 0 < n) :
    firstMomentAt (9 / 10) (fun _ => 1) n / (1 - (9 / 10 : ℝ) ^ n) -
      firstMomentAt (9 / 10) (fun _ => -1) n / (1 - (9 / 10 : ℝ) ^ n) = 2 := by
  rw [corrected_constant_input_difference _ _ _ _ (by norm_num) (by norm_num) hn]
  norm_num

example : 0 < (1000 : ℕ) := by norm_num

end Transformer.Grokking.AdamW
