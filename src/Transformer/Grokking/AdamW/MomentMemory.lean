import Transformer.Grokking.AdamW.MomentBounds
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Direct memory decay of native AdamW buffers

Source: PyTorch 2.14.1 AdamW at lab commit 0033b1b, exponential
moment recurrences. The fixed subsequent gradient stream is explicit.
Changing parameters would generally change that stream, so these are
direct buffer-memory estimates, not forgetting of a training trajectory.

The preserved modular-division GPTMini uses beta1=0.9 and beta2=0.98.
Prove causal erasure estimates and convergence of the contribution of
any old buffer, with exact numerical attenuation bounds for those betas.
An adaptive update divides by a changing second moment and epsilon;
raw-moment attenuation is not automatically an update-size bound or
an explanation of the thirty-thousand-step memorization interval.
-/

namespace Transformer.Grokking.AdamW

/-- Exact absolute direct-memory difference for a nonnegative beta.
Source: native AdamW at 0033b1b; only the initial buffers differ,
and every subsequent supplied input agrees. -/
theorem moment_initial_difference_abs (beta left right : ℝ) (input : ℕ → ℝ)
    (n : ℕ) (hb : 0 ≤ beta) :
    |momentAt beta input left n - momentAt beta input right n| =
      beta ^ n * |left - right| := by
  rw [moment_initial_difference, abs_mul, abs_of_nonneg (pow_nonneg hb n)]

example : 0 ≤ (9 / 10 : ℝ) := by norm_num

/-- Erasing a buffer at completed clock n and then feeding exactly the
same k gradients removes precisely beta^k times that old buffer.
Source: PyTorch 2.14.1 AdamW at 0033b1b, actual continued recurrence;
this resets the buffer only, not the completed-update bias clock. -/
theorem prefix_erasure_difference (beta : ℝ) (gradient : ℕ → ℝ) (n k : ℕ) :
    firstMomentAt beta gradient (n + k) -
        momentAt beta (fun j => gradient (n + j)) 0 k =
      beta ^ k * firstMomentAt beta gradient n := by
  unfold firstMomentAt
  rw [moment_continue, moment_initial_difference]
  ring

/-- A bounded past gradient prefix bounds the erased first-moment
contribution at every subsequent age k, irrespective of later inputs.
Source: native AdamW at 0033b1b; this is causal, and equal later
gradients are a supplied comparison condition, not learned dynamics. -/
theorem prefix_erasure_abs_bound (beta bound : ℝ) (gradient : ℕ → ℝ) (n k : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1) (hc : 0 ≤ bound)
    (hg : ∀ j : ℕ, j < n → |gradient j| ≤ bound) :
    |firstMomentAt beta gradient (n + k) -
        momentAt beta (fun j => gradient (n + j)) 0 k| ≤ beta ^ k * bound := by
  have hm := moment_prefix_abs_bound beta bound 0 gradient n hb h1
    (by simpa using hc) hg
  rw [prefix_erasure_difference, abs_mul, abs_of_nonneg (pow_nonneg hb k)]
  exact mul_le_mul_of_nonneg_left hm (pow_nonneg hb k)

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧
    (∀ j : ℕ, j < 3 → |(if j = 0 then (1 : ℝ) else -1)| ≤ 1) := by
  refine ⟨by norm_num, by norm_num, by norm_num, ?_⟩
  intro j hj
  split_ifs <;> norm_num

/-- Erasing the actual second moment has its own beta2 decay and
square-gradient bound. Source: PyTorch 2.14.1 non-AMSGrad AdamW
at 0033b1b. This is a raw variance estimate; the corresponding
denominator perturbation is not bounded by the same expression. -/
theorem second_prefix_erasure_abs_bound (beta bound : ℝ) (gradient : ℕ → ℝ) (n k : ℕ)
    (hb : 0 ≤ beta) (h1 : beta ≤ 1)
    (hg : ∀ j : ℕ, j < n → |gradient j| ≤ bound) :
    |secondMomentAt beta gradient (n + k) -
        momentAt beta (fun j => gradient (n + j) ^ 2) 0 k| ≤ beta ^ k * bound ^ 2 := by
  unfold secondMomentAt
  exact prefix_erasure_abs_bound beta (bound ^ 2) (fun j => gradient j ^ 2) n k hb h1
    (sq_nonneg _) (fun j hj => by
      rw [abs_of_nonneg (sq_nonneg _)]
      exact sq_le_sq' (abs_le.mp (hg j hj)).1 (abs_le.mp (hg j hj)).2)

example : 0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) ≤ 1 ∧
    (∀ j : ℕ, j < 3 → |(-1 : ℝ)| ≤ 1) := by norm_num

/-- Every old finite buffer difference vanishes at valid native betas
under a shared subsequent stream. Source: AdamW at 0033b1b, linear
recurrence, and the standard geometric-power limit. No bound or
convergence of the supplied gradients is required. -/
theorem moment_memory_tendsto (beta left right : ℝ) (input : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) :
    Filter.Tendsto (fun n => |momentAt beta input left n - momentAt beta input right n|)
      Filter.atTop (nhds 0) := by
  have hp := (tendsto_pow_atTop_nhds_zero_of_lt_one hb h1).mul_const |left - right|
  have ht : Filter.Tendsto (fun n : ℕ => beta ^ n * |left - right|)
      Filter.atTop (nhds 0) := by simpa using hp
  exact Filter.Tendsto.congr (fun n => (moment_initial_difference_abs beta left right input n hb).symm) ht

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 := by norm_num

/-- Direct buffer memory eventually falls below any positive tolerance.
Source: native AdamW at 0033b1b, previous exact recurrence limit;
the tolerance concerns buffers and does not certify prediction accuracy. -/
theorem moment_memory_eventually_small (beta left right tolerance : ℝ) (input : ℕ → ℝ)
    (hb : 0 ≤ beta) (h1 : beta < 1) (ht : 0 < tolerance) :
    ∃ clock : ℕ, ∀ n : ℕ, clock ≤ n →
      |momentAt beta input left n - momentAt beta input right n| < tolerance := by
  have hm := moment_memory_tendsto beta left right input hb h1
  exact Filter.eventually_atTop.mp (Filter.Tendsto.eventually_lt_const ht hm)

example : 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    (0 : ℝ) < 1 / 1000000 := by norm_num

/-- Exact attenuation factors for the preserved lab betas. Source:
native AdamW at 0033b1b, beta1=0.9 and beta2=0.98. These are raw
moment contributions after 100 and 500 shared-gradient insertions,
not estimates of training's generalization time. -/
theorem native_raw_memory_attenuation :
    (9 / 10 : ℝ) ^ 100 < 1 / 10000 ∧ (49 / 50 : ℝ) ^ 500 < 1 / 10000 := by
  refine ⟨by norm_num, ?_⟩
  have hblock : (49 / 50 : ℝ) ^ 50 ≤ 3 / 8 := by norm_num
  calc
    (49 / 50 : ℝ) ^ 500 = ((49 / 50 : ℝ) ^ 50) ^ 10 := pow_mul _ 50 10
    _ ≤ (3 / 8 : ℝ) ^ 10 := pow_le_pow_left₀ (by positivity) hblock 10
    _ < 1 / 10000 := by norm_num

/-- The first-buffer difference stays attenuated after the lab's
100-insertion bound, including a zero old difference. Source: native
AdamW at 0033b1b, beta1=0.9 and the exact shared-input recurrence. -/
theorem native_first_memory_uniform (left right : ℝ) (input : ℕ → ℝ) (n : ℕ)
    (hn : 100 ≤ n) :
    |momentAt (9 / 10) input left n - momentAt (9 / 10) input right n| ≤
      |left - right| / 10000 := by
  rw [moment_initial_difference_abs _ _ _ _ _ (by norm_num)]
  have hp := pow_le_pow_of_le_one (show (0 : ℝ) ≤ 9 / 10 by norm_num)
    (show (9 / 10 : ℝ) ≤ 1 by norm_num) hn
  have ha := native_raw_memory_attenuation.1
  have hm := mul_le_mul_of_nonneg_right (le_trans hp (le_of_lt ha)) (abs_nonneg (left - right))
  linarith

example : 100 ≤ (500 : ℕ) := by norm_num

/-- The second-buffer difference stays attenuated after 500 shared
gradient insertions. Source: native non-AMSGrad AdamW at 0033b1b,
beta2=0.98; old buffer signs are not needed for this raw difference. -/
theorem native_second_memory_uniform (left right : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (hn : 500 ≤ n) :
    |momentAt (49 / 50) (fun k => gradient k ^ 2) left n -
        momentAt (49 / 50) (fun k => gradient k ^ 2) right n| ≤ |left - right| / 10000 := by
  rw [moment_initial_difference_abs _ _ _ _ _ (by norm_num)]
  have hp := pow_le_pow_of_le_one (show (0 : ℝ) ≤ 49 / 50 by norm_num)
    (show (49 / 50 : ℝ) ≤ 1 by norm_num) hn
  have ha := native_raw_memory_attenuation.2
  have hm := mul_le_mul_of_nonneg_right (le_trans hp (le_of_lt ha)) (abs_nonneg (left - right))
  linarith

example : 500 ≤ (1000 : ℕ) := by norm_num

/-- The strict beta<1 condition is necessary for forgetting an initial
buffer. Source: endpoint counterexample to extrapolating the native
recurrence limit beyond the documented AdamW beta range. -/
theorem unit_beta_preserves_difference (input : ℕ → ℝ) (n : ℕ) :
    |momentAt 1 input 1 n - momentAt 1 input 0 n| = 1 := by
  rw [moment_initial_difference]
  norm_num

end Transformer.Grokking.AdamW
