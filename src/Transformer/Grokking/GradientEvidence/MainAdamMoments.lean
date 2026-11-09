module

public import Transformer.Grokking.GradientEvidence.AdaptiveForcing

/-!
# A gradient-independent bound for the main model's AdamW moments

Source: `review/flat_spectrum.md`, *Main-model AdamW motion*, AD1–AD2.
The exact-real recurrences use beta1=.9 and beta2=.95 from the copied
main-model kernels. Gradients may depend on the full tied model, past
states, minibatches and clipping. Explicit roundoff is a separate term.
This controls step size, not the sign or alignment of the direction.
-/

@[expose] public section
noncomputable section
namespace Transformer.Grokking.GradientEvidence

/-- AD1: the main model's first-moment update in exact real arithmetic. -/
def mainFirst (m g : ℝ) : ℝ := 9 / 10 * m + 1 / 10 * g

/-- AD1: the main model's second-moment update, including the squared gradient. -/
def mainSecond (v g : ℝ) : ℝ := 19 / 20 * v + 1 / 20 * g ^ 2

/-- AD1: actual moment buffers from zero for any sequence of coordinate gradients. -/
def mainMoments (g : ℕ → ℝ) : ℕ → ℝ × ℝ
  | 0 => (0, 0)
  | t + 1 => (mainFirst (mainMoments g t).1 (g t),
      mainSecond (mainMoments g t).2 (g t))

/-- AD1: the moment-cone slack is a decayed previous slack plus a square. -/
theorem main_moment_slack_identity (m v g : ℝ) :
    19 / 14 * mainSecond v g - (mainFirst m g) ^ 2 =
      19 / 20 * (19 / 14 * v - m ^ 2) + (14 * m - 9 * g) ^ 2 / 1400 := by
  unfold mainFirst mainSecond
  ring

/-- AD1: the explicit cone is invariant under every gradient value. -/
theorem main_moment_step_bound (m v g : ℝ) (h : m ^ 2 ≤ 19 / 14 * v) :
    (mainFirst m g) ^ 2 ≤ 19 / 14 * mainSecond v g := by
  have he := main_moment_slack_identity m v g
  nlinarith [sq_nonneg (14 * m - 9 * g)]

/-- AD1: no favorable-gradient or fixed-feature hypothesis is needed at any
time. This is an exact-real optimizer theorem for the specified betas. -/
theorem main_moment_invariant (g : ℕ → ℝ) (t : ℕ) :
    (mainMoments g t).1 ^ 2 ≤ 19 / 14 * (mainMoments g t).2 := by
  induction t with
  | zero => norm_num [mainMoments]
  | succ t ih => exact main_moment_step_bound _ _ _ ih

/-- AD1: the same invariant also implies nonnegative second moments. -/
theorem main_second_nonneg (g : ℕ → ℝ) (t : ℕ) :
    0 ≤ (mainMoments g t).2 := by
  have h := main_moment_invariant g t
  nlinarith [sq_nonneg (mainMoments g t).1]

/-- AD1: perturbations of the actual buffers are kept explicitly. Their
budget is not inferred from the exact-real recurrence. -/
theorem main_moment_roundoff_bound (m v g em ev : ℝ) :
    (mainFirst m g + em) ^ 2 - 19 / 14 * (mainSecond v g + ev) ≤
      19 / 20 * (m ^ 2 - 19 / 14 * v) +
      2 * |mainFirst m g| * |em| + em ^ 2 + 19 / 14 * |ev| := by
  have he := main_moment_slack_identity m v g
  have hp := le_abs_self (mainFirst m g * em)
  rw [abs_mul] at hp
  have hv := neg_le_abs ev
  nlinarith [sq_nonneg (14 * m - 9 * g)]

/-- AD2: a moment cone bounds normalization even when the variance is zero;
positive epsilon prevents division by a zero denominator. -/
theorem normalized_moment_bound (m v K eps : ℝ) (hv : 0 ≤ v) (hK : 0 ≤ K)
    (heps : 0 < eps) (h : m ^ 2 ≤ K ^ 2 * v) :
    |m / (Real.sqrt v + eps)| ≤ K := by
  have hs := Real.sq_sqrt hv
  have hm : |m| ≤ K * Real.sqrt v := by
    apply abs_le_of_sq_le_sq _ (mul_nonneg hK (Real.sqrt_nonneg v))
    nlinarith
  have hd : 0 < Real.sqrt v + eps := by positivity
  rw [abs_div, abs_of_pos hd, div_le_iff₀ hd]
  nlinarith

/-- AD2: include both bias corrections and epsilon, without replacing the
stored first moment by a current gradient. -/
theorem corrected_moment_bound (m v K b1 b2 eps : ℝ) (hv : 0 ≤ v) (hK : 0 ≤ K)
    (h1 : 0 < b1) (h2 : 0 < b2) (heps : 0 < eps)
    (h : m ^ 2 ≤ K ^ 2 * v) :
    |(m / b1) / (Real.sqrt v / b2 + eps)| ≤ K * b2 / b1 := by
  have hb := normalized_moment_bound m v K (b2 * eps) hv hK (mul_pos h2 heps) h
  have he : (m / b1) / (Real.sqrt v / b2 + eps) =
      (b2 / b1) * (m / (Real.sqrt v + b2 * eps)) := by
    field_simp
  rw [he, abs_mul, abs_of_pos (div_pos h2 h1)]
  have hm := mul_le_mul_of_nonneg_left hb (div_pos h2 h1).le
  convert hm using 1
  ring

/-- AD2: the time-dependent bound retains the exact bias-correction factors. -/
def mainDirectionBound (t : ℕ) : ℝ :=
  Real.sqrt (19 / 14) * Real.sqrt (1 - (19 / 20 : ℝ) ^ (t + 1)) /
    (1 - (9 / 10 : ℝ) ^ (t + 1))

/-- AD2: the main model's buffered adaptive direction from its actual moments. -/
def mainDirection (g : ℕ → ℝ) (eps : ℝ) (t : ℕ) : ℝ :=
  ((mainMoments g (t+1)).1 / (1 - (9 / 10 : ℝ) ^ (t+1))) /
    (Real.sqrt (mainMoments g (t+1)).2 /
      Real.sqrt (1 - (19 / 20 : ℝ) ^ (t+1)) + eps)

/-- AD2: every coordinate of exact main-model AdamW has a bounded adaptive
direction for every gradient history, including coupled feature learning. -/
theorem main_direction_bound (g : ℕ → ℝ) (eps : ℝ) (heps : 0 < eps) (t : ℕ) :
    |mainDirection g eps t| ≤ mainDirectionBound t := by
  have h1 : 0 < 1 - (9 / 10 : ℝ) ^ (t+1) := by
    have h := pow_lt_one₀ (by norm_num : (0 : ℝ) ≤ 9/10)
      (by norm_num : (9/10 : ℝ) < 1) (by omega : t+1 ≠ 0)
    linarith
  have h2 : 0 < Real.sqrt (1 - (19 / 20 : ℝ) ^ (t+1)) := by
    apply Real.sqrt_pos.mpr
    have h := pow_lt_one₀ (by norm_num : (0 : ℝ) ≤ 19/20)
      (by norm_num : (19/20 : ℝ) < 1) (by omega : t+1 ≠ 0)
    linarith
  apply corrected_moment_bound _ _ _ _ _ _ (main_second_nonneg g (t+1))
    (Real.sqrt_nonneg _) h1 h2 heps
  rw [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 19/14)]
  exact main_moment_invariant g (t+1)

example := main_moment_step_bound (9/14) ((9/14 : ℝ)^2/(19/14)) 1 (by norm_num)
example := normalized_moment_bound 1 1 2 (1/100000000)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example := corrected_moment_bound 1 1 2 (1/10) (1/5) (1/100000000)
  (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
example (t : ℕ) := main_direction_bound (fun n => if n % 2 = 0 then 1 else -2)
  (1/100000000) (by norm_num) t

end Transformer.Grokking.GradientEvidence
