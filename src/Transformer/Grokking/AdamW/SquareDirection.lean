import Transformer.Grokking.AdamW.SquareHistory

/-!
# Uniform actual corrected AdamW direction bounds

Source: PyTorch 2.14.1 non-AMSGrad AdamW, adam.py lines 445--475
and 529--547, ported at 88aa892/0033b1b. Zero native buffers and
their completed-update clock generate both corrected moments.
Combine the finite-history square budget with the exact two bias
corrections, without replacing either correction by its limit.

For beta1 squared below beta2, the absolute actual normalized
direction is bounded by the square root of the numerical coupled
scale. Epsilon is positive but its magnitude does not enter that
ceiling. No input bound, sign, convergence, clipping certificate or
future buffer inequality is a hypothesis. With the original
beta1=0.9/beta2=0.98, the ceiling is sqrt(49/17), strictly below 1.7.

The scalar-path application derives this same bound from actual
updates and zero initial buffers/clock. Inputs can be changing
learned-coordinate minibatch gradients if that exact recurrence
bridge is supplied. Neither a learned-head bridge nor stochastic
task generalization is implied by numerical boundedness. Exact-real
estimates do not certify floating-point kernels. Frozen optimizer
states and training schedules are unchanged.
-/

namespace Transformer.Grokking.AdamW

/-- Actual initialized corrected moments obey the constant square
scale at every positive clock. Source: both native bias corrections
at 0033b1b and the generated SquareHistory budget; no future moment
bound or asymptotic correction substitution is assumed. -/
theorem native_initialized_corrected_square (b1 b2 : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) (hn : 0 < n) :
    (firstMomentAt b1 gradient n / (1 - b1 ^ n)) ^ 2 ≤
      nativeMomentSquareScale b1 b2 * (secondMomentAt b2 gradient n / (1 - b2 ^ n)) := by
  have hb2 : 0 < b2 := by nlinarith only [hgap, sq_nonneg b1]
  have hc1 := bias_correction_positive b1 n hb1 h1 hn
  have hc2 := bias_correction_positive b2 n (le_of_lt hb2) h2 hn
  have hv := second_moment_nonneg b2 gradient n (le_of_lt hb2) (le_of_lt h2)
  have hk := native_moment_square_scale_pos b1 b2 h1 h2 hgap
  have hraw := native_initialized_history_square b1 b2 gradient h1 h2 hgap n
  have hleft := mul_le_mul_of_nonneg_right hraw (le_of_lt hc2)
  have hright := mul_le_mul_of_nonneg_left (native_history_correction_square b1 b2 n hb1 hb2)
    (mul_nonneg (le_of_lt hk) hv)
  have hcombined : firstMomentAt b1 gradient n ^ 2 * (1 - b2 ^ n) ≤
      nativeMomentSquareScale b1 b2 * secondMomentAt b2 gradient n * (1 - b1 ^ n) ^ 2 := by
    calc
      _ ≤ (nativeHistorySquareBudget b1 b2 n * secondMomentAt b2 gradient n) * (1 - b2 ^ n) := hleft
      _ = nativeMomentSquareScale b1 b2 * secondMomentAt b2 gradient n *
          ((1 - (b1 ^ 2 / b2) ^ n) * (1 - b2 ^ n)) := by unfold nativeHistorySquareBudget; ring
      _ ≤ _ := hright
  rw [div_pow]
  apply (div_le_iff₀ (sq_pos_of_pos hc1)).mpr
  have ht := (le_div_iff₀ hc2).mpr hcombined
  calc
    _ ≤ (nativeMomentSquareScale b1 b2 * secondMomentAt b2 gradient n * (1 - b1 ^ n) ^ 2) /
        (1 - b2 ^ n) := ht
    _ = _ := by ring

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧
    (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧ 0 < (3 : ℕ) := by norm_num

/-- The generated corrected native direction has an absolute
ceiling independent of epsilon and gradient magnitude. Source:
non-AMSGrad addcdiv at 0033b1b; the same actual retained histories
generate the numerator, variance and both corrections. -/
theorem native_initialized_direction_ceiling (b1 b2 eps : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2)
    (he : 0 < eps) (hn : 0 < n) :
    |historyDirection b1 b2 eps gradient n| ≤ Real.sqrt (nativeMomentSquareScale b1 b2) := by
  let moment := firstMomentAt b1 gradient n / (1 - b1 ^ n)
  let variance := secondMomentAt b2 gradient n / (1 - b2 ^ n)
  have hb2 : 0 < b2 := by nlinarith only [hgap, sq_nonneg b1]
  have hc2 := bias_correction_positive b2 n (le_of_lt hb2) h2 hn
  have hv : 0 ≤ variance := div_nonneg
    (second_moment_nonneg b2 gradient n (le_of_lt hb2) (le_of_lt h2)) (le_of_lt hc2)
  have hk := native_moment_square_scale_pos b1 b2 h1 h2 hgap
  have hm : moment ^ 2 ≤ nativeMomentSquareScale b1 b2 * variance :=
    native_initialized_corrected_square b1 b2 gradient n hb1 h1 h2 hgap hn
  have hs : (Real.sqrt (nativeMomentSquareScale b1 b2) * Real.sqrt variance) ^ 2 =
      nativeMomentSquareScale b1 b2 * variance := by
    rw [mul_pow, Real.sq_sqrt (le_of_lt hk), Real.sq_sqrt hv]
  have hprod := mul_nonneg (Real.sqrt_nonneg (nativeMomentSquareScale b1 b2)) (Real.sqrt_nonneg variance)
  have hab : |moment| ≤ Real.sqrt (nativeMomentSquareScale b1 b2) * Real.sqrt variance := by
    nlinarith only [hm, hs, hprod, abs_nonneg moment, sq_abs moment]
  have hd : 0 < Real.sqrt variance + eps := by linarith only [Real.sqrt_nonneg variance, he]
  change |moment / (Real.sqrt variance + eps)| ≤ _
  rw [abs_div, abs_of_pos hd]
  apply (div_le_iff₀ hd).mpr
  have hextra := mul_nonneg (Real.sqrt_nonneg (nativeMomentSquareScale b1 b2)) (le_of_lt he)
  nlinarith only [hab, hextra]

example : (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧
    (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧ (0 : ℝ) < 1 / 100000000 ∧ 0 < (3 : ℕ) := by norm_num

/-- Original retained betas give a strict rational direction
ceiling. Source: native betas at 0033b1b and the exact coupled
scale 49/17; the decimal 1.7 is the rational 17/10. -/
theorem native_original_square_root_ceiling :
    Real.sqrt (nativeMomentSquareScale (9 / 10) (49 / 50)) < (17 / 10 : ℝ) := by
  rw [native_original_moment_square_scale]
  have hs := Real.sq_sqrt (show (0 : ℝ) ≤ 49 / 17 by norm_num)
  nlinarith only [hs]

/-- Every original-beta corrected direction stays below 1.7,
regardless of gradient stream or positive epsilon magnitude.
Source: original native AdamW at 0033b1b; all completed clocks
and both retained moments remain in the actual expression. -/
theorem native_original_direction_ceiling (eps : ℝ) (gradient : ℕ → ℝ) (n : ℕ)
    (he : 0 < eps) (hn : 0 < n) : |historyDirection (9 / 10) (49 / 50) eps gradient n| < (17 / 10 : ℝ) := by
  exact lt_of_le_of_lt
    (native_initialized_direction_ceiling (9 / 10) (49 / 50) eps gradient n
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) he hn)
    native_original_square_root_ceiling

example : (0 : ℝ) < 1 / 100000000 ∧ 0 < (3 : ℕ) := by norm_num

/-- Actual zero-buffer/zero-clock scalar paths inherit the uniform
native direction ceiling at every update. Source: actual retained
scalar step at 88aa892; generated history fields remove every
future variance/direction guard from the hypotheses. -/
theorem scalar_initialized_direction_ceiling (b1 b2 eps decay rate : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) (he : 0 < eps) :
    ∀ n, |nextBufferDirection b1 b2 eps (state n).moment (state n).variance (gradient n) (state n).clock| ≤
      Real.sqrt (nativeMomentSquareScale b1 b2) := by
  intro n
  have hf := scalar_path_fields b1 b2 eps decay rate state gradient n hstep
  have ht := native_initialized_direction_ceiling b1 b2 eps gradient (n + 1) hb1 h1 h2 hgap he (by omega)
  simpa only [nextBufferDirection, hf.1, hf.2.1, hf.2.2, hm, hv, hc, zero_add,
    historyDirection, firstMomentAt, secondMomentAt, momentAt] using ht

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧
    (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧ (0 : ℝ) < 1 / 100000000 := by
  exact ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm,
    rfl, rfl, rfl, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- Every actual original-beta scalar insertion stays below the
rational direction ceiling. Source: retained native recurrence at
88aa892/0033b1b; zero initial moments/clock generate the needed
square history, with no future gradient or buffer bound. -/
theorem scalar_original_direction_ceiling (eps decay rate : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep (9 / 10) (49 / 50) eps decay rate (state n) (gradient n))
    (hm : (state 0).moment = 0) (hv : (state 0).variance = 0) (hc : (state 0).clock = 0) (he : 0 < eps) :
    ∀ n, |nextBufferDirection (9 / 10) (49 / 50) eps (state n).moment (state n).variance
      (gradient n) (state n).clock| < (17 / 10 : ℝ) := by
  intro n
  exact lt_of_le_of_lt
    (scalar_initialized_direction_ceiling (9 / 10) (49 / 50) eps decay rate state gradient
      hstep hm hv hc (by norm_num) (by norm_num) (by norm_num) (by norm_num) he n)
    native_original_square_root_ceiling

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (zeroScalarStateAt 0).moment = 0 ∧ (zeroScalarStateAt 0).variance = 0 ∧ (zeroScalarStateAt 0).clock = 0 ∧
    (0 : ℝ) < 1 / 100000000 := by
  exact ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm, rfl, rfl, rfl, by norm_num⟩

end Transformer.Grokking.AdamW
