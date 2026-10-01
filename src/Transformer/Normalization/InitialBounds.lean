/-
# Absolute constants for the initialization estimate

Elementary scalar estimates translating the numerator bound into the
rate in Theorem 4.2, §4.2 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.InitialAttention

namespace Transformer.Normalization

/-- A common threshold for every attention numerator. Source:
arXiv:2510.22026v2, Appendix B, self term, bias, and vector fluctuation
in the proof of Theorem 4.2. -/
noncomputable def initialThreshold (d n : ℕ) : ℝ :=
  3 + 2 * (n : ℝ) / d + 6 * Real.sqrt n + 12 * Real.sqrt ((n : ℝ) * Real.log n)

/-- For at least two tokens, the logarithm is bounded below by one half.
Source: arXiv:2510.22026v2, §4.2, constants in Theorem 4.2. -/
theorem half_le_log_nat {n : ℕ} (hn : 2 ≤ n) : (1 : ℝ) / 2 ≤ Real.log n := by
  have hn' : (2 : ℝ) ≤ n := by exact_mod_cast hn
  have h := Real.log_le_log (by norm_num : (0 : ℝ) < 2) hn'
  linarith [Real.log_two_gt_d9]

/-- Two tokens realize the token-count hypothesis. -/
example : 2 ≤ (2 : ℕ) := le_rfl

/-- The numerator threshold gives the paper's rate with `C=128`.
Source: arXiv:2510.22026v2, §4.2, Theorem 4.2. -/
theorem initialThreshold_rate {d n : ℕ} (hd : 0 < d) (hn : 2 ≤ n)
    (hdim : (d : ℝ) ≤ (n : ℝ) * Real.log n) :
    3 / (n : ℝ) * initialThreshold d n ≤
      128 * (Real.sqrt (Real.log n / n) + Real.log n / d) := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast (lt_of_lt_of_le (by decide : 0 < 2) hn)
  have hd' : (0 : ℝ) < d := by exact_mod_cast hd
  have hlog := half_le_log_nat hn
  have hl : 0 ≤ Real.log n := by linarith
  let a := Real.sqrt (Real.log n / n)
  let b := Real.log n / d
  have ha : 0 ≤ a := Real.sqrt_nonneg _
  have hb : 0 ≤ b := div_nonneg hl hd'.le
  have hroot : Real.sqrt ((n : ℝ) * Real.log n) = (n : ℝ) * a := by
    apply (sq_eq_sq₀ (Real.sqrt_nonneg _) (mul_nonneg hn'.le ha)).mp
    rw [Real.sq_sqrt (mul_nonneg hn'.le hl), mul_pow,
      show a ^ 2 = Real.log n / n from Real.sq_sqrt (div_nonneg hl hn'.le)]
    field_simp
  have hlogroot : (1 : ℝ) / 2 ≤ Real.sqrt (Real.log n) := by
    nlinarith [Real.sq_sqrt hl, Real.sqrt_nonneg (Real.log n)]
  have hmean : Real.sqrt n ≤ 2 * (n : ℝ) * a := by
    have h := mul_le_mul_of_nonneg_left hlogroot (Real.sqrt_nonneg (n : ℝ))
    rw [← Real.sqrt_mul hn'.le, hroot] at h
    linarith
  have hself : 9 / (n : ℝ) ≤ 9 * b := by
    dsimp [b]
    rw [← mul_div_assoc]
    apply (div_le_div_iff₀ hn' hd').mpr
    nlinarith
  have hmean' : 18 * Real.sqrt n / (n : ℝ) ≤ 36 * a := by
    apply (div_le_iff₀ hn').mpr
    nlinarith
  have hbias : 6 / (d : ℝ) ≤ 12 * b := by
    dsimp [b]
    rw [← mul_div_assoc]
    exact div_le_div_of_nonneg_right (by linarith) hd'.le
  have heq : 3 / (n : ℝ) * initialThreshold d n =
      9 / (n : ℝ) + 6 / (d : ℝ) + 18 * Real.sqrt n / (n : ℝ) + 36 * a := by
    dsimp [initialThreshold]
    rw [hroot]
    field_simp
    ring
  rw [heq]
  change _ ≤ 128 * (a + b)
  linarith

/-- Positive dimension and two tokens realize the hypotheses. -/
example : 0 < (1 : ℕ) ∧ 2 ≤ (2 : ℕ) ∧ (1 : ℝ) ≤ 2 * Real.log 2 :=
  ⟨by decide, le_rfl, by linarith [Real.log_two_gt_d9]⟩

/-- The scalar tail used for the other `n` tokens is at most `(n+1)^{-2}`.
Source: arXiv:2510.22026v2, Appendix B, the union-bound input in Theorem 4.2. -/
theorem initial_tail_exp_le {n : ℕ} (hn : 0 < n) :
    Real.exp (-(12 * Real.sqrt (((n + 1 : ℕ) : ℝ) * Real.log ((n + 1 : ℕ) : ℝ))) ^ 2 /
      (72 * (n : ℝ))) ≤ (((n + 1 : ℕ) : ℝ) ^ 2)⁻¹ := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have hN : (0 : ℝ) < (n + 1 : ℕ) := by positivity
  have hlog : 0 ≤ Real.log ((n + 1 : ℕ) : ℝ) :=
    (by linarith [half_le_log_nat (by omega : 2 ≤ n + 1)])
  have hexp : -(12 * Real.sqrt (((n + 1 : ℕ) : ℝ) * Real.log ((n + 1 : ℕ) : ℝ))) ^ 2 /
      (72 * (n : ℝ)) ≤ -(2 * Real.log ((n + 1 : ℕ) : ℝ)) := by
    rw [mul_pow, Real.sq_sqrt (mul_nonneg hN.le hlog)]
    apply (div_le_iff₀ (by positivity : (0 : ℝ) < 72 * (n : ℝ))).mpr
    push_cast at hlog ⊢
    nlinarith
  apply (Real.exp_le_exp.mpr hexp).trans_eq
  rw [Real.exp_neg, show (2 : ℝ) = (2 : ℕ) by norm_num,
    Real.exp_nat_mul, Real.exp_log hN]

/-- One remaining token realizes the positive-count hypothesis. -/
example : 0 < (1 : ℕ) := by decide

end Transformer.Normalization
