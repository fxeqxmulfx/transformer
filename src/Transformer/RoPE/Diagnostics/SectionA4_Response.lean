module

public import Transformer.RoPE.Scores
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

/-!
# Exact adjacent-position response bounds

Wu, Du and Peng, *RoPE at the End of Its Rope?*, arXiv:2609.39929v1,
Appendices A.2, A.4 and C.6. These deterministic bounds hold for arbitrary
fixed Q/K coefficients. They do not assume the calibrated-Gaussian conditions
of the paper's separate context-length theorem.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.RoPE.Diagnostics

/-- Appendix C.6: the exact response to moving a token by one position. -/
theorem adjacent_difference (a b omega distance : ℝ) :
    score a b omega (distance + 1) - score a b omega distance =
      2 * Real.sin (omega / 2) * score b (-a) omega (distance + 1 / 2) := by
  have hc := Real.cos_sub_cos (x := omega * (distance + 1)) (y := omega * distance)
  have hs := Real.sin_sub_sin (x := omega * (distance + 1)) (y := omega * distance)
  have hsum : (omega * (distance + 1) + omega * distance) / 2 =
      omega * (distance + 1 / 2) := by ring
  have hdiff : (omega * (distance + 1) - omega * distance) / 2 = omega / 2 := by ring
  rw [hsum, hdiff] at hc hs
  unfold score
  linear_combination a * hc + b * hs

/-- Appendix A.4/C.6: a single frequency's response is bounded by its
amplitude times the exact factor `2 |sin(omega/2)|`. -/
theorem adjacent_response_bound (a b omega distance : ℝ) :
    |score a b omega (distance + 1) - score a b omega distance| ≤
      2 * |Real.sin (omega / 2)| * Real.sqrt (a ^ 2 + b ^ 2) := by
  rw [adjacent_difference, abs_mul, abs_mul, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
  have hbound := score_amplitude_bound b (-a) omega (distance + 1 / 2)
  rw [show b ^ 2 + (-a) ^ 2 = a ^ 2 + b ^ 2 by ring] at hbound
  exact mul_le_mul_of_nonneg_left hbound (by positivity)

/-- Appendix C.6: low frequencies have at most frequency-proportional local
response, even if they cover a very long positional field. -/
theorem adjacent_response_frequency_bound (a b omega distance : ℝ) :
    |score a b omega (distance + 1) - score a b omega distance| ≤
      |omega| * Real.sqrt (a ^ 2 + b ^ 2) := by
  apply (adjacent_response_bound a b omega distance).trans
  have hs : 2 * |Real.sin (omega / 2)| ≤ |omega| := by
    have h := Real.abs_sin_le_abs (x := omega / 2)
    rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)] at h
    linarith
  exact mul_le_mul_of_nonneg_right hs (Real.sqrt_nonneg _)

/-- Appendix A.4: sum the exact bandwise bounds for a whole fully rotary head.
The coefficient magnitudes may differ arbitrarily between coordinates. -/
theorem head_adjacent_response_bound {n : ℕ} (a b omega : Fin n → ℝ) (distance : ℝ) :
    |(∑ i, score (a i) (b i) (omega i) (distance + 1)) -
      ∑ i, score (a i) (b i) (omega i) distance| ≤
        ∑ i, 2 * |Real.sin (omega i / 2)| * Real.sqrt (a i ^ 2 + b i ^ 2) := by
  rw [← Finset.sum_sub_distrib]
  exact (Finset.abs_sum_le_sum_abs _ _).trans
    (Finset.sum_le_sum fun i _ => adjacent_response_bound _ _ _ _)

end Transformer.RoPE.Diagnostics
