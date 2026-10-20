module

public import Transformer.RoPE.FrequencyMatching.Section3_Optimality

/-!
# Worst-case first-to-last dependence

Derived consequences of Wu et al., arXiv:2607.07678v1, §3, for a sequence of
`length + 1` tokens. The maximum token distance is `length`. The base threshold
below enforces the source's scalar half-period criterion only; it is not a
guarantee of retrieval accuracy or an optimum of language-model loss.
-/

@[expose] public section

open MeasureTheory Set

namespace Transformer.RoPE.FrequencyMatching

/-- Every token distance is bounded by the endpoint distance, and the first
and last token attain this bound. Zero-based indexing is explicit. -/
theorem token_distance_bound (length : ℕ) :
    (∀ i j : Fin (length + 1), |(i : ℝ) - (j : ℝ)| ≤ (length : ℝ)) ∧
      |((⟨length, by omega⟩ : Fin (length + 1)) : ℝ) -
        ((0 : Fin (length + 1)) : ℝ)| = (length : ℝ) := by
  constructor
  · intro i j
    have hi : (i : ℝ) ≤ (length : ℝ) := by exact_mod_cast (show i.val ≤ length by omega)
    have hj : (j : ℝ) ≤ (length : ℝ) := by exact_mod_cast (show j.val ≤ length by omega)
    have hi0 : 0 ≤ (i : ℝ) := Nat.cast_nonneg i.val
    have hj0 : 0 ≤ (j : ℝ) := Nat.cast_nonneg j.val
    exact abs_le.mpr ⟨by linarith, by linarith⟩
  · simp

/-- Worst-case dependence in a nontrivial finite sequence gives the exact
source optimizer `pi / length`, with `length = number of tokens - 1`. -/
theorem worst_case_optimum (length : ℕ) (hlength : 0 < length)
    (mu : Measure ℝ) [IsProbabilityMeasure mu]
    (hwindow : ∀ᵐ r ∂mu, r ∈ Icc 0 (length : ℝ))
    (hmass : 0 < mu (Ioc 0 (length : ℝ))) :
    Admissible (length : ℝ) (Real.pi / length) ∧
      ∀ omega, Admissible (length : ℝ) omega →
        utility mu omega ≤ utility mu (Real.pi / length) ∧
          (utility mu omega = utility mu (Real.pi / length) ↔ omega = Real.pi / length) := by
  exact unique_optimal_frequency mu (by exact_mod_cast hlength) hwindow hmass

example : Admissible 1 (Real.pi / 1) ∧
    ∀ omega, Admissible 1 omega →
      utility (Measure.dirac 1) omega ≤ utility (Measure.dirac 1) (Real.pi / 1) ∧
        (utility (Measure.dirac 1) omega = utility (Measure.dirac 1) (Real.pi / 1) ↔
          omega = Real.pi / 1) := by
  simpa only [Nat.cast_one] using
    worst_case_optimum 1 (by omega) (Measure.dirac 1) (by simp) (by simp)

/-- Solving half-period coverage for the slowest RoPE frequency gives the
exact base threshold. This is a derived scalar coverage statement, not Men
et al.'s different expected-score discrimination bound. -/
theorem slow_frequency_base_threshold {base dimension width : ℝ}
    (hbase : 0 < base) (hdimension : 2 < dimension) (hwidth : 0 < width) :
    Admissible width (slowFrequency base dimension) ↔
      (width / Real.pi) ^ (dimension / (dimension - 2)) ≤ base := by
  have hd : dimension ≠ 0 := ne_of_gt (lt_trans (by norm_num) hdimension)
  have hd2 : dimension - 2 ≠ 0 := ne_of_gt (by linarith)
  have hinv : 0 < dimension / (dimension - 2) := div_pos (by linarith) (by linarith)
  have hpow : 0 < base ^ ((dimension - 2) / dimension) := Real.rpow_pos_of_pos hbase _
  have hroot : (base ^ ((dimension - 2) / dimension)) ^
      (dimension / (dimension - 2)) = base := by
    rw [← Real.rpow_mul hbase.le]
    rw [show (dimension - 2) / dimension * (dimension / (dimension - 2)) = 1 by
      field_simp]
    exact Real.rpow_one base
  have horder := Real.rpow_le_rpow_iff (div_nonneg hwidth.le Real.pi_pos.le)
    hpow.le hinv
  rw [hroot] at horder
  rw [Admissible, slowFrequency, and_iff_right (Real.rpow_pos_of_pos hbase _)]
  rw [show -(dimension - 2) / dimension = -((dimension - 2) / dimension) by ring,
    Real.rpow_neg hbase.le]
  calc
    (base ^ ((dimension - 2) / dimension))⁻¹ * width ≤ Real.pi
        ↔ width ≤ Real.pi * base ^ ((dimension - 2) / dimension) := by
          rw [inv_mul_eq_div, div_le_iff₀ hpow]
    _ ↔ width / Real.pi ≤ base ^ ((dimension - 2) / dimension) :=
      by rw [div_le_iff₀ Real.pi_pos, mul_comm Real.pi]
    _ ↔ (width / Real.pi) ^ (dimension / (dimension - 2)) ≤ base := horder.symm

example : Admissible Real.pi (slowFrequency 2 4) ↔
    (Real.pi / Real.pi) ^ ((4 : ℝ) / (4 - 2)) ≤ 2 :=
  slow_frequency_base_threshold (by norm_num) (by norm_num) Real.pi_pos

end Transformer.RoPE.FrequencyMatching
