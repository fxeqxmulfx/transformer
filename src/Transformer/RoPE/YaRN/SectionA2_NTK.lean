module

public import Transformer.RoPE.Basic

/-!
# NTK-aware base scaling

Peng et al., *YaRN*, arXiv:2309.00071v3, Appendix A.2, Eqs. (19)–(22).
These are exact frequency identities. The manuscript explicitly says that
the resulting nominal extension factor need not equal the empirically usable
context extension. No loss or retrieval guarantee is inferred here.
-/

@[expose] public section

namespace Transformer.RoPE.YaRN

/-- YaRN Eq. (19), with a real rotary dimension for the algebraic identities. -/
noncomputable def adjustedBase (base scale dimension : ℝ) : ℝ :=
  base * scale ^ (dimension / (dimension - 2))

/-- Eq. (22): changing the base interpolates the slowest frequency by exactly
the nominal factor. The dimension must exceed two. -/
theorem slow_frequency_adjusted {base scale dimension : ℝ}
    (hbase : 0 < base) (hscale : 0 < scale) (hdimension : 2 < dimension) :
    slowFrequency (adjustedBase base scale dimension) dimension =
      slowFrequency base dimension / scale := by
  have hd : dimension ≠ 0 := ne_of_gt (by linarith)
  have hd2 : dimension - 2 ≠ 0 := ne_of_gt (by linarith)
  unfold slowFrequency adjustedBase
  rw [Real.mul_rpow hbase.le (Real.rpow_nonneg hscale.le _), ← Real.rpow_mul hscale.le]
  rw [show dimension / (dimension - 2) * (-(dimension - 2) / dimension) = -1 by
    field_simp]
  rw [Real.rpow_neg_one]
  simp only [div_eq_mul_inv]

example : slowFrequency (adjustedBase 10000 4 128) 128 =
    slowFrequency 10000 128 / 4 :=
  slow_frequency_adjusted (by norm_num) (by norm_num) (by norm_num)

/-- Appendix A.2: every coordinate has its own scaling factor. In particular,
this is not uniform position interpolation across the frequency grid. -/
theorem coordinate_frequency_adjusted {base scale : ℝ} {dimension : ℕ}
    (hbase : 0 < base) (hscale : 0 < scale) (hdimension : 2 < dimension) (index : ℕ) :
    frequency (adjustedBase base scale dimension) dimension index =
      frequency base dimension index / scale ^ (2 * (index : ℝ) / ((dimension : ℝ) - 2)) := by
  have hd : (dimension : ℝ) ≠ 0 := by exact_mod_cast (by omega : dimension ≠ 0)
  have hd2 : (dimension : ℝ) - 2 ≠ 0 := by
    have : (2 : ℝ) < dimension := by exact_mod_cast hdimension
    linarith
  unfold frequency adjustedBase
  rw [Real.mul_rpow hbase.le (Real.rpow_nonneg hscale.le _), ← Real.rpow_mul hscale.le]
  rw [show (dimension : ℝ) / ((dimension : ℝ) - 2) *
      (-(2 * (index : ℝ)) / (dimension : ℝ)) =
      -(2 * (index : ℝ) / ((dimension : ℝ) - 2)) by field_simp]
  rw [Real.rpow_neg hscale.le]
  simp only [div_eq_mul_inv]

example : frequency (adjustedBase 10000 4 128) 128 7 =
    frequency 10000 128 7 / (4 : ℝ) ^ (2 * (7 : ℝ) / (128 - 2)) := by
  simpa using coordinate_frequency_adjusted (by norm_num : (0 : ℝ) < 10000)
    (by norm_num : (0 : ℝ) < 4) (by omega : 2 < 128) 7

/-- Appendix A.2: the highest frequency is fixed even when the base changes. -/
theorem highest_frequency_preserved (base scale : ℝ) (dimension : ℕ) :
    frequency (adjustedBase base scale dimension) dimension 0 =
      frequency base dimension 0 := by
  simp [frequency]

/-- Eq. (21): matching the slowest phase range follows from the frequency
identity, without asserting that a trained model uses that coordinate. -/
theorem slow_phase_preserved {base scale dimension : ℝ}
    (hbase : 0 < base) (hscale : 0 < scale) (hdimension : 2 < dimension) (distance : ℝ) :
    slowFrequency (adjustedBase base scale dimension) dimension * (scale * distance) =
      slowFrequency base dimension * distance := by
  rw [slow_frequency_adjusted hbase hscale hdimension, interpolated_phase hscale.ne']

example : slowFrequency (adjustedBase 10000 4 128) 128 * (4 * 2047) =
    slowFrequency 10000 128 * 2047 :=
  slow_phase_preserved (by norm_num) (by norm_num) (by norm_num) 2047

end Transformer.RoPE.YaRN
