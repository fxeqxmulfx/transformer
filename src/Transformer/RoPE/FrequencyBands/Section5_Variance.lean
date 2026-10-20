module

public import Transformer.RoPE.ScalingLaws.Section3_Periods
public import Mathlib.Analysis.SpecialFunctions.Integrals.Basic

/-!
# Frequency-band variance and a source counterexample

Oka et al., *Frequency Bands in RoPE*, ICLR 2026, §5, Eqs. (4)–(9).
The normalized integrals are the true variance of `cos` under a uniform
phase on `[0,x]` for `x > 0`. The source's claim in Step 2 that this variance
is unimodal on `(0,infinity)` is false; an exact three-point counterexample
is proved below. The empirical predictor is not promoted to a theorem about
the learned norms of arbitrary models.
-/

@[expose] public section

open Set

namespace Transformer.RoPE.FrequencyBands

/-- §5.1: the second moment minus the square of the first moment under a
uniform phase on `[0,x]`. A positive interval length is required in its use. -/
noncomputable def cosineVariance (x : ℝ) : ℝ :=
  (∫ phase in 0..x, Real.cos phase ^ 2) / x -
    ((∫ phase in 0..x, Real.cos phase) / x) ^ 2

/-- Eq. (5), derived from actual integrals rather than taken as a definition. -/
theorem cosine_variance_formula {x : ℝ} (hx : x ≠ 0) :
    cosineVariance x = 1 / 2 + Real.sin (2 * x) / (4 * x) - (Real.sin x / x) ^ 2 := by
  simp only [cosineVariance, integral_cos_sq, integral_cos, Real.cos_zero,
    Real.sin_zero, mul_zero, sub_zero]
  rw [Real.sin_two_mul]
  field_simp
  ring

example : cosineVariance Real.pi =
    1 / 2 + Real.sin (2 * Real.pi) / (4 * Real.pi) - (Real.sin Real.pi / Real.pi) ^ 2 :=
  cosine_variance_formula Real.pi_pos.ne'

/-- Exact values at two complete half-periods surrounding the counterexample. -/
theorem variance_pi_two_pi :
    cosineVariance Real.pi = 1 / 2 ∧ cosineVariance (2 * Real.pi) = 1 / 2 := by
  constructor
  · rw [cosine_variance_formula Real.pi_pos.ne']
    norm_num
  · rw [cosine_variance_formula (by positivity : 2 * Real.pi ≠ 0)]
    rw [show 2 * (2 * Real.pi) = 2 * Real.pi + 2 * Real.pi by ring,
      Real.sin_add]
    norm_num

/-- The variance at the intervening phase is strictly smaller than at both
neighbors, contradicting §5.1 Step 2's global-unimodality assertion. -/
theorem variance_middle_dip : cosineVariance (3 * Real.pi / 2) < 1 / 2 := by
  have hsin : Real.sin (3 * Real.pi / 2) = -1 := by
    rw [show 3 * Real.pi / 2 = Real.pi + Real.pi / 2 by ring, Real.sin_add]
    norm_num
  have hcos : Real.cos (3 * Real.pi / 2) = 0 := by
    rw [show 3 * Real.pi / 2 = Real.pi + Real.pi / 2 by ring, Real.cos_add]
    norm_num
  rw [cosine_variance_formula (by positivity : 3 * Real.pi / 2 ≠ 0),
    Real.sin_two_mul, hsin, hcos]
  rw [show (-1 : ℝ) / (3 * Real.pi / 2) = -(1 / (3 * Real.pi / 2)) by ring, neg_sq]
  simp only [mul_zero, zero_div, add_zero]
  have hpositive : 0 < (1 / (3 * Real.pi / 2)) ^ 2 := by positivity
  nlinarith

/-- Counterexample to Oka et al., §5.1 Step 2: the variance is not monotone
up to a mode and antitone afterwards on the positive real axis. -/
theorem cosine_variance_not_unimodal :
    ¬ ∃ mode : ℝ, MonotoneOn cosineVariance (Ioc 0 mode) ∧
      AntitoneOn cosineVariance (Ici mode) := by
  rintro ⟨mode, hleft, hright⟩
  have hab : Real.pi ≤ 3 * Real.pi / 2 := by nlinarith [Real.pi_pos]
  have hbc : 3 * Real.pi / 2 ≤ 2 * Real.pi := by nlinarith [Real.pi_pos]
  rcases le_total (3 * Real.pi / 2) mode with h | h
  · have hle := hleft ⟨Real.pi_pos, hab.trans h⟩ ⟨by positivity, h⟩ hab
    rw [variance_pi_two_pi.1] at hle
    exact (not_le.mpr variance_middle_dip) hle
  · have hle := hright h (h.trans hbc) hbc
    rw [variance_pi_two_pi.2] at hle
    exact (not_le.mpr variance_middle_dip) hle

/-- Eq. (9): the continuous coordinate obtained by solving for a prescribed
phase span. Rounding this predictor and fitting a learned band are separate
empirical steps, and the numerical optimizer is not assumed here. -/
noncomputable def continuousBand (base dimension window phaseSpan : ℝ) : ℝ :=
  dimension / 2 * Real.logb base (window / phaseSpan)

/-- Eq. (9), exact algebraic content: the predicted continuous coordinate has
the specified phase span over the training window. -/
theorem continuous_band_matches {base dimension window phaseSpan : ℝ}
    (hbase : 1 < base) (hdimension : dimension ≠ 0)
    (hwindow : 0 < window) (hphase : 0 < phaseSpan) :
    base ^ (-(2 * continuousBand base dimension window phaseSpan) / dimension) * window =
      phaseSpan := by
  have hb : 0 < base := lt_trans (by norm_num) hbase
  unfold continuousBand
  rw [show -(2 * (dimension / 2 * Real.logb base (window / phaseSpan))) / dimension =
      -Real.logb base (window / phaseSpan) by field_simp]
  rw [Real.rpow_neg hb.le, Real.rpow_logb hb (ne_of_gt hbase)
    (div_pos hwindow hphase)]
  field_simp

example : (10000 : ℝ) ^
    (-(2 * continuousBand 10000 128 2048 Real.pi) / 128) * 2048 = Real.pi :=
  continuous_band_matches (by norm_num) (by norm_num) (by norm_num) Real.pi_pos

end Transformer.RoPE.FrequencyBands
