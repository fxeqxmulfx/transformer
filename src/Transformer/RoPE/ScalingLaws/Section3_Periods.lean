module

public import Transformer.RoPE.Basic
public import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# Periods and the critical coordinate

Liu et al., *Scaling Laws of RoPE-based Extrapolation*, arXiv:2310.05209v2,
§3.2, Eq. (4), and LongRoPE2, arXiv:2502.20082v1, §2.2, Eqs. (6)–(7).
The exact cutoff below is an integer and remains meaningful outside the
finite head: its intersection with the actual coordinate range gives the
observed split. No perplexity or universal extrapolation guarantee follows
from the cutoff alone.
-/

@[expose] public section

namespace Transformer.RoPE.ScalingLaws

/-- The first pair index whose full period is not strictly inside the phase
window. The source's feature-dimension cutoff is twice this pair index. -/
noncomputable def criticalPair (base : ℝ) (dimension : ℕ) (window : ℝ) : ℤ :=
  ⌈((dimension : ℝ) / 2) * Real.logb base (window / (2 * Real.pi))⌉

/-- §3.2: the full period is `2 pi * base^(2i/d)`. -/
theorem frequency_period {base : ℝ} (hbase : 0 < base) (dimension index : ℕ) :
    period (frequency base dimension index) =
      2 * Real.pi * base ^ (2 * (index : ℝ) / (dimension : ℝ)) := by
  unfold period frequency
  rw [show -(2 * (index : ℝ)) / (dimension : ℝ) =
      -(2 * (index : ℝ) / (dimension : ℝ)) by ring, Real.rpow_neg hbase.le]
  simp only [div_eq_mul_inv, inv_inv]

example : period (frequency 10000 128 63) =
    2 * Real.pi * (10000 : ℝ) ^ (2 * (63 : ℝ) / 128) := by
  simpa using frequency_period (by norm_num : (0 : ℝ) < 10000) 128 63

/-- Exact continuous threshold underlying Eq. (4). This statement concerns
phase-window coverage, not training quality of the coordinate. -/
theorem period_lt_window_iff {base window : ℝ} {dimension : ℕ}
    (hbase : 1 < base) (hwindow : 0 < window) (hdimension : 0 < dimension) (index : ℕ) :
    period (frequency base dimension index) < window ↔
      (index : ℝ) < (dimension : ℝ) / 2 * Real.logb base (window / (2 * Real.pi)) := by
  have hd : (0 : ℝ) < dimension := by exact_mod_cast hdimension
  have hp : 0 < 2 * Real.pi := by positivity
  rw [frequency_period (lt_trans (by norm_num) hbase)]
  calc
    2 * Real.pi * base ^ (2 * (index : ℝ) / (dimension : ℝ)) < window
        ↔ base ^ (2 * (index : ℝ) / (dimension : ℝ)) < window / (2 * Real.pi) := by
          rw [lt_div_iff₀ hp, mul_comm]
    _ ↔ 2 * (index : ℝ) / (dimension : ℝ) < Real.logb base (window / (2 * Real.pi)) :=
      (Real.lt_logb_iff_rpow_lt hbase (div_pos hwindow hp)).symm
    _ ↔ (index : ℝ) < (dimension : ℝ) / 2 * Real.logb base (window / (2 * Real.pi)) := by
      rw [div_lt_iff₀ hd]
      constructor <;> intro h <;> nlinarith

example : period (frequency 10000 128 7) < 4096 ↔
    (7 : ℝ) < (128 : ℝ) / 2 * Real.logb 10000 (4096 / (2 * Real.pi)) := by
  simpa using period_lt_window_iff (by norm_num : (1 : ℝ) < 10000)
    (by norm_num : (0 : ℝ) < 4096) (by omega : 0 < 128) 7

/-- Eq. (4): integer rounding gives exactly the first incomplete-period
coordinate, including the boundary case where a full period equals the window. -/
theorem period_lt_window_iff_critical {base window : ℝ} {dimension : ℕ}
    (hbase : 1 < base) (hwindow : 0 < window) (hdimension : 0 < dimension) (index : ℕ) :
    period (frequency base dimension index) < window ↔
      (index : ℤ) < criticalPair base dimension window := by
  rw [period_lt_window_iff hbase hwindow hdimension, criticalPair, Int.lt_ceil]
  norm_cast

example : period (frequency 10000 128 7) < 4096 ↔
    (7 : ℤ) < criticalPair 10000 128 4096 :=
  period_lt_window_iff_critical (by norm_num) (by norm_num) (by omega) 7

end Transformer.RoPE.ScalingLaws
