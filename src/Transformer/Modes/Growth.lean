/-
# The number of modes of a Gaussian KDE — comparison of growth rates

Auxiliary comparisons of powers and logarithms, in the form the bandwidth
regimes of arXiv:2412.09080v3 are written in: `β ≪ n^{2/5}`, `n^{2/5} ≪ β`,
`β ≪ n^2/log⁶ n`, and so on.  They carry no content of the paper; they are
what the satisfiability witnesses of those regimes are built from.
-/

import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

open Filter Asymptotics
open scoped Topology

namespace Transformer
namespace Modes

/-- `x ↦ x + 1` along the naturals runs off to infinity; the sequences the
regimes are witnessed by are built on it, so that `log` and `rpow` see a
positive argument at every index. -/
theorem tendsto_natSucc_atTop : Tendsto (fun k : ℕ => (k : ℝ) + 1) atTop atTop :=
  tendsto_atTop_add_const_right _ 1 tendsto_natCast_atTop_atTop

/-- A smaller power is negligible against a larger one. -/
theorem isLittleO_rpow_rpow_atTop {a b : ℝ} (hab : a < b) :
    (fun x : ℝ => x ^ a) =o[atTop] fun x : ℝ => x ^ b := by
  have hkey : Tendsto (fun x : ℝ => x ^ a / x ^ b) atTop (nhds 0) := by
    refine (tendsto_rpow_neg_atTop (y := b - a) (by linarith)).congr' ?_
    filter_upwards [eventually_gt_atTop (0 : ℝ)] with x hx
    rw [show -(b - a) = a - b by ring, Real.rpow_sub hx]
  refine (isLittleO_iff_tendsto' ?_).mpr hkey
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with x hx h0
  exact absurd h0 (Real.rpow_pos_of_pos hx b).ne'

/-- The same comparison along `k ↦ k + 1`. -/
theorem isLittleO_rpow_rpow_nat {a b : ℝ} (hab : a < b) :
    (fun k : ℕ => ((k : ℝ) + 1) ^ a) =o[atTop] fun k : ℕ => ((k : ℝ) + 1) ^ b :=
  (isLittleO_rpow_rpow_atTop hab).comp_tendsto tendsto_natSucc_atTop

/-- A power below `2` is negligible against `x² / log⁶ x`, the upper end of
the last regime of `thm:mammen`. -/
theorem isLittleO_rpow_sq_div_log_atTop {a : ℝ} (ha : a < 2) :
    (fun x : ℝ => x ^ a) =o[atTop] fun x : ℝ => x ^ 2 / Real.log x ^ 6 := by
  have hlog : (fun x : ℝ => Real.log x ^ (6 : ℝ)) =o[atTop] fun x : ℝ => x ^ (2 - a) :=
    isLittleO_log_rpow_rpow_atTop 6 (by linarith)
  have hmul : (fun x : ℝ => x ^ a * Real.log x ^ 6) =o[atTop] fun x : ℝ => x ^ 2 := by
    have h := (isBigO_refl (fun x : ℝ => x ^ a) atTop).mul_isLittleO hlog
    refine h.congr' ?_ ?_
    · filter_upwards [eventually_gt_atTop (0 : ℝ)] with x _
      rw [show (6 : ℝ) = ((6 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    · filter_upwards [eventually_gt_atTop (0 : ℝ)] with x hx
      rw [← Real.rpow_add hx]
      norm_num
  rw [isLittleO_iff]
  intro ε hε
  rw [isLittleO_iff] at hmul
  filter_upwards [hmul hε, eventually_gt_atTop (1 : ℝ)] with x hx hx1
  have hpos : (0 : ℝ) < Real.log x ^ 6 := pow_pos (Real.log_pos hx1) 6
  rw [norm_mul, Real.norm_of_nonneg hpos.le] at hx
  rw [norm_div, Real.norm_of_nonneg hpos.le, ← mul_div_assoc, le_div_iff₀ hpos]
  exact hx

/-- The same comparison along `k ↦ k + 1`. -/
theorem isLittleO_rpow_sq_div_log_nat {a : ℝ} (ha : a < 2) :
    (fun k : ℕ => ((k : ℝ) + 1) ^ a) =o[atTop]
      fun k : ℕ => ((k : ℝ) + 1) ^ 2 / Real.log ((k : ℝ) + 1) ^ 6 :=
  (isLittleO_rpow_sq_div_log_atTop ha).comp_tendsto tendsto_natSucc_atTop

/-- A positive power runs off to infinity along `k ↦ k + 1`. -/
theorem tendsto_rpow_natSucc_atTop {a : ℝ} (ha : 0 < a) :
    Tendsto (fun k : ℕ => ((k : ℝ) + 1) ^ a) atTop atTop :=
  (tendsto_rpow_atTop ha).comp tendsto_natSucc_atTop

end Modes
end Transformer
