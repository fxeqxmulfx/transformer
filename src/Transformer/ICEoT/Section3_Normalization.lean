/-
# IC-EoT: input-dependent layer normalization can break convexity

arXiv:2603.22095v2, §3.2, paragraph after Eq. (25). The source omits
layer normalization because it need not preserve input convexity. A standard
two-channel normalization with epsilon one and unit affine scale is already
non-convex along the affine input `(t,-t)`.
-/

import Transformer.ICEoT.Section3_Encoder
import Mathlib.Analysis.Real.Sqrt

noncomputable section

namespace Transformer.ICEoT

/-- The first coordinate of ordinary two-channel layer normalization:
population mean/variance, epsilon one, unit scale, zero bias. This supplies
an explicit witness for §3.2's paragraph after Eq. (25). -/
def twoChannelNorm (x : ℝ × ℝ) : ℝ :=
  let mean := (x.1 + x.2) / 2
  (x.1 - mean) / Real.sqrt (((x.1 - mean) ^ 2 + (x.2 - mean) ^ 2) / 2 + 1)

/-- Restriction of the normalization to `(t,-t)`; §3.2, after Eq. (25). -/
def normalizationSlice (t : ℝ) : ℝ := t / Real.sqrt (t ^ 2 + 1)

/-- The actual mean/variance calculation, §3.2's layer-normalization
obstruction. The initial features are affine in `t`. -/
theorem twoChannelNorm_slice (t : ℝ) : twoChannelNorm (t, -t) = normalizationSlice t := by
  simp only [twoChannelNorm, add_neg_cancel, zero_div, sub_zero, neg_sq, normalizationSlice]
  congr 2
  ring

/-- The scalar restriction fails Jensen at `0,4/3`; §3.2, after Eq. (25).
The source's omission is justified even with positive epsilon and scale. -/
theorem normalizationSlice_not_convex : ¬ ConvexOn ℝ Set.univ normalizationSlice := by
  have hz : normalizationSlice 0 = 0 := by simp [normalizationSlice]
  have he : normalizationSlice (4 / 3) = 4 / 5 := by
    have hs : Real.sqrt (((4 / 3 : ℝ) ^ 2 + 1)) = 5 / 3 := by
      convert Real.sqrt_sq_eq_abs (5 / 3 : ℝ) using 1 <;> norm_num
    rw [normalizationSlice, hs]
    norm_num
  have hp : 0 < Real.sqrt ((2 / 3 : ℝ) ^ 2 + 1) := Real.sqrt_pos.mpr (by norm_num)
  have hs := Real.sq_sqrt (by norm_num : 0 ≤ (2 / 3 : ℝ) ^ 2 + 1)
  have hlt : Real.sqrt ((2 / 3 : ℝ) ^ 2 + 1) < 5 / 3 := by nlinarith
  have hm : (2 / 5 : ℝ) < normalizationSlice (2 / 3) := by
    rw [normalizationSlice, lt_div_iff₀ hp]
    nlinarith
  intro h
  have hj := h.2 (Set.mem_univ (0 : ℝ)) (Set.mem_univ (4 / 3 : ℝ))
    (by norm_num : (0 : ℝ) ≤ 1 / 2) (by norm_num : (0 : ℝ) ≤ 1 / 2)
    (by norm_num : (1 / 2 : ℝ) + 1 / 2 = 1)
  rw [hz, he] at hj
  norm_num at hj
  linarith

/-- Ordinary layer normalization does not preserve input convexity even
when applied to affine features; §3.2, paragraph after Eq. (25). -/
theorem twoChannelNorm_not_convex : ¬ ConvexOn ℝ Set.univ twoChannelNorm := by
  intro h
  apply normalizationSlice_not_convex
  refine ⟨convex_univ, ?_⟩
  intro x hx y hy a b ha hb hab
  have hj := h.2 (Set.mem_univ (x, -x)) (Set.mem_univ (y, -y)) ha hb hab
  have hc : a • (x, -x) + b • (y, -y) = (a * x + b * y, -(a * x + b * y)) := by
    apply Prod.ext
    · rfl
    · change a * (-x) + b * (-y) = -(a * x + b * y)
      ring
  rw [hc, twoChannelNorm_slice, twoChannelNorm_slice, twoChannelNorm_slice] at hj
  exact hj

end Transformer.ICEoT
